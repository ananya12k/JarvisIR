import os
import sys
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from types import ModuleType

# --- 1. EMERGENCY MOCK (Bypass broken torchvision) ---
mock_tv = ModuleType('torchvision')
mock_tv_trans = ModuleType('torchvision.transforms')
class DummyCompose:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, img): return img
mock_tv_trans.Compose = DummyCompose
mock_tv.transforms = mock_tv_trans
sys.modules['torchvision'] = mock_tv
sys.modules['torchvision.transforms'] = mock_tv_trans

# --- 2. MONKEY PATCH (Fix for xformers) ---
import torch.nn.functional as F
def manual_attention(q, k, v, attn_bias=None, p=0.0, scale=None):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=p, scale=scale)
try:
    import xformers.ops
    xformers.ops.memory_efficient_attention = manual_attention
except ImportError:
    x_ops = ModuleType("ops")
    x_ops.memory_efficient_attention = manual_attention
    xf = ModuleType("xformers")
    xf.ops = x_ops
    sys.modules["xformers"] = xf
    sys.modules["xformers.ops"] = x_ops

# --- 3. VDA WRAPPER ---
class VDAWrapper:
    def __init__(self, encoder='vits', vda_root_path=None, device_id=0):
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        sys.path.append(vda_root_path)
        from video_depth_anything.video_depth import VideoDepthAnything
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        self.model = VideoDepthAnything(**model_configs[encoder])
        ckpt_path = os.path.join(vda_root_path, 'checkpoints', f'metric_video_depth_anything_{encoder}.pth')
        self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.model = self.model.to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def infer_batch(self, frame_batch_rgb):
        h_orig, w_orig = frame_batch_rgb[0].shape[:2]
        target_h, target_w = (h_orig // 14) * 14, (w_orig // 14) * 14
        frames_tensor = [torch.from_numpy(cv2.resize(img, (target_w, target_h))).permute(2, 0, 1).float() / 255.0 for img in frame_batch_rgb]
        batch_tensor = (torch.stack(frames_tensor).unsqueeze(0).to(self.device) - self.mean) / self.std
        with torch.no_grad():
            depth_pred = self.model(batch_tensor)
            if isinstance(depth_pred, (list, tuple)): depth_pred = depth_pred[-1]
            depth_out = depth_pred.squeeze().cpu().numpy()
        if depth_out.ndim == 2: depth_out = depth_out[np.newaxis, ...]
        return np.array([cv2.resize(d, (w_orig, h_orig)) for d in depth_out])

# --- 4. REALISTIC STABLE FOG ENGINE ---
class FogSystem:
    def __init__(self, beta=0.5, A_val=0.95):
        self.beta = beta
        self.A_val = A_val
        # Create a STATIC haze map once to prevent flickering
        noise = np.random.normal(0.9, 0.05, (128, 128)).astype(np.float32)
        self.static_haze = cv2.GaussianBlur(noise, (61, 61), 0)

    def apply_fog(self, frame_bgr, depth_map):
        h, w = frame_bgr.shape[:2]
        img_float = frame_bgr.astype(np.float32) / 255.0
        
        # --- 1. Normalize Depth for Consistency ---
        # Map depth to a 0-1 range to ensure beta works perfectly every time
        d_min, d_max = np.min(depth_map), np.max(depth_map)
        norm_depth = (depth_map - d_min) / (d_max - d_min + 1e-6)
        
        # --- 2. Create the Near-Visibility Curve ---
        # Instead of subtracting, we use a power curve. 
        # depth^2 makes the foreground clear and the background foggy very fast.
        transmission = np.exp(-self.beta * (norm_depth * 10)) 
        transmission = np.power(transmission, 1.5) # Sharpen the foreground transition
        transmission = np.clip(transmission, 0.0, 1.0)
        t_mask = np.stack([transmission] * 3, axis=-1)
        
        # --- 3. Static Fog Texture (Zero Flickering) ---
        haze_tex = cv2.resize(self.static_haze, (w, h))
        haze_3ch = np.stack([haze_tex] * 3, axis=-1)

        # --- 4. Depth-Blur (Distance Scattering) ---
        # The further away, the blurrier it gets
        blurred_img = cv2.GaussianBlur(img_float, (31, 31), 0)

        # --- 5. Atmospheric Composition ---
        # Color: Greyish-white cloudy tone
        A_color = np.array([0.94, 0.95, 0.96]) * self.A_val * haze_3ch
        
        # Combine: Sharp foreground + Blurred background + Atmospheric Color
        # J_scattered blends between sharp and blurred
        J_scattered = img_float * t_mask + blurred_img * (1 - t_mask)
        foggy = J_scattered * t_mask + A_color * (1 - t_mask)
        
        # Raise shadows (fog isn't pitch black)
        foggy = foggy * 0.9 + 0.08
        
        return np.clip(foggy * 255.0, 0, 255).astype(np.uint8)

# --- 5. EXECUTION ---
FOG_PRESETS = {
    'light':  {'beta': 0.1},
    'medium': {'beta': 0.4},
    'heavy':  {'beta': 1.2}, 
    'extreme': {'beta': 3.5} 
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--intensity', default='heavy', choices=['light', 'medium', 'heavy', 'extreme'])
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    vda_root = '/scratch/Ananya_Kulkarni/Video-Depth-Anything'
    imsize = (848, 480)
    
    pipe = VDAWrapper(encoder='vits', vda_root_path=vda_root, device_id=args.gpu_id)
    preset = FOG_PRESETS[args.intensity]
    engine = FogSystem(beta=preset['beta'])

    cap = cv2.VideoCapture(args.input)
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, imsize)
    
    pbar = tqdm(total=total, desc=f"Synthesizing Realistic Fog")
    buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        buffer.append(cv2.resize(frame, imsize))
        if len(buffer) == 4:
            batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in buffer]
            depths = pipe.infer_batch(batch_rgb)
            for i in range(len(buffer)):
                out.write(engine.apply_fog(buffer[i], depths[i]))
            pbar.update(4); buffer = []

    if buffer:
        batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in buffer]
        depths = pipe.infer_batch(batch_rgb)
        for i in range(len(buffer)):
            out.write(engine.apply_fog(buffer[i], depths[i]))
        pbar.update(len(buffer))

    cap.release(); out.release(); pbar.close()

if __name__ == '__main__':
    main()