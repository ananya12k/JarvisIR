
# import os
# import sys
# import cv2
# import torch
# import argparse
# import numpy as np
# from tqdm import tqdm

# # ---------------------------------------------------------------------------
# # 1. VDA WRAPPER (Unchanged)
# # ---------------------------------------------------------------------------
# class VDAWrapper:
#     def __init__(self, encoder='vits', vda_root_path=None, device_id=0):
#         self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
#         sys.path.append(vda_root_path)
#         from video_depth_anything.video_depth import VideoDepthAnything
#         model_configs = {
#             'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
#             'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
#         }
#         self.model = VideoDepthAnything(**model_configs[encoder])
#         ckpt_path = os.path.join(vda_root_path, 'checkpoints',
#                                  f'metric_video_depth_anything_{encoder}.pth')
#         self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
#         self.model = self.model.to(self.device).eval()
#         self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
#         self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

#     def infer_batch(self, frame_batch_rgb):
#         h_orig, w_orig = frame_batch_rgb[0].shape[:2]
#         target_h, target_w = (h_orig // 14) * 14, (w_orig // 14) * 14
#         frames_tensor = [
#             torch.from_numpy(cv2.resize(img, (target_w, target_h)))
#                  .permute(2, 0, 1).float() / 255.0
#             for img in frame_batch_rgb
#         ]
#         batch_tensor = (torch.stack(frames_tensor).unsqueeze(0).to(self.device)
#                         - self.mean) / self.std
#         with torch.no_grad():
#             depth_pred = self.model(batch_tensor)
#             if isinstance(depth_pred, (list, tuple)):
#                 depth_pred = depth_pred[-1]
#             depth_out = depth_pred.squeeze().cpu().numpy()
#         if depth_out.ndim == 2:
#             depth_out = depth_out[np.newaxis, ...]
#         return np.array([cv2.resize(d, (w_orig, h_orig)) for d in depth_out])


# # ---------------------------------------------------------------------------
# # 4. fBm NOISE FIELD (Smooth "Atmospheric" Noise)
# # ---------------------------------------------------------------------------
# def fbm_field(h, w, H=0.8, octaves=4, seed=42):
#     rng = np.random.default_rng(seed)
#     field = np.zeros((h, w), dtype=np.float32)
#     for k in range(octaves):
#         nh, nw = max(1, h // (2**k)), max(1, w // (2**k))
#         noise = rng.standard_normal((nh, nw)).astype(np.float32)
#         field += (0.5**k) * cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    
#     # Very heavy blur for smooth, non-synthetic transitions
#     field = cv2.GaussianBlur(field, (201, 201), 0)
#     field = (field - field.min()) / (field.max() - field.min() + 1e-8)
#     return field


# # ---------------------------------------------------------------------------
# # 6. PHYSICALLY ACCURATE FOG ENGINE (REFINED FOR REALISM)
# # ---------------------------------------------------------------------------
# class PhysicalFogEngine:
#     def __init__(self,
#                  beta_base   = 0.05,   
#                  A_inf       = 0.94,   
#                  wind_speed  = 0.03,   # Slower drift is more realistic
#                  cam_height  = 1.5):

#         self.beta_base   = beta_base
#         self.A_inf       = A_inf
#         self.wind_speed  = wind_speed
        
#         self._wind_offset = 0.0
#         self._fbm_field   = None 
#         self.fog_color   = np.array([0.90, 0.92, 0.94], dtype=np.float32) # BGR: Damp cloudy tone

#     def _get_fbm(self, h, w):
#         if self._fbm_field is None:
#             self._fbm_field = fbm_field(h, w * 4)
#         x0 = int(self._wind_offset) % (w * 3)
#         return self._fbm_field[:, x0 : x0 + w]

#     def _stratification_map(self, h, w):
#         y_coords = np.linspace(1.0, 0.0, h).reshape(h, 1)
#         # Softer stratification curve for more natural ground fog
#         strat = np.power(y_coords, 1.2)
#         return np.tile(strat, (1, w))

#     def apply_fog(self, frame_bgr, depth_meters):
#         h, w = frame_bgr.shape[:2]
#         img_float = frame_bgr.astype(np.float32) / 255.0

#         self._wind_offset += self.wind_speed
#         fbm = self._get_fbm(h, w)
#         strat = self._stratification_map(h, w)

#         # Scattering density field
#         beta_field = self.beta_base * (0.85 + 0.3 * fbm) * (0.7 + 0.3 * strat)

#         # Transmission calculation
#         transmission = np.exp(-beta_field * depth_meters)
#         transmission = np.clip(transmission, 0.0, 1.0)
        
#         # MINIMAL CHANGE 1: Smooth the transmission map to prevent hard halos around trees/signs
#         transmission = cv2.GaussianBlur(transmission, (15, 15), 0)
#         t3 = np.stack([transmission] * 3, axis=-1)

#         # MINIMAL CHANGE 2: Variable Blur (Physics-based)
#         # Real fog gets blurrier with distance. We blend multiple blur levels.
#         blur_light = cv2.GaussianBlur(img_float, (11, 11), 0)
#         blur_heavy = cv2.GaussianBlur(img_float, (31, 31), 0)
#         # Distance-based blur weight
#         blur_weight = (1.0 - transmission)[:, :, np.newaxis]
#         img_radiance = img_float * t3 + blur_light * (blur_weight * 0.5) + blur_heavy * (blur_weight * 0.5)

#         # Airlight (A)
#         A_final = self.fog_color * self.A_inf

#         # Final Composition: I = J*t + A*(1-t)
#         foggy = img_radiance * t3 + A_final * (1.0 - t3)
        
#         # MINIMAL CHANGE 3: Non-linear Shadow Lift
#         # Instead of just adding a flat value, we "compress" the blacks
#         # This gives that "milky" dashcam look without looking like a white veil
#         foggy = np.power(foggy, 0.85) * 0.9 + 0.05

#         return np.clip(foggy * 255.0, 0, 255).astype(np.uint8)


# # ---------------------------------------------------------------------------
# # 7. FOG PRESETS
# # ---------------------------------------------------------------------------
# FOG_PRESETS = {
#     'light':   {'beta': 0.012},
#     'medium':  {'beta': 0.055}, 
#     'heavy':   {'beta': 0.150}, 
#     'extreme': {'beta': 0.400}  
# }


# # ---------------------------------------------------------------------------
# # 8. MAIN
# # ---------------------------------------------------------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input',     required=True)
#     parser.add_argument('--output',    required=True)
#     parser.add_argument('--intensity', default='medium', choices=list(FOG_PRESETS.keys()))
#     parser.add_argument('--gpu_id',    type=int, default=0)
#     args = parser.parse_args()

#     vda_root = '/scratch/Ananya_Kulkarni/Video-Depth-Anything'
#     imsize   = (848, 480)

#     pipe = VDAWrapper(encoder='vits', vda_root_path=vda_root, device_id=args.gpu_id)
#     preset = FOG_PRESETS[args.intensity]
#     engine = PhysicalFogEngine(beta_base=preset['beta'])

#     cap   = cv2.VideoCapture(args.input)
#     fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     out   = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, imsize)

#     pbar   = tqdm(total=total, desc=f"Pro-Realistic Fog")
#     buffer = []

#     while True:
#         ret, frame = cap.read()
#         if not ret: break
#         buffer.append(cv2.resize(frame, imsize))
#         if len(buffer) == 4:
#             batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in buffer]
#             depths = pipe.infer_batch(batch_rgb)
#             for i in range(len(buffer)):
#                 out.write(engine.apply_fog(buffer[i], depths[i]))
#             pbar.update(4); buffer = []

#     if buffer:
#         batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in buffer]
#         depths = pipe.infer_batch(batch_rgb)
#         for i in range(len(buffer)):
#             out.write(engine.apply_fog(buffer[i], depths[i]))
#         pbar.update(len(buffer))

#     cap.release(); out.release(); pbar.close()

# if __name__ == '__main__':
#     main()

import os
import sys
import glob
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1. VDA WRAPPER (Unchanged)
# ---------------------------------------------------------------------------
class VDAWrapper:
    def __init__(self, encoder='vits', vda_root_path=None, device_id=0):
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        sys.path.append(vda_root_path)
        from video_depth_anything.video_depth import VideoDepthAnything
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        self.model = VideoDepthAnything(**model_configs[encoder])
        ckpt_path = os.path.join(vda_root_path, 'checkpoints', f'metric_video_depth_anything_{encoder}.pth')
        self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.model = self.model.to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

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

# ---------------------------------------------------------------------------
# 2. fBm GENERATOR (Physics-Based Spatially Correlated Noise)
# ---------------------------------------------------------------------------
def generate_fbm_field(h, w, H=0.7, octaves=5, seed=42):
    """
    H = Hurst exponent. 
    H ~ 0.8-0.9: Smooth radiation fog.
    H ~ 0.5-0.6: Turbulent patchy fog.
    """
    rng = np.random.default_rng(seed)
    field = np.zeros((h, w), dtype=np.float32)
    
    for i in range(octaves):
        freq = 2**i
        amp = 0.5**(i * H)
        # Create low-res noise and upscale with interpolation for smooth gradients
        nh, nw = max(1, h // freq), max(1, w // freq)
        noise = rng.standard_normal((nh, nw)).astype(np.float32)
        upsampled = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        field += amp * upsampled
        
    # Normalize to [0, 1]
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    # Final Gaussian pass to remove cubic interpolation artifacts
    return cv2.GaussianBlur(field, (15, 15), 0)

# ---------------------------------------------------------------------------
# 3. PHYSICAL FOG ENGINE
# ---------------------------------------------------------------------------
class PhysicalFogEngine:
    def __init__(self, beta_base=0.06, hurst=0.75):
        self.beta_base = beta_base
        self._wind_offset = 0.0
        
        # Build a 4x wide fBm field for smooth drifting in video
        self.fog_texture = generate_fbm_field(512, 2048, H=hurst, octaves=6)
        
        self.smooth_color = np.array([0.92, 0.93, 0.95])
        self.alpha = 0.02 

    def _get_stable_airlight(self, img_float):
        h, w = img_float.shape[:2]
        sky_region = img_float[0:int(h*0.12), int(w*0.25):int(w*0.75)]
        sampled_color = np.median(sky_region, axis=(0, 1))
        grey_val = np.mean(sampled_color)
        sampled_color = 0.75 * sampled_color + 0.25 * grey_val 
        self.smooth_color = (1 - self.alpha) * self.smooth_color + self.alpha * sampled_color
        return self.smooth_color

    def apply_fog(self, frame_bgr, depth_meters):
        h, w = frame_bgr.shape[:2]
        img_float = frame_bgr.astype(np.float32) / 255.0

        # Step 1: Spatiotemporal Update
        self._wind_offset += 0.04
        x0 = int(self._wind_offset) % (2048 - w)
        # Extract the fBm window for the current frame
        fbm = cv2.resize(self.fog_texture[:, x0 : x0 + w], (w, h))
        airlight_color = self._get_stable_airlight(img_float)

        # Step 2: Vertical Stratification (y)
        y_map = np.linspace(1.0, 0.0, h).reshape(h, 1)
        strat = np.power(y_map, 1.2)
        
        # Combined Beta field using fBm (Spatial Heterogeneity)
        beta_field = self.beta_base * (0.8 + 0.4 * fbm) * (0.6 + 0.4 * strat)

        # Step 3: DEFENSE - Multi-Scattering Gamma
        d_norm = depth_meters / (np.max(depth_meters) + 1e-6)
        gamma_d = 1.0 + 0.4 * d_norm 
        transmission = np.exp(-beta_field * depth_meters)
        transmission = np.power(np.clip(transmission, 1e-5, 1.0), gamma_d)
        
        t3 = np.stack([cv2.GaussianBlur(transmission, (15, 15), 0)] * 3, axis=-1)

        # Step 4: Volumetric Blur
        blur_val = int(max(1, 21 * self.beta_base * 10))
        if blur_val % 2 == 0: blur_val += 1
        blurred_bg = cv2.GaussianBlur(img_float, (blur_val, blur_val), 0)
        img_radiance = img_float * t3 + blurred_bg * (1.0 - t3)

        # Step 5: Final Composition
        # Airlight modulated by fBm ( correlated airlight)
        A_final = airlight_color * np.stack([fbm * 0.1 + 0.95]*3, axis=-1)
        
        # Secondary road glow (S)
        road_glow = 0.04 * np.stack([strat]*3, axis=-1) * (1.0 - t3)**2
        
        foggy = img_radiance * t3 + A_final * (1.0 - t3) + road_glow

        # Step 6: DEFENSE - Fog-Coupled Noise
        noise_sigma = 0.003 / (transmission + 0.15) 
        noise = np.random.normal(0, 1, foggy.shape).astype(np.float32) * np.stack([noise_sigma]*3, axis=-1)
        
        # Tone mapping
        foggy = np.power(np.clip(foggy + noise, 0, 1), 0.82) * 0.88 + 0.08

        return np.clip(foggy * 255.0, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------
FOG_PRESETS = {
    'light': 0.012, 'medium': 0.038, 'heavy': 0.100, 'extreme': 0.320
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--hurst', type=float, default=0.75, help="Hurst Exponent (0.5 patchy, 0.9 smooth)")
    parser.add_argument('--intensity', default='medium', choices=list(FOG_PRESETS.keys()))
    parser.add_argument('--encoder', default='vits', choices=['vits', 'vitl'])
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    vda_root = '/scratch/Ananya_Kulkarni/Video-Depth-Anything'
    imsize   = (848, 480)
    beta_val = args.beta if args.beta is not None else FOG_PRESETS[args.intensity]

    pipe = VDAWrapper(encoder=args.encoder, vda_root_path=vda_root, device_id=args.gpu_id)
    engine = PhysicalFogEngine(beta_base=beta_val, hurst=args.hurst)

    is_folder = os.path.isdir(args.input)
    if is_folder:
        img_paths = sorted(glob.glob(os.path.join(args.input, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(args.input, "*.[pP][nN][gG]")))
        os.makedirs(args.output, exist_ok=True)
        total = len(img_paths)
    else:
        cap = cv2.VideoCapture(args.input)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_vid = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), imsize)

    pbar = tqdm(total=total, desc=f"GPU {args.gpu_id} | Beta {beta_val:.3f} | H {args.hurst}")
    batch_size = 4
    
    for i in range(0, total, batch_size):
        batch_f, batch_n = [], []
        for j in range(i, min(i + batch_size, total)):
            if is_folder:
                img = cv2.imread(img_paths[j])
                batch_n.append(os.path.basename(img_paths[j]))
            else:
                ret, img = cap.read()
                if not ret: break
            if img is not None: batch_f.append(cv2.resize(img, imsize))

        if not batch_f: break
        depths = pipe.infer_batch([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_f])

        for k in range(len(batch_f)):
            res = engine.apply_fog(batch_f[k], depths[k])
            if is_folder: cv2.imwrite(os.path.join(args.output, batch_n[k]), res)
            else: out_vid.write(res)
        pbar.update(len(batch_f))

    if not is_folder: cap.release(); out_vid.release()
    pbar.close()

if __name__ == '__main__':
    main()