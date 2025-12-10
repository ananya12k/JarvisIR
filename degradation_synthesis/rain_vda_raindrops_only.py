# File: JarvisIR/degradation_synthesis/rain_syn_vda.py

import sys
import os
import cv2
import torch
import argparse
import numpy as np
import yaml
from tqdm import tqdm
from torchvision import transforms 
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__)) 
jarvis_ir_root = os.path.dirname(current_dir)
sys.path.append(current_dir)

possible_vda_paths = [
    '/scratch/Ananya_Kulkarni/Video-Depth-Anything',
    os.path.abspath(os.path.join(jarvis_ir_root, '../../Video-Depth-Anything')),
    os.path.abspath(os.path.join(jarvis_ir_root, '../Video-Depth-Anything'))
]

vda_root = None
for path in possible_vda_paths:
    if os.path.exists(os.path.join(path, 'video_depth_anything')):
        vda_root = path
        break

if vda_root is None:
    raise FileNotFoundError("Could not find Video-Depth-Anything.")

sys.path.append(vda_root)

from video_depth_anything.video_depth import VideoDepthAnything
from rainy.GuidedDisent.MUNIT.model_infer import MUNIT_infer
from rainy.GuidedDisent.droprenderer import DropModel
from rain_3d import Rain3DSystem, apply_global_darkening

# Tuned presets for "Bokeh/Blob" rain
RAIN_PRESETS = {
    'light':  {'density': 2000},
    'medium': {'density': 5000},
    'heavy':  {'density': 10000},
    'storm':  {'density': 20000}
}

class VDAWrapper:
    def __init__(self, encoder='vitl', vda_root_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        print(f"Loading VDA Metric Model ({encoder})...")
        self.model = VideoDepthAnything(**model_configs[encoder])
        ckpt_path = os.path.join(vda_root_path, 'checkpoints', f'metric_video_depth_anything_{encoder}.pth')
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing weights: {ckpt_path}")
            
        self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.model = self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def infer_batch(self, frame_batch_rgb):
        h_orig, w_orig = frame_batch_rgb[0].shape[:2]
        target_h = (h_orig // 14) * 14
        target_w = (w_orig // 14) * 14
        
        frames_tensor = []
        for img in frame_batch_rgb:
            if img.shape[0] != target_h or img.shape[1] != target_w:
                img = cv2.resize(img, (target_w, target_h))
            im_pil = Image.fromarray(img)
            t_img = self.transform(im_pil)
            frames_tensor.append(t_img)
            
        batch_tensor = torch.stack(frames_tensor).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            depth_pred = self.model(batch_tensor)
            if isinstance(depth_pred, (list, tuple)): depth_pred = depth_pred[-1]
            if depth_pred.ndim == 4 and depth_pred.shape[0] == 1: depth_pred = depth_pred.squeeze(0)
            if depth_pred.ndim == 4 and depth_pred.shape[1] == 1: depth_pred = depth_pred.squeeze(1)
            depth_out = depth_pred.cpu().numpy()

        final_depths = []
        for d in depth_out:
            if d.ndim > 2: d = d.squeeze()
            d = d.astype(np.float32)
            if d.shape[0] != h_orig or d.shape[1] != w_orig:
                d = cv2.resize(d, (int(w_orig), int(h_orig)), interpolation=cv2.INTER_LINEAR)
            final_depths.append(d)
        return np.array(final_depths)

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda')
        
        self.depth_model = VDAWrapper(encoder=config['vda_encoder'], vda_root_path=vda_root)
        self.munit = self._load_munit()
        self.drop_model = DropModel(imsize=(config['imsize'][1], config['imsize'][0]))
        self.preset = RAIN_PRESETS[config['intensity']]
        fov = 90.0 if config['wide_angle'] else 60.0
        
        self.rain_engine = Rain3DSystem(
            image_shape=(config['imsize'][1], config['imsize'][0]),
            max_particles=self.preset['density'],
            max_depth=50.0,
            fov_deg=fov
        )

    def _load_munit(self):
        with open(self.config['params_path'], 'r') as f: hparams = yaml.load(f, Loader=yaml.FullLoader)
        model = MUNIT_infer(hparams)
        weights = torch.load(self.config['weights_path'])
        model.gen_a.load_state_dict(weights['a'])
        model.gen_b.load_state_dict(weights['b'])
        return model.to(self.device).eval()

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, self.config['imsize'])
        
        car_v = np.array([0, 0, self.config['speed'] / 3.6])
        wind_v = np.array([self.config['wind'] / 3.6, 0, 0])
        
        batch_size = 8 
        frame_buffer = []
        
        print(f"Processing {input_path} -> {output_path} [{self.config['intensity']}]")
        pbar = tqdm(total=total_frames)
        
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.config['imsize'])
                frame_buffer.append(frame)
            
            if len(frame_buffer) == batch_size or (not ret and len(frame_buffer) > 0):
                batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frame_buffer]
                depth_maps = self.depth_model.infer_batch(batch_rgb)
                
                for i, frame_bgr in enumerate(frame_buffer):
                    f_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    t_input = transforms.ToTensor()(Image.fromarray(f_rgb)).unsqueeze(0).to(self.device)
                    with torch.no_grad(): wet_t = self.munit.forward(t_input)
                    wet_img = (wet_t[0].cpu().permute(1, 2, 0).numpy() + 1) / 2.0 * 255.0
                    wet_img_bgr = cv2.cvtColor(wet_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    
                    dark_img = apply_global_darkening(wet_img_bgr, factor=0.85)
                    self.rain_engine.update(1.0/fps, car_v, wind_v)
                    
                    final_comp = self.rain_engine.render(dark_img, depth_maps[i], {})
                    
                    c_rgb = cv2.cvtColor(final_comp, cv2.COLOR_BGR2RGB)
                    c_t = (transforms.ToTensor()(c_rgb) * 2.0 - 1.0).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        sigma = torch.tensor([3.0]).to(self.device)
                        final_t = self.drop_model.add_drops(c_t, sigma=sigma)
                    
                    final = ((final_t[0].cpu().permute(1, 2, 0).numpy() + 1) / 2.0 * 255.0).astype(np.uint8)
                    out.write(cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
                    pbar.update(1)
                
                frame_buffer = []
            
            if not ret: break
            
        cap.release()
        out.release()
        pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--speed', type=float, default=60)
    parser.add_argument('--wind', type=float, default=20)
    parser.add_argument('--intensity', type=str, default='medium', choices=['light', 'medium', 'heavy', 'storm'])
    parser.add_argument('--encoder', type=str, default='vitl')
    parser.add_argument('--wide_angle', action='store_true')
    args = parser.parse_args()
    
    rainy_dir = os.path.join(current_dir, 'rainy')
    
    config = {
        'imsize': (848, 480),
        'params_path': os.path.join(rainy_dir, 'GuidedDisent/configs/params_net.yaml'),
        'weights_path': os.path.join(rainy_dir, 'GuidedDisent/weights/pretrained.pth'),
        'speed': args.speed,
        'wind': args.wind,
        'intensity': args.intensity,
        'vda_encoder': args.encoder,
        'wide_angle': args.wide_angle
    }
    
    pipeline = Pipeline(config)
    pipeline.process_video(args.input, args.output)