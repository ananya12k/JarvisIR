# import os
# import sys
# import cv2
# import torch
# import argparse
# import numpy as np
# from tqdm import tqdm

# # ---------------------------------------------------------------------------
# # 1. VDA WRAPPER  (unchanged — your pipeline is correct)
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
#         target_h = (h_orig // 14) * 14
#         target_w = (w_orig // 14) * 14
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
# # 4. fBm NOISE FIELD
# #    UPDATED: Added GaussianBlur to remove dots and make it "cloudy"
# # ---------------------------------------------------------------------------
# def fbm_field(h, w, H=0.7, octaves=6, seed=None):
#     rng = np.random.default_rng(seed)
#     field = np.zeros((h, w), dtype=np.float32)
#     freq  = 1.0
#     amp   = 1.0
#     norm  = 0.0
#     for k in range(octaves):
#         nh = max(1, int(h / freq))
#         nw = max(1, int(w / freq))
#         noise = rng.standard_normal((nh, nw)).astype(np.float32)
#         upsampled = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
#         field += amp * upsampled
#         norm  += amp
#         freq  *= 2.0
#         amp   *= (0.5 ** H)
#     field /= norm
    
#     # MINIMAL CHANGE: Blur the noise field to make it uniform atmospheric volume
#     field = cv2.GaussianBlur(field, (101, 101), 0)
    
#     field = (field - field.min()) / (field.max() - field.min() + 1e-8)
#     return field


# # ---------------------------------------------------------------------------
# # 5. ORNSTEIN-UHLENBECK PROCESS (Unchanged)
# # ---------------------------------------------------------------------------
# class OrnsteinUhlenbeck:
#     def __init__(self, mu, theta=0.05, sigma=0.015):
#         self.mu    = mu
#         self.theta = theta
#         self.sigma = sigma
#         self.x     = mu

#     def step(self):
#         dx = self.theta * (self.mu - self.x) + self.sigma * np.random.randn()
#         self.x = np.clip(self.x + dx, self.mu * 0.3, self.mu * 2.5)
#         return self.x


# # ---------------------------------------------------------------------------
# # 6. PHYSICALLY ACCURATE FOG ENGINE
# #    UPDATED: Added Scattering Blur and cleaned up sensor noise
# # ---------------------------------------------------------------------------
# class PhysicalFogEngine:
#     def __init__(self,
#                  beta_base   = 0.06,
#                  A_inf       = 0.92,
#                  H_scale     = 20.0,
#                  hurst       = 0.72,
#                  fog_octaves = 6,
#                  k_gamma     = 0.4,
#                  S_strength  = 0.04,
#                  noise_sigma = 0.001,  # MINIMAL CHANGE: Reduced from 0.008 to remove dots
#                  wind_speed  = 0.3,
#                  ou_theta    = 0.04,
#                  ou_sigma    = 0.008,
#                  cam_height_m= 1.4,
#                  fov_v_deg   = 50.0,
#                  ):

#         self.A_inf       = A_inf
#         self.H_scale     = H_scale
#         self.hurst       = hurst
#         self.fog_octaves = fog_octaves
#         self.k_gamma     = k_gamma
#         self.S_strength  = S_strength
#         self.noise_sigma = noise_sigma
#         self.wind_speed  = wind_speed
#         self.cam_height  = cam_height_m
#         self.fov_v       = np.radians(fov_v_deg)

#         self.ou = OrnsteinUhlenbeck(mu=beta_base, theta=ou_theta, sigma=ou_sigma)
#         self._fbm_seed    = 42
#         self._fbm_field   = None
#         self._wind_offset = 0.0
#         self._frame_count = 0

#     def _get_fbm(self, h, w):
#         if (self._fbm_field is None
#                 or self._fbm_field.shape[0] < h
#                 or self._fbm_field.shape[1] < w * 3):
#             self._fbm_field = fbm_field(h, w * 3, H=self.hurst,
#                                         octaves=self.fog_octaves,
#                                         seed=self._fbm_seed)
#         x0 = int(self._wind_offset) % w
#         crop = np.take(self._fbm_field, np.arange(x0, x0 + w) % (w * 3), axis=1)
#         return crop

#     def _stratification_map(self, h, w):
#         horizon_row = int(h * 0.42)
#         rows = np.arange(h, dtype=np.float32)
#         pitch_rad = (rows - horizon_row) / h * self.fov_v
#         with np.errstate(divide='ignore', invalid='ignore'):
#             y_world = np.where(pitch_rad > 0.01,
#                                self.cam_height / np.tan(pitch_rad + 1e-6),
#                                self.cam_height + 50.0)
#         y_world = np.clip(y_world, 0.0, 200.0)
#         strat_weight = np.exp(-y_world / self.H_scale).astype(np.float32)
#         return np.tile(strat_weight[:, np.newaxis], (1, w))

#     def apply_fog(self, frame_bgr, depth_map_meters):
#         h, w = frame_bgr.shape[:2]
#         img_float = frame_bgr.astype(np.float32) / 255.0

#         beta_t = self.ou.step()
#         self._wind_offset += self.wind_speed
#         self._frame_count += 1

#         fbm  = self._get_fbm(h, w)
#         strat = self._stratification_map(h, w)
#         beta_field = beta_t * (0.6 + 0.8 * fbm) * (0.4 + 0.6 * strat)

#         d = np.clip(depth_map_meters, 0.1, 300.0).astype(np.float32)
#         transmission = np.exp(-beta_field * d)
#         transmission = np.clip(transmission, 0.0, 1.0)

#         # MINIMAL CHANGE: Add scattering blur. 
#         # Real fog acts as a low-pass filter. 
#         blurred_img = cv2.GaussianBlur(img_float, (21, 21), 0)

#         d_max = float(np.percentile(d, 95))
#         gamma_map = 1.0 + self.k_gamma * (d / (d_max + 1e-6))
#         t_gamma = np.power(transmission + 1e-8, gamma_map)
#         t3 = t_gamma[:, :, np.newaxis]

#         # MINIMAL CHANGE: Scene radiance J is now a blend between sharp and blurred based on depth
#         # This makes the background uniform and "foggy" instead of sharp white
#         img_radiance = img_float * t3 + blurred_img * (1.0 - t3)

#         beta_d = beta_field * d + 1e-6
#         A_weight = self.A_inf * (1.0 - np.exp(-beta_d)) / beta_d
#         A_weight *= (0.85 + 0.3 * fbm)
#         A_weight  = np.clip(A_weight, 0.0, 1.0)

#         range_factor = np.clip(d / (d_max + 1e-6), 0.0, 1.0)[:, :, np.newaxis]
#         A_color_base = np.array([0.91, 0.93, 0.96], dtype=np.float32)
#         A_color = A_color_base * (1.0 - 0.04 * range_factor)
#         A_final = A_weight[:, :, np.newaxis] * A_color

#         road_mask = np.clip(strat * 1.5, 0.0, 1.0)[:, :, np.newaxis]
#         S = self.S_strength * road_mask * A_color

#         # MINIMAL CHANGE: Use the blended radiance
#         foggy = img_radiance + A_final * (1.0 - t3) + S * (1.0 - transmission[:, :, np.newaxis])**2

#         # Step 8: Reduced noise impact significantly
#         noise_scale = self.noise_sigma / (transmission[:, :, np.newaxis] + 0.15)
#         noise = np.random.normal(0.0, 1.0, foggy.shape).astype(np.float32) * noise_scale
#         foggy = foggy + noise

#         foggy = foggy * 0.92 + 0.06
#         return np.clip(foggy * 255.0, 0, 255).astype(np.uint8)


# # ---------------------------------------------------------------------------
# # 7. FOG PRESETS (Unchanged)
# # ---------------------------------------------------------------------------
# FOG_PRESETS = {
#     'light':   {'beta_base': 0.010, 'hurst': 0.82, 'H_scale': 40.0, 'k_gamma': 0.2},
#     'medium':  {'beta_base': 0.040, 'hurst': 0.72, 'H_scale': 25.0, 'k_gamma': 0.35},
#     'heavy':   {'beta_base': 0.120, 'hurst': 0.65, 'H_scale': 12.0, 'k_gamma': 0.50},
#     'extreme': {'beta_base': 0.380, 'hurst': 0.58, 'H_scale':  6.0, 'k_gamma': 0.65},
# }


# # ---------------------------------------------------------------------------
# # 8. MAIN (Unchanged)
# # ---------------------------------------------------------------------------
# def main():
#     parser = argparse.ArgumentParser(description='Physically accurate fog synthesis for ego-centric video')
#     parser.add_argument('--input',     required=True,  help='Input video path')
#     parser.add_argument('--output',    required=True,  help='Output video path')
#     parser.add_argument('--intensity', default='heavy',
#                         choices=list(FOG_PRESETS.keys()))
#     parser.add_argument('--encoder',   default='vits', choices=['vits', 'vitl'])
#     parser.add_argument('--gpu_id',    type=int, default=0)
#     parser.add_argument('--wind',      type=float, default=0.4,
#                         help='Fog field drift speed (pixels/frame). 0 = static fog.')
#     parser.add_argument('--hurst',     type=float, default=None,
#                         help='Override Hurst exponent (0.5=patchy, 0.9=smooth)')
#     args = parser.parse_args()

#     vda_root = '/scratch/Ananya_Kulkarni/Video-Depth-Anything'
#     imsize   = (848, 480)

#     print(f"Loading Video Depth Anything ({args.encoder})...")
#     pipe = VDAWrapper(encoder=args.encoder, vda_root_path=vda_root, device_id=args.gpu_id)

#     preset = dict(FOG_PRESETS[args.intensity])
#     if args.hurst is not None:
#         preset['hurst'] = args.hurst

#     engine = PhysicalFogEngine(
#         beta_base   = preset['beta_base'],
#         H_scale     = preset['H_scale'],
#         hurst       = preset['hurst'],
#         k_gamma     = preset['k_gamma'],
#         wind_speed  = args.wind,
#     )

#     cap   = cv2.VideoCapture(args.input)
#     fps   = cap.get(cv2.CAP_PROP_FPS)
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     out   = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, imsize)

#     pbar   = tqdm(total=total, desc="Synthesising fog")
#     buffer = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         buffer.append(cv2.resize(frame, imsize))

#         if len(buffer) == 4:
#             batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in buffer]
#             depths    = pipe.infer_batch(batch_rgb)
#             for i in range(len(buffer)):
#                 out.write(engine.apply_fog(buffer[i], depths[i]))
#             pbar.update(4)
#             buffer = []

#     if buffer:
#         batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in buffer]
#         depths    = pipe.infer_batch(batch_rgb)
#         for i in range(len(buffer)):
#             out.write(engine.apply_fog(buffer[i], depths[i]))
#         pbar.update(len(buffer))

#     cap.release()
#     out.release()
#     pbar.close()

# if __name__ == '__main__':
#     main()

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
# 1. VDA WRAPPER (Standard)
# ---------------------------------------------------------------------------
class VDAWrapper:
    def __init__(self, encoder='vits', vda_root_path=None, device_id=0):
        use_cuda = torch.cuda.is_available() and device_id < torch.cuda.device_count()
        self.device = torch.device(f'cuda:{device_id}' if use_cuda else 'cpu')
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
# 2. PHYSICAL FOG ENGINE (Adaptive + Temporally Stable)
# ---------------------------------------------------------------------------
class PhysicalFogEngine:
    def __init__(self, beta_base=0.06):
        self.beta_base = beta_base
        self._wind_offset = 0.0
        self._fbm_field = None 
        # Persistent noise for fog wisps
        noise = np.random.normal(0.85, 0.1, (128, 512)).astype(np.float32)
        self.fog_texture = cv2.GaussianBlur(noise, (61, 61), 0)
        
        # Temporal Color Stability (Exponential Moving Average)
        self.smooth_color = np.array([0.92, 0.93, 0.95]) # Default Neutral
        self.alpha = 0.02 # How fast the color changes (Very slow = Stable)

    def _get_stable_airlight(self, img_float):
        """Samples horizon using Median + EMA + Neutral Constraint"""
        h, w = img_float.shape[:2]
        # Sample only the center-top sky (ignores side objects/signs)
        sky_region = img_float[0:int(h*0.12), int(w*0.2):int(w*0.8)]
        
        # Use Median to ignore colorful signs/objects
        sampled_color = np.median(sky_region, axis=(0, 1))
        
        # Neutral Constraint: Pull color toward grey if it's too saturated
        grey_val = np.mean(sampled_color)
        sampled_color = 0.7 * sampled_color + 0.3 * grey_val 
        
        # Temporal Smoothing: EMA prevents sudden jumps
        self.smooth_color = (1 - self.alpha) * self.smooth_color + self.alpha * sampled_color
        return self.smooth_color

    def apply_fog(self, frame_bgr, depth_meters):
        h, w = frame_bgr.shape[:2]
        img_float = frame_bgr.astype(np.float32) / 255.0

        # 1. Update Spatiotemporal state
        self._wind_offset += 0.03
        x0 = int(self._wind_offset) % (512 - 128)
        fbm = cv2.resize(self.fog_texture[:, x0:x0+128], (w, h))

        # 2. Get Stable Fog Color (Adaptive but stable)
        airlight_color = self._get_stable_airlight(img_float)

        # 3. Stratification & Physics Beta
        y_map = np.linspace(1.0, 0.0, h).reshape(h, 1)
        strat = np.power(y_map, 1.2) # Thicker at road level
        beta_field = self.beta_base * (0.85 + 0.3 * fbm) * (0.7 + 0.3 * strat)

        # 4. Transmission Calculation
        # Metric depth multiplication
        transmission = np.exp(-beta_field * depth_meters)
        transmission = np.clip(transmission, 0.0, 1.0)
        
        # Anti-Halo: Blur the transmission mask
        t3 = np.stack([cv2.GaussianBlur(transmission, (15, 15), 0)] * 3, axis=-1)

        # 5. Volumetric Scattering (Distance-based Blur)
        blurred_bg = cv2.GaussianBlur(img_float, (25, 25), 0)
        img_radiance = img_float * t3 + blurred_bg * (1.0 - t3)

        # 6. Composition
        foggy = img_radiance * t3 + (airlight_color * np.stack([fbm]*3, axis=-1)) * (1.0 - t3)

        # 7. Tone-Mapping (Shadow Lift + Contrast Compression)
        # This gives the Dashcam 'Damp' look
        foggy = np.power(foggy, 0.82) * 0.88 + 0.08

        return np.clip(foggy * 255.0, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# 3. PRESETS & MAIN
# ---------------------------------------------------------------------------
FOG_PRESETS = {
    'light':   {'beta': 0.015},
    'medium':  {'beta': 0.035}, 
    'heavy':   {'beta': 0.075}, 
    'extreme': {'beta': 0.100}  
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Video file or folder of images")
    parser.add_argument('--output', required=True, help="Output video path or folder")
    parser.add_argument('--beta', type=float, default=None, help="Specific beta value (Overrides intensity)")
    parser.add_argument('--intensity', default='medium', choices=list(FOG_PRESETS.keys()))
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    vda_root = '/scratch/Ananya_Kulkarni/Video-Depth-Anything'
    imsize   = (848, 480)

    # Use beta if provided, otherwise use preset
    beta_val = args.beta if args.beta is not None else FOG_PRESETS[args.intensity]

    pipe = VDAWrapper(encoder='vits', vda_root_path=vda_root, device_id=args.gpu_id)
    engine = PhysicalFogEngine(beta_base=beta_val)

    # Determine if input is a folder of images or a video file
    is_folder = os.path.isdir(args.input)
    
    if is_folder:
        img_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        frame_paths = []
        for ext in img_extensions:
            frame_paths.extend(glob.glob(os.path.join(args.input, ext)))
        frame_paths = sorted(frame_paths)
        total_frames = len(frame_paths)
        os.makedirs(args.output, exist_ok=True)
    else:
        cap = cv2.VideoCapture(args.input)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, imsize)

    pbar = tqdm(total=total_frames, desc=f"Fogging Beta={beta_val:.3f}")
    batch_size = 4
    
    for i in range(0, total_frames, batch_size):
        batch_frames = []
        batch_filenames = []
        
        # Load Batch
        for j in range(i, min(i + batch_size, total_frames)):
            if is_folder:
                frame = cv2.imread(frame_paths[j])
                batch_filenames.append(os.path.basename(frame_paths[j]))
            else:
                ret, frame = cap.read()
                if not ret: break
            
            if frame is not None:
                batch_frames.append(cv2.resize(frame, imsize))

        if not batch_frames: break

        # Inference
        batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_frames]
        depths = pipe.infer_batch(batch_rgb)

        # Process and Save
        for k in range(len(batch_frames)):
            foggy_res = engine.apply_fog(batch_frames[k], depths[k])
            
            if is_folder:
                cv2.imwrite(os.path.join(args.output, batch_filenames[k]), foggy_res)
            else:
                out.write(foggy_res)
        
        pbar.update(len(batch_frames))

    if not is_folder: cap.release(); out.release()
    pbar.close()

if __name__ == '__main__':
    main()
