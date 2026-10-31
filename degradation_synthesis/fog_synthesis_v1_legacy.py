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

def guided_filter(guide, src, radius=8, eps=1e-3):
    """
    Edge-aware refinement of `src` using `guide`'s edges (He et al., guided image
    filtering). VDA smears intermediate depth values across object silhouettes
    (foreground/background "depth bleeding"), which otherwise shows up as a bright
    halo ring around any near object once transmission is computed per-pixel — the
    fog appears to hug the object's outline rather than sit uniformly in the
    background. Snapping depth back to the RGB frame's true edges removes that ring.
    """
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    mean_g = cv2.boxFilter(guide, -1, (radius, radius))
    mean_s = cv2.boxFilter(src, -1, (radius, radius))
    mean_gs = cv2.boxFilter(guide * src, -1, (radius, radius))
    cov_gs = mean_gs - mean_g * mean_s
    mean_gg = cv2.boxFilter(guide * guide, -1, (radius, radius))
    var_g = mean_gg - mean_g * mean_g
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))
    return mean_a * guide + mean_b

# ---------------------------------------------------------------------------
# 3. PHYSICAL FOG ENGINE
# ---------------------------------------------------------------------------
class PhysicalFogEngine:
    def __init__(self, beta_base=0.06, hurst=0.75):
        self.beta_base = beta_base
        self._wind_offset = 0.0
        
        # Build a 4x wide fBm field for smooth drifting in video
        self.fog_texture = generate_fbm_field(512, 2048, H=hurst, octaves=6)
        
        # Fixed neutral-grey airlight — not sampled from the frame at all. Sampling from
        # the sky region made the fog color self-referential (sky gets replaced by a
        # color derived from the sky) and, combined with the near-white default, biased
        # fog toward white. A constant grey has no dependency on scene content.
        self.smooth_color = np.array([0.78, 0.79, 0.80])

    def apply_fog(self, frame_bgr, depth_meters):
        h, w = frame_bgr.shape[:2]
        img_float = frame_bgr.astype(np.float32) / 255.0

        # Refine depth edges against the actual image before anything else consumes
        # depth_meters — see guided_filter() docstring for why this matters.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        depth_max = depth_meters.max() + 1e-6
        depth_meters = guided_filter(gray, depth_meters / depth_max, radius=8, eps=1e-3) * depth_max
        depth_meters = np.clip(depth_meters, 0, None)

        # Step 1: Spatiotemporal Update
        self._wind_offset += 0.04
        x0 = int(self._wind_offset) % (2048 - w)
        # Extract the fBm window for the current frame
        fbm = cv2.resize(self.fog_texture[:, x0 : x0 + w], (w, h))
        airlight_color = self.smooth_color

        # Step 2: Vertical Stratification (y) — dense at the road (bottom), thin at the sky (top)
        y_map = np.linspace(0.0, 1.0, h).reshape(h, 1)
        strat = np.power(y_map, 1.2)

        # Combined Beta field using fBm (Spatial Heterogeneity)
        beta_field = self.beta_base * (0.8 + 0.4 * fbm) * (0.4 + 1.2 * strat)

        # Step 3: DEFENSE - Multi-Scattering Gamma
        d_norm = depth_meters / (np.max(depth_meters) + 1e-6)
        gamma_d = 1.0 + 0.4 * d_norm
        # Clamp depth to a fixed scene ceiling before the exponent. VDA assigns sky
        # several hundred meters, well past any confident metric estimate for a bounded
        # road scene; left uncapped (or capped at a beta-dependent V=3/beta, which
        # cancels beta_field's own beta_base factor), sky transmission stops responding
        # to the fog preset entirely. A fixed ceiling keeps beta_base in control of how
        # fast that capped depth saturates, so sky brightness actually tracks intensity.
        MAX_SCENE_DEPTH = 120.0
        depth_capped = np.clip(depth_meters, 0, MAX_SCENE_DEPTH)
        transmission = np.exp(-beta_field * depth_capped)
        transmission = np.power(np.clip(transmission, 1e-5, 1.0), gamma_d)
        
        # Small blur, just to anti-alias the transmission map — real edge alignment
        # (both the near-object silhouette and any small far-background patch, e.g. sky
        # glimpsed through a gap in foliage) is handled by the guided_filter applied to
        # depth_meters above. A wide blur here mixes transmission across screen-space
        # neighbors regardless of whether they're actually at similar depth — that's not
        # Koschmieder's law (a per-pixel, per-line-of-sight quantity), it's a non-physical
        # smearing that re-introduces a halo ring around any compact near object.
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

        # Tone mapping
        foggy = np.power(np.clip(foggy, 0, 1), 0.82) * 0.88 + 0.08

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
    beta_val = args.beta if args.beta is not None else FOG_PRESETS[args.intensity]

    pipe = VDAWrapper(encoder=args.encoder, vda_root_path=vda_root, device_id=args.gpu_id)
    engine = PhysicalFogEngine(beta_base=beta_val, hurst=args.hurst)

    # No forced resize to a fixed imsize — frames keep their native resolution end to
    # end. VDAWrapper.infer_batch already downsamples internally to satisfy the model's
    # /14 input constraint and upsamples the depth back to each frame's own shape, so
    # forcing every frame through a fixed 848x480 here only added an unnecessary
    # upscale/downscale blur pass with no benefit.
    is_folder = os.path.isdir(args.input)
    if is_folder:
        img_paths = sorted(glob.glob(os.path.join(args.input, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(args.input, "*.[pP][nN][gG]")))
        os.makedirs(args.output, exist_ok=True)
        total = len(img_paths)
    else:
        cap = cv2.VideoCapture(args.input)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out_vid = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), native_size)

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
            if img is not None: batch_f.append(img)

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