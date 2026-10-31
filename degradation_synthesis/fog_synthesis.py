"""
Physically-based video fog synthesis engine (EDFog / Defog_balanced_fogv2).

REBUILD (2026-11-01) of the engine that was purged from disk. It is written to
match the paper's Dataset section and the exact interface the surviving callers
use (VISIA_old/dataset_ablation/scripts/regen_from_manifest.py, regen_ladder.py):

    PhysicalFogEngine(beta_base, airlight_color, fog_layer_m, heterogeneity,
                      fbm_seed, speed_kmh, fps, scatter_frac_max=..., patchiness=...)
    engine.apply_fog(frame_bgr_uint8, depth_meters) -> foggy_bgr_uint8
    engine.traveled_m                      # cumulative ego distance (world advection)

It is NOT bit-identical to the original. Same physics and parameters, different
implementation details (noise hash, flow->speed conversion, filter radii).

Model (paper eq. numbers in brackets):
  d      = clip(GF(gray, d_hat / s_bar) * s_bar, 0, d_max)                [depth]
  X_c,Y_c= pinhole back-projection, FOV 65 deg                            [backproj]
  psi    = 1 if h<=0 else exp(-h / H_fog),  h = h_cam - Y_c               [strat]
  Phi    = 3D fBm value noise at (X_c, Y_c, d + s_ego) / ell              [fbm]
  b~     = b0 [1 + rho (2 Phi - 1)] (0.4 + 1.2 psi);  b = b~ b0 / mean(b~) [beta, renorm]
  t      = exp(-b d)                                                      [transmission]
  G      = (1-m) G9*J + m G13*J,  m = 1 - t                               [apsf]
  A_eff  = A0 (1 + 0.1 psi)                                               [airlight]
  I      = J t + G eta (1-t) + A_eff (1-eta)(1-t)                         [compose]
Weights sum to 1, no tone map: t == 1 gives I == J exactly.
"""
import os
import sys
import glob
import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 1. VDA WRAPPER
# ---------------------------------------------------------------------------
class VDAWrapper:
    def __init__(self, encoder='vits', vda_root_path=None, device_id=0):
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        sys.path.append(vda_root_path)
        from video_depth_anything.video_depth import VideoDepthAnything
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
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


def iter_scene_depth(vda, frame_paths, batch_size=4, overlap=2):
    """
    Yields (index, frame_bgr, depth) in order. Overlapping sliding window (window
    `batch_size`, overlap `overlap`) so VDA's temporal head never loses context at a
    batch boundary; only the newly covered frames' depth is kept. Streaming (not a
    list) so a 3500-frame full-res scene never has to sit in RAM as float depth.
    """
    n = len(frame_paths)
    window_frames, window_idxs = [], []
    first = True
    pos = 0
    while pos < n:
        n_new = batch_size if first else max(1, batch_size - overlap)
        new_idxs = list(range(pos, min(pos + n_new, n)))
        new_frames = [cv2.imread(frame_paths[j]) for j in new_idxs]
        if first:
            window_frames, window_idxs = new_frames, new_idxs
        else:
            window_frames = (window_frames[-overlap:] + new_frames) if overlap else new_frames
            window_idxs = (window_idxs[-overlap:] + new_idxs) if overlap else new_idxs
        batch_depths = vda.infer_batch([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in window_frames])
        keep_start = 0 if first else len(window_idxs) - len(new_idxs)
        for k in range(keep_start, len(window_idxs)):
            yield window_idxs[k], window_frames[k], batch_depths[k]
        pos += len(new_idxs)
        first = False


# ---------------------------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------------------------
def guided_filter(guide, src, radius=8, eps=1e-3):
    """
    Edge-aware refinement of `src` using `guide`'s edges (He et al., guided image
    filtering). VDA smears depth across object silhouettes; once transmission is
    computed per pixel that shows up as a bright halo ring around near objects.
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


def _hash01(ix, iy, iz, seed):
    """Integer lattice hash -> uniform [0,1). uint32 arithmetic wraps by design."""
    h = (ix.astype(np.uint32) * np.uint32(0x8DA6B343)) \
        ^ (iy.astype(np.uint32) * np.uint32(0xD8163841)) \
        ^ (iz.astype(np.uint32) * np.uint32(0xCB1AB31F)) \
        ^ np.uint32(seed)
    h ^= h >> np.uint32(13)
    h *= np.uint32(0x5BD1E995)
    h ^= h >> np.uint32(15)
    return h.astype(np.float32) * np.float32(1.0 / 4294967296.0)


def value_noise_3d(x, y, z, seed):
    """Trilinearly-interpolated hashed value noise (smoothstep fade), in [0,1)."""
    x0, y0, z0 = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = x - x0, y - y0, z - z0
    fx, fy, fz = (f * f * (3.0 - 2.0 * f) for f in (fx, fy, fz))
    ix, iy, iz = x0.astype(np.int64), y0.astype(np.int64), z0.astype(np.int64)
    out = np.zeros_like(x, dtype=np.float32)
    for dx in (0, 1):
        wx = fx if dx else 1.0 - fx
        for dy in (0, 1):
            wy = fy if dy else 1.0 - fy
            for dz in (0, 1):
                wz = fz if dz else 1.0 - fz
                out += wx * wy * wz * _hash01(ix + dx, iy + dy, iz + dz, seed)
    return out


# ---------------------------------------------------------------------------
# 3. PHYSICAL FOG ENGINE
# ---------------------------------------------------------------------------
# Paper Tab. "auxiliary synthesis parameters": (rho, octaves, correlation length m)
HETEROGENEITY_MODES = {
    'uniform': dict(patchiness=0.03, octaves=2, corr_len_m=100.0),
    'patchy':  dict(patchiness=0.15, octaves=3, corr_len_m=180.0),
}


class PhysicalFogEngine:
    def __init__(self, beta_base=0.03, airlight_color=(0.72, 0.72, 0.72), fog_layer_m=20.0,
                 heterogeneity='uniform', fbm_seed=None, speed_kmh=0.0, fps=30.0,
                 scatter_frac_max=0.08, patchiness=None, hurst=None,
                 fov_deg=65.0, cam_height_m=1.5, max_depth_m=120.0, road_glow=0.1,
                 depth_scale_ema=0.9, speed_ema=0.8, noise_downsample=4):
        if heterogeneity not in HETEROGENEITY_MODES:
            raise ValueError(f"heterogeneity must be one of {list(HETEROGENEITY_MODES)}")
        mode = HETEROGENEITY_MODES[heterogeneity]
        self.beta_base = float(beta_base)
        self.airlight_color = np.asarray(airlight_color, dtype=np.float32).reshape(1, 1, 3)
        self.fog_layer_m = float(fog_layer_m)
        self.heterogeneity = heterogeneity
        self.patchiness = mode['patchiness'] if patchiness is None else float(patchiness)
        self.octaves = mode['octaves']
        self.corr_len_m = mode['corr_len_m']
        self.scatter_frac = float(np.clip(scatter_frac_max, 0.0, 0.08))
        self.fov_deg = float(fov_deg)
        self.cam_height_m = float(cam_height_m)
        self.max_depth_m = float(max_depth_m)
        self.road_glow = float(road_glow)
        self.depth_scale_ema = float(depth_scale_ema)
        self.speed_ema = float(speed_ema)
        self.noise_downsample = max(1, int(noise_downsample))
        self.fps = float(fps)
        # speed_kmh > 0 -> constant ego speed; 0 (what every manifest row records)
        # -> per-frame optical-flow estimate, as in the paper.
        self.fixed_speed_mpf = float(speed_kmh) / 3.6 / self.fps if speed_kmh > 0 else None

        # Everything random about the noise field derives from fbm_seed, so a
        # manifest row (which stores fbm_seed, not Hurst) fully determines a clip.
        # fbm_seed=None -> OS entropy (regen_ladder L3 relies on this).
        if fbm_seed is None:
            fbm_seed = int(np.random.SeedSequence().generate_state(1)[0])
        self.fbm_seed = int(fbm_seed) & 0xFFFFFFFF
        rng = np.random.default_rng(self.fbm_seed)
        self.hurst = float(rng.uniform(0.5, 0.9)) if hurst is None else float(hurst)
        self.octave_offsets = rng.uniform(0.0, 1000.0, size=(self.octaves, 3)).astype(np.float32)
        self.octave_seeds = rng.integers(0, 2**32, size=self.octaves, dtype=np.uint64)

        self.traveled_m = 0.0
        self._depth_scale = None
        self._speed_mpf = 0.0
        self._prev_gray_small = None

    # -- ego motion ---------------------------------------------------------
    def _update_ego(self, gray, depth):
        if self.fixed_speed_mpf is not None:
            self.traveled_m += self.fixed_speed_mpf
            return
        h, w = gray.shape
        s = 4
        # Farneback needs 8-bit input; on [0,1] floats it returns all-zero flow.
        small = cv2.resize(np.clip(gray * 255.0, 0, 255).astype(np.uint8), (w // s, h // s),
                           interpolation=cv2.INTER_AREA)
        if self._prev_gray_small is not None:
            hs = small.shape[0] // 2
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray_small[hs:], small[hs:], None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag = np.linalg.norm(flow, axis=-1)
            # Forward translation: radial image flow r' = r * v / Z, so v = |flow| Z / r.
            # Uses the (lower, road) half only and the matching metric depth.
            ys, xs = np.mgrid[hs:small.shape[0], 0:small.shape[1]].astype(np.float32)
            r = np.hypot(xs - small.shape[1] / 2.0, ys - small.shape[0] / 2.0)
            z = cv2.resize(depth, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)[hs:]
            valid = (r > 0.15 * small.shape[1]) & (z > 1.0) & (z < 60.0) & (mag > 0.05)
            v = float(np.median(mag[valid] * z[valid] / r[valid])) if valid.sum() > 50 else 0.0
            v = float(np.clip(v, 0.0, 50.0 / self.fps))            # <= 180 km/h
            self._speed_mpf = self.speed_ema * self._speed_mpf + (1.0 - self.speed_ema) * v
            self.traveled_m += self._speed_mpf
        self._prev_gray_small = small

    # -- fBm in world space --------------------------------------------------
    def _fbm_world(self, X, Y, Z):
        p = np.stack([X, Y, Z + self.traveled_m], axis=0) / self.corr_len_m
        a = 2.0 ** (-self.hurst)
        total = np.zeros(X.shape, dtype=np.float32)
        norm = 0.0
        for i in range(self.octaves):
            f = 2.0 ** i
            o = self.octave_offsets[i]
            total += (a ** i) * value_noise_3d(f * p[0] + o[0], f * p[1] + o[1], f * p[2] + o[2],
                                               int(self.octave_seeds[i]))
            norm += a ** i
        return total / norm

    # -- main ---------------------------------------------------------------
    def apply_fog(self, frame_bgr, depth_meters):
        h, w = frame_bgr.shape[:2]
        J = frame_bgr.astype(np.float32) / 255.0
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        d_hat = np.clip(np.asarray(depth_meters, dtype=np.float32), 0.0, None)

        # Depth: EMA'd 95th-percentile scale, guided-filter edge snap, metric ceiling.
        p95 = float(np.percentile(d_hat, 95)) + 1e-6
        self._depth_scale = p95 if self._depth_scale is None else \
            self.depth_scale_ema * self._depth_scale + (1.0 - self.depth_scale_ema) * p95
        s_bar = self._depth_scale
        d = guided_filter(gray, d_hat / s_bar, radius=8, eps=1e-3) * s_bar
        d = np.clip(d, 0.0, self.max_depth_m).astype(np.float32)

        self._update_ego(gray, d)

        # World-space geometry (pinhole, horizontal FOV).
        f = w / (2.0 * np.tan(np.deg2rad(self.fov_deg) / 2.0))
        u = np.arange(w, dtype=np.float32)[None, :]
        v = np.arange(h, dtype=np.float32)[:, None]
        X_c = (u - w / 2.0) * d / f
        Y_c = (v - h / 2.0) * d / f                       # +down
        height = self.cam_height_m - Y_c
        psi = np.where(height <= 0.0, 1.0,
                       np.exp(-np.maximum(height, 0.0) / self.fog_layer_m)).astype(np.float32)

        # Heterogeneity (world-anchored fBm). The field is smooth on the scale of
        # the correlation length, so it is evaluated at reduced resolution.
        if self.patchiness > 0.0:
            s = self.noise_downsample
            hs, ws = max(1, h // s), max(1, w // s)
            small = lambda a: cv2.resize(a, (ws, hs), interpolation=cv2.INTER_AREA)
            phi = self._fbm_world(small(X_c), small(Y_c), small(d))
            phi = cv2.resize(phi, (w, h), interpolation=cv2.INTER_LINEAR)
            hetero = 1.0 + self.patchiness * (2.0 * phi - 1.0)
        else:
            hetero = 1.0

        beta_t = self.beta_base * hetero * (0.4 + 1.2 * psi)
        beta_field = beta_t * (self.beta_base / (float(np.mean(beta_t)) + 1e-12))

        t = np.exp(-beta_field * d).astype(np.float32)
        m = 1.0 - t

        # Forward scatter: depth-dependent APSF from two Gaussian scales.
        m3 = m[..., None]
        G = (1.0 - m3) * cv2.GaussianBlur(J, (9, 9), 0) + m3 * cv2.GaussianBlur(J, (13, 13), 0)

        A_eff = self.airlight_color * (1.0 + self.road_glow * psi)[..., None]

        s_sc = self.scatter_frac * m3
        s_air = m3 - s_sc
        I = J * t[..., None] + G * s_sc + A_eff * s_air

        return np.clip(np.rint(I * 255.0), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. MAIN (single clip; dataset builds go through build_fog_dataset.py)
# ---------------------------------------------------------------------------
MOR_LADDER_M = (50, 75, 100, 150, 200, 300)            # beta = 3 / MOR
FOG_PRESETS = {                                        # paper Tab. intensity bands (beta0)
    'light': 0.012, 'medium': 0.038, 'heavy': 0.100, 'extreme': 0.320,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='frame folder')
    parser.add_argument('--output', required=True)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--mor', type=float, default=None, help='visibility in m (beta = 3/MOR)')
    parser.add_argument('--intensity', default='medium', choices=list(FOG_PRESETS.keys()))
    parser.add_argument('--airlight', type=float, default=0.72)
    parser.add_argument('--fog_layer_m', type=float, default=20.0)
    parser.add_argument('--heterogeneity', default='uniform', choices=list(HETEROGENEITY_MODES))
    parser.add_argument('--fbm_seed', type=int, default=0)
    parser.add_argument('--speed_kmh', type=float, default=0.0, help='0 = optical-flow estimate')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--encoder', default='vits', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--vda_root', default='/scratch/Ananya_Kulkarni/Video-Depth-Anything')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    if args.beta is not None:
        beta_val = args.beta
    elif args.mor is not None:
        beta_val = 3.0 / args.mor
    else:
        beta_val = FOG_PRESETS[args.intensity]

    img_paths = sorted(glob.glob(os.path.join(args.input, "*.[jJ][pP][gG]")) +
                       glob.glob(os.path.join(args.input, "*.[pP][nN][gG]")))
    os.makedirs(args.output, exist_ok=True)

    pipe = VDAWrapper(encoder=args.encoder, vda_root_path=args.vda_root, device_id=args.gpu_id)
    engine = PhysicalFogEngine(beta_base=beta_val, airlight_color=(args.airlight,) * 3,
                               fog_layer_m=args.fog_layer_m, heterogeneity=args.heterogeneity,
                               fbm_seed=args.fbm_seed, speed_kmh=args.speed_kmh, fps=args.fps)
    for idx, frame, depth in tqdm(iter_scene_depth(pipe, img_paths), total=len(img_paths),
                                  desc=f"GPU {args.gpu_id} | Beta {beta_val:.4f}"):
        stem = os.path.splitext(os.path.basename(img_paths[idx]))[0]
        cv2.imwrite(os.path.join(args.output, stem + '.png'), engine.apply_fog(frame, depth))


if __name__ == '__main__':
    main()
