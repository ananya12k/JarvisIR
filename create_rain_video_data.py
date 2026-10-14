"""
create_rain_video_data.py
Rain (MUNIT wetness + DropModel raindrops) video synthesis on the
Defog_Dataset GT sequences, at two levels of temporal consistency.

Modes (all rendered from ONE MUNIT content-encoding pass per frame):
  frame_indep      Level 1 baseline. MUNIT and DropModel used unchanged: a fresh
                   wet style code and a fresh droplet map for EVERY frame.
                   Easy, but flickers (style + drops jump frame to frame).
  temporal_static  Level 2. One wet style code per clip, one droplet map per
                   clip reused for every frame (drops stuck on the lens).
  temporal_moving  Level 2 + controlled motion. Same clip style and the same
                   stuck drops as temporal_static, plus a second, sparser
                   droplet layer that slides across the lens at a constant
                   per-clip velocity (down, with a little sideways drift).
  temporal_dynamic Level 2 + simulated water. Same clip style; droplets are
                   individually tracked (rainy/temporal_dynamic.py): Poisson
                   arrivals fade in, grow, slide once heavy (stick-slip,
                   wobble, teardrop stretch), leave fading trails, merge and
                   leave the frame. Rendered via DropModel.apply_drops().

Variations: --num_conditions independent draws per clip (cond0..condN-1).
Each (clip, cond) samples its own drop size / frequency / shape / blur,
wet style and slide velocity ONCE from continuous ranges, so conds are a
severity spectrum, not named presets. cond_k in every mode shares the same
drop params, so the modes are directly comparable.

Temporal stability check: for each clip/cond, the mean absolute
consecutive-frame difference of every output is divided by that of the GT
(flicker ratio; ~1.0 = no flicker beyond real scene motion). It is reported
for the full rainy outputs and for the MUNIT-only wet image (wet_indep vs
wet_clip_style), so MUNIT's own temporal stability is checked separately
from the drops.

Output layout:
  {out_base}/
    {split}/
      frame_indep/cond{k}/{seq_id}/000001.jpg ...
      temporal_static/cond{k}/{seq_id}/...
      temporal_moving/cond{k}/{seq_id}/...
      wet/cond{k}/{seq_id}/...            (only with --save_wet; clip-style MUNIT output)
      params/{seq_id}.json                sampled per-clip params per cond
      stats/{seq_id}.json                 flicker ratios per cond / mode
    videos/{split}_{seq_id}/side_by_side.mp4   GT | frame_indep | static | moving | dynamic | dynamic water map (cond0)
    videos/{split}_{seq_id}/{GT,mode}.mp4      full-res per mode (cond0)
    {split}/stats/{seq_id}_{cond}_dynamic.csv  per-frame simulator diagnostics

GT stays at: {gt_base}/{split}/GT/{seq_id}/

Usage:
    python create_rain_video_data.py --num_gpus 4
    python create_rain_video_data.py --num_gpus 2 --splits val --max_seqs 2 --first_n 100 --num_conditions 1   # 2-video test
    python create_rain_video_data.py ... --modes temporal_dynamic --sim_seed 7   # re-run only the simulator with another seed
"""

import os
import sys
import cv2
import json
import glob
import random
import hashlib
import time
import argparse
import numpy as np
import torch
import yaml
from tqdm import tqdm
import torch.multiprocessing as mp

SYN_ROOT = '/scratch/Ananya_Kulkarni/JarvisIR/degradation_synthesis'
sys.path.insert(0, SYN_ROOT)

# ---------------------------------------------------------------------------
GT_BASE  = '/scratch/Ananya_Kulkarni/Defog_Dataset'
OUT_BASE = '/scratch/Ananya_Kulkarni/Rain_video'
PARAMS_PATH  = os.path.join(SYN_ROOT, 'rainy/GuidedDisent/configs/params_net.yaml')
WEIGHTS_PATH = os.path.join(SYN_ROOT, 'rainy/GuidedDisent/weights/pretrained.pth')
BATCH = 8
NUM_CONDITIONS = 3
MAX_FRAMES_PER_SEQ = 2000   # longer GoPro sequences are evenly strided down
PREVIEW_FRAMES = 150
MODES = ['frame_indep', 'temporal_static', 'temporal_moving', 'temporal_dynamic']
DYNAMIC_CONFIG = os.path.join(SYN_ROOT, 'rainy/temporal_dynamic.yaml')
# ---------------------------------------------------------------------------


def _seed(global_seed: int, key: str) -> int:
    """Deterministic seed from a string key: same key -> same draw every run."""
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return (global_seed ^ h) & 0xFFFFFFFF


def _seed_all(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)


def _select_frames(img_paths, max_frames):
    n = len(img_paths)
    if n <= max_frames:
        return img_paths
    stride = max(1, n // max_frames)
    return img_paths[::stride]


def sample_rain_clip(rng: np.random.Generator):
    """Per-(clip, cond) params from continuous ranges (defaults: 50 / 8 / 0.6 / 4)."""
    return {
        'drop_size':      float(np.exp(rng.uniform(np.log(30), np.log(80)))),  # smaller = bigger drops
        'drop_frequency': float(rng.uniform(4, 12)),                          # higher = more drops
        'drop_shape':     float(rng.uniform(0.5, 0.8)),                       # higher = rounder
        'drop_sigma':     float(rng.uniform(2.0, 6.0)),                       # refraction blur
        'slide_vy':       float(rng.uniform(0.5, 3.0)),                       # px / frame, downward
        'slide_vx':       float(rng.uniform(-0.6, 0.6)),                      # px / frame, wind drift
        'rain_intensity': float(rng.uniform(0.6, 1.4)),                       # temporal_dynamic spawn rate scale
    }


def load_munit(device):
    from rainy.GuidedDisent.MUNIT.model_infer import MUNIT_infer
    with open(PARAMS_PATH) as f:
        hp = yaml.safe_load(f)
    model = MUNIT_infer(hp)
    w = torch.load(WEIGHTS_PATH, map_location='cpu')
    model.gen_a.load_state_dict(w['a'])
    model.gen_b.load_state_dict(w['b'])
    return model.to(device).eval()


def shift_map(normal_map, imsize, dy, dx):
    """Translate a (v, mask) droplet map with wrap-around (drops re-enter at the top)."""
    v, mask = normal_map
    h, w = imsize
    v = torch.roll(v.view(1, h, w, 3), shifts=(dy, dx), dims=(1, 2)).view(1, h * w, 3)
    mask = torch.roll(mask.view(1, h, w, 1), shifts=(dy, dx), dims=(1, 2)).view(1, h * w, 1)
    return v, mask


def combine_maps(front, back):
    """Overlay two droplet maps: `front` drops win where they exist (same rule
    DropModel uses between its own size layers)."""
    v_f, m_f = front
    v_b, m_b = back
    keep = (m_f > 0)
    return torch.where(keep, v_f, v_b), torch.where(keep, m_f, m_b)


def to_bgr_uint8(t):
    """[-1,1] 3xHxW RGB tensor -> HxWx3 BGR uint8."""
    x = ((t.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return x[:, :, ::-1]


class FlickerMeter:
    """Mean |x_t - x_{t-1}| over a clip, accumulated batch by batch."""
    def __init__(self):
        self.prev, self.total, self.n = None, 0.0, 0

    def update(self, frames):  # frames: Bx3xHxW in [-1,1]
        if self.prev is not None:
            frames_ext = torch.cat([self.prev, frames], 0)
        else:
            frames_ext = frames
        if frames_ext.size(0) > 1:
            d = (frames_ext[1:] - frames_ext[:-1]).abs().mean(dim=(1, 2, 3))
            self.total += d.sum().item()
            self.n += d.numel()
        self.prev = frames[-1:].clone()

    def value(self):
        return self.total / max(self.n, 1)


# categorical series colours (dataviz skill reference palette, fixed slot order)
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']


def field_frame(v, mk):
    """Refraction field as colour: hue = offset direction, brightness = offset size
    (px), weighted by the water mask. v: 1xHxWx3, mk: 1xHxWx1 (torch)."""
    H, W = v.shape[1:3]
    off = (-v[0, :, :, :2] * v[0, :, :, 2:3]).cpu().numpy()          # normalised (row, col)
    dy, dx = off[..., 0] * H / 2, off[..., 1] * W / 2                 # pixels
    mag = np.hypot(dx, dy)
    ang = (np.degrees(np.arctan2(dy, dx)) % 360) / 2                   # OpenCV hue 0..180
    val = np.clip(mag / 60.0, 0, 1) * np.clip(mk[0, :, :, 0].cpu().numpy() * 2, 0, 1)
    hsv = np.stack([ang, np.full_like(ang, 255), val * 255], -1).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.putText(out, 'refraction field: hue = direction, brightness = offset (0-60 px) x water mask',
                (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return out


def track_frame(img, drops, hist):
    """Overlay drop tracks: pinned = blue, sliding = orange; last ~20 drawn positions."""
    out = img.copy()
    col = {'pinned': (214, 120, 42), 'sliding': (52, 104, 235)}       # BGR of slots 1 / 2
    for (i, x, y, r, state) in drops:
        pts = hist.get(i, [])
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col['sliding'], 1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), max(2, int(r)), col[state], 1, cv2.LINE_AA)
    cv2.putText(out, 'tracks: blue = pinned, orange = sliding (+ last 20 positions)', (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return out


def plot_diagnostics(path, title, coverage, frame_ms, modes):
    """Two small multiples on their own y-axes (never dual-axis): water coverage per
    frame and runtime per frame, one line per mode, legend + direct end labels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = '#1f1f1e', '#6b6b66', '#e6e5e0', '#fcfcfb'
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=surface)
    for ax, data, ylab in ((axes[0], coverage, 'water coverage (fraction of frame)'),
                           (axes[1], frame_ms, 'runtime per frame (ms)')):
        ax.set_facecolor(surface)
        for k, m in enumerate(modes):
            y = data[m]
            c = SERIES_COLORS[k % len(SERIES_COLORS)]
            ax.plot(np.arange(len(y)), y, color=c, lw=2, label=m.replace('temporal_dynamic_', 'dynamic: '))
        ax.set_xlabel('frame', color=muted)
        ax.set_ylabel(ylab, color=muted)
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(grid)
        ax.tick_params(colors=muted)
        ax.set_ylim(bottom=0)
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False, labelcolor=ink)
    fig.suptitle(title, color=ink, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=surface)
    plt.close(fig)


def summarize_dynamic(log):
    """Per-clip summary of a temporal_dynamic per-frame stats log: mean of every
    numeric field, max of every `max_*` field, totals of event counts."""
    keys = [k for k in log[0] if k != 'frame']
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in log], float)
        out[f'{k}_max' if k.startswith('max_') else f'{k}_mean'] = float(vals.max() if k.startswith('max_') else vals.mean())
    for k in ('arrivals', 'merges', 'impacts'):
        if k in keys:
            out[f'{k}_total'] = int(sum(r[k] for r in log))
    return out


def worker(rank, jobs, args):
    from rainy.GuidedDisent.droprenderer import DropModel
    from rainy.temporal_dynamic import TemporalDynamicRain, load_rain_config

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    munit = load_munit(device)
    style_dim = munit.style_dim
    one = torch.ones(1, device=device)  # droprenderer's default lives on cuda:0
    dyn_cfgs = {}
    if 'temporal_dynamic' in args.modes:
        dyn_cfgs = {f'temporal_dynamic_{sc}': load_rain_config(args.dynamic_config, sc) for sc in args.scenarios}
    dyn_modes = list(dyn_cfgs)
    modes = [m for m in args.modes if m != 'temporal_dynamic'] + dyn_modes
    cond_names = [f'cond{i}' for i in range(args.num_conditions)]
    out_modes = modes + (['wet'] if args.save_wet else [])

    for split, seq_id in jobs:
        gt_dir = os.path.join(GT_BASE, split, 'GT', seq_id)
        img_paths = sorted(glob.glob(os.path.join(gt_dir, '*.jpg')) +
                           glob.glob(os.path.join(gt_dir, '*.png')))
        if not img_paths:
            continue
        if args.first_n:
            img_paths = img_paths[:args.first_n]   # consecutive frames (no striding)
        img_paths = _select_frames(img_paths, args.max_frames_per_seq)

        h, w = cv2.imread(img_paths[0]).shape[:2]
        h4, w4 = h - h % 4, w - w % 4   # MUNIT needs H, W divisible by 4
        imsize = (h4, w4)

        # ---- per-(clip, cond) draws: params, wet style, droplet maps / simulator ----
        clips = {}
        for name in cond_names:
            seed = _seed(args.seed, f'{split}/{seq_id}/{name}')
            p = sample_rain_clip(np.random.default_rng(seed))
            _seed_all(seed)
            dm = DropModel(imsize=imsize, size_threshold=1 / p['drop_size'],
                           frequency_threshold=p['drop_frequency'], shape_threshold=p['drop_shape'])
            dm_slide = DropModel(imsize=imsize, size_threshold=1 / p['drop_size'],
                                 frequency_threshold=max(4.0, p['drop_frequency'] - 3),  # <=2 never spawns a drop (get_numdrops loops forever)
                                 shape_threshold=p['drop_shape'])
            with torch.no_grad():
                clips[name] = {
                    'params': p,
                    'dm': dm,
                    'sigma': torch.full((1,), p['drop_sigma'], device=device),
                    'style': torch.randn(1, style_dim, 1, 1, device=device),
                    'static_map': dm.get_normal_map(),
                    'slide_map': dm_slide.get_normal_map(),
                }
            del dm_slide
            # one simulator per scenario; all scenarios share the clip's seed, style and
            # drop shape, so they differ only by their preset (controlled comparison)
            clips[name]['sims'], clips[name]['sim_logs'] = {}, {}
            for m, dcfg in dyn_cfgs.items():
                sim_seed = _seed(args.sim_seed if args.sim_seed is not None else args.seed,
                                 f'{split}/{seq_id}/{name}/dynamic')
                sim = TemporalDynamicRain(w4, h4, dcfg, rain_intensity=p['rain_intensity'],
                                          shape_threshold=p['drop_shape'], seed=sim_seed, device=device,
                                          scenario=m[len('temporal_dynamic_'):])
                sim.warmup(dcfg['warmup_frames'])
                if sim.profile is not None:
                    sim.profile.clear()
                clips[name]['sims'][m] = sim
                clips[name]['sim_logs'][m] = []
            torch.cuda.empty_cache()

        for m in out_modes:
            for name in cond_names:
                os.makedirs(os.path.join(args.out_base, split, m, name, seq_id), exist_ok=True)
        for sub in ('params', 'stats'):
            os.makedirs(os.path.join(args.out_base, split, sub), exist_ok=True)
        with open(os.path.join(args.out_base, split, 'params', f'{seq_id}.json'), 'w') as f:
            json.dump({n: c['params'] for n, c in clips.items()}, f, indent=2)

        meters = {'gt': FlickerMeter()}
        coverage = {}
        mode_time = {m: 0.0 for m in modes}
        frame_ms = {m: [] for m in modes}           # per frame, cond0 only
        cur = {'cond': None}
        mode_peak = {m: 0.0 for m in modes}
        for name in cond_names:
            for k in modes + ['wet_indep', 'wet_clip_style']:
                meters[f'{name}/{k}'] = FlickerMeter()
            for m in modes:
                coverage[f'{name}/{m}'] = []

        # ---- videos (cond0): side-by-side grid + one full-res mp4 per mode ----
        writers = {}
        if args.preview_frames > 0:
            vid_dir = os.path.join(args.out_base, 'videos', f'{split}_{seq_id}')
            os.makedirs(vid_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            pw, ph = w4 // 2, h4 // 2
            # water map of the heaviest selected scenario as the last panel
            panels = ['GT'] + modes + ([f'water map ({dyn_modes[-1][len("temporal_dynamic_"):]})'] if dyn_modes else [])
            cols = 3 if len(panels) > 4 else 2
            rows = (len(panels) + cols - 1) // cols
            writers['grid'] = cv2.VideoWriter(os.path.join(vid_dir, 'side_by_side.mp4'), fourcc, args.fps,
                                              (pw * cols, ph * rows))
            for m in ['GT'] + modes:
                writers[m] = cv2.VideoWriter(os.path.join(vid_dir, f'{m}.mp4'), fourcc, args.fps, (w4, h4))
            if args.diagnostics:
                for m in dyn_modes:
                    sc = m[len('temporal_dynamic_'):]
                    writers[f'tracks/{m}'] = cv2.VideoWriter(os.path.join(vid_dir, f'tracks_{sc}.mp4'), fourcc, args.fps, (w4, h4))
                    writers[f'field/{m}'] = cv2.VideoWriter(os.path.join(vid_dir, f'field_{sc}.mp4'), fourcc, args.fps, (w4, h4))
        track_hist = {m: {} for m in dyn_modes}

        def timed(m, fn):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            out = fn()
            torch.cuda.synchronize(device)
            mode_time[m] += time.perf_counter() - t0
            if cur['cond'] == cond_names[0]:
                frame_ms[m].append(1000 * (time.perf_counter() - t0))
            mode_peak[m] = max(mode_peak[m], torch.cuda.max_memory_allocated(device) / 1e9)
            return out

        pbar = tqdm(total=len(img_paths), desc=f'GPU{rank} | {split}/{seq_id}', position=rank, leave=True)
        for i in range(0, len(img_paths), args.batch):
            batch_paths = img_paths[i:i + args.batch]
            frames = [cv2.imread(p)[:h4, :w4, ::-1] for p in batch_paths]
            x = torch.from_numpy(np.stack(frames).copy()).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1
            b = x.size(0)
            meters['gt'].update(x)

            with torch.no_grad():
                content, _ = munit.gen_a.encode(x)
                for name in cond_names:
                    c = clips[name]
                    cur['cond'] = name
                    # Level 1: fresh style per frame
                    styles = []
                    for p in batch_paths:
                        _seed_all(_seed(args.seed, f'{p}/{name}/style'))
                        styles.append(torch.randn(1, style_dim, 1, 1, device=device))
                    wet_indep = munit.gen_b.decode(content, torch.cat(styles, 0))
                    # Level 2: one style for the whole clip
                    wet_clip = munit.gen_b.decode(content, c['style'].expand(b, -1, -1, -1))

                    outs = {m: [] for m in modes}
                    dyn_masks = []
                    diag = {m: [] for m in dyn_modes}
                    for j, p in enumerate(batch_paths):
                        t = i + j
                        kw = dict(sigma=c['sigma'], transparency=one)
                        if 'frame_indep' in modes:
                            def run():
                                _seed_all(_seed(args.seed, f'{p}/{name}/drops'))
                                v, mk = c['dm'].get_normal_map()
                                return c['dm'].apply_drops(wet_indep[j:j + 1], v, mk, **kw), mk
                            o, mk = timed('frame_indep', run)
                            outs['frame_indep'].append(o)
                            coverage[f'{name}/frame_indep'].append((mk > 0.05).float().mean().item())
                        if 'temporal_static' in modes:
                            v, mk = c['static_map']
                            outs['temporal_static'].append(timed('temporal_static', lambda: c['dm'].apply_drops(wet_clip[j:j + 1], v, mk, **kw)))
                            coverage[f'{name}/temporal_static'].append((mk > 0.05).float().mean().item())
                        if 'temporal_moving' in modes:
                            def run():
                                slid = shift_map(c['slide_map'], imsize,
                                                 int(round(t * c['params']['slide_vy'])),
                                                 int(round(t * c['params']['slide_vx'])))
                                v, mk = combine_maps(c['static_map'], slid)
                                return c['dm'].apply_drops(wet_clip[j:j + 1], v, mk, **kw), mk
                            o, mk = timed('temporal_moving', run)
                            outs['temporal_moving'].append(o)
                            coverage[f'{name}/temporal_moving'].append((mk > 0.05).float().mean().item())
                        for m in dyn_modes:
                            sim = c['sims'][m]
                            def run():
                                sim.step()
                                v, mk = sim.render_maps()
                                return c['dm'].apply_drops(wet_clip[j:j + 1], v, mk, **kw), mk
                            o, mk = timed(m, run)
                            outs[m].append(o)
                            if m == dyn_modes[-1]:
                                dyn_masks.append(mk)
                            if args.diagnostics and name == cond_names[0] and i + j < args.preview_frames:
                                vmap, _ = sim.last_maps
                                drops = [(d.id, d.rx, d.ry, d.render_radius, d.state) for d in sim.droplets]
                                diag[m].append((field_frame(vmap, mk), drops))
                            coverage[f'{name}/{m}'].append(sim.stats['coverage'])
                            c['sim_logs'][m].append({'frame': os.path.basename(p), **sim.stats})
                    outs = {m: torch.cat(v, 0) for m, v in outs.items()}

                    meters[f'{name}/wet_indep'].update(wet_indep)
                    meters[f'{name}/wet_clip_style'].update(wet_clip)
                    for m in modes:
                        meters[f'{name}/{m}'].update(outs[m])
                    if args.save_wet:
                        outs['wet'] = wet_clip

                    for m, tens in outs.items():
                        for j, p in enumerate(batch_paths):
                            out_path = os.path.join(args.out_base, split, m, name, seq_id, os.path.basename(p))
                            cv2.imwrite(out_path, to_bgr_uint8(tens[j]), [cv2.IMWRITE_JPEG_QUALITY, 95])

                    if writers and name == cond_names[0]:
                        for j in range(b):
                            if i + j >= args.preview_frames:
                                break
                            full = {'GT': np.ascontiguousarray(frames[j][:, :, ::-1])}
                            full.update({m: np.ascontiguousarray(to_bgr_uint8(outs[m][j])) for m in modes})
                            for m, img in full.items():
                                writers[m].write(img)
                            tiles = [cv2.resize(full[m], (pw, ph)) for m in ['GT'] + modes]
                            if dyn_masks:
                                mm = (dyn_masks[j][0, :, :, 0].clamp(0, 1) * 255).byte().cpu().numpy()
                                tiles.append(cv2.cvtColor(cv2.resize(mm, (pw, ph)), cv2.COLOR_GRAY2BGR))
                            for tile, label in zip(tiles, panels):
                                cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            tiles += [np.zeros_like(tiles[0])] * (rows * cols - len(tiles))
                            writers['grid'].write(np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]))
                            if args.diagnostics:
                                for m in dyn_modes:
                                    fimg, drops = diag[m][j]
                                    hist = track_hist[m]
                                    for (di, x, y, _, _) in drops:
                                        hist.setdefault(di, []).append((x, y))
                                        del hist[di][:-20]
                                    alive = {d[0] for d in drops}
                                    for di in [k for k in hist if k not in alive]:
                                        del hist[di]
                                    writers[f'tracks/{m}'].write(track_frame(full[m], drops, hist))
                                    writers[f'field/{m}'].write(fimg)
            pbar.update(b)
        pbar.close()
        for wr in writers.values():
            wr.release()

        n_frames = len(img_paths)
        if args.diagnostics:
            os.makedirs(os.path.join(args.out_base, 'plots'), exist_ok=True)
            plot_diagnostics(os.path.join(args.out_base, 'plots', f'{split}_{seq_id}.png'),
                             f'{split}/{seq_id} ({cond_names[0]}): water coverage and runtime per frame',
                             {m: coverage[f'{cond_names[0]}/{m}'] for m in modes},
                             frame_ms, modes)
        gt_flicker = meters['gt'].value()
        stats = {'gt_mean_abs_frame_diff': gt_flicker, 'num_frames': n_frames,
                 'ms_per_frame_per_cond': {m: 1000 * mode_time[m] / (n_frames * len(cond_names)) for m in modes},
                 'peak_gpu_gb': mode_peak}
        for name in cond_names:
            stats[name] = {
                'flicker_ratio': {k: meters[f'{name}/{k}'].value() / max(gt_flicker, 1e-8)
                                  for k in modes + ['wet_indep', 'wet_clip_style']},
                'coverage_mean': {m: float(np.mean(coverage[f'{name}/{m}'])) for m in modes},
                'coverage_first_last': {m: [coverage[f'{name}/{m}'][0], coverage[f'{name}/{m}'][-1]] for m in modes},
            }
            stats[name]['dynamic'] = {}
            for m, log in clips[name]['sim_logs'].items():
                stats[name]['dynamic'][m] = summarize_dynamic(log)
                prof = clips[name]['sims'][m].profile
                if prof:
                    stats[name]['dynamic'][m]['render_ms_per_frame'] = {k: 1000 * v / n_frames for k, v in prof.items()}
                with open(os.path.join(args.out_base, split, 'stats', f'{seq_id}_{name}_{m}.csv'), 'w') as f:
                    keys = list(log[0].keys())
                    f.write(','.join(keys) + '\n')
                    for r in log:
                        f.write(','.join(str(r[k]) for k in keys) + '\n')
        with open(os.path.join(args.out_base, split, 'stats', f'{seq_id}.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        del clips
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Rain video synthesis (frame-independent vs temporally consistent)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--out_base', type=str, default=OUT_BASE)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_conditions', type=int, default=NUM_CONDITIONS,
                        help=f'Independent param draws per clip (default: {NUM_CONDITIONS})')
    parser.add_argument('--max_frames_per_seq', type=int, default=MAX_FRAMES_PER_SEQ)
    parser.add_argument('--max_seqs', type=int, default=None, help='Limit sequences per split (testing)')
    parser.add_argument('--first_n', type=int, default=None,
                        help='Use only the first N consecutive frames of each sequence (testing)')
    parser.add_argument('--batch', type=int, default=BATCH)
    parser.add_argument('--preview_frames', type=int, default=PREVIEW_FRAMES,
                        help='Frames in the per-sequence comparison mp4 (0 disables)')
    parser.add_argument('--modes', nargs='+', default=MODES, choices=MODES)
    parser.add_argument('--dynamic_config', type=str, default=DYNAMIC_CONFIG)
    parser.add_argument('--scenarios', nargs='+', default=['light', 'steady', 'heavy', 'torrential'],
                        help='temporal_dynamic rain scenarios (presets in --dynamic_config); one output mode each')
    parser.add_argument('--sim_seed', type=int, default=None,
                        help='Reseed only the temporal_dynamic simulator (everything else keeps --seed)')
    parser.add_argument('--fps', type=int, default=30, help='Output video frame rate')
    parser.add_argument('--diagnostics', action='store_true',
                        help='Per scenario (cond0): track-overlay + refraction-field videos, coverage/runtime plot')
    parser.add_argument('--save_wet', action='store_true', help='Also save the MUNIT-only wet frames')
    args = parser.parse_args()

    all_jobs = []
    for split in args.splits:
        gt_dir = os.path.join(GT_BASE, split, 'GT')
        if not os.path.isdir(gt_dir):
            print(f'[skip] {gt_dir} not found')
            continue
        seqs = sorted(os.listdir(gt_dir))[:args.max_seqs]
        all_jobs += [(split, s) for s in seqs]

    print(f'\nTotal sequences : {len(all_jobs)}  |  GPUs: {args.num_gpus}')
    print(f'Source GT       : {GT_BASE}/{{split}}/GT/{{seq_id}}/')
    print(f'Output root     : {args.out_base}/{{split}}/{{mode}}/{{cond}}/{{seq_id}}/')
    print(f'Modes           : {args.modes}' + (' + wet' if args.save_wet else '')
          + (f'  |  scenarios: {args.scenarios}' if 'temporal_dynamic' in args.modes else ''))

    # reproducibility: the exact resolved simulator config of every scenario + all CLI args
    from rainy.temporal_dynamic import load_rain_config
    os.makedirs(os.path.join(args.out_base, 'configs'), exist_ok=True)
    with open(os.path.join(args.out_base, 'configs', 'run_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    if 'temporal_dynamic' in args.modes:
        for sc in args.scenarios:
            with open(os.path.join(args.out_base, 'configs', f'temporal_dynamic_{sc}.yaml'), 'w') as f:
                yaml.safe_dump(load_rain_config(args.dynamic_config, sc), f, sort_keys=False)
    print(f'Conditions      : {args.num_conditions} per clip  |  max frames/seq: {args.max_frames_per_seq}\n')

    worker_jobs = [[] for _ in range(args.num_gpus)]
    for i, job in enumerate(all_jobs):
        worker_jobs[i % args.num_gpus].append(job)
    for rank, jobs in enumerate(worker_jobs):
        print(f'  GPU {rank}: {[f"{s}/{q}" for s, q in jobs]}')

    procs = []
    for rank in range(args.num_gpus):
        if not worker_jobs[rank]:
            continue
        p = mp.Process(target=worker, args=(rank, worker_jobs[rank], args))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    print('\nAll done.')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
