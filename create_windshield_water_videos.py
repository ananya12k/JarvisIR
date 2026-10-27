"""
create_windshield_water_videos.py
Rainy windshield videos from the Defog_Dataset GT with the stateful windshield-water
simulator (degradation_synthesis/windshield_water). Independent of
create_rain_video_data.py (frame_indep / temporal_* modes), which is left unchanged.

Per clip:
  Module A  MUNIT wet appearance, ONE style code per clip (s ~ N(0, I), reused every frame)
  Module B-D one WindshieldWaterSimulator per scenario (light / steady / heavy / torrential),
            stepped once per frame (dt = 1 / fps), rendered into maps, applied to the wet frame

Outputs ({out_base}):
  videos/{split}_{seq}/side_by_side.mp4   GT | light | steady | heavy | torrential | water map (torrential)
  videos/{split}_{seq}/integration.mp4    original | MUNIT only | water only (steady) | MUNIT + water (steady)
  videos/{split}_{seq}/{scenario}.mp4, GT.mp4
  frames/{split}/{scenario}/{seq}/000001.jpg ...           (--save_frames)
  metrics/{split}_{seq}_{scenario}.json / .csv              simulator + rendered-output metrics
  configs/{scenario}.json, run_args.json

Usage:
  python create_windshield_water_videos.py --splits val --seqs 02 06 --first_n 100 --num_gpus 2
"""

import argparse
import dataclasses
import glob
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from tqdm import tqdm

SYN_ROOT = '/scratch/Ananya_Kulkarni/JarvisIR/degradation_synthesis'
sys.path.insert(0, SYN_ROOT)

GT_BASE = '/scratch/Ananya_Kulkarni/Defog_Dataset'
OUT_BASE = '/scratch/Ananya_Kulkarni/Windshield_water'
PARAMS_PATH = os.path.join(SYN_ROOT, 'rainy/GuidedDisent/configs/params_net.yaml')
WEIGHTS_PATH = os.path.join(SYN_ROOT, 'rainy/GuidedDisent/weights/pretrained.pth')


def seed_of(global_seed, key):
    return (global_seed ^ int(hashlib.md5(key.encode()).hexdigest(), 16)) & 0xFFFFFFFF


def load_munit(device):
    from rainy.GuidedDisent.MUNIT.model_infer import MUNIT_infer
    with open(PARAMS_PATH) as f:
        model = MUNIT_infer(yaml.safe_load(f))
    w = torch.load(WEIGHTS_PATH, map_location='cpu')
    model.gen_a.load_state_dict(w['a'])
    model.gen_b.load_state_dict(w['b'])
    return model.to(device).eval()


def to_bgr(t):
    return np.ascontiguousarray(((t.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1])


def label(img, text):
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img


def grid(tiles, cols=3):
    tiles = tiles + [np.zeros_like(tiles[0])] * ((-len(tiles)) % cols)
    return np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])


class Flicker:
    """Mean |x_t - x_{t-1}| of a stream (rendered output); reported relative to GT."""
    def __init__(self):
        self.prev, self.total, self.n = None, 0.0, 0

    def update(self, x):
        if self.prev is not None:
            self.total += (x - self.prev).abs().mean().item()
            self.n += 1
        self.prev = x.clone()

    def value(self):
        return self.total / max(self.n, 1)


def run_clip(split, seq, args, gpu, munit):
    from windshield_water import WindshieldWaterSimulator, MetricsRecorder, preset
    from windshield_water.renderer import WaterRenderer

    device = torch.device(f'cuda:{gpu}')
    paths = sorted(glob.glob(os.path.join(GT_BASE, split, 'GT', seq, '*.jpg')))
    if args.first_n:
        paths = paths[:args.first_n]
    h, w = cv2.imread(paths[0]).shape[:2]
    h4, w4 = h - h % 4, w - w % 4
    clip_seed = seed_of(args.seed, f'{split}/{seq}')
    g = torch.Generator(device=device).manual_seed(clip_seed)
    style = torch.randn(1, munit.style_dim, 1, 1, device=device, generator=g)   # one style per clip
    sims = {}
    for sc in args.scenarios:
        sims[sc] = WindshieldWaterSimulator(w4, h4, preset(sc), seed=clip_seed, device=device)
        sims[sc].warmup()
    integ_sim = sims.get('steady')
    renderer = WaterRenderer(h4, w4, device=device)
    recs = {sc: MetricsRecorder(args.fps) for sc in args.scenarios}
    flick = {k: Flicker() for k in ['gt', 'munit'] + list(args.scenarios)}

    vdir = os.path.join(args.out_base, 'videos', f'{split}_{seq}')
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(os.path.join(args.out_base, 'metrics'), exist_ok=True)
    four = cv2.VideoWriter_fourcc(*'mp4v')
    pw, ph = w4 // 2, h4 // 2
    wr = {'grid': cv2.VideoWriter(os.path.join(vdir, 'side_by_side.mp4'), four, args.fps, (pw * 3, ph * 2)),
          'GT': cv2.VideoWriter(os.path.join(vdir, 'GT.mp4'), four, args.fps, (w4, h4))}
    for sc in args.scenarios:
        wr[sc] = cv2.VideoWriter(os.path.join(vdir, f'{sc}.mp4'), four, args.fps, (w4, h4))
    if integ_sim is not None:
        wr['integ'] = cv2.VideoWriter(os.path.join(vdir, 'integration.mp4'), four, args.fps, (pw * 2, ph * 2))

    for p in tqdm(paths, desc=f'GPU{gpu} {split}/{seq}', position=gpu, leave=True):
        img = cv2.imread(p)[:h4, :w4, ::-1]
        x = torch.from_numpy(img.copy()).to(device).permute(2, 0, 1)[None].float() / 127.5 - 1
        flick['gt'].update(x)
        with torch.no_grad():
            content, _ = munit.gen_a.encode(x)
            wet = munit.gen_b.decode(content, style)                       # Module A
        flick['munit'].update(wet)
        gt_bgr = np.ascontiguousarray(img[:, :, ::-1])
        wr['GT'].write(gt_bgr)
        outs, maps_last = {}, None
        for sc, sim in sims.items():                                       # Modules B-D
            t0 = time.perf_counter()
            sim.step()
            maps = sim.render_maps()
            with torch.no_grad():
                out = renderer.apply_maps(wet, maps, sigma=args.sigma, cfg=sim.cfg)
            torch.cuda.synchronize(device)
            recs[sc].record(sim, maps, step_ms=1000 * (time.perf_counter() - t0))
            flick[sc].update(out)
            outs[sc] = to_bgr(out[0])
            wr[sc].write(outs[sc])
            maps_last = maps
            if args.save_frames:
                d = os.path.join(args.out_base, 'frames', split, sc, seq)
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, os.path.basename(p)), outs[sc], [cv2.IMWRITE_JPEG_QUALITY, 95])
        tiles = [label(cv2.resize(gt_bgr, (pw, ph)), 'GT (clean)')]
        tiles += [label(cv2.resize(outs[sc], (pw, ph)), f'{sc} rain') for sc in args.scenarios]
        a = (maps_last.alpha.clamp(0, 1) * 255).byte().cpu().numpy()
        tiles.append(label(cv2.cvtColor(cv2.resize(a, (pw, ph)), cv2.COLOR_GRAY2BGR), f'water map ({args.scenarios[-1]})'))
        wr['grid'].write(grid(tiles[:6], 3))
        if integ_sim is not None:
            with torch.no_grad():
                water_only = renderer.apply_maps(x, integ_sim.last_maps, sigma=args.sigma, cfg=integ_sim.cfg)
            it = [label(cv2.resize(gt_bgr, (pw, ph)), '1. original'),
                  label(cv2.resize(to_bgr(wet[0]), (pw, ph)), '2. MUNIT only'),
                  label(cv2.resize(to_bgr(water_only[0]), (pw, ph)), '3. water only (steady)'),
                  label(cv2.resize(outs['steady'], (pw, ph)), '4. MUNIT + water (steady)')]
            wr['integ'].write(grid(it, 2))
    for v in wr.values():
        v.release()
    gt_f = flick['gt'].value()
    for sc in args.scenarios:
        base = os.path.join(args.out_base, 'metrics', f'{split}_{seq}_{sc}')
        recs[sc].save(base + '.json', base + '.csv')
        with open(base + '.json') as f:
            summ = json.load(f)
        summ['flicker_ratio_rendered'] = flick[sc].value() / max(gt_f, 1e-8)
        summ['flicker_ratio_munit_only'] = flick['munit'].value() / max(gt_f, 1e-8)
        with open(base + '.json', 'w') as f:
            json.dump(summ, f, indent=2)


def worker(gpu, queue, args):
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    munit = load_munit(torch.device(f'cuda:{gpu}'))
    while True:
        job = queue.get()
        if job is None:
            break
        run_clip(*job, args, gpu, munit)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--splits', nargs='+', default=['val'])
    ap.add_argument('--seqs', nargs='+', default=None)
    ap.add_argument('--max_seqs', type=int, default=None)
    ap.add_argument('--first_n', type=int, default=100)
    ap.add_argument('--scenarios', nargs='+', default=['light', 'steady', 'heavy', 'torrential'])
    ap.add_argument('--out_base', default=OUT_BASE)
    ap.add_argument('--num_gpus', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--sigma', type=float, default=3.5, help='refraction blur (DropModel optics)')
    ap.add_argument('--save_frames', action='store_true')
    args = ap.parse_args()

    from windshield_water import preset
    os.makedirs(os.path.join(args.out_base, 'configs'), exist_ok=True)
    for sc in args.scenarios:
        with open(os.path.join(args.out_base, 'configs', f'{sc}.json'), 'w') as f:
            json.dump(dataclasses.asdict(preset(sc)), f, indent=2)
    with open(os.path.join(args.out_base, 'configs', 'run_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    jobs = []
    for split in args.splits:
        seqs = sorted(os.listdir(os.path.join(GT_BASE, split, 'GT')))
        if args.seqs:
            seqs = [s for s in seqs if s in args.seqs]
        jobs += [(split, s) for s in seqs[:args.max_seqs]]
    print(f'{len(jobs)} clips x {len(args.scenarios)} scenarios on {args.num_gpus} GPU(s) -> {args.out_base}')
    queue = mp.Queue()
    for j in jobs:
        queue.put(j)
    n = min(args.num_gpus, len(jobs))
    for _ in range(n):
        queue.put(None)
    procs = [mp.Process(target=worker, args=(g, queue, args)) for g in range(n)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print('All done.')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
