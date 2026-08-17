"""
create_night_data.py
Generates N continuously-sampled night-severity conditions for every frame
in Defog_balanced.

Strategy:
  - N workers, one per GPU (--num_gpus)
  - Sequences split round-robin across workers
  - Each worker: VDA depth inference (GPU) → NightEngine conditions (CPU)
  - All conditions generated from ONE depth pass per frame → no redundant VDA calls
  - Per-CLIP: each (sequence, cond slot) draws its own engine params ONCE
    from a CONTINUOUS log-uniform range (sample_night_clip) — ambient /
    headlight strength / beam geometry / discrete-light positions differ
    clip to clip AND slot to slot, so the dataset is a smooth severity
    spectrum rather than a handful of discrete named presets a model could
    key off of. Params stay fixed for every frame within that clip →
    temporally coherent, not flickering.
  - Per-FRAME: only the sensor noise is reseeded per frame
    (global_seed XOR frame_path hash), so noise is temporally independent
    the way real camera noise is, while the lighting itself is not.

Output layout (ready for NightDataset — no reorganisation needed):
  {out_base}/
    {split}/
      cond0 .. cond{N-1}/   ← N independently-sampled severity draws per clip
        {seq_id}/
          000001.jpg ...
Each cond slot is NOT a fixed named condition (e.g. cond0 isn't always
"dark") — every clip draws fresh params per slot, so slot identity carries
no information a model could learn to exploit.

GT lives separately at: {gt_base}/{split}/GT/{seq_id}/
Use --defog-root {gt_base} in train_night_enhance.py to point at it.

Usage:
    python create_night_data.py --num_gpus 4
    python create_night_data.py --num_gpus 4 --num_conditions 5
    python create_night_data.py --num_gpus 2 --out_base /scratch/.../Night_balanced_v2
"""

import os
import sys
import cv2
import glob
import hashlib
import argparse
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp

sys.path.insert(0, '/scratch/Ananya_Kulkarni/JarvisIR/degradation_synthesis')
from night_synthesis import VDAWrapper, NightEngine, sample_night_clip

# ---------------------------------------------------------------------------
GT_BASE  = '/scratch/Ananya_Kulkarni/Defog_balanced'
OUT_BASE = '/scratch/Ananya_Kulkarni/Night_balanced'
VDA_ROOT = '/scratch/Ananya_Kulkarni/Video-Depth-Anything'
IMSIZE   = (480, 270)    # (W, H) — matches GT resolution
BATCH    = 4
NUM_CONDITIONS   = 6       # independent continuous draws per clip (default)
MAX_FRAMES_PER_SEQ = 2000  # cap per sequence; shorter clips are used in full,
                           # longer ones (some GoPro dumps run 7-9k frames) are
                           # evenly strided down to this so no single sequence
                           # dominates the dataset just by being longer.
# ---------------------------------------------------------------------------


def _frame_seed(global_seed: int, path: str) -> int:
    """Deterministic per-frame seed: same path → same noise every run."""
    h = int(hashlib.md5(path.encode()).hexdigest(), 16)
    return (global_seed ^ h) & 0xFFFFFFFF


def _select_frames(img_paths, max_frames):
    """Even stride down to ~max_frames, spanning the full clip duration.
    Sequences already at or under the cap are returned unchanged."""
    n = len(img_paths)
    if n <= max_frames:
        return img_paths
    stride = max(1, n // max_frames)
    return img_paths[::stride]


def worker(rank, jobs, out_base, global_seed, num_conditions, max_frames_per_seq):
    pipe = VDAWrapper(encoder='vits', vda_root_path=VDA_ROOT, device_id=rank)
    cond_names = [f'cond{i}' for i in range(num_conditions)]

    for split, seq_id in jobs:
        gt_dir    = os.path.join(GT_BASE, split, 'GT', seq_id)
        img_paths = sorted(
            glob.glob(os.path.join(gt_dir, '*.[jJ][pP][gG]')) +
            glob.glob(os.path.join(gt_dir, '*.[pP][nN][gG]'))
        )
        if not img_paths:
            continue
        img_paths = _select_frames(img_paths, max_frames_per_seq)

        # One continuously-sampled NightEngine per (clip, cond slot) —
        # drawn ONCE here from the full severity spectrum, then reused for
        # every frame of this clip below. Temporally coherent within a
        # sequence, and no two clips (or slots) share the same params —
        # unlike the old fixed-named-preset approach, cond0 doesn't mean
        # "dark" every time; it's just one of num_conditions independent draws.
        clip_engines = {}
        for name in cond_names:
            clip_seed = _frame_seed(global_seed, f'{split}/{seq_id}/{name}')
            clip_rng  = np.random.default_rng(clip_seed)
            clip_engines[name] = NightEngine(**sample_night_clip(clip_rng))

        # Output dirs: {out_base}/{split}/{cond}/{seq_id}/
        for name in cond_names:
            os.makedirs(os.path.join(out_base, split, name, seq_id), exist_ok=True)

        total_batches = (len(img_paths) + BATCH - 1) // BATCH
        pbar = tqdm(
            total=total_batches,
            desc=f"GPU{rank} | {split}/{seq_id}",
            position=rank,
            leave=True,
        )

        for i in range(0, len(img_paths), BATCH):
            batch_paths  = img_paths[i : i + BATCH]
            batch_frames = []
            valid_paths  = []

            for p in batch_paths:
                img = cv2.imread(p)
                if img is not None:
                    batch_frames.append(cv2.resize(img, IMSIZE))
                    valid_paths.append(p)

            if not batch_frames:
                pbar.update(1)
                continue

            # ONE VDA call → depth for the whole batch
            batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_frames]
            depths    = pipe.infer_batch(batch_rgb)

            # num_conditions NightEngine draws per frame (CPU) — lighting
            # fixed per clip (clip_engines), only sensor noise reseeded per frame.
            for frame, depth, path in zip(batch_frames, depths, valid_paths):
                fname = os.path.basename(path)
                rng   = np.random.default_rng(_frame_seed(global_seed, path))
                for name, eng in clip_engines.items():
                    out_path = os.path.join(out_base, split, name, seq_id, fname)
                    if not os.path.exists(out_path):   # resume-safe
                        cv2.imwrite(out_path, eng.apply_night(frame, depth, rng=rng))

            pbar.update(1)

        pbar.close()


def main():
    parser = argparse.ArgumentParser(description='Night synthesis — multi-GPU batch job')
    parser.add_argument('--splits',     nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--num_gpus',   type=int,  default=4)
    parser.add_argument('--out_base',   type=str,  default=OUT_BASE,
                        help='Root output directory (default: Night_balanced)')
    parser.add_argument('--seed',       type=int,  default=42,
                        help='Global RNG seed for reproducible noise (default: 42)')
    parser.add_argument('--num_conditions', type=int, default=NUM_CONDITIONS,
                        help=f'Independent continuous severity draws per clip (default: {NUM_CONDITIONS})')
    parser.add_argument('--max_frames_per_seq', type=int, default=MAX_FRAMES_PER_SEQ,
                        help=f'Cap frames per sequence, evenly strided if longer (default: {MAX_FRAMES_PER_SEQ})')
    args = parser.parse_args()

    # Collect all (split, seq_id) jobs
    all_jobs = []
    for split in args.splits:
        gt_dir = os.path.join(GT_BASE, split, 'GT')
        if not os.path.isdir(gt_dir):
            print(f"[skip] {gt_dir} not found")
            continue
        for seq_id in sorted(os.listdir(gt_dir)):
            all_jobs.append((split, seq_id))

    cond_names = [f'cond{i}' for i in range(args.num_conditions)]
    print(f"\nTotal sequences : {len(all_jobs)}  |  GPUs: {args.num_gpus}")
    print(f"Source GT       : {GT_BASE}/{{split}}/GT/{{seq_id}}/")
    print(f"Output root     : {args.out_base}/{{split}}/{{condition}}/{{seq_id}}/")
    print(f"Conditions      : {cond_names}  (each an independent continuous severity draw, not a fixed named preset)")
    print(f"Max frames/seq  : {args.max_frames_per_seq}  (evenly strided if a sequence is longer)")
    print(f"Noise seed      : {args.seed}  (deterministic per frame path; lighting sampled per clip)\n")

    # Round-robin assignment
    worker_jobs = [[] for _ in range(args.num_gpus)]
    for i, job in enumerate(all_jobs):
        worker_jobs[i % args.num_gpus].append(job)

    for rank, jobs in enumerate(worker_jobs):
        print(f"  GPU {rank}: {[f'{s}/{q}' for s, q in jobs]}")

    procs = []
    for rank in range(args.num_gpus):
        if not worker_jobs[rank]:
            continue
        p = mp.Process(target=worker,
                       args=(rank, worker_jobs[rank], args.out_base, args.seed,
                             args.num_conditions, args.max_frames_per_seq))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("\nAll done.")
    print(f"\nTo train with this data:")
    print(f"  --dataset-root {args.out_base}")
    print(f"  --defog-root   {GT_BASE}")
    print(f"  --input-conditions {' '.join(cond_names)}")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
