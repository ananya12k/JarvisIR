"""
Stage 1: build the frame-subset plan + per-clip severity plan, and copy GT into
a new, separate, self-contained dataset root. Does NOT touch Defog_balanced.

Frame selection: sequences with <=MAX_FRAMES_PER_SEQ frames use every frame;
longer sequences (the GoPro dumps that run 7-9k frames) are evenly subsampled
down to that cap, spanning the full clip, so temporal coverage isn't just the
first N frames and no single sequence dominates the dataset by raw length.

Severity: NUM_CONDITIONS independent continuous draws per (split, scene) from
night_synthesis.sample_night_clip — persisted to severity_plan.json so Stage 2
(generate_uniform_night.py) renders exactly this plan instead of drawing fresh
random params, and re-running Stage 1 doesn't silently reshuffle an
already-generated dataset. Mirrors build_uniform_dataset.py's beta_plan.json
convention for fog.
"""
import os
import sys
import glob
import json
import random
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from night_synthesis import sample_night_clip

SRC_ROOT = '/scratch/Ananya_Kulkarni/Defog_balanced'
DST_ROOT = '/scratch/Ananya_Kulkarni/Night_balanced_uniform'
SPLITS = ['train', 'val', 'test']
MAX_FRAMES_PER_SEQ = 2000
NUM_CONDITIONS = 6


def select_frames(frame_paths, max_frames):
    """Even stride down to max_frames, spanning the full clip duration.
    Sequences already at or under the cap are returned unchanged."""
    n = len(frame_paths)
    if n <= max_frames:
        return frame_paths
    idx = np.linspace(0, n - 1, max_frames, dtype=int)
    return [frame_paths[i] for i in idx]


def to_jsonable(params):
    """Cast numpy scalars (float32 etc, not JSON-serialisable) to plain python types."""
    out = {}
    for k, v in params.items():
        if k == 'lights':
            out[k] = [{lk: float(lv) for lk, lv in light.items()} for light in v]
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


frame_plan = {}
severity_plan = {}

for split in SPLITS:
    gt_root = os.path.join(SRC_ROOT, split, 'GT')
    if not os.path.isdir(gt_root):
        print(f'[skip] {gt_root} not found')
        continue
    scenes = sorted(os.listdir(gt_root))
    frame_plan[split] = {}
    severity_plan[split] = {}

    for scene in scenes:
        gt_dir = os.path.join(gt_root, scene)
        frame_paths = sorted(
            glob.glob(os.path.join(gt_dir, '*.[jJ][pP][gG]')) +
            glob.glob(os.path.join(gt_dir, '*.[pP][nN][gG]'))
        )
        n = len(frame_paths)
        chosen = select_frames(frame_paths, MAX_FRAMES_PER_SEQ)
        basenames = [os.path.basename(p) for p in chosen]
        frame_plan[split][scene] = basenames

        # Deterministic per (split, scene, cond) — random.Random with a string
        # seed hashes via sha512 internally, so (unlike the builtin hash()) this
        # is reproducible across runs/processes regardless of PYTHONHASHSEED.
        # Same convention as build_uniform_dataset.py's beta/airlight/fbm plans.
        severity_plan[split][scene] = {}
        for i in range(NUM_CONDITIONS):
            cond = f'cond{i}'
            seed_rng = random.Random(f'{split}/{scene}/{cond}')
            np_rng = np.random.default_rng(seed_rng.getrandbits(32))
            params = sample_night_clip(np_rng)
            severity_plan[split][scene][cond] = to_jsonable(params)

        # Copy GT into the new self-contained root
        dst_gt_dir = os.path.join(DST_ROOT, split, 'GT', scene)
        os.makedirs(dst_gt_dir, exist_ok=True)
        for bn in basenames:
            src = os.path.join(gt_dir, bn)
            dst = os.path.join(dst_gt_dir, bn)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        print(f'{split}/{scene}: {n} available -> {len(chosen)} selected')

os.makedirs(DST_ROOT, exist_ok=True)
with open(os.path.join(DST_ROOT, 'frame_plan.json'), 'w') as f:
    json.dump(frame_plan, f)
with open(os.path.join(DST_ROOT, 'severity_plan.json'), 'w') as f:
    json.dump(severity_plan, f, indent=2)

total = sum(len(v) for split in frame_plan.values() for v in split.values())
print(f'\nTotal GT frames copied  : {total}')
print(f'Conditions per sequence : {NUM_CONDITIONS}')
print(f'Frame plan saved to     {os.path.join(DST_ROOT, "frame_plan.json")}')
print(f'Severity plan saved to  {os.path.join(DST_ROOT, "severity_plan.json")}')
print(f'\nNext: python generate_uniform_night.py')
