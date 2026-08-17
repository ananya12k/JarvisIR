"""
Stage 1: build the frame-subset plan and copy GT + Semantic into a new, separate
dataset root. Does NOT touch the original Defog_balanced tree at all.

Frame selection: scenes with <=3500 frames use every frame; longer scenes (the 3
GoPro clips) are evenly subsampled down to 3500 indices spanning the full clip, so
temporal coverage isn't just the first N frames.
"""
import os
import glob
import shutil
import json
import random
import numpy as np

SRC_ROOT = '/scratch/Ananya_Kulkarni/Defog_balanced'
DST_ROOT = '/scratch/Ananya_Kulkarni/Defog_balanced_fixed'
FRAME_CAP = 3500
SPLITS = ['train', 'val', 'test']

# Ranges, not fixed points — each scene gets its own randomly sampled beta within
# its level's range, same approach the original dataset used (e.g. "light" was never
# exactly one beta across scenes). light_medium's range exactly fills the gap that
# used to sit between light and medium; no extreme range (dropped per current scope).
LEVEL_RANGES = {
    'light':        (0.008, 0.020),
    'light_medium': (0.020, 0.030),
    'medium':       (0.030, 0.050),
    'heavy':        (0.070, 0.120),
}

# Named airlight colors spanning realistic fog/haze lighting conditions. Kept
# grey-ish (no wild hues) so every sample stays physically plausible, but the
# hue tilt + jitter give the model real color variation to generalize from,
# instead of every scene using the same fixed grey (PhysicalFogEngine's default).
# One color per scene (not per level) since airlight is about ambient lighting/
# atmosphere at the time the scene was shot, not fog density.
AIRLIGHT_PRESETS = {
    'overcast':    (0.78, 0.79, 0.80),  # neutral grey, overcast daylight
    'warm_dusk':   (0.82, 0.76, 0.68),  # sunset/dusk haze, warm tint
    'cool_dawn':   (0.75, 0.79, 0.85),  # dawn haze, cool blue tint
    'bright_haze': (0.85, 0.85, 0.83),  # bright midday, near-white
    'yellow_haze': (0.80, 0.77, 0.65),  # polluted/industrial haze, yellow-brown tint
}

plan = {}
beta_plan = {}
airlight_plan = {}
fbm_seed_plan = {}

for split in SPLITS:
    gt_root = os.path.join(SRC_ROOT, split, 'GT')
    sem_root = os.path.join(SRC_ROOT, split, 'Semantic')
    scenes = sorted(os.listdir(gt_root))
    plan[split] = {}
    beta_plan[split] = {}
    airlight_plan[split] = {}
    fbm_seed_plan[split] = {}

    for scene in scenes:
        # Deterministic per (split, scene, level) so re-running this script always
        # reproduces the same beta assignments rather than drawing new ones each time.
        rng = random.Random(f'{split}/{scene}')
        beta_plan[split][scene] = {
            name: round(rng.uniform(*rng_range), 4) for name, rng_range in LEVEL_RANGES.items()
        }

        # Separate deterministic rng (distinct seed string) so airlight draws don't
        # correlate with beta draws for the same scene.
        airlight_rng = random.Random(f'{split}/{scene}/airlight')
        airlight_name = airlight_rng.choice(list(AIRLIGHT_PRESETS.keys()))
        base_color = AIRLIGHT_PRESETS[airlight_name]
        color = [
            round(min(0.95, max(0.5, c + airlight_rng.uniform(-0.03, 0.03))), 4)
            for c in base_color
        ]
        airlight_plan[split][scene] = {'name': airlight_name, 'color': color}

        # Per-scene fBm seed (distinct rng again, separate from beta/airlight draws).
        # One seed per scene, shared across all fog levels of that scene — the fBm
        # field is still generated once and only spatially windowed as it drifts
        # frame-to-frame (see PhysicalFogEngine), so this does not introduce any
        # per-frame variation or touch temporal consistency at all. It only stops
        # every scene in the dataset from reusing the exact same seed=42 texture.
        seed_rng = random.Random(f'{split}/{scene}/fbm_seed')
        fbm_seed_plan[split][scene] = seed_rng.randint(0, 2**31 - 1)

        gt_dir = os.path.join(gt_root, scene)
        sem_dir = os.path.join(sem_root, scene)
        frame_paths = sorted(glob.glob(os.path.join(gt_dir, '*.jpg')))
        n = len(frame_paths)

        if n <= FRAME_CAP:
            chosen = frame_paths
        else:
            idx = np.linspace(0, n - 1, FRAME_CAP, dtype=int)
            chosen = [frame_paths[i] for i in idx]

        basenames = [os.path.basename(p) for p in chosen]
        plan[split][scene] = basenames

        dst_gt_dir = os.path.join(DST_ROOT, split, 'GT', scene)
        dst_sem_dir = os.path.join(DST_ROOT, split, 'Semantic', scene)
        os.makedirs(dst_gt_dir, exist_ok=True)
        os.makedirs(dst_sem_dir, exist_ok=True)

        for bn in basenames:
            src_gt = os.path.join(gt_dir, bn)
            dst_gt = os.path.join(dst_gt_dir, bn)
            if not os.path.exists(dst_gt):
                shutil.copy2(src_gt, dst_gt)

            sem_bn = os.path.splitext(bn)[0] + '.npy'
            src_sem = os.path.join(sem_dir, sem_bn)
            dst_sem = os.path.join(dst_sem_dir, sem_bn)
            if os.path.exists(src_sem) and not os.path.exists(dst_sem):
                shutil.copy2(src_sem, dst_sem)

        print(f'{split}/{scene}: {n} available -> {len(chosen)} selected')

with open(os.path.join(DST_ROOT, 'frame_plan.json'), 'w') as f:
    json.dump(plan, f)

with open(os.path.join(DST_ROOT, 'beta_plan.json'), 'w') as f:
    json.dump(beta_plan, f, indent=2)

with open(os.path.join(DST_ROOT, 'airlight_plan.json'), 'w') as f:
    json.dump(airlight_plan, f, indent=2)

with open(os.path.join(DST_ROOT, 'fbm_seed_plan.json'), 'w') as f:
    json.dump(fbm_seed_plan, f, indent=2)

total = sum(len(v) for split in plan.values() for v in split.values())
print(f'\nTotal GT/Semantic frames copied: {total}')
print(f'Plan saved to {os.path.join(DST_ROOT, "frame_plan.json")}')
print(f'Beta assignments saved to {os.path.join(DST_ROOT, "beta_plan.json")}')
print(f'Airlight assignments saved to {os.path.join(DST_ROOT, "airlight_plan.json")}')
print(f'fBm seed assignments saved to {os.path.join(DST_ROOT, "fbm_seed_plan.json")}')
