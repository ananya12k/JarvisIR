"""
Stage 2: full-factorial night generation. Every (split, scene) x every
condition slot (cond0..condN-1), using the current NightEngine. Reads
frame_plan.json (which frames) and severity_plan.json (exact per-clip params
per condition, sampled once by build_uniform_night_dataset.py) so night
frames line up 1:1 by filename with the GT already copied into DST_ROOT.

Lighting (severity_plan) is fixed per clip -> temporally coherent within a
sequence. Sensor noise is reseeded per frame (see _frame_noise_rng below),
independent of clip lighting -> not temporally static, the way real camera
noise behaves.
"""
import os
import sys
import json
import random
import cv2
import torch
import numpy as np
from multiprocessing import Process, Queue

sys.path.insert(0, os.path.dirname(__file__))
from night_synthesis import NightEngine, VDAWrapper

DST_ROOT = '/scratch/Ananya_Kulkarni/Night_balanced_uniform'
VDA_ROOT = '/scratch/Ananya_Kulkarni/Video-Depth-Anything'
# Auto-detect however many GPUs are actually visible (SLURM's --gres=gpu:N sets
# CUDA_VISIBLE_DEVICES accordingly), instead of a hardcoded count that can
# mismatch whatever the SBATCH request actually grants and crash on an
# out-of-range device.
GPUS = list(range(torch.cuda.device_count()))
BATCH_SIZE = 4
IMSIZE = (480, 270)   # (W, H) — matches GT resolution

with open(os.path.join(DST_ROOT, 'frame_plan.json')) as f:
    frame_plan = json.load(f)
with open(os.path.join(DST_ROOT, 'severity_plan.json')) as f:
    severity_plan = json.load(f)


def _frame_noise_rng(split, scene, fname):
    """Deterministic per-frame noise seed, independent of clip lighting/cond —
    same convention as build_uniform_night_dataset.py's string-seeded random.Random."""
    seed_rng = random.Random(f'{split}/{scene}/{fname}/noise')
    return np.random.default_rng(seed_rng.getrandbits(32))


def worker(gpu_id, task_queue, done_queue):
    pipe = VDAWrapper(encoder='vits', vda_root_path=VDA_ROOT, device_id=gpu_id)

    while not task_queue.empty():
        try:
            split, scene = task_queue.get(timeout=2)
        except Exception:
            break

        basenames = frame_plan[split][scene]
        scene_conds = severity_plan[split][scene]

        # Resumability: skip conditions whose output folder already has every
        # frame from a prior (possibly interrupted) run.
        engines, out_dirs = {}, {}
        for cond, params in scene_conds.items():
            d = os.path.join(DST_ROOT, split, cond, scene)
            os.makedirs(d, exist_ok=True)
            existing = len([f for f in os.listdir(d) if f.endswith(('.jpg', '.png'))])
            if existing >= len(basenames):
                continue
            engines[cond] = NightEngine(**params)
            out_dirs[cond] = d

        if not engines:
            print(f'[GPU {gpu_id}] skip {split}/{scene} (already complete)', flush=True)
            done_queue.put((split, scene, 0))
            continue

        gt_dir = os.path.join(DST_ROOT, split, 'GT', scene)
        frames = [cv2.resize(cv2.imread(os.path.join(gt_dir, bn)), IMSIZE) for bn in basenames]

        for i in range(0, len(frames), BATCH_SIZE):
            batch = frames[i:i + BATCH_SIZE]
            batch_names = basenames[i:i + BATCH_SIZE]
            depths = pipe.infer_batch([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch])
            for cond, eng in engines.items():
                for k, frame in enumerate(batch):
                    rng = _frame_noise_rng(split, scene, batch_names[k])
                    res = eng.apply_night(frame, depths[k], rng=rng)
                    cv2.imwrite(os.path.join(out_dirs[cond], batch_names[k]), res)

        done_queue.put((split, scene, len(frames)))
        print(f'[GPU {gpu_id}] done {split}/{scene} '
              f'({len(frames)} frames x {len(scene_conds)} conditions)', flush=True)


if __name__ == '__main__':
    task_queue = Queue()
    done_queue = Queue()
    total_tasks = 0
    for split, scenes in frame_plan.items():
        for scene in scenes:
            task_queue.put((split, scene))
            total_tasks += 1

    n_conditions = len(next(iter(next(iter(severity_plan.values())).values())))
    print(f'Detected {len(GPUS)} visible GPU(s): {GPUS}', flush=True)
    print(f'Total scene tasks: {total_tasks} (x{n_conditions} conditions each)', flush=True)

    procs = [Process(target=worker, args=(gid, task_queue, done_queue)) for gid in GPUS]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    print('ALL DONE', flush=True)
