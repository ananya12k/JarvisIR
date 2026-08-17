"""
Stage 2: full-factorial Haze generation. Every scene x every fog level, using the
current (fixed) PhysicalFogEngine. Reads frame_plan.json (which frames) and
beta_plan.json (which exact beta per scene+level, randomly sampled within each
level's range by build_uniform_dataset.py — not a fixed point value shared by every
scene) so Haze frames line up 1:1 by filename with the GT/Semantic already copied
into DST_ROOT.
"""
import os
import sys
import json
import cv2
import torch
import numpy as np
from multiprocessing import Process, Queue

sys.path.insert(0, os.path.dirname(__file__))
from fog_synthesis import PhysicalFogEngine, VDAWrapper

DST_ROOT = '/scratch/Ananya_Kulkarni/Defog_balanced_fixed'
VDA_ROOT = '/scratch/Ananya_Kulkarni/Video-Depth-Anything'
# Auto-detect however many GPUs are actually visible (SLURM's --gres=gpu:N sets
# CUDA_VISIBLE_DEVICES accordingly), instead of a hardcoded count that can mismatch
# whatever the SBATCH request actually grants and crash on an out-of-range device.
GPUS = list(range(torch.cuda.device_count()))
BATCH_SIZE = 4

with open(os.path.join(DST_ROOT, 'frame_plan.json')) as f:
    plan = json.load(f)

with open(os.path.join(DST_ROOT, 'beta_plan.json')) as f:
    beta_plan = json.load(f)

with open(os.path.join(DST_ROOT, 'airlight_plan.json')) as f:
    airlight_plan = json.load(f)

with open(os.path.join(DST_ROOT, 'fbm_seed_plan.json')) as f:
    fbm_seed_plan = json.load(f)


def worker(gpu_id, task_queue, done_queue):
    pipe = VDAWrapper(encoder='vits', vda_root_path=VDA_ROOT, device_id=gpu_id)

    while not task_queue.empty():
        try:
            split, scene = task_queue.get(timeout=2)
        except Exception:
            break

        basenames = plan[split][scene]
        scene_betas = beta_plan[split][scene]
        scene_airlight = airlight_plan[split][scene]['color']
        scene_fbm_seed = fbm_seed_plan[split][scene]

        # Resumability: skip levels whose output folder already has every frame from
        # a prior (possibly interrupted) run, so re-running this script is cheap.
        engines, out_dirs = {}, {}
        for name, beta in scene_betas.items():
            d = os.path.join(DST_ROOT, split, 'Haze', f'{scene}_{name}_beta{beta:.4f}')
            os.makedirs(d, exist_ok=True)
            existing = len([f for f in os.listdir(d) if f.endswith('.jpg')])
            if existing >= len(basenames):
                continue
            engines[name] = PhysicalFogEngine(beta_base=beta, hurst=0.75, airlight_color=scene_airlight, fbm_seed=scene_fbm_seed)
            out_dirs[name] = d

        if not engines:
            print(f'[GPU {gpu_id}] skip {split}/{scene} (already complete)', flush=True)
            done_queue.put((split, scene, 0))
            continue

        gt_dir = os.path.join(DST_ROOT, split, 'GT', scene)
        frames = [cv2.imread(os.path.join(gt_dir, bn)) for bn in basenames]

        for i in range(0, len(frames), BATCH_SIZE):
            batch = frames[i:i + BATCH_SIZE]
            batch_names = basenames[i:i + BATCH_SIZE]
            depths = pipe.infer_batch([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch])
            for name in engines:
                for k, frame in enumerate(batch):
                    res = engines[name].apply_fog(frame, depths[k])
                    cv2.imwrite(os.path.join(out_dirs[name], batch_names[k]), res)

        done_queue.put((split, scene, len(frames)))
        print(f'[GPU {gpu_id}] done {split}/{scene} ({len(frames)} frames x {len(scene_betas)} levels)', flush=True)


if __name__ == '__main__':
    task_queue = Queue()
    done_queue = Queue()
    total_tasks = 0
    for split, scenes in plan.items():
        for scene in scenes:
            task_queue.put((split, scene))
            total_tasks += 1

    n_levels = len(next(iter(next(iter(beta_plan.values())).values())))
    print(f'Detected {len(GPUS)} visible GPU(s): {GPUS}', flush=True)
    print(f'Total scene tasks: {total_tasks} (x{n_levels} levels each)', flush=True)

    procs = [Process(target=worker, args=(gid, task_queue, done_queue)) for gid in GPUS]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    print('ALL DONE', flush=True)
