"""
Build the paired EDFog video dataset (foggy / clean) with fog_synthesis.py.

REBUILD (2026-11-01) of the purged driver. CLI flags match the surviving call in
VISIA_old/build_diverse_mor_edfog.sh; output layout and manifest columns match
what regen_from_manifest.py / regen_ladder.py / video_dataset.py consume.

Two ways to get per-clip parameters:
  sample (default)   stratified per split: the MOR ladder, airlight levels,
                     fog-layer heights and heterogeneity modes are each spread as
                     evenly as possible over the split's (scene,config) pairs, so
                     val and test contain every fog variant that train does; no
                     MOR repeats within a scene. fbm_seed = SHA-256(global_seed|
                     scene|config).
  --from_manifest    replay an existing manifest.csv row-for-row (e.g. the
                     recovered train/val rows of the shipped EDFog).

Input: one folder per split holding <scene>/<frames>, e.g.
  --train_root .../Defog_balanced_fogv2/train/GT

Output:
  <out_root>/<split>/GT/<scene>/<frame>                     (symlinks, or --copy_gt)
  <out_root>/<split>/Haze/<scene>_cfg<i>_beta<b:.4f>/<frame>.png + meta.json
  <out_root>/manifest.csv

Depth is computed ONCE per scene (overlapping VDA sliding window) and every
config of that scene is rendered from the same depth stream in the same pass.
Multi-GPU: a single run uses every visible GPU. Scene jobs go into one shared
queue, longest first, and --jobs_per_gpu workers per GPU each pull the next
scene when free, so no GPU idles while work remains. (--shard_index/--num_shards
still split the scene list across separate runs if needed.)
"""
import argparse
import csv
import glob
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import shutil
import sys

import cv2
import numpy as np
from tqdm import tqdm

MANIFEST_FIELDS = ["split", "scene", "config_idx", "beta_base", "mor", "airlight_color",
                   "fog_layer_m", "heterogeneity", "fbm_seed", "speed_kmh", "fps",
                   "n_frames", "gt_path", "haze_path"]
AIRLIGHT_LEVELS = (0.64, 0.72, 0.82)
FOG_LAYER_LEVELS_M = (8.0, 20.0, 40.0)
DEFAULT_HETEROGENEITY_MIX = "patchy:0.1,uniform:0.9"      # paper: patchy p=0.1


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big")


def load_engine(engine_file, vda_root):
    # VDA does `from utils.util import ...`; VDA root must win sys.path[0] and cwd.
    sys.path.insert(0, vda_root)
    os.chdir(vda_root)
    spec = importlib.util.spec_from_file_location("_fog_engine", engine_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def list_frames(d):
    return sorted(glob.glob(os.path.join(d, "*.[jJ][pP][gG]")) +
                  glob.glob(os.path.join(d, "*.[pP][nN][gG]")))


def clip_name(scene, cfg, beta):
    return f"{scene}_cfg{cfg}_beta{beta:.4f}"


def parse_mix(spec):
    mix = [(name.strip(), float(p)) for name, p in (item.split(":") for item in spec.split(",") if item.strip())]
    total = sum(p for _, p in mix)
    return [(name, p / total) for name, p in mix]


def balanced(levels, n, rng):
    """n values covering every level as evenly as possible (counts differ by <= 1), shuffled."""
    reps = -(-n // len(levels))
    return list(rng.permutation(np.concatenate([rng.permutation(levels) for _ in range(reps)])[:n]))


def mix_counts(mix, n):
    """Clip count per heterogeneity mode: proportional to p, but >= 1 for every mode with
    p > 0 whenever the split has enough clips, so each split contains every fog type."""
    counts = {name: int(round(p * n)) for name, p in mix}
    if n >= len(mix):
        for name, p in mix:
            if p > 0 and counts[name] == 0:
                counts[name] = 1
    while sum(counts.values()) > n:
        counts[max(counts, key=counts.get)] -= 1
    while sum(counts.values()) < n:
        counts[max(mix, key=lambda m: m[1])[0]] += 1
    return counts


def assign_mors(tasks, ladder, rng):
    """Balanced MOR ladder over the split (each level an equal share, +-1) with no MOR
    repeated within a scene when the scene has <= len(ladder) configs."""
    pool = {m: c for m, c in zip(*np.unique(balanced(ladder, len(tasks), rng), return_counts=True))}
    by_scene = {}
    for scene, cfg in tasks:
        by_scene.setdefault(scene, []).append(cfg)
    out = {}
    for scene in rng.permutation(sorted(by_scene)):
        used = set()
        for cfg in by_scene[scene]:
            cands = [m for m, c in pool.items() if c > 0 and m not in used] or [m for m, c in pool.items() if c > 0]
            top = max(pool[m] for m in cands)
            m = rng.choice(sorted(m for m in cands if pool[m] == top))
            pool[m] -= 1
            used.add(m)
            out[(scene, cfg)] = float(m)
    return out


def sample_plan(scenes_by_split, n_cfg, mor_ladder, global_seed, fps, mix):
    """
    Stratified, per split: every split (train, val AND test) covers every MOR level,
    every heterogeneity mode, every airlight level and every fog-layer height as
    evenly as its clip count allows, instead of i.i.d. draws that can leave a level
    out of a small eval split. Deterministic from global_seed; fbm_seed stays a
    SHA-256 function of (scene, config).
    """
    rows = []
    for split, scenes in scenes_by_split.items():
        tasks = [(scene, cfg) for scene in sorted(scenes) for cfg in range(n_cfg[split])]
        if not tasks:
            continue
        n = len(tasks)
        rng = np.random.default_rng(stable_seed(global_seed, "plan", split))
        mors = assign_mors(tasks, list(mor_ladder), rng)
        airlights = balanced(list(AIRLIGHT_LEVELS), n, rng)
        layers = balanced(list(FOG_LAYER_LEVELS_M), n, rng)
        hets = list(rng.permutation([name for name, c in mix_counts(mix, n).items() for _ in range(c)]))
        for i, (scene, cfg) in enumerate(tasks):
            mor = mors[(scene, cfg)]
            a = float(airlights[i])
            rows.append(dict(
                split=split, scene=scene, config_idx=cfg,
                beta_base=round(3.0 / mor, 6), mor=mor,
                airlight_color=str([a, a, a]),
                fog_layer_m=float(layers[i]),
                heterogeneity=str(hets[i]),
                fbm_seed=int(np.random.default_rng(stable_seed(global_seed, scene, cfg)).integers(0, 2**32)),
                speed_kmh=0.0, fps=fps,
            ))
    return rows


def manifest_plan(path, splits):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in splits]
    for r in rows:
        r["config_idx"] = int(r["config_idx"])
        for k in ("beta_base", "mor", "fog_layer_m", "speed_kmh", "fps"):
            r[k] = float(r[k])
        r["fbm_seed"] = int(r["fbm_seed"])
    return rows


def materialize_gt(frame_paths, dst_dir, copy, resize=None):
    """Writes the clean reference; returns the frame paths rendering should read."""
    os.makedirs(dst_dir, exist_ok=True)
    out = []
    for p in frame_paths:
        if resize:
            dst = os.path.join(dst_dir, os.path.splitext(os.path.basename(p))[0] + ".png")
            if not os.path.exists(dst):
                cv2.imwrite(dst, cv2.resize(cv2.imread(p), resize, interpolation=cv2.INTER_AREA))
            out.append(dst)
            continue
        dst = os.path.join(dst_dir, os.path.basename(p))
        if not os.path.lexists(dst):
            if copy:
                shutil.copy2(p, dst)
            else:
                os.symlink(os.path.abspath(p), dst)
        out.append(p)
    return out


def write_manifest(rows, out_root, name="manifest.csv"):
    path = os.path.join(out_root, name)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine_file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fog_synthesis.py"))
    p.add_argument("--train_root", default=None)
    p.add_argument("--val_root", default=None)
    p.add_argument("--test_root", default=None)
    p.add_argument("--out_root", required=True)
    p.add_argument("--from_manifest", default=None, help="replay this manifest instead of sampling")
    p.add_argument("--n_configs_train", type=int, default=4)
    p.add_argument("--n_configs_eval", type=int, default=2)
    p.add_argument("--n_configs_test", type=int, default=None, help="default: --n_configs_eval")
    p.add_argument("--mor_ladder", default="50,75,100,150,200,300")
    p.add_argument("--heterogeneity_mix", default=DEFAULT_HETEROGENEITY_MIX,
                   help="e.g. 'uniform:0.4,patchy_strong:0.6'")
    p.add_argument("--resize", default="", help="WxH, e.g. 480x270 (the shipped EDFog size); default native")
    p.add_argument("--global_seed", default="edfog")
    p.add_argument("--scenes", default="", help="comma list restricting scenes (all splits)")
    p.add_argument("--max_frames", type=int, default=0, help="first N frames per scene; 0 = all")
    p.add_argument("--vda_root", default="/scratch/Ananya_Kulkarni/Video-Depth-Anything")
    p.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    p.add_argument("--num_gpus", type=int, default=0, help="0 = all visible GPUs")
    p.add_argument("--jobs_per_gpu", type=int, default=2,
                   help="workers per GPU; >1 overlaps one worker's CPU fog maths with another's VDA")
    p.add_argument("--threads_per_worker", type=int, default=2)
    p.add_argument("--gpu_id", type=int, default=0, help="GPU to use when --num_gpus 1")
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--vda_batch", type=int, default=4)
    p.add_argument("--vda_overlap", type=int, default=2)
    p.add_argument("--copy_gt", action="store_true", help="copy GT frames instead of symlinking")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--manifest_name", default="manifest.csv")
    p.add_argument("--plan_only", action="store_true", help="write manifest.csv and exit")
    args = p.parse_args()

    out_root = os.path.abspath(args.out_root)
    resize = tuple(int(v) for v in args.resize.lower().split("x")) if args.resize else None
    split_roots = {s: os.path.abspath(r) for s, r in
                   (("train", args.train_root), ("val", args.val_root), ("test", args.test_root)) if r}
    scene_filter = {s.strip() for s in args.scenes.split(",") if s.strip()}

    if args.from_manifest:
        rows = manifest_plan(args.from_manifest, set(split_roots) or {"train", "val", "test"})
        if scene_filter:
            rows = [r for r in rows if r["scene"] in scene_filter]
        # Source frames come from --<split>_root if given, else the row's own gt_path.
        src_dir = {(r["split"], r["scene"]): os.path.join(split_roots[r["split"]], r["scene"])
                   if r["split"] in split_roots else r["gt_path"] for r in rows}
    else:
        if not split_roots:
            p.error("give at least one of --train_root / --val_root / --test_root")
        scenes_by_split = {}
        for split, root in split_roots.items():
            scenes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
            scenes_by_split[split] = [s for s in scenes if not scene_filter or s in scene_filter]
        n_test = args.n_configs_eval if args.n_configs_test is None else args.n_configs_test
        n_cfg = {"train": args.n_configs_train, "val": args.n_configs_eval, "test": n_test}
        mor_ladder = [float(m) for m in args.mor_ladder.split(",")]
        rows = sample_plan(scenes_by_split, n_cfg, mor_ladder, args.global_seed, args.fps,
                           parse_mix(args.heterogeneity_mix))
        src_dir = {(r["split"], r["scene"]): os.path.join(split_roots[r["split"]], r["scene"]) for r in rows}

    frames_by_scene = {}
    for key, d in src_dir.items():
        fr = list_frames(d)
        frames_by_scene[key] = fr[:args.max_frames] if args.max_frames > 0 else fr
    for r in rows:
        key = (r["split"], r["scene"])
        r["n_frames"] = len(frames_by_scene[key])
        r["gt_path"] = os.path.join(out_root, r["split"], "GT", r["scene"])
        r["haze_path"] = os.path.join(out_root, r["split"], "Haze",
                                      clip_name(r["scene"], r["config_idx"], r["beta_base"]))

    os.makedirs(out_root, exist_ok=True)
    write_manifest(rows, out_root, args.manifest_name)
    print(f"[plan] {len(rows)} clip(s) over {len(src_dir)} scene(s) -> {out_root}/{args.manifest_name}", flush=True)
    for split in ("train", "val", "test"):
        mors = sorted(int(r["mor"]) for r in rows if r["split"] == split)
        if mors:
            print(f"[plan] {split}: MOR counts " +
                  ", ".join(f"{m}m x{mors.count(m)}" for m in sorted(set(mors))), flush=True)
    if args.plan_only:
        return

    by_scene = {}
    for r in rows:
        by_scene.setdefault((r["split"], r["scene"]), []).append(r)
    scene_keys = sorted(by_scene)[args.shard_index::args.num_shards]
    # Longest first (cost ~ frames x (depth pass + one fog pass per config)), so
    # short scenes fill the gaps at the end and no worker idles while others run.
    jobs = [(split, scene, frames_by_scene[(split, scene)], by_scene[(split, scene)])
            for split, scene in scene_keys]
    jobs.sort(key=lambda j: -len(j[2]) * (1 + len(j[3])))
    if not jobs:
        return

    if args.num_gpus == 1:
        gpus = [args.gpu_id]
    else:
        import torch
        gpus = list(range(args.num_gpus or torch.cuda.device_count())) or [0]
    slots = [g for g in gpus for _ in range(args.jobs_per_gpu)]
    print(f"[queue] {len(jobs)} scene job(s) | {len(gpus)} GPU(s) x {args.jobs_per_gpu} worker(s)", flush=True)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    for j in jobs:
        queue.put(j)
    for _ in slots:
        queue.put(None)
    procs = [ctx.Process(target=worker, args=(rank, gpu, queue, args, out_root, resize))
             for rank, gpu in enumerate(slots)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    failed = os.path.join(out_root, "failed_jobs.txt")
    if os.path.exists(failed):
        print(f"[queue] some scenes FAILED, see {failed}", flush=True)
    print("ALL DONE", flush=True)


def worker(rank, gpu, queue, args, out_root, resize):
    """One worker = one VDA instance on `gpu`; pulls scene jobs until the sentinel."""
    cv2.setNumThreads(args.threads_per_worker)
    import torch
    torch.set_num_threads(args.threads_per_worker)
    mod = load_engine(args.engine_file, args.vda_root)
    vda = None
    tag = f"[GPU {gpu}.{rank}]"
    while True:
        job = queue.get()
        if job is None:
            return
        split, scene = job[0], job[1]
        try:
            if vda is None:
                vda = mod.VDAWrapper(encoder=args.encoder, vda_root_path=args.vda_root, device_id=gpu)
            render_scene(mod, vda, job, args, out_root, resize, tag)
        except Exception as e:  # keep the other scenes going; report at the end
            import traceback
            traceback.print_exc()
            with open(os.path.join(out_root, "failed_jobs.txt"), "a") as f:
                f.write(f"{split}/{scene}\t{type(e).__name__}: {e}\n")


def render_scene(mod, vda, job, args, out_root, resize, tag):
    split, scene, frame_paths, scene_rows = job
    if not frame_paths:
        print(f"WARNING: no frames for {split}/{scene}", flush=True)
        return
    frame_paths = materialize_gt(frame_paths, os.path.join(out_root, split, "GT", scene),
                                 args.copy_gt, resize)

    todo = []
    for r in scene_rows:
        if args.skip_existing and len(glob.glob(os.path.join(r["haze_path"], "*.png"))) == len(frame_paths):
            print(f"{tag} skip {split}/{os.path.basename(r['haze_path'])}", flush=True)
            continue
        todo.append(r)
    if not todo:
        return

    engines = []
    for r in todo:
        os.makedirs(r["haze_path"], exist_ok=True)
        engines.append(mod.PhysicalFogEngine(
            beta_base=r["beta_base"], airlight_color=json.loads(r["airlight_color"]),
            fog_layer_m=r["fog_layer_m"], heterogeneity=r["heterogeneity"],
            fbm_seed=r["fbm_seed"], speed_kmh=r["speed_kmh"], fps=r["fps"]))

    for idx, frame, depth in tqdm(mod.iter_scene_depth(vda, frame_paths, args.vda_batch, args.vda_overlap),
                                  total=len(frame_paths), desc=f"{tag} {split}/{scene} x{len(todo)}",
                                  mininterval=30):
        stem = os.path.splitext(os.path.basename(frame_paths[idx]))[0]
        for r, eng in zip(todo, engines):
            cv2.imwrite(os.path.join(r["haze_path"], stem + ".png"), eng.apply_fog(frame, depth))

    for r, eng in zip(todo, engines):
        meta = {k: r[k] for k in MANIFEST_FIELDS}
        meta.update(hurst=eng.hurst, patchiness=eng.patchiness, octaves=eng.octaves,
                    corr_len_m=eng.corr_len_m, scatter_frac=eng.scatter_frac,
                    traveled_m=eng.traveled_m, engine_file=os.path.abspath(args.engine_file))
        with open(os.path.join(r["haze_path"], "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    print(f"{tag} done {split}/{scene} ({len(frame_paths)} frames x {len(todo)} configs)", flush=True)


if __name__ == "__main__":
    main()
