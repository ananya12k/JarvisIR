"""
Build the paired EDFog video dataset (foggy / clean) with fog_synthesis.py.

REBUILD (2026-11-01) of the purged driver. CLI flags match the surviving call in
VISIA_old/build_diverse_mor_edfog.sh; output layout and manifest columns match
what regen_from_manifest.py / regen_ladder.py / video_dataset.py consume.

Two ways to get per-clip parameters:
  sample (default)   balanced MOR ladder {50,75,100,150,200,300} m, repeated and
                     deterministically shuffled across the split's (scene,config)
                     pairs; airlight / fog layer / heterogeneity / fbm_seed drawn
                     from SHA-256(global_seed|scene|config). The seed deliberately
                     excludes the split (the shipped manifest shows train/GH019009
                     and val/GH019009 sharing fbm_seeds per config).
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
Multi-GPU: one process per GPU with --gpu_id/--shard_index/--num_shards (sharded
by scene). The manifest is a pure function of the arguments, so every shard
writes the same file.
"""
import argparse
import csv
import glob
import hashlib
import importlib.util
import json
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
P_PATCHY = 0.1


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


def sample_plan(scenes_by_split, n_cfg, mor_ladder, global_seed, fps):
    rows = []
    for split, scenes in scenes_by_split.items():
        tasks = [(scene, cfg) for scene in sorted(scenes) for cfg in range(n_cfg[split])]
        if not tasks:
            continue
        reps = -(-len(tasks) // len(mor_ladder))
        mors = np.array(list(mor_ladder) * reps)[:len(tasks)]
        mors = np.random.default_rng(stable_seed(global_seed, "mor_ladder", split)).permutation(mors)
        for (scene, cfg), mor in zip(tasks, mors):
            rng = np.random.default_rng(stable_seed(global_seed, scene, cfg))
            a = float(rng.choice(AIRLIGHT_LEVELS))
            rows.append(dict(
                split=split, scene=scene, config_idx=cfg,
                beta_base=round(3.0 / float(mor), 6), mor=float(mor),
                airlight_color=str([a, a, a]),
                fog_layer_m=float(rng.choice(FOG_LAYER_LEVELS_M)),
                heterogeneity="patchy" if rng.random() < P_PATCHY else "uniform",
                fbm_seed=int(rng.integers(0, 2**32)),
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


def materialize_gt(frame_paths, dst_dir, copy):
    os.makedirs(dst_dir, exist_ok=True)
    for p in frame_paths:
        dst = os.path.join(dst_dir, os.path.basename(p))
        if os.path.lexists(dst):
            continue
        if copy:
            shutil.copy2(p, dst)
        else:
            os.symlink(os.path.abspath(p), dst)


def write_manifest(rows, out_root):
    path = os.path.join(out_root, "manifest.csv")
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
    p.add_argument("--mor_ladder", default="50,75,100,150,200,300")
    p.add_argument("--global_seed", default="edfog")
    p.add_argument("--scenes", default="", help="comma list restricting scenes (all splits)")
    p.add_argument("--max_frames", type=int, default=0, help="first N frames per scene; 0 = all")
    p.add_argument("--vda_root", default="/scratch/Ananya_Kulkarni/Video-Depth-Anything")
    p.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--vda_batch", type=int, default=4)
    p.add_argument("--vda_overlap", type=int, default=2)
    p.add_argument("--copy_gt", action="store_true", help="copy GT frames instead of symlinking")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--plan_only", action="store_true", help="write manifest.csv and exit")
    args = p.parse_args()

    out_root = os.path.abspath(args.out_root)
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
        n_cfg = {"train": args.n_configs_train, "val": args.n_configs_eval, "test": args.n_configs_eval}
        mor_ladder = [float(m) for m in args.mor_ladder.split(",")]
        rows = sample_plan(scenes_by_split, n_cfg, mor_ladder, args.global_seed, args.fps)
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
    write_manifest(rows, out_root)
    print(f"[plan] {len(rows)} clip(s) over {len(src_dir)} scene(s) -> {out_root}/manifest.csv", flush=True)
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
    if not scene_keys:
        return

    mod = load_engine(args.engine_file, args.vda_root)
    vda = None

    for split, scene in scene_keys:
        frame_paths = frames_by_scene[(split, scene)]
        if not frame_paths:
            print(f"WARNING: no frames for {split}/{scene} at {src_dir[(split, scene)]}", flush=True)
            continue
        materialize_gt(frame_paths, os.path.join(out_root, split, "GT", scene), args.copy_gt)

        todo = []
        for r in by_scene[(split, scene)]:
            if args.skip_existing and len(glob.glob(os.path.join(r["haze_path"], "*.png"))) == len(frame_paths):
                print(f"[GPU {args.gpu_id}] skip {split}/{os.path.basename(r['haze_path'])}", flush=True)
                continue
            todo.append(r)
        if not todo:
            continue

        engines = []
        for r in todo:
            os.makedirs(r["haze_path"], exist_ok=True)
            engines.append(mod.PhysicalFogEngine(
                beta_base=r["beta_base"], airlight_color=json.loads(r["airlight_color"]),
                fog_layer_m=r["fog_layer_m"], heterogeneity=r["heterogeneity"],
                fbm_seed=r["fbm_seed"], speed_kmh=r["speed_kmh"], fps=r["fps"]))

        if vda is None:
            vda = mod.VDAWrapper(encoder=args.encoder, vda_root_path=args.vda_root, device_id=args.gpu_id)
        for idx, frame, depth in tqdm(mod.iter_scene_depth(vda, frame_paths, args.vda_batch, args.vda_overlap),
                                      total=len(frame_paths), desc=f"GPU {args.gpu_id} {split}/{scene} x{len(todo)}"):
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
        print(f"[GPU {args.gpu_id}] done {split}/{scene} ({len(frame_paths)} frames x {len(todo)} configs)", flush=True)

    print(f"[shard {args.shard_index}/{args.num_shards}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
