"""
Leak-free train/val/test split of the clean source scenes, and a leak checker.

Some "scenes" are consecutive pieces of one continuous recording (e.g. 00 -> 09:
the last frame of one is the first frame of the next), and a forward-facing
camera sees the road it is about to drive 100+ m ahead. Splitting by scene name
alone therefore leaks. This script

  1. detects continuity between scenes from the frames (last frame of A vs first
     frame of B) and lays every continuous recording out on one timeline,
  2. applies a scene -> split assignment (JSON; "exclude" drops a scene, e.g. as
     a buffer; a list of parts splits one recording across splits) and guarantees that kept frames of different splits are at least
     --buffer_s apart on that timeline, trimming the lower-priority side if needed
     (train before val before test, so eval sets stay whole),
  3. writes a source tree <out>/<split>/GT/<scene>/ of symlinks to the kept frames
     (consumed by build_fog_dataset.py, which copies the bytes),
  4. checks the result and exits non-zero on any leak:
       - a scene name in more than one split,
       - kept frames of different splits closer than the buffer on a timeline,
       - a cross-split near-duplicate frame (thumbnail cosine >= --dup_threshold),
       - the SAME PLACE in two splits (e.g. two recordings driving the same road):
         ORB features + RANSAC fundamental matrix over every cross-split pair of
         frames sampled every --place_step frames; a scene pair leaks with one
         match >= --place_strong inliers or >= --place_min_hits matches >=
         --place_inliers (isolated 15-22 inlier matches on pylons/wires are noise).
         (Measured on this data: unrelated places <= 9 inliers, the same place
         2 s apart 12-52, frames at a cut ~900.)

  python leakfree_split.py --source .../Defog_Dataset --split_json splits/edfog_leakfree.json --out .../src
  python leakfree_split.py --check_only --root <dataset root>      # re-check any built dataset
"""
import argparse
import glob
import itertools
import json
import multiprocessing as mp
import os
import sys

import cv2
import numpy as np

SPLITS = ("train", "val", "test")
KEEP_PRIORITY = {"test": 2, "val": 1, "train": 0}


def list_frames(d):
    return sorted(glob.glob(os.path.join(d, "*.[jJ][pP][gG]")) +
                  glob.glob(os.path.join(d, "*.[pP][nN][gG]")))


def thumb(path):
    g = cv2.cvtColor(cv2.resize(cv2.imread(path), (64, 36), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY).astype(np.float32)
    g -= g.mean()
    return g.ravel() / (np.linalg.norm(g) + 1e-6)


def readable(path):
    return os.path.getsize(path) > 0 and cv2.imread(path) is not None


def longest_readable_run(frames):
    """(lo, hi) of the longest run of consecutive readable frames, and the unreadable names."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        ok = list(ex.map(readable, frames))
    best, start = (0, 0), None
    for i, good in enumerate(ok + [False]):
        if good and start is None:
            start = i
        elif not good and start is not None:
            best = max(best, (i - start, start))
            start = None
    n, lo = best
    return (lo, lo + n), [os.path.basename(f) for f, good in zip(frames, ok) if not good]


def scan_source(source, broken=None):
    """{scene: all frame paths} plus {scene: readable (lo, hi)} across the source's split folders.
    A broken frame would be a temporal jump, so only the longest readable run is usable."""
    scenes, usable = {}, {}
    for split in SPLITS:
        root = os.path.join(source, split, "GT")
        if not os.path.isdir(root):
            continue
        for scene in sorted(os.listdir(root)):
            if scene in scenes:
                sys.exit(f"source scene name {scene!r} appears twice; names must be unique")
            frames = list_frames(os.path.join(root, scene))
            (lo, hi), bad = longest_readable_run(frames)
            if bad and broken is not None:
                broken[scene] = dict(unreadable=bad, kept=[os.path.basename(frames[lo]), os.path.basename(frames[hi - 1])])
                print(f"[split] {scene}: {len(bad)} unreadable frame(s) {bad[:5]}; "
                      f"usable: longest clean run {broken[scene]['kept']}")
            scenes[scene], usable[scene] = frames, (lo, hi)
    return scenes, usable


def continuity(frames_by_scene, threshold):
    """[(a, b, score)]: the end of scene a continues seamlessly into the start of b."""
    ends = {s: thumb(f[-1]) for s, f in frames_by_scene.items() if f and readable(f[-1])}
    starts = {s: thumb(f[0]) for s, f in frames_by_scene.items() if f and readable(f[0])}
    links = []
    for a in ends:
        for b in starts:
            if a != b:
                score = float(ends[a] @ starts[b])
                if score >= threshold:
                    links.append((a, b, score))
    return links


def timelines(frames, links):
    """{scene: (timeline id, offset of its first frame)}: continuous scenes share a timeline."""
    succ, pred = {}, {}
    for a, b, score in sorted(links, key=lambda l: -l[2]):   # keep the best link per scene
        if a not in succ and b not in pred:
            succ[a], pred[b] = b, a
    pos = {}
    for head in sorted(s for s in frames if s not in pred):
        off, s = 0, head
        while s is not None and s not in pos:
            pos[s] = (head, off)
            off += len(frames[s])
            s = succ.get(s)
    for s in frames:                                          # cycles (should not happen)
        pos.setdefault(s, (s, 0))
    return pos


def _segments(units, pos):
    """Kept units laid out on their recording's timeline: (timeline, start, end, unit), sorted."""
    return sorted((pos[u["scene"]][0], pos[u["scene"]][1] + u["lo"], pos[u["scene"]][1] + u["hi"], name)
                  for name, u in units.items() if u["hi"] > u["lo"])


def enforce_buffers(units, pos, buf):
    """Trim units so kept frames of different splits are >= buf apart on every timeline."""
    actions = []
    changed = True
    while changed:
        changed = False
        segs = _segments(units, pos)
        for (t1, _, e1, a), (t2, s2, _, b) in zip(segs, segs[1:]):
            sa, sb = units[a]["split"], units[b]["split"]
            if t1 != t2 or sa == sb:
                continue
            gap = s2 - e1
            if gap < buf:
                need = buf - gap
                if KEEP_PRIORITY[sa] < KEEP_PRIORITY[sb]:
                    units[a]["hi"] -= need
                    actions.append(f"trim last {need} frames of {a} ({sa}) before {b} ({sb})")
                else:
                    units[b]["lo"] += need
                    actions.append(f"trim first {need} frames of {b} ({sb}) after {a} ({sa})")
                changed = True
                break
    return actions


def min_cross_split_gaps(units, pos):
    """[(gap_frames, a, b)] for neighbouring kept units of different splits on a timeline."""
    segs = _segments(units, pos)
    return [(s2 - e1, a, b) for (t1, _, e1, a), (t2, s2, _, b) in zip(segs, segs[1:])
            if t1 == t2 and units[a]["split"] != units[b]["split"]]


def expand_assignment(spec, frames, usable):
    """{unit: dict(scene, split, lo, hi)} from the split JSON. A scene maps to a split
    name, to "exclude", or to a list of parts [{"name", "split", "frames": [lo, hi]}]
    (one recording split across splits; buffers between parts are enforced later).
    "keep_frames" {scene: [lo, hi]} restricts a single-split scene."""
    assign, keep_frames = spec["assignment"], spec.get("keep_frames", {})
    unknown = sorted(set(frames) - set(assign))
    missing = sorted(set(assign) - set(frames))
    if unknown or missing:
        sys.exit(f"split JSON mismatch: unassigned source scenes {unknown}, unknown scenes {missing}")
    units = {}
    for scene, v in assign.items():
        lo0, hi0 = usable[scene]
        parts = v if isinstance(v, list) else [dict(name=scene, split=v, frames=keep_frames.get(scene, [0, hi0]))]
        for part in parts:
            if part["split"] == "exclude":
                continue
            if part["split"] not in SPLITS:
                sys.exit(f"invalid split {part['split']!r} for {scene}")
            if part["name"] in units:
                sys.exit(f"duplicate unit name {part['name']!r}")
            lo, hi = part["frames"]
            units[part["name"]] = dict(scene=scene, split=part["split"], lo=max(lo, lo0), hi=min(hi, hi0))
    return units


def near_duplicates(frames_by_key, step, threshold, limit=20):
    """Cross-split frame pairs with thumbnail cosine >= threshold. Keys are (split, scene)."""
    feats = {k: (np.stack([thumb(p) for p in f[::step]]), f[::step]) for k, f in frames_by_key.items() if f}
    hits, worst = [], []
    for a, b in itertools.combinations(sorted(feats), 2):
        if a[0] == b[0]:
            continue
        S = feats[a][0] @ feats[b][0].T
        i, j = np.unravel_index(int(S.argmax()), S.shape)
        worst.append((float(S[i, j]), f"{a[0]}/{a[1]}/{os.path.basename(feats[a][1][i])}",
                      f"{b[0]}/{b[1]}/{os.path.basename(feats[b][1][j])}"))
        for i, j in zip(*np.nonzero(S >= threshold)):
            hits.append((float(S[i, j]), f"{a[0]}/{a[1]}/{os.path.basename(feats[a][1][i])}",
                         f"{b[0]}/{b[1]}/{os.path.basename(feats[b][1][j])}"))
    return sorted(hits, reverse=True)[:limit], sorted(worst, reverse=True)[:5]


# -- same-place check (ORB + RANSAC) -----------------------------------------
_ORB_N = 1500
_FEATS = None


def _orb(path):
    cv2.setNumThreads(1)
    k, d = cv2.ORB_create(_ORB_N).detectAndCompute(cv2.imread(path, cv2.IMREAD_GRAYSCALE), None)
    return np.float32([p.pt for p in k]) if k else np.zeros((0, 2), np.float32), d


def epipolar_inliers(A, B, thresh_px=1.5):
    """Inliers of a RANSAC fundamental matrix, counted from the Sampson distance ourselves:
    OpenCV 4.11's returned mask can hold garbage values, so it is not trusted."""
    F, _ = cv2.findFundamentalMat(A, B, cv2.FM_RANSAC, thresh_px, 0.999)
    if F is None or F.shape != (3, 3):
        return 0
    a = np.hstack([A, np.ones((len(A), 1), np.float32)]).astype(np.float64)
    b = np.hstack([B, np.ones((len(B), 1), np.float32)]).astype(np.float64)
    Fa, Ftb = a @ F.T, b @ F
    num = np.einsum("ij,ij->i", b, Fa) ** 2
    den = Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Ftb[:, 0] ** 2 + Ftb[:, 1] ** 2 + 1e-12
    return int(np.count_nonzero(num / den < thresh_px ** 2))


def _init_pool(feats):
    global _FEATS
    _FEATS = feats
    cv2.setNumThreads(1)


def _best_match(args):
    """Max geometric inliers between eval frame i and every frame of the other split."""
    i, others = args
    pi, di = _FEATS[i]
    if di is None or len(pi) < 15:
        return i, 0, -1
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    best, best_j = 0, -1
    for j in others:
        pj, dj = _FEATS[j]
        if dj is None or len(pj) < 15:
            continue
        m = [a for a, b in (x for x in bf.knnMatch(di, dj, k=2) if len(x) == 2) if a.distance < 0.75 * b.distance]
        if len(m) < 15 or len(m) <= best:
            continue
        A = pi[[x.queryIdx for x in m]]
        B = pj[[x.trainIdx for x in m]]
        n = epipolar_inliers(A, B)
        if n > best:
            best, best_j = n, j
    return i, best, best_j


def same_place_pairs(frames_by_key, step, threshold, workers):
    """Every cross-split pair of sampled frames, ORB + RANSAC verified."""
    samples = [(k, p) for k, f in sorted(frames_by_key.items()) for p in f[::step]]
    with mp.get_context("fork").Pool(workers) as pool:
        feats = pool.map(_orb, [p for _, p in samples], chunksize=16)
    split_of = [k[0] for k, _ in samples]
    eval_idx = [i for i, s in enumerate(split_of) if s != "train"]
    jobs = [(i, [j for j, s in enumerate(split_of) if s != split_of[i] and (s != "test" or split_of[i] == "train")])
            for i in eval_idx]                       # eval vs train, val vs test (each pair once)
    with mp.get_context("fork").Pool(workers, initializer=_init_pool, initargs=(feats,)) as pool:
        res = pool.map(_best_match, jobs, chunksize=1)
    name = lambda i: f"{samples[i][0][0]}/{samples[i][0][1]}/{os.path.basename(samples[i][1])}"
    pairs = sorted(((n, name(i), name(j)) for i, n, j in res if j >= 0), reverse=True)
    return pairs, len(samples)


def place_leaks(pairs, threshold, strong, min_hits):
    """A cross-split scene pair leaks if one sampled frame pair is a strong match
    (>= strong inliers) or >= min_hits frames match at >= threshold. Thin repetitive
    structures (pylons, wires, poles) give isolated false matches around 15-22 inliers;
    a real revisit of a place matches persistently over consecutive samples."""
    by_pair = {}
    for n, a, b in pairs:
        if n >= threshold:
            by_pair.setdefault((a.rsplit("/", 1)[0], b.rsplit("/", 1)[0]), []).append((n, a, b))
    return {k: sorted(v, reverse=True) for k, v in by_pair.items()
            if v and (max(x[0] for x in v) >= strong or len(v) >= min_hits)}, by_pair


def check(kept, args, report, gaps=None):
    ok = True
    names = {}
    for split, scene in kept:
        names.setdefault(scene, set()).add(split)
    shared = {s: sorted(v) for s, v in names.items() if len(v) > 1}
    print(f"[check] scene names in more than one split: {shared or 'none'}")
    ok &= not shared

    if gaps is not None:
        buf = int(round(args.buffer_s * args.fps))
        short = [(g, a, b) for g, a, b in gaps if g < buf]
        print(f"[check] cross-split neighbours on a continuous recording: "
              + (", ".join(f"{a}|{b} {g / args.fps:.0f}s apart" for g, a, b in gaps) or "none"))
        print(f"[check] ... closer than the {args.buffer_s:g} s buffer: {short or 'none'}")
        ok &= not short

    hits, worst = near_duplicates(kept, args.sample_step, args.dup_threshold)
    print(f"[check] cross-split near-duplicate frames (thumbnail cosine >= {args.dup_threshold}): {len(hits) or 'none'}"
          f"   [highest: {worst[0][0]:.3f}]" if worst else "")
    for s, a, b in hits[:10]:
        print(f"         {s:.3f}  {a}  <->  {b}")
    ok &= not hits

    pairs, n_samples = same_place_pairs(kept, args.place_step, args.place_inliers, args.workers)
    leaks, candidates = place_leaks(pairs, args.place_inliers, args.place_strong, args.place_min_hits)
    print(f"[check] same place in two splits (ORB+RANSAC on {n_samples} frames sampled every {args.place_step}; "
          f"leak = a match >= {args.place_strong} inliers or >= {args.place_min_hits} matches >= {args.place_inliers}): "
          f"{len(leaks) or 'none'}")
    for (a, b), v in leaks.items():
        print(f"         LEAK {a} <-> {b}: {len(v)} matching frame(s), strongest {v[0][0]} ({v[0][1]} <-> {v[0][2]})")
    print("[check] strongest cross-split matches (unrelated places score <= ~10):")
    for n, a, b in pairs[:5]:
        print(f"         {n:4d}  {a}  <->  {b}")
    place_hits = {f"{a} <-> {b}": v for (a, b), v in leaks.items()}
    place_top = pairs[:8]
    ok &= not leaks

    report["check"] = dict(passed=bool(ok), shared_scene_names=shared, cross_split_gaps_frames=gaps,
                           near_duplicates=hits, most_similar_thumbnails=worst,
                           same_place_leaks=place_hits, strongest_place_matches=place_top,
                           same_place_candidates={f"{a} <-> {b}": v for (a, b), v in candidates.items()},
                           thresholds=dict(dup=args.dup_threshold, place_inliers=args.place_inliers,
                                           place_strong=args.place_strong, place_min_hits=args.place_min_hits,
                                           place_step=args.place_step, buffer_s=args.buffer_s))
    print("[check] PASSED: no leak" if ok else "[check] FAILED: leak detected")
    return ok


def make_split(args):
    with open(args.split_json) as f:
        spec = json.load(f)
    broken = {}
    frames, usable = scan_source(args.source, broken)
    units = expand_assignment(spec, frames, usable)

    buf = int(round(args.buffer_s * args.fps))
    links = continuity(frames, args.continuity_threshold)
    pos = timelines(frames, links)
    actions = enforce_buffers(units, pos, buf)
    for a in actions:
        print(f"[split] {a}")

    kept = {}
    for name, u in units.items():
        if u["hi"] - u["lo"] < args.fps:
            sys.exit(f"buffer leaves {name} with {u['hi'] - u['lo']} frames; reassign it or exclude it")
        kept[(u["split"], name)] = frames[u["scene"]][u["lo"]:u["hi"]]

    for (split, unit), fr in kept.items():
        d = os.path.join(args.out, split, "GT", unit)
        os.makedirs(d, exist_ok=True)
        keep = {os.path.basename(p) for p in fr}
        for p in fr:
            dst = os.path.join(d, os.path.basename(p))
            if not os.path.lexists(dst):
                os.symlink(os.path.realpath(p), dst)
        for name in os.listdir(d):                          # drop stale links from older runs
            if name not in keep:
                os.remove(os.path.join(d, name))
    for split in SPLITS:                                    # drop units no longer in this split
        root = os.path.join(args.out, split, "GT")
        if os.path.isdir(root):
            for unit in os.listdir(root):
                if (split, unit) not in kept:
                    import shutil
                    shutil.rmtree(os.path.join(root, unit))

    chains = {}
    for sc, (t, off) in pos.items():
        chains.setdefault(t, []).append((off, sc))
    totals = {sp: sum(len(fr) for (s2, _), fr in kept.items() if s2 == sp) for sp in SPLITS}
    grand = sum(totals.values()) or 1
    used = {sc for u in units.values() for sc in [u["scene"]]}
    report = dict(source=os.path.abspath(args.source), split_json=os.path.abspath(args.split_json),
                  buffer_s=args.buffer_s, fps=args.fps, unreadable_source_frames=broken,
                  continuous_recordings=[[sc for _, sc in sorted(v)] for v in chains.values() if len(v) > 1],
                  excluded_scenes=sorted(set(frames) - used),
                  buffer_actions=actions,
                  split_frames=totals,
                  split_ratio={sp: round(100.0 * n / grand, 1) for sp, n in totals.items()},
                  units={f"{u['split']}/{name}": dict(source_scene=u["scene"], source_frames=len(frames[u["scene"]]),
                                                      kept=[u["lo"], u["hi"]], n_kept=u["hi"] - u["lo"])
                         for name, u in sorted(units.items())})
    print(f"[split] continuous recordings: {report['continuous_recordings']} | "
          f"excluded scenes: {report['excluded_scenes']}")
    print("[split] frames per split: " + ", ".join(f"{sp} {totals[sp]} ({report['split_ratio'][sp]}%)" for sp in SPLITS))
    ok = check(kept, args, report, gaps=min_cross_split_gaps(units, pos))
    with open(os.path.join(args.out, "split_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[split] report -> {os.path.join(args.out, 'split_report.json')}")
    return ok


def check_dataset(args):
    """Re-check a built dataset: GT scenes, Haze clips and Semantic all per split."""
    kept, ok = {}, True
    for split in SPLITS:
        gt = os.path.join(args.root, split, "GT")
        if not os.path.isdir(gt):
            continue
        scenes = set(os.listdir(gt))
        for scene in scenes:
            kept[(split, scene)] = list_frames(os.path.join(gt, scene))
        for sub in ("Haze", "Semantic"):
            d = os.path.join(args.root, split, sub)
            if os.path.isdir(d):
                orphans = [c for c in os.listdir(d) if (c.split("_cfg")[0] if sub == "Haze" else c) not in scenes]
                if orphans:
                    print(f"[check] {split}/{sub} entries without a {split}/GT scene: {orphans}")
                    ok = False
    report = {}
    ok &= check(kept, args, report)
    with open(os.path.join(args.root, "leak_check.json"), "w") as f:
        json.dump(report, f, indent=2)
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", help="clean source with <split>/GT/<scene>/ (any split layout)")
    p.add_argument("--split_json", help='{"assignment": {scene: "train"|"val"|"test"|"exclude"|[parts]}}')
    p.add_argument("--out", help="where to write the leak-free source tree")
    p.add_argument("--check_only", action="store_true")
    p.add_argument("--root", help="dataset root to check (with --check_only)")
    p.add_argument("--buffer_s", type=float, default=60.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--continuity_threshold", type=float, default=0.97)
    p.add_argument("--dup_threshold", type=float, default=0.97)
    p.add_argument("--sample_step", type=int, default=5)
    p.add_argument("--place_step", type=int, default=30, help="frames between same-place samples (1 s)")
    p.add_argument("--place_inliers", type=int, default=20)
    p.add_argument("--place_strong", type=int, default=40)
    p.add_argument("--place_min_hits", type=int, default=3)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()

    if args.check_only:
        ok = check_dataset(args)
    else:
        if not (args.source and args.split_json and args.out):
            p.error("--source, --split_json and --out are required (or use --check_only --root)")
        ok = make_split(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
