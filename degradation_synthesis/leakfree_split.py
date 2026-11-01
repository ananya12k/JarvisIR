"""
Leak-free train/val/test split of the clean source scenes, and a leak checker.

Some "scenes" are consecutive pieces of one continuous recording (e.g. 00 -> 09:
the last frame of one is the first frame of the next). Splitting by scene name
alone then leaks: the frames on both sides of a cut are near-identical. This
script

  1. detects continuity between every pair of scenes from the frames themselves
     (last frame of A vs first frame of B),
  2. applies a scene -> split assignment (JSON) and, at every continuous
     boundary that crosses splits, removes a --buffer_s buffer of frames from the
     lower-priority side (train before val before test, so eval sets stay whole),
  3. writes a source tree <out>/<split>/GT/<scene>/ of symlinks to the kept frames
     (consumed by build_fog_dataset.py, which copies the bytes),
  4. checks the result and exits non-zero on any leak:
       - a scene name in more than one split,
       - a continuous cross-split boundary closer than the buffer,
       - any cross-split frame pair that is a near-duplicate (thumbnail cosine
         >= --dup_threshold, sampled every --sample_step frames).

  python leakfree_split.py --source .../Defog_Dataset --split_json splits/edfog_leakfree.json --out .../src
  python leakfree_split.py --check_only --root <dataset root>      # re-check any built dataset
"""
import argparse
import glob
import itertools
import json
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
    """Longest run of consecutive readable frames (a broken frame would be a temporal jump)."""
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
    return frames[lo:lo + n], [os.path.basename(f) for f, good in zip(frames, ok) if not good]


def scan_source(source, broken=None):
    """{scene: frame paths} across all split folders of the source (names must be unique).
    Scenes with unreadable frames are cut to their longest readable run (logged in `broken`)."""
    scenes = {}
    for split in SPLITS:
        root = os.path.join(source, split, "GT")
        if not os.path.isdir(root):
            continue
        for scene in sorted(os.listdir(root)):
            if scene in scenes:
                sys.exit(f"source scene name {scene!r} appears twice; names must be unique")
            frames, bad = longest_readable_run(list_frames(os.path.join(root, scene)))
            if bad and broken is not None:
                broken[scene] = dict(unreadable=bad, kept=[os.path.basename(frames[0]), os.path.basename(frames[-1])])
                print(f"[split] {scene}: {len(bad)} unreadable frame(s) {bad[:5]}; "
                      f"keeping longest clean run {broken[scene]['kept']}")
            scenes[scene] = frames
    return scenes


def continuity(frames_by_scene, threshold):
    """[(a, b, score)] where the end of scene a continues seamlessly into the start of b."""
    ends = {s: thumb(f[-1]) for s, f in frames_by_scene.items() if f}
    starts = {s: thumb(f[0]) for s, f in frames_by_scene.items() if f}
    links = []
    for a in ends:
        for b in starts:
            if a != b:
                score = float(ends[a] @ starts[b])
                if score >= threshold:
                    links.append((a, b, score))
    return links


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


def make_split(args):
    with open(args.split_json) as f:
        assign = json.load(f)["assignment"]                 # {scene: split}
    broken = {}
    frames = scan_source(args.source, broken)
    unknown = sorted(set(frames) - set(assign))
    missing = sorted(set(assign) - set(frames))
    if unknown or missing:
        sys.exit(f"split JSON mismatch: unassigned source scenes {unknown}, unknown scenes {missing}")
    bad = {s: v for s, v in assign.items() if v not in SPLITS}
    if bad:
        sys.exit(f"invalid split names: {bad}")

    buf = int(round(args.buffer_s * args.fps))
    links = continuity(frames, args.continuity_threshold)
    trim = {s: [0, len(f)] for s, f in frames.items()}      # kept frame range [lo, hi)
    report_links = []
    for a, b, score in links:
        sa, sb = assign[a], assign[b]
        entry = dict(end_of=a, start_of=b, score=round(score, 4), splits=[sa, sb], trimmed=None)
        if sa != sb:
            if KEEP_PRIORITY[sa] < KEEP_PRIORITY[sb]:        # trim the end of a
                trim[a][1] = min(trim[a][1], len(frames[a]) - buf)
                entry["trimmed"] = f"last {buf} frames of {a}"
            else:                                           # trim the start of b
                trim[b][0] = max(trim[b][0], buf)
                entry["trimmed"] = f"first {buf} frames of {b}"
        report_links.append(entry)

    kept = {}
    for s, (lo, hi) in trim.items():
        if hi - lo < args.fps:
            sys.exit(f"buffer leaves scene {s} with {hi - lo} frames; lower --buffer_s or reassign")
        kept[(assign[s], s)] = frames[s][lo:hi]

    for (split, scene), fr in kept.items():
        d = os.path.join(args.out, split, "GT", scene)
        os.makedirs(d, exist_ok=True)
        for p in fr:
            dst = os.path.join(d, os.path.basename(p))
            if not os.path.lexists(dst):
                os.symlink(os.path.realpath(p), dst)
        for name in os.listdir(d):                          # drop stale links from older runs
            if name not in {os.path.basename(p) for p in fr}:
                os.remove(os.path.join(d, name))

    report = dict(source=os.path.abspath(args.source), split_json=os.path.abspath(args.split_json),
                  buffer_s=args.buffer_s, fps=args.fps, unreadable_source_frames=broken,
                  scenes={f"{sp}/{sc}": dict(source_frames=len(frames[sc]), kept=[trim[sc][0], trim[sc][1]],
                                             n_kept=len(fr)) for (sp, sc), fr in sorted(kept.items())},
                  continuity_links=report_links)
    ok = check(kept, args, report, links_by_split=[(assign[a], assign[b], a, b) for a, b, _ in links],
               trim=trim, buf=buf, lengths={sc: len(f) for sc, f in frames.items()})
    with open(os.path.join(args.out, "split_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[split] report -> {os.path.join(args.out, 'split_report.json')}")
    return ok


def check(kept, args, report, links_by_split=None, trim=None, buf=None, lengths=None):
    ok = True
    names = {}
    for split, scene in kept:
        names.setdefault(scene, set()).add(split)
    shared = {s: sorted(v) for s, v in names.items() if len(v) > 1}
    print(f"[check] scene names in more than one split: {shared or 'none'}")
    ok &= not shared

    if links_by_split is not None:
        unbuffered = [(a, b) for sa, sb, a, b in links_by_split
                      if sa != sb and not (buf > 0 and (trim[a][1] <= lengths[a] - buf or trim[b][0] >= buf))]
        print(f"[check] continuous cross-split boundaries without a {args.buffer_s:g} s buffer: {unbuffered or 'none'}")
        ok &= not unbuffered

    hits, worst = near_duplicates(kept, args.sample_step, args.dup_threshold)
    print(f"[check] cross-split near-duplicate frames (cosine >= {args.dup_threshold}): {len(hits) or 'none'}")
    for s, a, b in hits[:10]:
        print(f"         {s:.3f}  {a}  <->  {b}")
    print("[check] most similar cross-split pairs (for reference):")
    for s, a, b in worst:
        print(f"         {s:.3f}  {a}  <->  {b}")
    ok &= not hits
    report["check"] = dict(passed=bool(ok), shared_scene_names=shared, near_duplicates=hits, most_similar=worst,
                           dup_threshold=args.dup_threshold, sample_step=args.sample_step)
    print("[check] PASSED: no leak" if ok else "[check] FAILED: leak detected")
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
    p.add_argument("--split_json", help='{"assignment": {scene: "train"|"val"|"test"}}')
    p.add_argument("--out", help="where to write the leak-free source tree")
    p.add_argument("--check_only", action="store_true")
    p.add_argument("--root", help="dataset root to check (with --check_only)")
    p.add_argument("--buffer_s", type=float, default=15.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--continuity_threshold", type=float, default=0.97)
    p.add_argument("--dup_threshold", type=float, default=0.97)
    p.add_argument("--sample_step", type=int, default=5)
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
