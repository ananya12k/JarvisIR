"""
compose_rain_videos.py
Assemble side-by-side comparison videos from per-mode mp4s written by separate
create_rain_video_data.py runs (e.g. baselines and temporal_dynamic split across
GPUs with --modes / --seqs).

Layouts (half-resolution panels, 3 per row):
  all      GT, 3 baselines, 4 dynamic scenarios, torrential refraction field
  methods  GT -> frame_indep -> temporal_static -> temporal_moving -> temporal_dynamic (steady) + its field
  levels   GT -> light -> steady -> heavy -> torrential (dynamic) + torrential field

Usage:
    python compose_rain_videos.py --runs RUN_A RUN_B --out OUT_DIR [--seqs val_02 val_06] [--layout methods --name methods --flat]
"""

import os
import argparse
import cv2
import numpy as np

LAYOUTS = {
    # every mode (default)
    'all': [
        ('GT.mp4', 'GT'),
        ('frame_indep.mp4', 'frame_indep'),
        ('temporal_static.mp4', 'temporal_static'),
        ('temporal_moving.mp4', 'temporal_moving'),
        ('temporal_dynamic_light.mp4', 'dynamic: light'),
        ('temporal_dynamic_steady.mp4', 'dynamic: steady'),
        ('temporal_dynamic_heavy.mp4', 'dynamic: heavy'),
        ('temporal_dynamic_torrential.mp4', 'dynamic: torrential'),
        ('field_torrential.mp4', 'refraction field (torrential)'),
    ],
    # GT -> each synthesis method (dynamic shown at the steady preset)
    'methods': [
        ('GT.mp4', 'GT (clean)'),
        ('frame_indep.mp4', '1. frame-independent'),
        ('temporal_static.mp4', '2. temporal static'),
        ('temporal_moving.mp4', '3. temporal moving'),
        ('temporal_dynamic_steady.mp4', '4. temporal dynamic (ours)'),
        ('field_steady.mp4', '4. dynamic: refraction field'),
    ],
    # GT -> rain intensities of the dynamic simulator
    'levels': [
        ('GT.mp4', 'GT (clean)'),
        ('temporal_dynamic_light.mp4', 'light rain'),
        ('temporal_dynamic_steady.mp4', 'steady rain'),
        ('temporal_dynamic_heavy.mp4', 'heavy rain'),
        ('temporal_dynamic_torrential.mp4', 'torrential rain'),
        ('field_torrential.mp4', 'torrential: refraction field'),
    ],
}


def find(runs, seq, name):
    for r in runs:
        p = os.path.join(r, 'videos', seq, name)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True, help='out_base dirs of the runs to combine')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seqs', nargs='+', default=None, help='e.g. val_02 (default: all found)')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--layout', choices=list(LAYOUTS), default='all')
    ap.add_argument('--name', default='side_by_side', help='output file name (without .mp4)')
    ap.add_argument('--flat', action='store_true', help='write OUT/<seq>_<name>.mp4 instead of OUT/<seq>/<name>.mp4')
    args = ap.parse_args()

    seqs = args.seqs or sorted({s for r in args.runs if os.path.isdir(os.path.join(r, 'videos'))
                                for s in os.listdir(os.path.join(r, 'videos'))})
    for seq in seqs:
        panels = [(find(args.runs, seq, f), label) for f, label in LAYOUTS[args.layout]]
        panels = [(p, label) for p, label in panels if p]
        caps = [cv2.VideoCapture(p) for p, _ in panels]
        n = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps)
        w, h = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)) // 2, int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
        cols = 3
        rows = (len(panels) + cols - 1) // cols
        out_path = (os.path.join(args.out, f'{seq}_{args.name}.mp4') if args.flat
                    else os.path.join(args.out, seq, f'{args.name}.mp4'))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w * cols, h * rows))
        for _ in range(n):
            tiles = []
            for c, (_, label) in zip(caps, panels):
                ok, f = c.read()
                t = cv2.resize(f, (w, h))
                cv2.putText(t, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                tiles.append(t)
            tiles += [np.zeros_like(tiles[0])] * (rows * cols - len(tiles))
            wr.write(np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]))
        wr.release()
        for c in caps:
            c.release()
        print(f'{seq}: {n} frames, {len(panels)} panels -> {out_path}')


if __name__ == '__main__':
    main()
