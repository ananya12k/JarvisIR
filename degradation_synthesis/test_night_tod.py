"""
Extreme-night variant test for NightEngine.

Saves for each of 10 frames:
  - A horizontal comparison strip: GT | extreme_unlit | extreme_headlight | extreme_glare
  - Individual images per variant in sub-folders

All three variants are equally dark (same illuminant, tungsten) — they
differ in severity/light layout, not colour. Each variant is sampled with
sample_variant_params() using a fixed seed here, so beam geometry / light
positions are reproducible across runs of this script but not identical to
what create_night_data.py would generate for a real clip.

Usage:
    conda run -n visia python test_night_tod.py --gpu_id 0
"""

import os
import sys
import cv2
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from night_synthesis import VDAWrapper, NightEngine, EXTREME_VARIANTS, sample_variant_params

# ---------------------------------------------------------------------------
# One jittered sample per variant, seeded for reproducibility of this test.
# ---------------------------------------------------------------------------
_rng = np.random.default_rng(0)
TIME_OF_DAY = {name: sample_variant_params(base, _rng) for name, base in EXTREME_VARIANTS.items()}

FRAME_NAMES = [
    '000001.jpg', '000234.jpg', '000468.jpg', '000702.jpg', '000936.jpg',
    '001170.jpg', '001404.jpg', '001638.jpg', '001872.jpg', '002106.jpg',
]

GT_DIR    = '/scratch/Ananya_Kulkarni/Defog_balanced/train/GT/00'
IMSIZE    = (480, 270)   # (W, H)
VDA_ROOT  = '/scratch/Ananya_Kulkarni/Video-Depth-Anything'


def add_label(img_bgr, text, font_scale=0.55, thickness=1):
    """Burn a white label with black shadow onto the top-left corner."""
    out = img_bgr.copy()
    pos = (8, 22)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def make_strip(gt_bgr, variants):
    """Horizontal concat: GT | stage1 | stage2 | ..."""
    panels = [add_label(gt_bgr, 'GT (day)')]
    for name, img in variants.items():
        panels.append(add_label(img, name))
    strip = np.concatenate(panels, axis=1)
    # thin white dividers
    h = strip.shape[0]
    w_panel = IMSIZE[0]
    for i in range(1, len(panels)):
        x = i * w_panel
        strip[:, x-1:x+1] = 255
    return strip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--output', default='/tmp/night_tod_test')
    parser.add_argument('--encoder', default='vits', choices=['vits', 'vitl'])
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    strips_dir = os.path.join(args.output, 'comparison_strips')
    os.makedirs(strips_dir, exist_ok=True)
    for tod in TIME_OF_DAY:
        os.makedirs(os.path.join(args.output, tod), exist_ok=True)

    print(f"Loading VDA ({args.encoder}) on GPU {args.gpu_id} ...")
    pipe = VDAWrapper(encoder=args.encoder, vda_root_path=VDA_ROOT, device_id=args.gpu_id)

    engines = {name: NightEngine(**params) for name, params in TIME_OF_DAY.items()}

    # Load all 10 frames
    frames_bgr = []
    for fname in FRAME_NAMES:
        img = cv2.imread(os.path.join(GT_DIR, fname))
        frames_bgr.append(cv2.resize(img, IMSIZE))

    print("Running VDA on 10 frames ...")
    batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    depths    = pipe.infer_batch(batch_rgb)   # (10, H, W)

    print("Generating time-of-day variants and saving ...")
    for i, (gt_bgr, depth, fname) in enumerate(zip(frames_bgr, depths, FRAME_NAMES)):
        stem = os.path.splitext(fname)[0]
        variants = {}
        for tod_name, eng in engines.items():
            out = eng.apply_night(gt_bgr, depth)
            variants[tod_name] = out
            cv2.imwrite(os.path.join(args.output, tod_name, f'{stem}.jpg'), out)

        strip = make_strip(gt_bgr, variants)
        cv2.imwrite(os.path.join(strips_dir, f'{stem}_strip.jpg'), strip)
        print(f"  [{i+1:2d}/10]  {fname}  depth range [{depth.min():.1f}, {depth.max():.1f}] m")

    print(f"\nDone. Results in: {args.output}")
    print(f"  comparison_strips/  — horizontal GT + all variants (4 panels wide)")
    for tod in TIME_OF_DAY:
        print(f"  {tod}/              — individual frames")


if __name__ == '__main__':
    main()
