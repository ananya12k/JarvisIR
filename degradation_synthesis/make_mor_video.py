"""
Side-by-side MOR comparison video: one clean clip rendered at every rung of the
MOR ladder, all other fog parameters (airlight, fog layer, heterogeneity, seed)
held fixed, plus the clean reference and the VDA depth.

  clean   | 300 m | 200 m | 150 m
  100 m   |  75 m |  50 m | depth

Usage:
  python make_mor_video.py --input /scratch/Ananya_Kulkarni/Defog_Dataset/test/GT/03 \
      --output mor_ladder_03.mp4 --seconds 15
"""
import argparse
import glob
import os
import subprocess
import sys

import cv2
import numpy as np
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def depth_tile(depth, max_depth_m):
    d = np.clip(depth, 0, max_depth_m) / max_depth_m
    return cv2.applyColorMap((255 * (1.0 - d)).astype(np.uint8), cv2.COLORMAP_INFERNO)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/scratch/Ananya_Kulkarni/Defog_Dataset/test/GT/03")
    p.add_argument("--output", default=os.path.join(HERE, "mor_ladder_video.mp4"))
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--start", type=int, default=0, help="first frame index")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--mor_ladder", default="300,200,150,100,75,50")
    p.add_argument("--airlight", type=float, default=0.72)
    p.add_argument("--fog_layer_m", type=float, default=20.0)
    p.add_argument("--heterogeneity", default="uniform", choices=["uniform", "patchy"])
    p.add_argument("--fbm_seed", type=int, default=12345)
    p.add_argument("--vda_root", default="/scratch/Ananya_Kulkarni/Video-Depth-Anything")
    p.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--crf", type=int, default=18)
    args = p.parse_args()

    frames = sorted(glob.glob(os.path.join(args.input, "*.[jJ][pP][gG]")) +
                    glob.glob(os.path.join(args.input, "*.[pP][nN][gG]")))
    n = int(round(args.seconds * args.fps))
    frames = frames[args.start:args.start + n]
    if len(frames) < n:
        print(f"WARNING: only {len(frames)} frames available ({len(frames) / args.fps:.1f} s)")
    output = os.path.abspath(args.output)

    sys.path.insert(0, args.vda_root)
    os.chdir(args.vda_root)
    sys.path.insert(0, HERE)
    import fog_synthesis as fs

    mors = [float(m) for m in args.mor_ladder.split(",")]
    engines = [fs.PhysicalFogEngine(beta_base=3.0 / m, airlight_color=(args.airlight,) * 3,
                                    fog_layer_m=args.fog_layer_m, heterogeneity=args.heterogeneity,
                                    fbm_seed=args.fbm_seed, speed_kmh=0.0, fps=args.fps)
               for m in mors]
    vda = fs.VDAWrapper(encoder=args.encoder, vda_root_path=args.vda_root, device_id=args.gpu_id)

    h, w = cv2.imread(frames[0]).shape[:2]
    W, H = 4 * w, 2 * h
    W, H = W - W % 2, H - H % 2
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf), "-preset", "medium",
         "-movflags", "+faststart", output],
        stdin=subprocess.PIPE)

    for idx, frame, depth in tqdm(fs.iter_scene_depth(vda, frames), total=len(frames), desc="MOR video"):
        tiles = [label(frame, "clean")]
        for m, eng in zip(mors, engines):
            tiles.append(label(eng.apply_fog(frame, depth), f"MOR {m:.0f} m   beta {3.0 / m:.3f}"))
        tiles.append(label(depth_tile(depth, engines[0].max_depth_m), "VDA depth (0-120 m)"))
        grid = np.vstack([np.hstack(tiles[0:4]), np.hstack(tiles[4:8])])[:H, :W]
        ffmpeg.stdin.write(np.ascontiguousarray(grid).tobytes())

    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"Saved {output}  ({len(frames)} frames, {len(frames) / args.fps:.1f} s, {W}x{H})")


if __name__ == "__main__":
    main()
