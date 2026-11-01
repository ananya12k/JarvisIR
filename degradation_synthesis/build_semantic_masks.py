"""
Cityscapes semantic masks for every clean (GT) frame of a fog dataset, with
SegFormer-B5 (nvidia/segformer-b5-finetuned-cityscapes-1024-1024).

  <root>/<split>/GT/<scene>/<frame>.(jpg|png)
  -> <root>/<split>/Semantic/<scene>/<frame>.npy     uint8 Cityscapes trainIds 0..18

Masks are stored as uint8 (not float logits) at exactly the GT frame's
resolution, so they align pixel-for-pixel with GT and Haze. `.npy` is what
VISIA's training/video_dataset.py reads; `--format png` stores the same labels
losslessly at ~1/10 the size.

Quality: frames are upscaled to --infer_width (default 2048, the Cityscapes
training width) for inference, logits are bilinearly resized back to the GT
size and argmax'd there. Labels are never resized after argmax.

All visible GPUs are used: scenes go into one shared queue, longest first, and
--jobs_per_gpu workers per GPU pull the next scene when free.
"""
import argparse
import glob
import multiprocessing as mp
import os

import cv2
import numpy as np
from tqdm import tqdm

MODEL_ID = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def list_frames(d):
    return sorted(glob.glob(os.path.join(d, "*.[jJ][pP][gG]")) +
                  glob.glob(os.path.join(d, "*.[pP][nN][gG]")))


def out_path(out_dir, frame_path, fmt):
    return os.path.join(out_dir, os.path.splitext(os.path.basename(frame_path))[0] + "." + fmt)


def save_mask(path, mask, fmt):
    tmp = path + ".tmp." + fmt
    if fmt == "npy":
        with open(tmp, "wb") as f:
            np.save(f, mask)
    else:
        cv2.imwrite(tmp, mask)
    os.replace(tmp, path)


def worker(rank, gpu, queue, args):
    import torch
    import torch.nn.functional as F
    from transformers import SegformerForSemanticSegmentation

    torch.set_num_threads(args.threads_per_worker)
    cv2.setNumThreads(args.threads_per_worker)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    tag = f"[GPU {gpu}.{rank}]"

    while True:
        job = queue.get()
        if job is None:
            return
        split, scene, frames, out_dir = job
        try:
            os.makedirs(out_dir, exist_ok=True)
            for i in tqdm(range(0, len(frames), args.batch), desc=f"{tag} {split}/{scene}", mininterval=30):
                chunk = frames[i:i + args.batch]
                imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in chunk]
                h, w = imgs[0].shape[:2]
                ih = int(round(h * args.infer_width / w / 32)) * 32
                x = torch.from_numpy(np.stack(imgs)).to(device).permute(0, 3, 1, 2).float() / 255.0
                x = F.interpolate(x, size=(ih, args.infer_width), mode="bicubic", align_corners=False)
                x = (x.clamp(0, 1) - mean) / std
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16,
                                                     enabled=device.type == "cuda"):
                    logits = model(pixel_values=x).logits
                logits = F.interpolate(logits.float(), size=(h, w), mode="bilinear", align_corners=False)
                masks = logits.argmax(1).to(torch.uint8).cpu().numpy()
                for p, m in zip(chunk, masks):
                    save_mask(out_path(out_dir, p, args.format), m, args.format)
            print(f"{tag} done {split}/{scene} ({len(frames)} frames)", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            with open(os.path.join(args.root, "failed_semantic_jobs.txt"), "a") as f:
                f.write(f"{split}/{scene}\t{type(e).__name__}: {e}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="dataset root holding <split>/GT/<scene>/")
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--format", default="npy", choices=["npy", "png"])
    p.add_argument("--infer_width", type=int, default=2048)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--num_gpus", type=int, default=0, help="0 = all visible GPUs")
    p.add_argument("--jobs_per_gpu", type=int, default=1)
    p.add_argument("--threads_per_worker", type=int, default=2)
    p.add_argument("--max_frames", type=int, default=0, help="first N frames per scene (testing); 0 = all")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()
    args.root = os.path.abspath(args.root)

    jobs = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        gt_root = os.path.join(args.root, split, "GT")
        if not os.path.isdir(gt_root):
            continue
        for scene in sorted(os.listdir(gt_root)):
            frames = list_frames(os.path.join(gt_root, scene))
            if args.max_frames > 0:
                frames = frames[:args.max_frames]
            out_dir = os.path.join(args.root, split, "Semantic", scene)
            if args.skip_existing:
                frames = [f for f in frames if not os.path.exists(out_path(out_dir, f, args.format))]
            if frames:
                jobs.append((split, scene, frames, out_dir))
    jobs.sort(key=lambda j: -len(j[2]))
    if not jobs:
        print("[semantic] nothing to do")
        return

    import torch
    gpus = list(range(args.num_gpus or torch.cuda.device_count())) or [0]
    slots = [g for g in gpus for _ in range(args.jobs_per_gpu)]
    print(f"[semantic] {len(jobs)} scene(s), {sum(len(j[2]) for j in jobs)} frames | "
          f"{len(gpus)} GPU(s) x {args.jobs_per_gpu} worker(s) | {args.format} uint8", flush=True)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    for j in jobs:
        queue.put(j)
    for _ in slots:
        queue.put(None)
    procs = [ctx.Process(target=worker, args=(rank, gpu, queue, args)) for rank, gpu in enumerate(slots)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    if os.path.exists(os.path.join(args.root, "failed_semantic_jobs.txt")):
        print(f"[semantic] some scenes FAILED, see {args.root}/failed_semantic_jobs.txt", flush=True)
    print("[semantic] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
