import argparse
import csv
import glob
import itertools
import multiprocessing as mp
import os
import shutil
import sys
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm


# Keep this script standalone and non-invasive to existing training/inference code.
IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
MODULE_ORDER = [
    "volumetric_blur",
    "heterogeneous_density",
    "nonlinear_gamma",
    "secondary_scattering",
    "adaptive_airlight",
    "fog_coupled_noise",
]


def list_images(folder):
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
        paths.extend(glob.glob(os.path.join(folder, ext.upper())))
    return sorted(paths)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_target_frame_list(src_paths, target_frames):
    if target_frames <= 0 or len(src_paths) >= target_frames:
        used = src_paths if target_frames <= 0 else src_paths[:target_frames]
        prepared = []
        for i, p in enumerate(used):
            ext = os.path.splitext(p)[1].lower() or ".png"
            prepared.append((p, f"frame_{i:06d}{ext}"))
        return prepared

    prepared = []
    n = len(src_paths)
    for i in range(target_frames):
        p = src_paths[i % n]
        ext = os.path.splitext(p)[1].lower() or ".png"
        prepared.append((p, f"frame_{i:06d}{ext}"))
    return prepared


def np_psnr(a, b):
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def np_ssim(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    if a.ndim == 3 and a.shape[2] == 3:
        ssim_vals = [ssim_single(a[..., c], b[..., c]) for c in range(3)]
        return float(np.mean(ssim_vals))
    return float(ssim_single(a, b))


def ssim_single(a, b):
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    a = cv2.GaussianBlur(a, (11, 11), 1.5)
    b = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu1 = a
    mu2 = b
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur((a - mu1) ** 2, (11, 11), 1.5)
    sigma2_sq = cv2.GaussianBlur((b - mu2) ** 2, (11, 11), 1.5)
    sigma12 = cv2.GaussianBlur((a - mu1) * (b - mu2), (11, 11), 1.5)

    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return np.mean(num / (den + 1e-8))


def load_vda(vda_root, encoder, device_id):
    from degradation_synthesis.fog_synthesis import VDAWrapper

    return VDAWrapper(encoder=encoder, vda_root_path=vda_root, device_id=device_id)


def generate_fbm_field(h, w, hurst=0.75, octaves=6, seed=42):
    rng = np.random.default_rng(seed)
    field = np.zeros((h, w), dtype=np.float32)
    for i in range(octaves):
        freq = 2 ** i
        amp = 0.5 ** (i * hurst)
        nh = max(1, h // freq)
        nw = max(1, w // freq)
        noise = rng.standard_normal((nh, nw)).astype(np.float32)
        up = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        field += amp * up
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    return cv2.GaussianBlur(field, (15, 15), 0)


class FogAblationEngine:
    def __init__(self, beta_base=0.038, hurst=0.75, wind_speed=0.04, alpha=0.02):
        self.beta_base = beta_base
        self.wind_speed = wind_speed
        self.alpha = alpha
        self.fog_texture = generate_fbm_field(512, 2048, hurst=hurst, octaves=6, seed=42)
        self.fog_color = np.array([0.92, 0.93, 0.95], dtype=np.float32)
        self.wind_offset = 0.0

    def _strat(self, h, w):
        y = np.linspace(1.0, 0.0, h, dtype=np.float32).reshape(h, 1)
        s = np.power(y, 1.2)
        return np.tile(s, (1, w))

    def _airlight(self, img_float, adaptive):
        if not adaptive:
            return self.fog_color
        h, w = img_float.shape[:2]
        sky = img_float[0:max(1, int(h * 0.12)), int(w * 0.25):max(int(w * 0.25) + 1, int(w * 0.75))]
        sampled = np.median(sky, axis=(0, 1))
        grey = np.mean(sampled)
        sampled = 0.75 * sampled + 0.25 * grey
        self.fog_color = (1.0 - self.alpha) * self.fog_color + self.alpha * sampled
        return self.fog_color

    def apply(self, frame_bgr, depth_meters, cfg):
        h, w = frame_bgr.shape[:2]
        img = frame_bgr.astype(np.float32) / 255.0

        self.wind_offset += self.wind_speed
        x0 = int(self.wind_offset) % (2048 - w)
        fbm = cv2.resize(self.fog_texture[:, x0:x0 + w], (w, h))
        strat = self._strat(h, w)

        phi = fbm if cfg["heterogeneous_density"] else np.ones((h, w), dtype=np.float32)
        psi = strat if cfg["heterogeneous_density"] else np.ones((h, w), dtype=np.float32)

        beta_field = self.beta_base * (0.8 + 0.4 * phi) * (0.6 + 0.4 * psi)

        transmission = np.exp(-beta_field * depth_meters)
        transmission = np.clip(transmission, 1e-5, 1.0)

        if cfg["nonlinear_gamma"]:
            d_norm = depth_meters / (np.max(depth_meters) + 1e-6)
            gamma_d = 1.0 + 0.4 * d_norm
            transmission = np.power(transmission, gamma_d)

        transmission = cv2.GaussianBlur(transmission, (15, 15), 0)
        t3 = np.stack([transmission, transmission, transmission], axis=-1)

        if cfg["volumetric_blur"]:
            blur_val = int(max(1, 21 * self.beta_base * 10))
            if blur_val % 2 == 0:
                blur_val += 1
            blurred = cv2.GaussianBlur(img, (blur_val, blur_val), 0)
            j_vol = img * t3 + blurred * (1.0 - t3)
        else:
            j_vol = img

        airlight = self._airlight(img, cfg["adaptive_airlight"])
        if cfg["heterogeneous_density"]:
            phi3 = np.stack([phi * 0.1 + 0.95, phi * 0.1 + 0.95, phi * 0.1 + 0.95], axis=-1)
        else:
            phi3 = np.ones_like(t3)

        s_road = np.zeros_like(t3)
        if cfg["secondary_scattering"]:
            s_road = 0.04 * np.stack([strat, strat, strat], axis=-1) * (1.0 - t3) ** 2

        foggy = j_vol * t3 + (airlight * phi3) * (1.0 - t3) + s_road

        if cfg["fog_coupled_noise"]:
            sigma = 0.003 / (transmission + 0.15)
            noise = np.random.normal(0.0, 1.0, foggy.shape).astype(np.float32)
            foggy = foggy + noise * np.stack([sigma, sigma, sigma], axis=-1)

        foggy = np.power(np.clip(foggy, 0.0, 1.0), 0.82) * 0.88 + 0.08
        out = np.clip(foggy * 255.0, 0, 255).astype(np.uint8)

        stats = {
            "mean_transmission": float(np.mean(transmission)),
            "std_transmission": float(np.std(transmission)),
        }
        return out, stats


def get_variants():
    module_names = MODULE_ORDER

    base = {
        "volumetric_blur": False,
        "heterogeneous_density": False,
        "nonlinear_gamma": False,
        "secondary_scattering": False,
        "adaptive_airlight": False,
        "fog_coupled_noise": False,
    }

    variants = {"clean": None}

    for r in range(1, len(module_names) + 1):
        for combo in itertools.combinations(module_names, r):
            cfg = dict(base)
            for m in combo:
                cfg[m] = True
            vname = "combo_" + "__".join(combo)
            variants[vname] = cfg

    return variants


def load_visia_model(visia_root, checkpoint_path, seq_len, device):
    if visia_root not in sys.path:
        sys.path.insert(0, visia_root)

    from student_train_inference.models.msbdn_rdff_dual_p3d import Net as Net4x

    model = Net4x(res_blocks=18, num_classes=19, T=seq_len).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    clean_sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd)
    model.eval()
    return model


def get_feather_mask(size, overlap, device):
    h, w = size
    mask = torch.ones((1, 1, 1, h, w), device=device)
    ramp = torch.linspace(0, 1, overlap, device=device)
    for i in range(overlap):
        mask[..., i, :] *= ramp[i]
        mask[..., h - 1 - i, :] *= ramp[i]
        mask[..., :, i] *= ramp[i]
        mask[..., :, w - 1 - i] *= ramp[i]
    return mask


@torch.no_grad()
def tiled_predict(model, input_seq, mask, tile_size, overlap, device):
    b, t, c, h, w = input_seq.shape
    output = torch.zeros_like(input_seq)
    count_map = torch.zeros((1, 1, 1, h, w), device=device)

    padded = torch.nn.functional.pad(
        input_seq.reshape(b * t, c, h, w),
        (overlap, overlap, overlap, overlap),
        mode="reflect",
    ).reshape(b, t, c, h + 2 * overlap, w + 2 * overlap)

    stride = tile_size - overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = y_end - tile_size
            x_start = x_end - tile_size

            tile = padded[:, :, :, y_start:y_end + 2 * overlap, x_start:x_end + 2 * overlap]
            th, tw = tile.shape[3], tile.shape[4]
            ph = (16 - th % 16) % 16
            pw = (16 - tw % 16) % 16

            tile_in = torch.nn.functional.pad(tile.reshape(-1, c, th, tw), (0, pw, 0, ph), mode="reflect")

            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    res_flat, _ = model(tile_in)
            else:
                res_flat, _ = model(tile_in)

            res = res_flat.reshape(b, t, c, tile_in.shape[2], tile_in.shape[3])
            res_cropped = res[:, :, :, overlap:overlap + tile_size, overlap:overlap + tile_size]

            output[:, :, :, y_start:y_end, x_start:x_end] += res_cropped * mask
            count_map[:, :, :, y_start:y_end, x_start:x_end] += mask

    return output / (count_map + 1e-8)


def bgr_to_tensor(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return TF.to_tensor(Image.fromarray(frame_rgb))


def run_inference_on_variant(model, frame_paths, output_dir, seq_len, tile_size, overlap, device):
    ensure_dir(output_dir)
    half = seq_len // 2
    mask = get_feather_mask((tile_size, tile_size), overlap, device)

    frames = [cv2.imread(p, cv2.IMREAD_COLOR) for p in frame_paths]
    tensors = [bgr_to_tensor(f) for f in frames]

    n = len(frame_paths)
    for idx in tqdm(range(n), desc=f"Infer {os.path.basename(output_dir)}"):
        start = idx - half
        end = idx + half
        window = []
        for wi in range(start, end + 1):
            ci = min(max(wi, 0), n - 1)
            window.append(tensors[ci])

        input_seq = torch.stack(window, dim=0).unsqueeze(0).to(device)
        out_seq = tiled_predict(model, input_seq, mask, tile_size, overlap, device)
        clean = out_seq[0, half].clamp(0, 1).cpu()

        np_out = (clean.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        out_bgr = cv2.cvtColor(np_out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(frame_paths[idx])), out_bgr)

        del input_seq, out_seq, clean
        if device.type == "cuda":
            torch.cuda.empty_cache()


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_ranking_rows(summary_rows):
    records = []
    for row in summary_rows:
        rec = {
            "variant": row["variant"],
            "num_frames": int(row["num_frames"]),
            "avg_psnr_degraded_vs_gt": float(row["avg_psnr_degraded_vs_gt"]),
            "avg_ssim_degraded_vs_gt": float(row["avg_ssim_degraded_vs_gt"]),
            "avg_psnr_restored_vs_gt": float(row["avg_psnr_restored_vs_gt"]),
            "avg_ssim_restored_vs_gt": float(row["avg_ssim_restored_vs_gt"]),
            "avg_delta_psnr": float(row["avg_delta_psnr"]),
            "avg_delta_ssim": float(row["avg_delta_ssim"]),
            "avg_mean_transmission": float(row["avg_mean_transmission"]),
            "avg_std_transmission": float(row["avg_std_transmission"]),
        }
        records.append(rec)

    by_dpsnr = sorted(records, key=lambda x: x["avg_delta_psnr"], reverse=True)
    rank_dpsnr = {r["variant"]: i + 1 for i, r in enumerate(by_dpsnr)}

    by_dssim = sorted(records, key=lambda x: x["avg_delta_ssim"], reverse=True)
    rank_dssim = {r["variant"]: i + 1 for i, r in enumerate(by_dssim)}

    by_psnr = sorted(records, key=lambda x: x["avg_psnr_restored_vs_gt"], reverse=True)
    rank_psnr = {r["variant"]: i + 1 for i, r in enumerate(by_psnr)}

    by_ssim = sorted(records, key=lambda x: x["avg_ssim_restored_vs_gt"], reverse=True)
    rank_ssim = {r["variant"]: i + 1 for i, r in enumerate(by_ssim)}

    ranked = []
    for rec in records:
        rd = rank_dpsnr[rec["variant"]]
        rs = rank_dssim[rec["variant"]]
        rp = rank_psnr[rec["variant"]]
        rr = rank_ssim[rec["variant"]]
        mean_rank = (rd + rs + rp + rr) / 4.0
        ranked.append({
            "variant": rec["variant"],
            "num_frames": rec["num_frames"],
            "avg_delta_psnr": f"{rec['avg_delta_psnr']:.6f}",
            "avg_delta_ssim": f"{rec['avg_delta_ssim']:.6f}",
            "avg_psnr_restored_vs_gt": f"{rec['avg_psnr_restored_vs_gt']:.6f}",
            "avg_ssim_restored_vs_gt": f"{rec['avg_ssim_restored_vs_gt']:.6f}",
            "rank_delta_psnr": rd,
            "rank_delta_ssim": rs,
            "rank_psnr_restored": rp,
            "rank_ssim_restored": rr,
            "mean_rank": f"{mean_rank:.4f}",
        })

    ranked.sort(key=lambda x: float(x["mean_rank"]))
    return ranked


def parse_gpu_ids(gpu_ids_text):
    ids = []
    for t in gpu_ids_text.split(","):
        t = t.strip()
        if t:
            ids.append(int(t))
    if not ids:
        ids = [0]
    return ids


def build_stagewise_cfgs():
    base = {m: False for m in MODULE_ORDER}
    stage_cfgs = [("stage_0_clean", None)]
    enabled = []
    for i, m in enumerate(MODULE_ORDER, start=1):
        enabled.append(m)
        cfg = dict(base)
        for e in enabled:
            cfg[e] = True
        stage_name = f"stage_{i}_{'__'.join(enabled)}"
        stage_cfgs.append((stage_name, cfg))
    return stage_cfgs


def generate_stagewise_degradation(clean_frames, depths, frame_names, out_root, beta, hurst):
    ensure_dir(out_root)
    stage_cfgs = build_stagewise_cfgs()

    for stage_name, cfg in stage_cfgs:
        out_dir = os.path.join(out_root, stage_name)
        ensure_dir(out_dir)

        if cfg is None:
            for img, n in zip(clean_frames, frame_names):
                cv2.imwrite(os.path.join(out_dir, n), img)
            print(f"Saved stagewise {stage_name}: {len(frame_names)}/{len(frame_names)}")
            continue

        engine = FogAblationEngine(beta_base=beta, hurst=hurst)
        for img, dep, n in tqdm(
            zip(clean_frames, depths, frame_names),
            total=len(frame_names),
            desc=f"Stagewise {stage_name}",
        ):
            foggy, _ = engine.apply(img, dep, cfg)
            cv2.imwrite(os.path.join(out_dir, n), foggy)

        saved = len(list_images(out_dir))
        print(f"Saved stagewise {stage_name}: {saved}/{len(frame_names)}")


def inference_worker(
    worker_name,
    gpu_id,
    visia_root,
    checkpoint,
    seq_len,
    tile_size,
    overlap,
    degraded_root,
    restored_root,
    frame_names,
    variant_queue,
):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[{worker_name}] Starting on device {device}")
    model = load_visia_model(visia_root, checkpoint, seq_len, device)

    while True:
        vname = variant_queue.get()
        if vname is None:
            print(f"[{worker_name}] Stop signal received")
            break

        print(f"[{worker_name}] Inference for variant: {vname}")
        in_dir = os.path.join(degraded_root, vname)
        out_dir = os.path.join(restored_root, vname)
        v_paths = [os.path.join(in_dir, n) for n in frame_names]
        run_inference_on_variant(
            model=model,
            frame_paths=v_paths,
            output_dir=out_dir,
            seq_len=seq_len,
            tile_size=tile_size,
            overlap=overlap,
            device=device,
        )


def main():
    parser = argparse.ArgumentParser(description="Fog module ablation + VISIA checkpoint inference on frame folders")
    parser.add_argument("--input-frames", type=str, required=True, help="Folder containing clean frames")
    parser.add_argument("--checkpoint", type=str, default="/scratch/Ananya_Kulkarni/VISIA/experiments/p3d_defog_v1/best_model.pth")
    parser.add_argument("--visia-root", type=str, default="/scratch/Ananya_Kulkarni/VISIA")
    parser.add_argument("--vda-root", type=str, default="/scratch/Ananya_Kulkarni/Video-Depth-Anything")
    parser.add_argument("--output-root", type=str, default="/scratch/Ananya_Kulkarni/JarvisIR/ablation_outputs")
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitl"])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--infer-gpu-ids", type=str, default="0,1", help="Comma-separated GPU ids for parallel inference")

    parser.add_argument("--resize-width", type=int, default=960)
    parser.add_argument("--resize-height", type=int, default=540)
    parser.add_argument("--target-frames", type=int, default=900, help="Target frame count for the experiment")

    parser.add_argument("--beta", type=float, default=0.038)
    parser.add_argument("--hurst", type=float, default=0.75)

    parser.add_argument("--seq-len", type=int, default=7)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    args = parser.parse_args()

    if args.seq_len % 2 == 0:
        raise ValueError("--seq-len must be odd to select a center frame")

    if not os.path.isdir(args.input_frames):
        raise FileNotFoundError(f"Input folder not found: {args.input_frames}")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"fog_ablation_{ts}"
    run_root = os.path.join(args.output_root, run_name)

    gt_dir = os.path.join(run_root, "gt_frames")
    degraded_root = os.path.join(run_root, "degraded")
    degraded_stagewise_root = os.path.join(run_root, "degraded_stagewise")
    restored_root = os.path.join(run_root, "restored")
    analysis_dir = os.path.join(run_root, "analysis")

    for p in [gt_dir, degraded_root, degraded_stagewise_root, restored_root, analysis_dir]:
        ensure_dir(p)

    print(f"Run root: {run_root}")

    src_paths = list_images(args.input_frames)
    if len(src_paths) == 0:
        raise RuntimeError("No images found in input folder")

    prepared_frames = build_target_frame_list(src_paths, args.target_frames)

    print(f"Preparing {len(prepared_frames)} frames from {len(src_paths)} source frames")

    frame_names = []
    clean_frames = []
    for p, name in tqdm(prepared_frames, desc="Load/resize GT"):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (args.resize_width, args.resize_height), interpolation=cv2.INTER_AREA)
        out_path = os.path.join(gt_dir, name)
        cv2.imwrite(out_path, img)
        frame_names.append(name)
        clean_frames.append(img)

    vda_device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Depth device: {vda_device}")

    print("Loading VDA depth model...")
    depth_pipe = load_vda(args.vda_root, args.encoder, args.gpu_id)

    print("Computing depth maps...")
    depths = []
    batch_size = 4
    for i in tqdm(range(0, len(clean_frames), batch_size), desc="Depth"):
        batch = clean_frames[i:i + batch_size]
        batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch]
        d = depth_pipe.infer_batch(batch_rgb)
        for dd in d:
            depths.append(dd.astype(np.float32))

    variants = get_variants()
    print(f"Total variants to test: {len(variants)}")
    variant_stats = {}

    print("Generating stage-wise cumulative degradation previews...")
    generate_stagewise_degradation(
        clean_frames=clean_frames,
        depths=depths,
        frame_names=frame_names,
        out_root=degraded_stagewise_root,
        beta=args.beta,
        hurst=args.hurst,
    )

    print("Generating fog ablation variants...")
    for vname, cfg in variants.items():
        out_dir = os.path.join(degraded_root, vname)
        ensure_dir(out_dir)

        if cfg is None:
            for img, n in zip(clean_frames, frame_names):
                cv2.imwrite(os.path.join(out_dir, n), img)
            variant_stats[vname] = [{"mean_transmission": 1.0, "std_transmission": 0.0}] * len(frame_names)
            continue

        engine = FogAblationEngine(beta_base=args.beta, hurst=args.hurst)
        stats_this = []
        for img, dep, n in tqdm(zip(clean_frames, depths, frame_names), total=len(frame_names), desc=f"Fog {vname}"):
            foggy, st = engine.apply(img, dep, cfg)
            cv2.imwrite(os.path.join(out_dir, n), foggy)
            stats_this.append(st)
        variant_stats[vname] = stats_this

    gpu_ids = parse_gpu_ids(args.infer_gpu_ids)

    if not torch.cuda.is_available():
        gpu_ids = [0]

    print(f"Running VISIA inference across GPUs: {gpu_ids}")
    if len(gpu_ids) == 1:
        # Single-GPU fallback
        device_single = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
        model_single = load_visia_model(args.visia_root, args.checkpoint, args.seq_len, device_single)
        for vname in variants.keys():
            in_dir = os.path.join(degraded_root, vname)
            out_dir = os.path.join(restored_root, vname)
            v_paths = [os.path.join(in_dir, n) for n in frame_names]
            run_inference_on_variant(
                model=model_single,
                frame_paths=v_paths,
                output_dir=out_dir,
                seq_len=args.seq_len,
                tile_size=args.tile_size,
                overlap=args.overlap,
                device=device_single,
            )
    else:
        variant_queue = mp.Queue()
        for vname in variants.keys():
            variant_queue.put(vname)
        for _ in gpu_ids:
            variant_queue.put(None)

        procs = []
        for wi, gid in enumerate(gpu_ids):
            p = mp.Process(
                target=inference_worker,
                args=(
                    f"worker-{wi}",
                    gid,
                    args.visia_root,
                    args.checkpoint,
                    args.seq_len,
                    args.tile_size,
                    args.overlap,
                    degraded_root,
                    restored_root,
                    frame_names,
                    variant_queue,
                ),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"A worker exited with non-zero code: {p.exitcode}")

    print("Computing per-frame metrics and analysis CSVs...")
    per_frame_rows = []
    summary_map = {}

    for vname in variants.keys():
        sum_key = vname
        summary_map[sum_key] = {
            "count": 0,
            "psnr_deg": 0.0,
            "ssim_deg": 0.0,
            "psnr_res": 0.0,
            "ssim_res": 0.0,
            "dpsnr": 0.0,
            "dssim": 0.0,
            "mt": 0.0,
            "st": 0.0,
        }

        for idx, n in enumerate(frame_names):
            gt = cv2.imread(os.path.join(gt_dir, n), cv2.IMREAD_COLOR)
            deg = cv2.imread(os.path.join(degraded_root, vname, n), cv2.IMREAD_COLOR)
            res = cv2.imread(os.path.join(restored_root, vname, n), cv2.IMREAD_COLOR)

            psnr_deg = np_psnr(deg, gt)
            ssim_deg = np_ssim(deg, gt)
            psnr_res = np_psnr(res, gt)
            ssim_res = np_ssim(res, gt)
            dpsnr = psnr_res - psnr_deg
            dssim = ssim_res - ssim_deg

            st = variant_stats[vname][idx]
            mt = st["mean_transmission"]
            stdt = st["std_transmission"]

            per_frame_rows.append({
                "variant": vname,
                "frame": n,
                "psnr_degraded_vs_gt": f"{psnr_deg:.6f}",
                "ssim_degraded_vs_gt": f"{ssim_deg:.6f}",
                "psnr_restored_vs_gt": f"{psnr_res:.6f}",
                "ssim_restored_vs_gt": f"{ssim_res:.6f}",
                "delta_psnr": f"{dpsnr:.6f}",
                "delta_ssim": f"{dssim:.6f}",
                "mean_transmission": f"{mt:.6f}",
                "std_transmission": f"{stdt:.6f}",
            })

            acc = summary_map[sum_key]
            acc["count"] += 1
            acc["psnr_deg"] += psnr_deg
            acc["ssim_deg"] += ssim_deg
            acc["psnr_res"] += psnr_res
            acc["ssim_res"] += ssim_res
            acc["dpsnr"] += dpsnr
            acc["dssim"] += dssim
            acc["mt"] += mt
            acc["st"] += stdt

    per_frame_csv = os.path.join(analysis_dir, "per_frame_metrics.csv")
    write_csv(
        per_frame_csv,
        [
            "variant",
            "frame",
            "psnr_degraded_vs_gt",
            "ssim_degraded_vs_gt",
            "psnr_restored_vs_gt",
            "ssim_restored_vs_gt",
            "delta_psnr",
            "delta_ssim",
            "mean_transmission",
            "std_transmission",
        ],
        per_frame_rows,
    )

    summary_rows = []
    for vname, acc in summary_map.items():
        c = max(acc["count"], 1)
        summary_rows.append({
            "variant": vname,
            "num_frames": acc["count"],
            "avg_psnr_degraded_vs_gt": f"{(acc['psnr_deg'] / c):.6f}",
            "avg_ssim_degraded_vs_gt": f"{(acc['ssim_deg'] / c):.6f}",
            "avg_psnr_restored_vs_gt": f"{(acc['psnr_res'] / c):.6f}",
            "avg_ssim_restored_vs_gt": f"{(acc['ssim_res'] / c):.6f}",
            "avg_delta_psnr": f"{(acc['dpsnr'] / c):.6f}",
            "avg_delta_ssim": f"{(acc['dssim'] / c):.6f}",
            "avg_mean_transmission": f"{(acc['mt'] / c):.6f}",
            "avg_std_transmission": f"{(acc['st'] / c):.6f}",
        })

    summary_csv = os.path.join(analysis_dir, "summary_metrics.csv")
    write_csv(
        summary_csv,
        [
            "variant",
            "num_frames",
            "avg_psnr_degraded_vs_gt",
            "avg_ssim_degraded_vs_gt",
            "avg_psnr_restored_vs_gt",
            "avg_ssim_restored_vs_gt",
            "avg_delta_psnr",
            "avg_delta_ssim",
            "avg_mean_transmission",
            "avg_std_transmission",
        ],
        summary_rows,
    )

    ranking_rows = build_ranking_rows(summary_rows)
    ranking_csv = os.path.join(analysis_dir, "ranking_metrics.csv")
    write_csv(
        ranking_csv,
        [
            "variant",
            "num_frames",
            "avg_delta_psnr",
            "avg_delta_ssim",
            "avg_psnr_restored_vs_gt",
            "avg_ssim_restored_vs_gt",
            "rank_delta_psnr",
            "rank_delta_ssim",
            "rank_psnr_restored",
            "rank_ssim_restored",
            "mean_rank",
        ],
        ranking_rows,
    )

    top5_rows = ranking_rows[:5]
    top5_csv = os.path.join(analysis_dir, "ranking_top5.csv")
    write_csv(
        top5_csv,
        [
            "variant",
            "num_frames",
            "avg_delta_psnr",
            "avg_delta_ssim",
            "avg_psnr_restored_vs_gt",
            "avg_ssim_restored_vs_gt",
            "rank_delta_psnr",
            "rank_delta_ssim",
            "rank_psnr_restored",
            "rank_ssim_restored",
            "mean_rank",
        ],
        top5_rows,
    )

    print("Done.")
    print(f"GT frames: {gt_dir}")
    print(f"Degraded variants: {degraded_root}")
    print(f"Degraded stagewise: {degraded_stagewise_root}")
    print(f"Restored outputs: {restored_root}")
    print(f"Per-frame CSV: {per_frame_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Ranking CSV: {ranking_csv}")
    print(f"Top-5 CSV: {top5_csv}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
