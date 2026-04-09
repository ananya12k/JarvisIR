import os
import glob
import random
import shutil
import subprocess
from multiprocessing import Process, Queue
import time
import torch

# ================= CONFIGURATION =================
# Where your 15 original 1-minute folders are
ORIGINAL_INPUT_ROOT = "/scratch/Ananya_Kulkarni/Video_fog_data"

# Where the clean, split 30-second GT clips will be saved
GT_ROOT = "/scratch/Ananya_Kulkarni/Fog_GT_Frames"

# Where the final foggy results will be saved
OUTPUT_ROOT = "/scratch/Ananya_Kulkarni/Fog_Results_Frames_30sec_clips"

# Path to your synthesis script
SCRIPT_PATH = "degradation_synthesis/fog_synthesis.py"

# Available GPUs
GPUS = [0, 1, 2]
# =================================================

def sample_params():
    """Sample physical fog parameters for high diversity."""
    category = random.choice(['light', 'medium', 'heavy', 'extreme'])
    if category == 'light':
        beta = random.uniform(0.008, 0.02)
    elif category == 'medium':
        beta = random.uniform(0.03, 0.05)
    elif category == 'heavy':
        beta = random.uniform(0.07, 0.12)
    else: # extreme
        beta = random.uniform(0.20, 0.40)
    
    # Hurst exponent (H): 0.5 (Patchy/Turbulent) to 0.9 (Smooth/Uniform)
    hurst = random.uniform(0.5, 0.9)
    return category, beta, hurst

def prepare_split_gt():
    """Stage 1: Split 15 folders into 30 GT folders."""
    if not os.path.exists(GT_ROOT):
        os.makedirs(GT_ROOT)
    
    original_folders = sorted([d for d in os.listdir(ORIGINAL_INPUT_ROOT) 
                               if os.path.isdir(os.path.join(ORIGINAL_INPUT_ROOT, d))])
    
    split_folders = []
    print(f"--- STAGE 1: Splitting {len(original_folders)} folders into GT Parts ---")

    for folder_name in original_folders:
        src_path = os.path.join(ORIGINAL_INPUT_ROOT, folder_name)
        frames = sorted(glob.glob(os.path.join(src_path, "*.[jJ][pP][gG]")) + 
                        glob.glob(os.path.join(src_path, "*.[pP][nN][gG]")))
        
        mid = len(frames) // 2
        
        # Define Part 1 and Part 2
        parts = [
            (frames[:mid], "Part1"),
            (frames[mid:], "Part2")
        ]
        
        for frame_list, part_suffix in parts:
            new_folder_name = f"{folder_name}_{part_suffix}"
            dst_path = os.path.join(GT_ROOT, new_folder_name)
            
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
                # Copy frames to new GT location
                for f in frame_list:
                    shutil.copy2(f, dst_path)
            
            split_folders.append(dst_path)
            print(f"Created GT: {new_folder_name} ({len(frame_list)} frames)")
            
    return split_folders

def worker(gpu_id, task_queue):
    """Stage 2: Process fog rendering per split folder."""
    print(f"🚀 Worker started on GPU {gpu_id}")
    
    # Hide other GPUs from this process
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    while not task_queue.empty():
        try:
            task = task_queue.get(timeout=2)
        except:
            break

        input_folder = task['input'] # This is the GT folder
        beta = task['beta']
        hurst = task['hurst']
        intensity = task['intensity']
        
        video_part_name = os.path.basename(input_folder)
        
        # Final output name includes parameters for dataset tracking
        output_name = f"{video_part_name}_{intensity}_B{beta:.3f}_H{hurst:.2f}"
        output_folder = os.path.join(OUTPUT_ROOT, output_name)
        os.makedirs(output_folder, exist_ok=True)

        print(f"[GPU {gpu_id}] Rendering: {video_part_name} | β={beta:.3f} H={hurst:.2f}")

        # Construct Command
        cmd = [
            "python", SCRIPT_PATH,
            "--input", input_folder,
            "--output", output_folder,
            "--beta", str(beta),
            "--hurst", str(hurst),
            "--gpu_id", "0", # Always 0 because we masked with CUDA_VISIBLE_DEVICES
            "--encoder", "vits"
        ]

        try:
            subprocess.run(cmd, check=True, env=env)
            print(f"✅ Finished: {output_name}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed: {video_part_name}")

if __name__ == "__main__":
    # Stage 1: Split the data
    gt_folders = prepare_split_gt()
    
    # Stage 2: Queue Fog Synthesis
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    task_queue = Queue()

    print(f"\n--- STAGE 2: Queuing {len(gt_folders)} tasks for Fog Rendering ---")
    for folder in gt_folders:
        intensity, beta, hurst = sample_params()
        task_queue.put({
            'input': folder,
            'intensity': intensity,
            'beta': beta,
            'hurst': hurst
        })

    # Start multi-GPU processing
    processes = []
    for gid in GPUS:
        p = Process(target=worker, args=(gid, task_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n🎉 DATASET GENERATION COMPLETE!")
    print(f"Clean GT in: {GT_ROOT}")
    print(f"Foggy Results in: {OUTPUT_ROOT}")