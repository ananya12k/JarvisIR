import os
import glob
import random
import subprocess
from multiprocessing import Process, Queue
import time

import torch

# ================= CONFIGURATION =================
# Root directory where your source image folders are located
INPUT_ROOT = "/scratch/Ananya_Kulkarni/Video_fog_data"

# Root directory where processed foggy frames will be saved
OUTPUT_ROOT = "/scratch/Ananya_Kulkarni/Fog_Results_Frames"

# Path to the fog synthesis script
SCRIPT_PATH = "degradation_synthesis/fog_synthesis.py"

# List of GPU IDs to utilize
GPUS = [0, 1,2]

# Multi-GPU batch size (number of frames fog_synthesis.py processes at once)
BATCH_SIZE = 4
# =================================================


def sample_beta():
    """
    Samples a beta value based on physical visibility ranges.
    Formula: beta = 3 / Visibility(meters)
    """
    category = random.choice(['light', 'medium', 'heavy', 'extreme'])
    
    if category == 'light':
        return random.uniform(0.008, 0.02)   # V: 150m - 350m
    elif category == 'medium':
        return random.uniform(0.03, 0.06)    # V: 50m - 100m
    elif category == 'heavy':
        return random.uniform(0.08, 0.15)    # V: 20m - 40m
    else: # extreme
        return random.uniform(0.20, 0.45)    # V: 6m - 15m


def sample_intensity_and_beta():
    """Sample a fog intensity label together with its beta value."""
    category = random.choice(['light', 'medium', 'heavy', 'extreme'])

    if category == 'light':
        beta = random.uniform(0.008, 0.02)
    elif category == 'medium':
        beta = random.uniform(0.03, 0.06)
    elif category == 'heavy':
        beta = random.uniform(0.08, 0.15)
    else:
        beta = random.uniform(0.20, 0.45)

    return category, beta


def get_worker_gpus():
    """Use only CUDA devices that are actually visible to this process."""
    if torch.cuda.is_available():
        visible_count = torch.cuda.device_count()
        return GPUS[:visible_count] if visible_count > 0 else []
    return [0]


def worker(gpu_id, task_queue):
    """
    Process that lives on a specific GPU and pulls folders from the queue.
    """
    print(f" Worker started on GPU {gpu_id}")

    while not task_queue.empty():
        try:
            # timeout=2 prevents hanging if the queue check and get are not atomic
            task = task_queue.get(timeout=2)
        except:
            break

        input_folder = task['input']
        beta = task['beta']
        intensity = task['intensity']
        
        # Determine naming
        video_name = os.path.basename(input_folder)
        beta_str = f"{beta:.4f}"
        
        # Create unique output folder: folder_name_intensity_beta_0.0500
        output_folder = os.path.join(OUTPUT_ROOT, f"{video_name}_{intensity}_beta{beta_str}")
        os.makedirs(output_folder, exist_ok=True)

        print(f"[GPU {gpu_id}] Processing: {video_name} | {intensity} | β={beta_str}")

        # Construct the command
        # We pass the gpu_id directly to the script's --gpu_id argument
        cmd = [
            "python", SCRIPT_PATH,
            "--input", input_folder,
            "--output", output_folder,
            "--beta", str(beta),
            "--gpu_id", str(gpu_id)
        ]

        try:
            # Run the synthesis script and wait for it to finish
            start_time = time.time()
            subprocess.run(cmd, check=True)
            elapsed = time.time() - start_time
            print(f"Finished: {video_name} on GPU {gpu_id} in {elapsed:.1f}s")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {video_name} on GPU {gpu_id}: {e}")


if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 1. Collect all folders in the input root
    # Filters out any files, only takes directories
    all_folders = sorted([
        os.path.join(INPUT_ROOT, d) 
        for d in os.listdir(INPUT_ROOT) 
        if os.path.isdir(os.path.join(INPUT_ROOT, d))
    ])

    print(f"Found {len(all_folders)} video folders to process.")

    # 2. Fill the Task Queue
    task_queue = Queue()
    for folder in all_folders:
        intensity, beta = sample_intensity_and_beta()
        task_queue.put({
            'input': folder,
            'intensity': intensity,
            'beta': beta,
        })

    # 3. Spawn Worker Processes
    processes = []
    worker_gpus = get_worker_gpus()
    if not worker_gpus:
        worker_gpus = [0]

    for gpu_id in worker_gpus:
        p = Process(target=worker, args=(gpu_id, task_queue))
        p.start()
        processes.append(p)

    # 4. Wait for all processes to finish
    for p in processes:
        p.join()

    print("\ ALL FOG GENERATION TASKS COMPLETE!")