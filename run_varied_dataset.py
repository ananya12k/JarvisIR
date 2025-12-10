# File: JarvisIR/run_varied_dataset.py

import os
import subprocess
import random
from glob import glob
from multiprocessing import Process, Queue

# ================= CONFIGURATION =================
INPUT_DIR = "/scratch/Ananya_Kulkarni/Jarvis/JarvisIR/Synthetic_rain_video_data"
OUTPUT_DIR = "/scratch/Ananya_Kulkarni/Jarvis/JarvisIR/Rain_Results_Varied"
SCRIPT_PATH = "degradation_synthesis/rain_syn_vda.py"

GPUS = [0, 1, 2, 3]
INTENSITIES = ['light', 'medium', 'heavy', 'storm']
# =================================================

def worker(gpu_id, task_queue):
    print(f"🚀 Worker started on GPU {gpu_id}")
    
    # CRITICAL: Isolate GPU to prevent MUNIT device errors
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    while not task_queue.empty():
        try:
            task = task_queue.get(timeout=2)
        except:
            break
            
        input_file = task['input']
        p = task['params']
        
        filename = os.path.basename(input_file)
        name, ext = os.path.splitext(filename)
        out_name = f"{name}_{p['intensity']}_wind{p['wind']}{ext}"
        output_file = os.path.join(OUTPUT_DIR, out_name)
        
        print(f"  [GPU {gpu_id}] Processing: {filename} -> {p['intensity'].upper()} (Spd:{p['speed']}, Wind:{p['wind']})")
        
        cmd = (
            f"python {SCRIPT_PATH} "
            f"--input \"{input_file}\" "
            f"--output \"{output_file}\" "
            f"--encoder vitl "
            f"--wide_angle "
            f"--intensity {p['intensity']} "
            f"--speed {p['speed']} "
            f"--wind {p['wind']}"
        )
        
        try:
            # Run without capture_output to see errors if any
            subprocess.run(cmd, shell=True, check=True, env=env)
        except subprocess.CalledProcessError:
            print(f"  ❌ [GPU {gpu_id}] Failed: {filename}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    video_files = sorted(glob(os.path.join(INPUT_DIR, "*.mp4")) + glob(os.path.join(INPUT_DIR, "*.MP4")))
    
    if not video_files:
        print(f"No video files found in {INPUT_DIR}")
        exit()
        
    print(f"Found {len(video_files)} videos. Queueing tasks...")
    
    task_queue = Queue()
    
    for i, video_path in enumerate(video_files):
        # 1. Round Robin Intensity
        intensity = INTENSITIES[i % 4]
        
        # 2. Random Parameters (Varied Numbers)
        if intensity == 'light':
            wind = random.randint(0, 5)
            speed = random.choice([0, 10, 20])
        elif intensity == 'medium':
            wind = random.randint(5, 15)
            speed = random.choice([30, 40, 50])
        elif intensity == 'heavy':
            wind = random.randint(15, 30)
            speed = random.choice([50, 70, 90])
        else: # Storm
            wind = random.randint(30, 50)
            speed = random.choice([0, 80, 100])
            
        task = {
            'input': video_path,
            'params': {
                'intensity': intensity,
                'speed': speed,
                'wind': wind
            }
        }
        task_queue.put(task)
        
    processes = []
    for gpu_id in GPUS:
        p = Process(target=worker, args=(gpu_id, task_queue))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print(f"\n✅ All videos processed! Results in: {OUTPUT_DIR}")