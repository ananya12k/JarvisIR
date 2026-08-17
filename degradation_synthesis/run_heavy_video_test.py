import os
import sys
import glob
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from fog_synthesis import PhysicalFogEngine, VDAWrapper

SRC_FOLDER = '/scratch/Ananya_Kulkarni/Defog_balanced/val/GT/GH019005'
OUT_ROOT   = '/scratch/Ananya_Kulkarni/JarvisIR/degradation_synthesis/fog_video_heavy_test'
FRAMES_DIR = os.path.join(OUT_ROOT, 'frames')
VIDEO_PATH = os.path.join(OUT_ROOT, 'heavy_fog.mp4')
NUM_FRAMES = 400
FPS = 30
BETA_HEAVY = 0.100

os.makedirs(FRAMES_DIR, exist_ok=True)

img_paths = sorted(glob.glob(os.path.join(SRC_FOLDER, '*.jpg')))[:NUM_FRAMES]
frames = [cv2.imread(p) for p in img_paths]
h, w = frames[0].shape[:2]

pipe = VDAWrapper(encoder='vits', vda_root_path='/scratch/Ananya_Kulkarni/Video-Depth-Anything', device_id=0)
engine = PhysicalFogEngine(beta_base=BETA_HEAVY, hurst=0.75)

out_vid = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))

batch_size = 4
for i in tqdm(range(0, len(frames), batch_size), desc='heavy fog'):
    batch = frames[i:i+batch_size]
    depths = pipe.infer_batch([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch])
    for k, frame in enumerate(batch):
        res = engine.apply_fog(frame, depths[k])
        frame_idx = i + k
        cv2.imwrite(os.path.join(FRAMES_DIR, f'{frame_idx:06d}.png'), res)
        out_vid.write(res)

out_vid.release()
print(f'Saved {len(frames)} frames to {FRAMES_DIR}')
print(f'Saved video to {VIDEO_PATH}')
