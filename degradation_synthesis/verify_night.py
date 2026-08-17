import sys, types
sys.path.insert(0, '/scratch/Ananya_Kulkarni/JarvisIR/degradation_synthesis')
vda_mod = types.ModuleType('video_depth_anything')
sys.modules['video_depth_anything'] = vda_mod
sys.modules['video_depth_anything.video_depth'] = vda_mod

import numpy as np, cv2
from PIL import Image
from night_synthesis import NightEngine, EXTREME_VARIANTS, sample_variant_params

img = np.array(Image.open('/scratch/Ananya_Kulkarni/Defog_balanced/train/GT/00/000001.jpg'))
frame_bgr = img[:, :, ::-1]
h, w = frame_bgr.shape[:2]
y = np.linspace(80, 5, h).reshape(h, 1)
depth = np.tile(y, (1, w)).astype(np.float32)

print("Stage          mean  noise  B_mean  R_mean  colour")
rng = np.random.default_rng(0)
for name, base in EXTREME_VARIANTS.items():
    params = sample_variant_params(base, rng)
    eng = NightEngine(**params)
    out = eng.apply_night(frame_bgr, depth)
    noise = np.std(out.astype(float) - cv2.GaussianBlur(out,(5,5),0).astype(float))
    b, r = out[...,0].mean(), out[...,2].mean()
    warm = "warm" if r > b else "cool/cyan"
    print(f"{name:14s}  {out.mean():5.1f}  {noise:5.2f}  {b:6.1f}  {r:6.1f}  {warm}")
print(f"GT              {img.mean():5.1f}  --     {img[...,2].mean():6.1f}  {img[...,0].mean():6.1f}")
