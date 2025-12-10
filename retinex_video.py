import cv2
import torch
import torch.nn.functional as F
import numpy as np
import time

# --------- GPU Retinex function --------- #
def gpu_retinex_torch(frame, sigma=50):
    img = torch.from_numpy(frame.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()
    img = img + 1.0

    size = int(6*sigma + 1)
    if size % 2 == 0:
        size += 1

    x = torch.arange(size).cuda() - size // 2
    gauss_1d = torch.exp(-(x**2) / (2*sigma*sigma))
    gauss_1d = gauss_1d / gauss_1d.sum()

    gauss_kernel_x = gauss_1d.view(1,1,1,-1)
    gauss_kernel_y = gauss_1d.view(1,1,-1,1)

    for c in range(3):
        channel = img[:, c:c+1]
        channel = F.conv2d(channel, gauss_kernel_x, padding=(0, size//2))
        channel = F.conv2d(channel, gauss_kernel_y, padding=(size//2, 0))
        if c == 0:
            blur_out = channel
        else:
            blur_out = torch.cat([blur_out, channel], dim=1)

    blur = blur_out

    retinex = torch.log(img) - torch.log(blur + 1.0)

    r_min = retinex.min()
    r_max = retinex.max()
    retinex = (retinex - r_min) / (r_max - r_min + 1e-6)

    out = retinex * 255.0
    out = out.squeeze(0).permute(1,2,0).clamp(0,255).cpu().numpy().astype(np.uint8)

    return out


# ---------------- VIDEO PROCESSING ---------------- #

input_video = "00_clean_night_rain_2.mp4"
output_video = "00_clean.mp4"

cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = frame_count_total / fps

print(f"📌 Input video FPS: {fps}")
print(f"📌 Total frames: {frame_count_total}")
print(f"📌 Video duration: {video_duration:.2f} sec")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

frame_id = 0
start_time = time.time()

print("🔥 Running GPU Retinex...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    enhanced = gpu_retinex_torch(frame)
    writer.write(enhanced)

    frame_id += 1
    if frame_id % 30 == 0:
        print(f"Processed {frame_id}/{frame_count_total} frames...")

end_time = time.time()

processing_time = end_time - start_time
real_time_factor = video_duration / processing_time

cap.release()
writer.release()

print("\n================ RESULTS ================")
print(f"⏱ Original video duration : {video_duration:.2f} sec")
print(f"⏱ Processing time         : {processing_time:.2f} sec")
print(f"⚡ Real-time factor (RTF)  : {real_time_factor:.2f}x")
print(f"📁 Saved output            : {output_video}")
print("==========================================")
