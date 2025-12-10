# File: JarvisIR/degradation_synthesis/rain_engine_3d.py

import numpy as np
import cv2
import random
import math

class Rain3DSystem:
    def __init__(self, image_shape, max_particles=10000, max_depth=50.0, fov_deg=70):
        self.h, self.w = image_shape[:2]
        self.max_particles = max_particles
        self.max_depth = max_depth
        
        # Intrinsics
        self.cx = self.w / 2.0
        self.cy = self.h / 2.0
        fov_rad = math.radians(fov_deg)
        self.focal_length = self.w / (2 * math.tan(fov_rad / 2))
        
        self.particles = np.zeros((max_particles, 8), dtype=np.float32)
        
        self.time_counter = 0.0
        self._spawn(max_particles)

    def _spawn(self, count, idxs=None):
        z = np.random.uniform(0.5, self.max_depth, count)
        view_w = (self.w * z) / self.focal_length
        view_h = (self.h * z) / self.focal_length
        
        x = np.random.uniform(-view_w * 1.5, view_w * 1.5, count)
        y = np.random.uniform(-view_h * 1.5, view_h * 1.5, count)
            
        # --- FIX: SHORTER LENGTHS ---
        # Was 0.8-1.4 -> Now 0.15-0.35
        # This makes them look like short dashes (real shutter speed)
        r_len = np.random.uniform(0.15, 0.35, count)
        r_bright = np.random.uniform(0.5, 1.0, count)
        
        if idxs is None:
            self.particles[:, 0] = x
            self.particles[:, 1] = y
            self.particles[:, 2] = z
            self.particles[:, 3:6] = self.particles[:, 0:3]
            self.particles[:, 6] = r_len
            self.particles[:, 7] = r_bright
        else:
            self.particles[idxs, 0] = x
            self.particles[idxs, 1] = y
            self.particles[idxs, 2] = z
            self.particles[idxs, 3:6] = self.particles[idxs, 0:3]
            self.particles[idxs, 6] = r_len
            self.particles[idxs, 7] = r_bright

    def update(self, dt, car_vel_vec, wind_vel_vec):
        self.time_counter += dt
        
        # Physics
        gust = np.sin(self.time_counter * 3.0) * 2.5 
        current_wind = np.array(wind_vel_vec)
        current_wind[0] += gust 

        gravity = np.array([0, 25.0, 0]) 
        rel_vel = (current_wind + gravity) - np.array(car_vel_vec)
        
        self.particles[:, 3:6] = self.particles[:, 0:3] 
        turbulence = np.random.normal(0, 0.5, (self.max_particles, 3))
        self.particles[:, 0:3] += (rel_vel + turbulence) * dt
        
        # Infinite Wrap
        z = self.particles[:, 2]
        factor = z / self.focal_length
        limit_w = self.w * factor * 1.5 
        
        mask_r = self.particles[:, 0] > limit_w
        self.particles[mask_r, 0] -= (2 * limit_w[mask_r])
        self.particles[mask_r, 3] = self.particles[mask_r, 0]
        
        mask_l = self.particles[:, 0] < -limit_w
        self.particles[mask_l, 0] += (2 * limit_w[mask_l])
        self.particles[mask_l, 3] = self.particles[mask_l, 0]

        limit_y = (self.h * z) / (2 * self.focal_length) * 1.5
        mask_reset = (self.particles[:, 2] < 0.2) | \
                     (self.particles[:, 2] > self.max_depth) | \
                     (self.particles[:, 1] > limit_y)
        
        if np.any(mask_reset):
            self._spawn(np.sum(mask_reset), np.where(mask_reset)[0])

    def render(self, background_img, depth_map_meters, style_params):
        h, w = background_img.shape[:2]
        
        # Draw on a blank mask
        rain_mask = np.zeros((h, w), dtype=np.uint8)
        
        z = np.maximum(self.particles[:, 2], 0.1)
        prev_z = np.maximum(self.particles[:, 5], 0.1)
        
        u = (self.particles[:, 0] * self.focal_length) / z + self.cx
        v = (self.particles[:, 1] * self.focal_length) / z + self.cy
        prev_u = (self.particles[:, 3] * self.focal_length) / prev_z + self.cx
        prev_v = (self.particles[:, 4] * self.focal_length) / prev_z + self.cy
        
        r_lens = self.particles[:, 6]
        
        valid_idxs = np.where((u >= -50) & (u < w+50) & (v >= -50) & (v < h+50))[0]
        
        for i in valid_idxs:
            ui, vi = int(u[i]), int(v[i])
            d_u, d_v = np.clip(ui, 0, w-1), np.clip(vi, 0, h-1)
            
            # Depth Occlusion
            if z[i] > (depth_map_meters[d_v, d_u] * 0.95): continue

            # Length Calculation
            dx = prev_u[i] - u[i]
            dy = prev_v[i] - v[i]
            
            # Shorten tail using random factor (0.15-0.35)
            tail_x = int(u[i] + dx * r_lens[i])
            tail_y = int(v[i] + dy * r_lens[i])
            
            # --- FIX: HARD LENGTH CAP ---
            # If a streak is too long (warp speed effect), chop it.
            # This prevents the "long laser" look.
            dist = math.sqrt((tail_x - ui)**2 + (tail_y - vi)**2)
            if dist > 30: 
                # Normalize and scale to 30px
                scale = 30.0 / dist
                tail_x = int(ui + (tail_x - ui) * scale)
                tail_y = int(vi + (tail_y - vi) * scale)
            
            if dist < 2: continue

            # Thickness: Strictly 1px unless super close (<2m)
            thick = 2 if z[i] < 2.0 else 1
            
            # Draw WHITE on the MASK
            cv2.line(rain_mask, (tail_x, tail_y), (ui, vi), 255, thick)
            
        # Blur the mask
        rain_blur = cv2.GaussianBlur(rain_mask, (3, 3), 0)
        
        # --- SUBTRACTIVE BLENDING ---
        # Same as before: darken BG + add grey tint
        alpha = (rain_blur.astype(np.float32) / 255.0) * 0.4
        alpha = np.stack([alpha, alpha, alpha], axis=2)
        
        bg_float = background_img.astype(np.float32)
        darkened_bg = bg_float * (1.0 - (0.3 * alpha))
        rain_color = np.ones_like(bg_float) * 110.0 # Dark Grey
        
        final = darkened_bg * (1.0 - alpha) + rain_color * alpha
        
        return np.clip(final, 0, 255).astype(np.uint8)

def apply_global_darkening(img, factor=0.85):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = (hsv[:, :, 2] * factor).astype(np.uint8)
    hsv[:, :, 1] = (hsv[:, :, 1] * 0.9).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)