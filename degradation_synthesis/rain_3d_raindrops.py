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
        
        self.cx = self.w / 2.0
        self.cy = self.h / 2.0
        fov_rad = math.radians(fov_deg)
        self.focal_length = self.w / (2 * math.tan(fov_rad / 2))
        
        # [x, y, z, prev_x, prev_y, prev_z, rand_size, rand_opacity]
        self.particles = np.zeros((max_particles, 8), dtype=np.float32)
        self.time_counter = 0.0
        self._spawn(max_particles)

    def _spawn(self, count, idxs=None):
        z = np.random.uniform(0.5, self.max_depth, count)
        view_w = (self.w * z) / self.focal_length
        view_h = (self.h * z) / self.focal_length
        
        x = np.random.uniform(-view_w * 1.5, view_w * 1.5, count)
        y = np.random.uniform(-view_h * 1.5, view_h * 1.5, count)
        
        # Increased size variation for "Bokeh" look
        r_size = np.random.uniform(0.5, 2.0, count) 
        r_opac = np.random.uniform(0.4, 0.9, count)
        
        if idxs is None:
            self.particles[:, 0] = x
            self.particles[:, 1] = y
            self.particles[:, 2] = z
            self.particles[:, 3:6] = self.particles[:, 0:3]
            self.particles[:, 6] = r_size
            self.particles[:, 7] = r_opac
        else:
            self.particles[idxs, 0] = x
            self.particles[idxs, 1] = y
            self.particles[idxs, 2] = z
            self.particles[idxs, 3:6] = self.particles[idxs, 0:3]
            self.particles[idxs, 6] = r_size
            self.particles[idxs, 7] = r_opac

    def update(self, dt, car_vel_vec, wind_vel_vec):
        self.time_counter += dt
        
        # Physics
        gust = np.sin(self.time_counter * 2.0) * 1.5 
        current_wind = np.array(wind_vel_vec)
        current_wind[0] += gust 

        gravity = np.array([0, 20.0, 0]) 
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
        
        mask_l = self.particles[:, 0] < -limit_w
        self.particles[mask_l, 0] += (2 * limit_w[mask_l])

        limit_y = (self.h * z) / (2 * self.focal_length) * 1.5
        mask_reset = (self.particles[:, 2] < 0.2) | \
                     (self.particles[:, 2] > self.max_depth) | \
                     (self.particles[:, 1] > limit_y)
        
        if np.any(mask_reset):
            self._spawn(np.sum(mask_reset), np.where(mask_reset)[0])

    def render(self, background_img, depth_map_meters, style_params):
        h, w = background_img.shape[:2]
        rain_layer = np.zeros(background_img.shape, dtype=np.uint8)
        
        z = np.maximum(self.particles[:, 2], 0.1)
        
        u = (self.particles[:, 0] * self.focal_length) / z + self.cx
        v = (self.particles[:, 1] * self.focal_length) / z + self.cy
        
        rand_sizes = self.particles[:, 6]
        rand_opacs = self.particles[:, 7]
        
        valid_idxs = np.where((u >= -20) & (u < w+20) & (v >= -20) & (v < h+20))[0]
        
        for i in valid_idxs:
            ui, vi = int(u[i]), int(v[i])
            
            d_u, d_v = np.clip(ui, 0, w-1), np.clip(vi, 0, h-1)
            if z[i] > (depth_map_meters[d_v, d_u] * 0.95): continue

            # --- BOKEH BLOB RENDERING ---
            
            # Base Radius from depth
            # Close = 5px, Far = 2px. (Never 1px/point)
            if z[i] < 3.0:
                base_radius = 5.0
            elif z[i] < 10.0:
                base_radius = 3.5
            else:
                base_radius = 2.0
            
            # Apply random variation factor (0.5x to 2.0x)
            final_radius = int(base_radius * rand_sizes[i])
            final_radius = max(2, final_radius) # Ensure minimum size is 2
            
            # Color: Watery Grey
            base_col = 180
            col_val = int(np.clip(base_col * rand_opacs[i], 140, 220))
            color = (col_val, col_val, col_val)
            
            # Draw Circle (Filled)
            cv2.circle(rain_layer, (ui, vi), final_radius, color, -1)
            
        # Heavy Gaussian Blur to create "Out of Focus" look
        rain_blur = cv2.GaussianBlur(rain_layer, (5, 5), 0)
        
        # Soft Blend
        return cv2.addWeighted(background_img, 1.0, rain_blur, 0.35, 0)

def apply_global_darkening(img, factor=0.85):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = (hsv[:, :, 2] * factor).astype(np.uint8)
    hsv[:, :, 1] = (hsv[:, :, 1] * 0.9).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)