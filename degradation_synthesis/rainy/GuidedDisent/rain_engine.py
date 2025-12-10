# File: rain_engine.py (The Definitive Streak Engine)

import cv2
import numpy as np
import random
import math

def create_perlin_noise_map(shape, scale=100.0):
    """Creates a Perlin-like noise map to control rain density."""
    height, width = shape
    low_res_h, low_res_w = max(1, height // int(scale)), max(1, width // int(scale))
    noise = np.random.rand(low_res_h, low_res_w)
    perlin_map = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
    return perlin_map

def generate_streak_layer(
    image_shape, 
    vehicle_speed_kmh=30,
    rainfall_rate_mm_hr=50, 
    wind_speed_kmh=0
):
    """
    Definitive Engine: Generates ONLY the falling rain streaks as a transparent RGBA layer,
    using a noise-driven approach for natural, chaotic distribution.
    """
    height, width = image_shape
    
    # --- PHYSICAL & PERSPECTIVE PARAMETERS ---
    PIXELS_PER_METER = 50 
    vehicle_speed_ms = vehicle_speed_kmh * 1000 / 3600
    base_rain_vy_pps = 400
    wind_vx_pps = (wind_speed_kmh * 1000 / 3600) * PIXELS_PER_METER
    focus_of_expansion = (width // 2, height // 2)

    streaks_layer_rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # --- NOISE-DRIVEN STREAK GENERATION ---
    density_map = create_perlin_noise_map((height, width), scale=random.uniform(80, 120))
    num_streaks = int((rainfall_rate_mm_hr / 100.0) * (width * height * 0.1))

    for _ in range(num_streaks):
        x1 = random.randint(0, width - 1)
        y1 = random.randint(0, height - 1)
        
        # Only draw a streak if the noise map value is high enough
        if density_map[y1, x1] < 0.4:
            continue

        # --- "Dark Core" Rendering for Contrast ---
        dx = x1 - focus_of_expansion[0]
        perspective_vx = dx * 0.01 * vehicle_speed_ms
        perceived_vx = wind_vx_pps + perspective_vx
        
        shutter_speed = 1/60
        streak_length = int(base_rain_vy_pps * shutter_speed * random.uniform(0.7, 1.5))
        if streak_length < 4: continue
            
        x2 = x1 + int(perceived_vx * shutter_speed)
        y2 = y1 + streak_length
        
        # 1. Darker, thicker, more transparent "body"
        body_opacity = random.randint(5, 20)
        body_color = (100, 100, 100, body_opacity)
        cv2.line(streaks_layer_rgba, (x1, y1), (x2, y2), body_color, 2)

        # 2. Brighter, thinner "highlight"
        highlight_opacity = random.randint(30, 80)
        highlight_color = (230, 230, 230, highlight_opacity)
        cv2.line(streaks_layer_rgba, (x1, y1), (x2, y2), highlight_color, 1)
        
    # Apply a final soft blur
    streaks_layer_rgba = cv2.GaussianBlur(streaks_layer_rgba, (3, 3), 0)

    return streaks_layer_rgba