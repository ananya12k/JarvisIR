import numpy as np
import cv2

class FogSystem:
    def __init__(self, beta=0.1, A=0.92):
        self.beta = beta
        self.A = A
        self.drift = 0

    def apply_fog(self, frame_bgr, depth_map_meters, wind_v=None):
        h, w = frame_bgr.shape[:2]
        img_float = frame_bgr.astype(np.float32) / 255.0
        
        # 1. Physics-based Transmission (t = e^-beta*d)
        transmission = np.exp(-self.beta * depth_map_meters)
        transmission = np.clip(transmission, 0.0, 1.0)
        
        # 2. CREATE SOFT MIST CLOUDS (Low Frequency)
        # We create noise at a TINY resolution and scale it up to make it look like "puffy clouds"
        noise_small = np.random.normal(0.5, 0.2, (8, 12)).astype(np.float32)
        if wind_v:
            self.drift = (self.drift + 0.05) % 12
            noise_small = np.roll(noise_small, int(self.drift), axis=1)
        
        # Gaussian blur the tiny noise to make it super smooth mist
        mist_map = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_CUBIC)
        mist_map = cv2.GaussianBlur(mist_map, (101, 101), 0)
        mist_map = np.clip(mist_map, 0.4, 1.0) # Ensure it's not too "patchy"
        
        # 3. DEPTH-BASED BLURRING (The "Scattering" Effect)
        # Real fog makes things further away look "out of focus"
        # We create a blurred version of the whole image
        blurred_img = cv2.GaussianBlur(img_float, (15, 15), 0)
        
        # Blend between sharp and blurred based on distance
        # Things far away (low transmission) get more blur
        t_mask = np.stack([transmission] * 3, axis=-1)
        img_scattered = img_float * t_mask + blurred_img * (1 - t_mask)

        # 4. FINAL COMPOSITION
        # Atmospheric Light Color (Slightly blue-grey, not pure white)
        # This makes it look like real air moisture
        A_color = np.array([0.90, 0.92, 0.94]).reshape(1, 1, 3) # BGR
        
        # The 'A' value is modulated by our soft mist map
        A_final = A_color * self.A * np.stack([mist_map]*3, axis=-1)
        
        # Standard Fog Equation: I = J*t + A*(1-t)
        foggy = img_scattered * t_mask + A_final * (1 - t_mask)
        
        # 5. POST-PROCESS: Lower contrast slightly
        # This prevents that "pasted on" look
        foggy = foggy * 0.9 + 0.05 
        
        return np.clip(foggy * 255.0, 0, 255).astype(np.uint8)

def apply_color_cast(img):
    """Deepen the fog look by reducing contrast slightly"""
    # Optional: make it look more 'damp' and grey
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:,:,0] *= 0.95 # Darken L channel slightly
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)