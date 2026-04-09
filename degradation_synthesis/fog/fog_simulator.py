import os
import cv2
import random
import numpy as np
from torch.utils import data as data
import torch
from scipy.linalg import orth
from torchvision import transforms
import matplotlib
from PIL import Image

# Import Hugging Face transformers
from transformers import pipeline

def uint2single(img):
    return np.float32(img/255.)

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def random_resize(img, scale_factor=1.):
    return cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

def save_image(tensor, path):
    transform = transforms.ToPILImage()
    image = transform(tensor)
    image.save(path)

def get_fog_levels(beta_range, A_range, color_p, color_range):
    """Generate fog parameters based on configuration"""
    # Generate transmission coefficient beta
    beta = np.random.rand() * (beta_range[1] - beta_range[0]) + beta_range[0]
    
    # Generate atmospheric light A
    A = np.random.rand() * (A_range[1] - A_range[0]) + A_range[0]
    
    # Add color shift
    if np.random.rand() < color_p:
        A_random = np.random.rand(3) * (color_range[1] - color_range[0]) + color_range[0]
        A = A + A_random
    
    return beta, A

class ImageProcessor:
    def __init__(self, degration_cfg):
        self.degration_cfg = degration_cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize depth estimation model
        model_name = degration_cfg.get('depth_model', 'depth-anything/Depth-Anything-V2-Large-hf')
        print(f"Loading depth estimation model: {model_name}")
        
        # Load model using pipeline
        self.pipe = pipeline(
            task="depth-estimation", 
            model=model_name,
            device=self   .device
        )
        print("Depth estimation model loaded successfully")
        
        # Set depth map colormap (for visualization)
        self.cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        
        # Set fog parameters
        self.beta_range = degration_cfg.get('beta_range', [0.3, 1.5])
        self.A_range = degration_cfg.get('A_range', [0.25, 1.0])
        self.color_p = degration_cfg.get('color_p', 1.0)
        self.color_range = degration_cfg.get('color_range', [-0.025, 0.025])
        
    def generate_depth(self, img):
        """
        Generate depth map using HuggingFace depth estimation model
        img: numpy array, shape (H, W, C), BGR format
        returns: numpy array, shape (H, W), range [0, 1]
        """
        # Convert BGR to RGB and create PIL image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Use pipeline for depth estimation
        depth = self.pipe(pil_img)["depth"]
        
        # Convert PIL image to numpy array
        depth_np = np.array(depth)
        
        # Normalize to [0, 1]
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-6)
        
        return depth_np
        
    def visualize_depth(self, depth):
        """
        将深度图可视化为彩色图像
        depth: numpy array, shape (H, W), range [0, 1]
        returns: numpy array, shape (H, W, 3), range [0, 1]
        """
        # 应用颜色映射
        colored_depth = self.cmap(depth)
        
        # 移除alpha通道
        colored_depth = colored_depth[..., :3]
        
        return colored_depth
        
    def add_Gaussian_noise(self, img, noise_level1=2, noise_level2=25):
        """Add Gaussian noise"""
        noise_level = random.randint(noise_level1, noise_level2)
        rnum = np.random.rand()
        if rnum > 0.6:   # Add colored Gaussian noise
            img += np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
        elif rnum < 0.4: # Add grayscale Gaussian noise
            img += np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
        else:            # Add correlated noise
            L = noise_level2/255.
            D = np.diag(np.random.rand(3))
            U = orth(np.random.rand(3,3))
            conv = np.dot(np.dot(np.transpose(U), D), U)
            img += np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
        return np.clip(img, 0.0, 1.0)

    def add_JPEG_noise(self, img):
        """Add JPEG compression noise"""
        quality_factor = random.randint(30, 95)
        img = (img.clip(0, 1) * 255.).round().astype(np.uint8)
        result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
        img = cv2.imdecode(encimg, 1)
        return img.astype(np.float32) / 255.

    def Fog_Degrading(self, imgs, img_meta=None):
        """
        Add fog effect to images
        imgs: numpy array or torch tensor, shape (B, C, H, W) or (C, H, W), range [0, 1]
        """
        if torch.is_tensor(imgs):
            imgs = imgs.cpu().numpy()
        if len(imgs.shape) == 3:
            imgs = imgs[None, ...]
            
        # Generate depth maps using depth estimation model
        depth_maps = []
        for i in range(imgs.shape[0]):
            img = imgs[i].transpose(1, 2, 0)  # CHW -> HWC
            
            # Convert float image to uint8 and RGB to BGR
            img_uint8 = (img * 255).astype(np.uint8)
            if img.shape[2] == 3:  # If RGB image
                img_uint8 = img_uint8[:, :, ::-1]  # RGB -> BGR
            
            # Generate depth map
            depth = self.generate_depth(img_uint8)
            depth_maps.append(depth)
            
        depth_maps = np.stack(depth_maps, axis=0)
        depth_maps = depth_maps[:, None, :, :]  # Add channel dimension
            
        # Normalize depth maps to [0, 1]
        depth_maps = (depth_maps - depth_maps.min()) / (depth_maps.max() - depth_maps.min() + 1e-6)
            
        results = []
        for i in range(imgs.shape[0]):
            img = imgs[i].transpose(1, 2, 0)  # CHW -> HWC
            depth = depth_maps[i, 0]  # Take first channel as depth map
            
            # Generate fog parameters
            beta, A = get_fog_levels(self.beta_range, self.A_range, self.color_p, self.color_range)
            
            # Calculate transmission
            t = np.exp(-(1- depth) * 2.0 * beta)
            t = t[..., None]  # Expand dimensions for broadcasting
            
            # Adjust brightness
            if np.random.rand() < 0.5:
                img = np.power(img, np.random.rand(1) * 3.5 + 3.5)

            # Add Gaussian noise
            if np.random.rand() < 0.5:
                img = self.add_Gaussian_noise(img)
            
            # Add fog effect
            img_fog = img * t + A * (1 - t)
            
            # Add JPEG compression noise
            if np.random.rand() < 0.5:
                img_fog = self.add_JPEG_noise(img_fog)
                
            # Convert back to CHW format
            img_fog = img_fog.transpose(2, 0, 1)
            results.append(img_fog)
            
        results = np.stack(results, axis=0)
        return torch.from_numpy(results).float()
