# File: JarvisIR/degradation_synthesis/night_utils.py

import torch
import random
import numpy as np
from scipy import stats

class NightProcessor:
    def __init__(self, device):
        self.device = device
        self.config = {
            'gamma_range': [2.0, 3.5],
            'rgb_range': [0.8, 0.1],
            'red_range': [1.9, 2.4],
            'blue_range': [1.5, 1.9],
            # Slightly brighter range so it's not pitch black without noise
            'darkness_range': [0.08, 0.18], 
            'quantisation': [2, 4, 6]
        }
        
        self.xyz2cams = torch.tensor([[[1.0234, -0.2969, -0.2266],
                                  [-0.5625, 1.6328, -0.0469],
                                  [-0.0703, 0.2188, 0.6406]],
                                 [[0.4913, -0.0541, -0.0202],
                                  [-0.613, 1.3513, 0.2906],
                                  [-0.1564, 0.2151, 0.7183]],
                                 [[0.838, -0.263, -0.0639],
                                  [-0.2887, 1.0725, 0.2496],
                                  [-0.0627, 0.1427, 0.5438]],
                                 [[0.6596, -0.2079, -0.0562],
                                  [-0.4782, 1.3016, 0.1933],
                                  [-0.097, 0.1581, 0.5181]]], device=device)
        self.rgb2xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                                [0.2126729, 0.7151522, 0.0721750],
                                [0.0193339, 0.1191920, 0.9503041]], device=device)

    def apply_ccm(self, image, ccm):
        image = image.float()
        ccm = ccm.float()
        shape = image.shape
        image = image.reshape(-1, 3)
        image = torch.matmul(image, ccm.T)
        return image.view(shape)

    def degrade_batch(self, imgs):
        B, C, H, W = imgs.shape
        imgs = imgs.permute(0, 2, 3, 1)

        # Inverse tone mapping
        imgs = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * imgs) / 3.0)
        
        # Inverse gamma
        epsilon = torch.tensor([1e-8], device=self.device)
        gamma = torch.rand(B, 1, 1, 1, device=self.device) * (self.config['gamma_range'][1] - self.config['gamma_range'][0]) + self.config['gamma_range'][0]
        imgs = torch.max(imgs, epsilon) ** gamma

        # sRGB to cRGB
        idx = random.randint(0, self.xyz2cams.shape[0] - 1)
        xyz2cam = self.xyz2cams[idx]
        rgb2cam = torch.matmul(xyz2cam, self.rgb2xyz)
        rgb2cam = rgb2cam / rgb2cam.sum(dim=-1, keepdim=True)
        imgs = self.apply_ccm(imgs, rgb2cam)

        # Inverse white balance
        rgb_gain = torch.normal(mean=self.config['rgb_range'][0], std=self.config['rgb_range'][1], size=(B, 1, 1, 1), device=self.device)
        red_gain = torch.rand(B, 1, 1, 1, device=self.device) * (self.config['red_range'][1] - self.config['red_range'][0]) + self.config['red_range'][0]
        blue_gain = torch.rand(B, 1, 1, 1, device=self.device) * (self.config['blue_range'][1] - self.config['blue_range'][0]) + self.config['blue_range'][0]

        gains1 = torch.cat([1.0 / red_gain, torch.ones_like(red_gain), 1.0 / blue_gain], dim=-1) * rgb_gain
        imgs = imgs * gains1

        # Darkness (Clean darkening, NO NOISE ADDITION)
        lower, upper = self.config['darkness_range']
        mu, sigma = 0.06, 0.04
        darkness = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(size=B)
        darkness = torch.tensor(darkness, device=self.device).view(B, 1, 1, 1).float()
        imgs = imgs * darkness
        
        # --- DISABLED NOISE STEPS ---
        # shot_noise, read_noise = ...
        # quan_noise = ...
        
        # Forward White Balance
        gains2 = torch.cat([red_gain, torch.ones_like(red_gain), blue_gain], dim=-1)
        imgs = imgs * gains2

        # cRGB to sRGB
        cam2rgb = torch.inverse(rgb2cam)
        imgs = self.apply_ccm(imgs, cam2rgb)
        
        # Gamma Correction
        imgs = torch.max(imgs, epsilon) ** (1 / gamma)
        imgs = imgs.permute(0, 3, 1, 2)
        
        return torch.clamp(imgs, 0.0, 1.0)