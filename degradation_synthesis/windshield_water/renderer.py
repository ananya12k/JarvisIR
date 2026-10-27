"""
Optical rendering: apply WaterMaps to a frame.

Reuses the existing DropModel optical path unchanged (refraction by grid_sample,
Gaussian blur of the refracted image, alpha blending) - only the maps now come from
the persistent simulation instead of a fresh random draw per frame.
"""

import os
import sys

import torch

_SYN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SYN_ROOT not in sys.path:
    sys.path.insert(0, _SYN_ROOT)
from rainy.GuidedDisent.droprenderer import DropModel  # noqa: E402


class WaterRenderer:
    def __init__(self, height, width, device='cuda'):
        self.device = torch.device(device)
        self.dm = DropModel(imsize=(height, width))
        self.one = torch.ones(1, device=self.device)

    def apply_maps(self, frame, maps, sigma=3.5):
        """frame: 1 x 3 x H x W in [-1, 1]; maps: WaterMaps. Returns the degraded frame."""
        v, mask = maps.drop_model_inputs()
        sig = torch.full((1,), float(sigma), device=self.device)
        return self.dm.apply_drops(frame, v, mask, sigma=sig, transparency=self.one)
