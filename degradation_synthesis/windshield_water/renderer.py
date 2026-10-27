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

    def apply_maps(self, frame, maps, sigma=3.5, cfg=None):
        """frame: 1 x 3 x H x W in [-1, 1]; maps: WaterMaps. Returns the degraded frame:
        DropModel refraction + blur + blend, then shading (if cfg.render enables it)."""
        v, mask = maps.drop_model_inputs()
        sig = torch.full((1,), float(sigma), device=self.device)
        out = self.dm.apply_drops(frame, v, mask, sigma=sig, transparency=self.one)
        if cfg is not None:
            out = self.shade(out, maps, cfg.render)
        return out

    @staticmethod
    def shade(img, maps, rc):
        """Light on the water surface, weighted by water presence (dry glass untouched):
           sheen     = per-layer sky reflection (splash sheets, film)
           highlight = max(0, (n.h)^s - h_z^s)   Blinn-Phong, flat-glass baseline removed
           rim       = clip(gain (1 - n_z), 0, 1) steep edges -> dark meniscus"""
        if rc.specular <= 0 and rc.rim_dark <= 0:
            return img
        n = maps.normal                                          # (n_x, n_y, n_z)
        l = torch.tensor(rc.light_dir, device=n.device, dtype=n.dtype)
        l = l / l.norm()
        hv = l + torch.tensor([0.0, 0.0, 1.0], device=n.device, dtype=n.dtype)
        hv = hv / hv.norm()
        ndoth = (n * hv.view(3, 1, 1)).sum(0).clamp(min=0)
        spec = (ndoth ** rc.shininess - hv[2] ** rc.shininess).clamp(min=0)
        rim = (rc.rim_gain * (1 - n[2])).clamp(0, 1)
        a = maps.alpha
        for name, k in rc.sheen:                                 # wet sheets reflect the bright sky
            if k > 0 and name in maps.layers:
                img = img + (k * maps.layers[name])[None, None] * (1 - img)
        img = img - (rc.rim_dark * rim * a)[None, None] * (img + 1)        # darken towards black
        img = img + (rc.specular * spec * a)[None, None] * (1 - img)       # brighten towards white
        return img.clamp(-1, 1)
