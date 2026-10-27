"""
Simulation state -> image-space maps.

Droplets are rasterised as spherical-cap height patches in local bounding boxes
(batched per patch-size bucket on the GPU, composited with a per-pixel max - never
averaged). Footprint: ellipse a = r s (along motion), b = r / s (across), rotated to
theta = atan2(v_y, v_x), trailing edge tapered, boundary r(phi, t) = r0 [1 + eps(phi, t)].

All water layers share one height field h(x, y):
    n = (-dh/dx, -dh/dy, 1) / sqrt(dh/dx^2 + dh/dy^2 + 1)
    offset (px) = -k n_xy                         I_warp(p) = I(p + offset)
Masks / transparency are separate maps; they are not interchangeable with normals
or offsets.
"""

import math
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class WaterMaps:
    height: torch.Tensor                 # H x W, px
    normal: torch.Tensor                 # 3 x H x W, unit normals
    offset_px: torch.Tensor              # 2 x H x W, (dy, dx) sampling offsets in px
    alpha: torch.Tensor                  # H x W, blend weight (transparency) in [0, 1]
    layers: Dict[str, torch.Tensor] = field(default_factory=dict)   # per-layer alpha (H x W)
    layer_heights: Dict[str, torch.Tensor] = field(default_factory=dict)   # per-layer height (H x W)

    def drop_model_inputs(self):
        """(v 1xHxWx3, mask 1xHxWx1) in DropModel.apply_drops' convention:
        refraction = -v[..., :2] * v[..., 2] in normalised [-1, 1] coordinates."""
        H, W = self.height.shape
        v = torch.stack([-self.offset_px[0] / (H / 2), -self.offset_px[1] / (W / 2),
                         torch.ones_like(self.height)], -1)[None]
        return v, self.alpha[None, :, :, None]


def droplet_layer(drops, H, W, cfg, device):
    """Height and alpha of all droplets (H x W each), batched patches + scatter-max."""
    h_map = torch.zeros(H * W, device=device)
    a_map = torch.zeros(H * W, device=device)
    if not drops:
        return h_map.view(H, W), a_map.view(H, W)
    sh, rc = cfg.shape, cfg.render
    f32 = dict(device=device, dtype=torch.float32)
    K = sh.n_harmonics
    r = torch.tensor([d.radius for d in drops], **f32)
    s = torch.tensor([max(d.stretch, 1.0) for d in drops], **f32)
    ux = torch.tensor([d.dir_x for d in drops], **f32)
    uy = torch.tensor([d.dir_y for d in drops], **f32)
    xs = torch.tensor([d.rx for d in drops], **f32)
    ys = torch.tensor([d.ry for d in drops], **f32)
    op = torch.tensor([d.opacity for d in drops], **f32)
    ca = torch.tensor(np.array([d.shape_a for d in drops]).reshape(len(drops), K), **f32)
    cb = torch.tensor(np.array([d.shape_b for d in drops]).reshape(len(drops), K), **f32)
    kk = torch.arange(2, K + 2, **f32)

    eps_max = (ca.abs() + cb.abs()).sum(1)
    ext = r * s * (1 + eps_max) * 1.2 + 3
    need = (2 * torch.ceil(ext) + 1).clamp(max=rc.max_patch)
    buckets = [b for b in rc.patch_buckets if b < rc.max_patch] + [rc.max_patch]
    flats, hs, als = [], [], []
    lower = 0
    for P in buckets:
        sel = torch.nonzero((need > lower) & (need <= P)).flatten()
        lower = P
        if sel.numel() == 0:
            continue
        half = P // 2
        off = torch.arange(P, **f32) - half
        cx, cy = xs[sel].round(), ys[sel].round()
        dx = off.view(1, 1, P) - (xs[sel] - cx).view(-1, 1, 1)          # sub-pixel centre
        dy = off.view(1, P, 1) - (ys[sel] - cy).view(-1, 1, 1)
        u_, v_ = ux[sel].view(-1, 1, 1), uy[sel].view(-1, 1, 1)
        along = dx * u_ + dy * v_
        across = -dx * v_ + dy * u_
        rr, ss = r[sel].view(-1, 1, 1), s[sel].view(-1, 1, 1)
        a_ax, b_ax = rr * ss, rr / ss                                    # a = r s, b = r / s
        un = along / a_ax
        taper = 1 + sh.taper * (ss - 1) * (-un).clamp(0, 1)              # narrower trailing edge
        wn = across / (b_ax / taper)
        phi = torch.atan2(wn, un)
        eps = (ca[sel].view(-1, K, 1, 1) * torch.cos(kk.view(1, K, 1, 1) * phi[:, None])
               + cb[sel].view(-1, K, 1, 1) * torch.sin(kk.view(1, K, 1, 1) * phi[:, None])).sum(1)
        rho = torch.sqrt(un ** 2 + wn ** 2) / (1 + eps).clamp(min=0.3)
        hgt = sh.cap_height * rr * torch.sqrt((1 - rho ** 2).clamp(min=0))       # spherical cap
        edge = (1 - rho) * torch.minimum(a_ax, b_ax) / rc.edge_px                 # AA edge in px
        alp = edge.clamp(0, 1) * op[sel].view(-1, 1, 1)
        hgt = hgt * op[sel].view(-1, 1, 1)
        ar = torch.arange(P, device=device)
        gy = (cy.long() - half).view(-1, 1, 1) + ar.view(1, P, 1)
        gx = (cx.long() - half).view(-1, 1, 1) + ar.view(1, 1, P)
        valid = (alp > 0) & (gy >= 0) & (gy < H) & (gx >= 0) & (gx < W)
        flats.append((gy * W + gx).expand_as(alp)[valid])
        hs.append(hgt[valid])
        als.append(alp[valid])
    if flats:
        flat = torch.cat(flats)
        h_map.scatter_reduce_(0, flat, torch.cat(hs), reduce='amax')
        a_map.scatter_reduce_(0, flat, torch.cat(als), reduce='amax')
    return h_map.view(H, W), a_map.view(H, W)


def height_to_maps(height, alpha, cfg, layers=None) -> WaterMaps:
    """Normals from the height field, then sampling offsets (px)."""
    hx = torch.zeros_like(height)
    hy = torch.zeros_like(height)
    hx[:, 1:-1] = (height[:, 2:] - height[:, :-2]) / 2
    hy[1:-1] = (height[2:] - height[:-2]) / 2
    norm = torch.sqrt(hx ** 2 + hy ** 2 + 1)
    n = torch.stack([-hx / norm, -hy / norm, 1 / norm])                  # (n_x, n_y, n_z)
    k = cfg.render.refraction_px
    offset = torch.stack([-k * n[1], -k * n[0]])                         # (dy, dx) = -k (n_y, n_x)
    return WaterMaps(height=height, normal=n, offset_px=offset, alpha=alpha.clamp(0, 1), layers=layers or {})
