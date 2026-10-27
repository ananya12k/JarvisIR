"""
Measurements for the windshield-water simulator.

The water itself is measured from the simulation state and maps (not from rendered
pixels), because real background motion also changes pixels: a temporal artefact
shows up as an unexplained change of the water MASK, a jump of a tracked drop, or a
volume error at a merge.
"""

import json
from typing import Dict, List

import numpy as np
import torch


def mask_change(prev_alpha: torch.Tensor, alpha: torch.Tensor, thr=0.05) -> float:
    """1 - IoU of consecutive water masks (0 = identical, 1 = completely different)."""
    a, b = prev_alpha > thr, alpha > thr
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return 1.0 - inter / union if union else 0.0


class MetricsRecorder:
    """Collects the simulator's per-frame stats plus temporal mask change and timing."""

    def __init__(self, fps: float):
        self.fps = fps
        self.rows: List[Dict] = []
        self._prev_alpha = None

    def record(self, sim, maps=None, step_ms=None, **extra):
        row = dict(frame=sim.frame, **sim.stats, **extra)
        if step_ms is not None:
            row['step_ms'] = step_ms
        if maps is not None:
            if self._prev_alpha is not None:
                row['mask_change'] = mask_change(self._prev_alpha, maps.alpha)
            self._prev_alpha = maps.alpha
        if torch.cuda.is_available():
            row['gpu_mem_gb'] = torch.cuda.max_memory_allocated() / 1e9
        self.rows.append(row)
        return row

    def summary(self) -> Dict:
        if not self.rows:
            return {}
        keys = sorted({k for r in self.rows for k in r if isinstance(r[k], (int, float)) and k != 'frame'})
        out = {'frames': len(self.rows)}
        for k in keys:
            v = np.array([r[k] for r in self.rows if k in r], float)
            if k.startswith('max_') or k == 'gpu_mem_gb':
                out[f'{k}'] = float(v.max())
            else:
                out[f'{k}_mean'] = float(v.mean())
        n = len(self.rows)
        for k in ('arrivals', 'merges', 'impacts', 'splashes_new'):
            if any(k in r for r in self.rows):
                out[f'{k}_per_s'] = sum(r.get(k, 0) for r in self.rows) * self.fps / n
        return out

    def save(self, path_json, path_csv=None):
        with open(path_json, 'w') as f:
            json.dump(self.summary(), f, indent=2)
        if path_csv:
            keys = sorted({k for r in self.rows for k in r})
            with open(path_csv, 'w') as f:
                f.write(','.join(keys) + '\n')
                for r in self.rows:
                    f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')


def save_layer_maps(sim, out_dir, tag='', maps=None):
    """Write each layer's height and alpha, plus the composite height, alpha and offset
    magnitude, as PNGs (shared height scale), to tell simulation from rendering issues."""
    import os
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    layers = sim.layer_maps()
    maps = maps or sim.render_maps()
    hmax = max([float(h.max()) for h, _ in layers.values()] + [float(maps.height.max()), 1e-6])

    def png(name, t, scale):
        img = (np.clip(t.detach().float().cpu().numpy() / scale, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f'{tag}{name}.png'), cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS))
    for k, (h, a) in layers.items():
        png(f'{k}_height', h, hmax)
        png(f'{k}_alpha', a, 1.0)
    png('composite_height', maps.height, hmax)
    png('composite_alpha', maps.alpha, 1.0)
    png('composite_offset', maps.offset_px.norm(dim=0), sim.cfg.render.refraction_px)
    return {k: (float(h.max()), float((a > 0.05).float().mean())) for k, (h, a) in layers.items()}
