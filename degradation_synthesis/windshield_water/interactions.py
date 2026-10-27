"""
Collision detection (spatial grid) and merging.

Contact test:     |p_i - p_j| <= c (r_i + r_j)      (c = contact_frac < 1: real overlap)
Volume:           V_new = V_i + V_j   ->  r_new = (r_i^3 + r_j^3)^(1/3)   (V = k r^3)
Position / vel.:  volume-weighted means of the parents

Lifecycle: candidate -> confirmed contact -> child with a NEW id -> both parents
retired (never drawn together with the child) -> trails stay separate objects.
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .droplet import Droplet


class SpatialGrid:
    """Uniform grid hash: candidate pairs only from the same / neighbouring cells."""

    def __init__(self, cell_px: float):
        self.cell = max(cell_px, 1.0)

    def candidate_pairs(self, drops: Dict[int, Droplet]) -> List[Tuple[int, int]]:
        cells = defaultdict(list)
        for d in drops.values():
            cells[(int(d.x // self.cell), int(d.y // self.cell))].append(d.drop_id)
        pairs = []
        for (cx, cy), ids in cells.items():
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    other = cells.get((cx + dx, cy + dy))
                    if not other:
                        continue
                    for i in ids:
                        for j in other:
                            if i < j:
                                pairs.append((i, j))
        return pairs


def merge_droplets(drops: Dict[int, Droplet], cfg, new_id, rng) -> List[dict]:
    """Merge touching drops in place. `new_id()` returns a fresh id. Returns one event
    dict per merge with the volumes before / after (for the volume-error metric)."""
    if not cfg.merge.enabled or len(drops) < 2:
        return []
    r_max = max(d.radius for d in drops.values())
    grid = SpatialGrid(2 * r_max * cfg.merge.contact_frac + 1.0)
    contacts = []
    for i, j in grid.candidate_pairs(drops):
        a, b = drops[i], drops[j]
        dist = math.hypot(a.x - b.x, a.y - b.y)
        if dist <= cfg.merge.contact_frac * (a.radius + b.radius):
            contacts.append((dist, i, j))
    contacts.sort()                                   # closest pairs first
    events = []
    retired = set()
    for dist, i, j in contacts:
        if i in retired or j in retired:              # each drop merges at most once per step
            continue
        a, b = drops.pop(i), drops.pop(j)
        retired.update((i, j))
        big, small = (a, b) if a.volume >= b.volume else (b, a)
        v = a.volume + b.volume
        child = Droplet(
            drop_id=new_id(), k=big.k,
            x=(a.x * a.volume + b.x * b.volume) / v,
            y=(a.y * a.volume + b.y * b.volume) / v,
            volume=v,
            vx=(a.vx * a.volume + b.vx * b.volume) / v,
            vy=(a.vy * a.volume + b.vy * b.volume) / v,
            age=max(a.age, b.age),
            state='sliding' if (a.state == 'sliding' or b.state == 'sliding') else 'pinned',
            opacity=max(a.opacity, b.opacity),
            stretch=big.stretch, dir_x=big.dir_x, dir_y=big.dir_y,
            slide_volume=big.slide_volume,
            shape_a=big.shape_a.copy(), shape_b=big.shape_b.copy(),   # the bigger shape persists
            trail_id=-1,                                              # parents' trails stay as they are
            parents=(a.drop_id, b.drop_id),
        )
        if child.volume >= child.slide_volume:
            child.state = 'sliding'
        # drawn position starts where the bigger parent was drawn, eases to the new centre
        ease = cfg.merge.ease_s
        if ease > 0:
            child.rox, child.roy = big.rx - child.x, big.ry - child.y
            child.ease_left_s = child.ease_total_s = ease
        drops[child.drop_id] = child
        events.append({
            'parents': (a.drop_id, b.drop_id), 'child': child.drop_id,
            'volume_before': a.volume + b.volume, 'volume_after': child.volume,
            'shift_px': math.hypot(child.x - big.x, child.y - big.y),
        })
    return events


def volume_error(event: dict, eps: float = 1e-9) -> float:
    """e_V = |V_after - V_before| / (V_before + eps)."""
    return abs(event['volume_after'] - event['volume_before']) / (event['volume_before'] + eps)
