"""
Persistent trails and rivulets.

Trails are objects of their own (they outlive the drop that laid them):
    T_j = {p_1, ..., p_n}, each point with width w and strength D,
    D(t + dt) = D(t) exp(-mu dt).
A sliding drop appends a point every `point_spacing_px` of travel and loses the
water it lays down (volume accounting).

Rivulets live on a coarse deposition field F(x, y, t):
    F_{t+dt} = max(0, F_t - mu dt) + S_t
with local spreading (small blur, never global), downward transport, and
connected components above a threshold. Sliding drops on a rivulet transfer
water into it continuously.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np


@dataclass
class Trail:
    trail_id: int
    parent_id: int
    points: List[List[float]] = field(default_factory=list)   # [x, y, width, strength]
    since_last: float = 0.0                                     # px travelled since last point


class TrailSystem:
    def __init__(self, cfg):
        self.cfg = cfg
        self.trails: Dict[int, Trail] = {}
        self._next = 0
        self.deposited: List[tuple] = []        # this step's (x, y, water) -> rivulet field

    def start(self, parent_id: int, x: float, y: float, width: float) -> int:
        t = Trail(self._next, parent_id, [[x, y, width, 1.0]])
        self.trails[t.trail_id] = t
        self._next += 1
        return t.trail_id

    def deposit(self, drop, x0, y0, dt) -> float:
        """Drop moved from (x0, y0) to (drop.x, drop.y): append points every spacing px.
        Returns the volume the drop loses into the trail."""
        tc = self.cfg.trails
        if not tc.enabled:
            return 0.0
        step = math.hypot(drop.x - x0, drop.y - y0)
        if step <= 1e-6:
            return 0.0
        r = drop.radius
        # wider behind larger drops; faster drops deposit less -> thinner
        width = tc.width_frac * r * (tc.width_speed_ref_px_s / (tc.width_speed_ref_px_s + drop.speed)) ** 0.3
        if drop.trail_id not in self.trails:
            drop.trail_id = self.start(drop.drop_id, x0, y0, width)
        tr = self.trails[drop.trail_id]
        tr.since_last += step
        while tr.since_last >= tc.point_spacing_px:
            tr.since_last -= tc.point_spacing_px
            f = 1.0 - tr.since_last / step if step > 0 else 1.0
            px, py = x0 + (drop.x - x0) * min(max(f, 0), 1), y0 + (drop.y - y0) * min(max(f, 0), 1)
            tr.points.append([px, py, width, 1.0])
        lost = min(tc.deposit_per_px * r * step, 0.5 * drop.volume)
        self.deposited.append((drop.x, drop.y, lost))
        return lost

    def update(self, dt):
        """Decay every trail by its own lifecycle; drop faded points / empty trails."""
        tc = self.cfg.trails
        k = math.exp(-tc.decay_per_s * dt)
        dead = []
        for tid, tr in self.trails.items():
            for p in tr.points:
                p[3] *= k
            tr.points = [p for p in tr.points if p[3] >= tc.min_strength]
            if not tr.points:
                dead.append(tid)
        for tid in dead:
            del self.trails[tid]

    def rasterize(self, H, W, levels=16) -> np.ndarray:
        """Trail height field (H x W, float32). Segments are grouped by quantised
        strength and width and drawn with one cv2.polylines call per group, weakest
        first (stronger water drawn on top, a max-like composite)."""
        tc = self.cfg.trails
        h = np.zeros((H, W), np.float32)
        groups = {}
        for tr in self.trails.values():
            p = np.asarray(tr.points, np.float32)
            if len(p) < 2:
                continue
            strength = np.minimum(p[:-1, 3], p[1:, 3])
            width = np.maximum(1, np.round(0.5 * (p[:-1, 2] + p[1:, 2]))).astype(int)
            q = np.minimum((strength * levels).astype(int), levels - 1)
            seg = np.stack([p[:-1, :2], p[1:, :2]], 1).round().astype(np.int32)    # n x 2 x 2
            for key in set(zip(q.tolist(), width.tolist())):
                m = (q == key[0]) & (width == key[1])
                groups.setdefault(key, []).extend(list(seg[m]))
        for (qi, wi) in sorted(groups):
            cv2.polylines(h, groups[(qi, wi)], False, float((qi + 0.5) / levels * tc.height),
                          thickness=int(wi), lineType=cv2.LINE_AA)
        return cv2.GaussianBlur(h, (0, 0), 1.0) if groups else h

    def total_points(self):
        return sum(len(t.points) for t in self.trails.values())


class DepositionField:
    """Rivulets: F_{t+dt} = max(0, F_t - mu dt) + S_t, local spreading, downward flow."""

    def __init__(self, cfg, H, W):
        self.cfg = cfg
        rc = cfg.rivulets
        self.g = rc.grid_px
        self.gh, self.gw = math.ceil(H / self.g), math.ceil(W / self.g)
        self.F = np.zeros((self.gh, self.gw), np.float32)
        self._gy, self._gx = np.mgrid[0:self.gh, 0:self.gw].astype(np.float32)
        self.absorbed = 0.0

    def cell(self, x, y):
        return min(max(int(y / self.g), 0), self.gh - 1), min(max(int(x / self.g), 0), self.gw - 1)

    def level_at(self, x, y) -> float:
        return float(self.F[self.cell(x, y)])

    def add(self, x, y, water, radius_px=None):
        rc = self.cfg.rivulets
        cy, cx = self.cell(x, y)
        if radius_px is None or radius_px <= self.g:
            self.F[cy, cx] += water * rc.deposit_gain
            return
        rr = max(1, int(round(radius_px / self.g)))
        splat = np.zeros_like(self.F)
        cv2.circle(splat, (cx, cy), rr, 1.0, -1)
        self.F += splat * (water * rc.deposit_gain / max(splat.sum(), 1e-6))

    def update(self, dt, deposits, drops, rng) -> int:
        """Advance one step. deposits: trail water this step. Sliding drops sitting on a
        rivulet transfer water into it (dV = -absorb * V * dt). Returns #drops absorbed."""
        rc = self.cfg.rivulets
        S = np.zeros_like(self.F)
        for (x, y, w) in deposits:
            cy, cx = self.cell(x, y)
            S[cy, cx] += w * rc.deposit_gain
        self.F = np.maximum(0.0, self.F - rc.decay_per_s * dt) + S
        # drops reaching a rivulet feed it (continuous transfer: no size pop)
        absorbed = 0
        for d in list(drops.values()):
            if d.state != 'sliding':
                continue
            cy, cx = self.cell(d.x, d.y + d.radius)
            if self.F[cy, cx] > rc.threshold:
                dv = min(d.volume, rc.absorb_per_s * d.volume * dt)
                d.volume -= dv
                self.add(d.x, d.y, dv, radius_px=d.radius)
                absorbed += 1
        # local spreading only (blend with a small blur), downward transport (thicker = faster)
        a = 1.0 - math.exp(-rc.spread_per_s * dt)
        self.F = (1 - a) * self.F + a * cv2.GaussianBlur(self.F, (0, 0), rc.spread_sigma_cells)
        v = (rc.flow_px_s * dt / self.g) * np.clip(self.F / max(rc.threshold, 1e-6), 0.3, 1.5)
        self.F = cv2.remap(self.F, self._gx, self._gy - v.astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return absorbed

    def mask(self) -> np.ndarray:
        return self.F > self.cfg.rivulets.threshold

    def components(self) -> int:
        n, _, st, _ = cv2.connectedComponentsWithStats(self.mask().astype(np.uint8), connectivity=8)
        return int((st[1:, cv2.CC_STAT_AREA] >= self.cfg.rivulets.min_component_cells).sum())

    def rasterize(self, H, W) -> np.ndarray:
        rc = self.cfg.rivulets
        lvl = np.clip((self.F - 0.5 * rc.threshold) / (rc.threshold + 1e-6), 0, 1.5)
        h = cv2.resize(lvl.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        return np.clip(h, 0, None) * rc.height
