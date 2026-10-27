"""
Droplet state.

    S_i(t) = (p_i, v_i, r_i, V_i, q_i, tau_i, alpha_i, theta_i) + shape

Positions are floats (sub-pixel motion); velocities in px / s; volume V = k r^3.
A droplet keeps its id for its whole life; growth and motion modify it in place.
Merging retires both parents and creates a child with a NEW id (see interactions.py).
"""

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class Droplet:
    drop_id: int
    x: float
    y: float
    volume: float
    k: float = 1.0                       # V = k r^3
    vx: float = 0.0                      # px / s
    vy: float = 0.0
    age: float = 0.0                     # s
    state: str = 'pinned'                # 'pinned' | 'sliding'
    opacity: float = 0.0                 # rises over fade-in
    stretch: float = 1.0                 # footprint a = r s, b = r / s
    dir_x: float = 0.0                   # smoothed motion direction (unit) -> orientation theta
    dir_y: float = 1.0
    slide_volume: float = 0.0            # V_slide for this drop
    pause_left: float = 0.0              # s of stick-slip pinning left
    wobble: float = 0.0                  # lateral OU state, px / s
    # boundary perturbation eps(phi, t) = sum_k a_k cos(k phi) + b_k sin(k phi), k = 2..K+1
    shape_a: np.ndarray = field(default_factory=lambda: np.zeros(0))
    shape_b: np.ndarray = field(default_factory=lambda: np.zeros(0))
    trail_id: int = -1                   # trail this drop is currently laying
    parents: Tuple[int, ...] = ()
    # drawn = physical + offset; the offset eases to 0 after a merge (no visual jump)
    rox: float = 0.0
    roy: float = 0.0
    ease_left_s: float = 0.0
    ease_total_s: float = 0.0

    @property
    def radius(self) -> float:
        return (max(self.volume, 0.0) / self.k) ** (1.0 / 3.0)

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def orientation(self) -> float:
        """theta = atan2(v_y, v_x) of the smoothed motion direction."""
        return math.atan2(self.dir_y, self.dir_x)

    @property
    def rx(self) -> float:
        return self.x + self.rox

    @property
    def ry(self) -> float:
        return self.y + self.roy


def radius_to_volume(r: float, k: float = 1.0) -> float:
    return k * r ** 3
