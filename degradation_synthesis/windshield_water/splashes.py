"""
Splash events: short-lived impacts that spread, finger, shed satellites, relax and
leave water behind.

    R(t) = R0 + (Rmax - R0) (1 - exp(-t / tau))
    boundary  R(phi, t) = R(t) [1 + relax(t) (eps(phi) + fingers(phi))],  relax = exp(-t / t_relax)

Impacts are rate-limited (Poisson) with a minimum spacing / refractory interval.
When an event ends, its water is transferred to a residual droplet and the film.
"""

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class SplashEvent:
    x: float
    y: float
    water_amount: float
    age: float
    duration: float
    initial_radius: float
    strength: float
    max_radius: float = 0.0
    harm_a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    harm_b: np.ndarray = field(default_factory=lambda: np.zeros(3))
    finger_angles: np.ndarray = field(default_factory=lambda: np.zeros(0))
    finger_widths: np.ndarray = field(default_factory=lambda: np.zeros(0))
    satellites_done: bool = False

    def radius(self, tau):
        return self.initial_radius + (self.max_radius - self.initial_radius) * (1 - math.exp(-self.age / tau))

    def boundary(self, phi, tau, relax_s, finger_amp):
        """R(phi, t) for an array of angles phi."""
        relax = math.exp(-self.age / relax_s)
        eps = sum(self.harm_a[k] * np.cos((k + 2) * phi) + self.harm_b[k] * np.sin((k + 2) * phi)
                  for k in range(len(self.harm_a)))
        fing = np.zeros_like(phi)
        for ang, wid in zip(self.finger_angles, self.finger_widths):
            d = np.angle(np.exp(1j * (phi - ang)))
            fing = fing + finger_amp * np.exp(-(d / wid) ** 2)
        return self.radius(tau) * (1 + relax * (eps + fing))

    @property
    def fade(self):
        """Opacity / thickness over the lifetime: fast rise, slow fade."""
        x = self.age / self.duration
        return min(1.0, x / 0.05) * max(0.0, 1.0 - x) ** 0.7


class SplashSystem:
    def __init__(self, cfg):
        self.cfg = cfg
        self.events: List[SplashEvent] = []
        self.recent = []                     # (x, y, time) for the refractory interval
        self.t = 0.0

    def spawn(self, rng, W, H, dt, rain_sampler=None):
        sc = self.cfg.splash
        if not sc.enabled:
            return 0
        n = int(rng.poisson(sc.rate_per_s * dt))
        made = 0
        self.recent = [r for r in self.recent if self.t - r[2] < sc.refractory_s]
        for _ in range(n):
            x, y = rain_sampler(rng) if rain_sampler else (rng.uniform(0, W), rng.uniform(0, H))
            if any(math.hypot(x - ex.x, y - ex.y) < sc.min_spacing_px for ex in self.events) or \
               any(math.hypot(x - rx, y - ry) < sc.min_spacing_px for rx, ry, _ in self.recent):
                continue                     # minimum spacing / refractory interval
            rmax = float(rng.uniform(*sc.r_max_px))
            nf = int(rng.integers(sc.fingers[0], sc.fingers[1] + 1))
            ev = SplashEvent(
                x=x, y=y, water_amount=sc.water_per_px2 * rmax ** 2, age=0.0,
                duration=sc.duration_s * float(rng.uniform(0.8, 1.2)),
                initial_radius=sc.r0_frac * rmax, strength=1.0, max_radius=rmax,
                harm_a=rng.normal(0, 0.08, 3), harm_b=rng.normal(0, 0.08, 3),
                finger_angles=rng.uniform(-math.pi, math.pi, nf),
                finger_widths=rng.uniform(0.08, 0.2, nf))
            self.events.append(ev)
            self.recent.append((x, y, self.t))
            made += 1
        return made

    def update(self, dt, rng, make_drop, film=None):
        """Age events; shed satellites once the rim has formed; at the end transfer the
        remaining water to a residual droplet and the film. make_drop(x, y, volume)."""
        sc = self.cfg.splash
        self.t += dt
        keep = []
        for ev in self.events:
            ev.age += dt
            if not ev.satellites_done and ev.age >= 1.5 * sc.tau_s:
                ev.satellites_done = True
                for _ in range(int(rng.poisson(sc.satellites_mean))):
                    ang = (rng.choice(ev.finger_angles) if len(ev.finger_angles) else rng.uniform(-math.pi, math.pi))
                    dist = ev.radius(sc.tau_s) * (1 + sc.finger_amp) * float(rng.uniform(1.05, 1.4))
                    r = float(rng.uniform(*sc.satellite_r_px))
                    make_drop(ev.x + dist * math.cos(ang), ev.y + dist * math.sin(ang), r ** 3)
                    ev.water_amount = max(0.0, ev.water_amount - r ** 3)
            if ev.age >= ev.duration:
                if ev.water_amount > 0:
                    make_drop(ev.x, ev.y, sc.residual_drop_frac * ev.water_amount)
                    if film is not None:
                        film.add(ev.x, ev.y, sc.residual_film_frac * min(1.0, ev.water_amount / 200.0),
                                 ev.max_radius)
                continue
            keep.append(ev)
        self.events = keep

    def rasterize(self, H, W) -> np.ndarray:
        """Thin, irregular, fading water sheet per event (height field)."""
        sc = self.cfg.splash
        h = np.zeros((H, W), np.float32)
        for ev in self.events:
            R = ev.max_radius * (1 + sc.finger_amp) * 1.3
            x0, x1 = int(max(ev.x - R, 0)), int(min(ev.x + R + 1, W))
            y0, y1 = int(max(ev.y - R, 0)), int(min(ev.y + R + 1, H))
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            dx, dy = xx - ev.x, yy - ev.y
            phi = np.arctan2(dy, dx)
            rho = np.hypot(dx, dy) / np.maximum(ev.boundary(phi, sc.tau_s, sc.relax_s, sc.finger_amp), 1e-3)
            prof = np.clip(1 - rho ** 4, 0, 1) * (0.6 + 0.4 * rho)       # thicker rim than centre
            hv = (sc.height * ev.strength * ev.fade * prof).astype(np.float32)
            h[y0:y1, x0:x1] = np.maximum(h[y0:y1, x0:x1], hv)
        return h
