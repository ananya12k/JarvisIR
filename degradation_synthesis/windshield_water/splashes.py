"""
Splash events, triggered by real rain impacts (plus optional ambient events).

    trigger(x, y, impact_volume, impact_speed, water_budget)

The impact sets the event: strength s = min(1, (V / V_ref)(u / u_ref)^2) (a kinetic-
energy proxy), size R_max ~ V^(1/3) (u / u_ref)^(1/2), duration ~ sqrt(R_max / R_ref),
number of fingers / satellites ~ s, and a water budget taken from the arriving drop.

Stages (all from the event's age; the boundary is drawn once at creation, never
re-randomised):
  impact      compact irregular wet patch           R ~ R0, fingers just forming
  expansion   R(t) = R0 + (Rmax - R0)(1 - exp(-t / tau)), uneven rim, fingers grow
  relaxation  irregularity and fingers decay with exp(-t / t_relax)
  transfer    the sheet recedes and fades; the remaining budget becomes a residual
              droplet + film when the event ends
Water is conserved: budget = satellites + residual droplet + film share (ledger).
Height scales with the event size (capped); alpha is a separate map.
"""

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class SplashEvent:
    x: float
    y: float
    water_amount: float                 # remaining budget (volume units)
    age: float
    duration: float
    initial_radius: float
    strength: float
    max_radius: float = 0.0
    budget: float = 0.0                 # initial budget (for the ledger)
    harm_a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    harm_b: np.ndarray = field(default_factory=lambda: np.zeros(3))
    finger_angles: np.ndarray = field(default_factory=lambda: np.zeros(0))
    finger_widths: np.ndarray = field(default_factory=lambda: np.zeros(0))
    satellites_done: bool = False
    source: str = 'impact'              # 'impact' | 'ambient'

    def radius(self, tau):
        r = self.initial_radius + (self.max_radius - self.initial_radius) * (1 - math.exp(-self.age / tau))
        # transfer stage (last 30 %): the sheet recedes towards the centre, where the
        # residual droplet forms, instead of vanishing at full size
        x = min(self.age / self.duration, 1.0)
        z = min(max((x - 0.7) / 0.3, 0.0), 1.0)
        return r * (1 - 0.6 * z * z * (3 - 2 * z))

    def irregularity(self, tau, relax_s):
        """Envelope of rim unevenness / fingers: grows during expansion, then relaxes."""
        return (1 - math.exp(-self.age / (0.5 * tau))) * math.exp(-self.age / relax_s)

    def boundary(self, phi, tau, relax_s, finger_amp):
        """R(phi, t) for an array of angles phi (fixed shape, age-dependent envelope)."""
        env = self.irregularity(tau, relax_s)
        eps = sum(self.harm_a[k] * np.cos((k + 2) * phi) + self.harm_b[k] * np.sin((k + 2) * phi)
                  for k in range(len(self.harm_a)))
        fing = np.zeros_like(phi)
        for ang, wid in zip(self.finger_angles, self.finger_widths):
            d = np.angle(np.exp(1j * (phi - ang)))
            fing = fing + finger_amp * self.strength * np.exp(-(d / wid) ** 2)
        return self.radius(tau) * (1 + env * (eps + fing))

    def fade(self, dt):
        """0 -> 1 over two frames (no pop), then a smooth fade that is ~0 well before the
        event is removed (no sudden disappearance)."""
        rise = min(1.0, self.age / max(2 * dt, 1e-6))
        x = min(self.age / self.duration, 1.0)
        return rise * (1.0 - x) ** 2


class SplashSystem:
    def __init__(self, cfg):
        self.cfg = cfg
        self.events: List[SplashEvent] = []
        self.recent = []                     # (x, y, time) for the refractory interval
        self.t = 0.0
        self.finished = []                   # (duration, mean area) of events that ended this step
        self.ledger = {'budget': 0.0, 'to_drops': 0.0, 'to_film': 0.0}

    # ---------------------------------------------------------------- creation
    def _blocked(self, x, y):
        sc = self.cfg.splash
        return any(math.hypot(x - e.x, y - e.y) < sc.min_spacing_px for e in self.events) or \
            any(math.hypot(x - rx, y - ry) < sc.min_spacing_px for rx, ry, _ in self.recent)

    def trigger(self, rng, x, y, impact_volume, impact_speed, water_budget, source='impact'):
        """Create an event from an impact. Returns the event, or None when the minimum
        spacing / refractory interval blocks it (the caller then keeps the water)."""
        sc, k = self.cfg.splash, self.cfg.growth.volume_k
        if not sc.enabled or self._blocked(x, y):
            return None
        r_imp = (impact_volume / k) ** (1 / 3)
        u = impact_speed / sc.impact_speed_ref
        strength = min(1.0, (r_imp / sc.impact_radius_ref) ** 3 * u ** 2)
        rmax = float(np.clip(sc.radius_per_drop_radius * r_imp * math.sqrt(u), *sc.r_max_px))
        nf = int(round(sc.fingers[0] + strength * (sc.fingers[1] - sc.fingers[0])))
        ev = SplashEvent(
            x=x, y=y, water_amount=water_budget, budget=water_budget, age=0.0,
            duration=sc.duration_s * math.sqrt(rmax / sc.radius_ref_px) * float(rng.uniform(0.85, 1.15)),
            initial_radius=sc.r0_frac * rmax, strength=strength, max_radius=rmax,
            harm_a=rng.normal(0, 0.08, 3), harm_b=rng.normal(0, 0.08, 3),
            finger_angles=rng.uniform(-math.pi, math.pi, nf),
            finger_widths=rng.uniform(0.08, 0.2, nf), source=source)
        self.events.append(ev)
        self.recent.append((x, y, self.t))
        self.ledger['budget'] += water_budget
        return ev

    def spawn_ambient(self, rng, W, H, dt, rain_sampler=None):
        """Optional independent events (default rate 0): only if a model calls for them."""
        sc = self.cfg.splash
        if not sc.enabled or sc.ambient_rate_per_s <= 0:
            return 0
        made = 0
        for _ in range(int(rng.poisson(sc.ambient_rate_per_s * dt))):
            x, y = rain_sampler(rng) if rain_sampler else (rng.uniform(0, W), rng.uniform(0, H))
            r = float(rng.uniform(4, 9))
            if self.trigger(rng, x, y, self.cfg.growth.volume_k * r ** 3, sc.impact_speed_ref,
                            sc.ambient_water_per_px2 * r ** 2, source='ambient'):
                made += 1
        return made

    # ---------------------------------------------------------------- lifecycle
    def update(self, dt, rng, make_drop, film=None):
        """Age events; shed satellites once the rim has formed; at the end transfer the
        remaining budget to a residual droplet and the film. make_drop(x, y, volume)."""
        sc = self.cfg.splash
        self.t += dt
        self.recent = [r for r in self.recent if self.t - r[2] < sc.refractory_s]
        self.finished = []
        keep = []
        for ev in self.events:
            ev.age += dt
            if not ev.satellites_done and ev.age >= 1.5 * sc.tau_s:
                ev.satellites_done = True
                for _ in range(int(rng.poisson(sc.satellites_mean * ev.strength))):
                    r = float(rng.uniform(*sc.satellite_r_px))
                    v = self.cfg.growth.volume_k * r ** 3
                    if v > ev.water_amount:
                        break                                  # never create water
                    ang = rng.choice(ev.finger_angles) if len(ev.finger_angles) else rng.uniform(-math.pi, math.pi)
                    dist = ev.radius(sc.tau_s) * (1 + sc.finger_amp) * float(rng.uniform(1.05, 1.4))
                    make_drop(ev.x + dist * math.cos(ang), ev.y + dist * math.sin(ang), v)
                    ev.water_amount -= v
                    self.ledger['to_drops'] += v
            if ev.age >= ev.duration:
                to_drop = sc.residual_drop_frac * ev.water_amount
                to_film = ev.water_amount - to_drop
                r_res = (to_drop / self.cfg.growth.volume_k) ** (1 / 3)
                if r_res >= self.cfg.growth.min_radius_px:
                    make_drop(ev.x, ev.y, to_drop)
                    self.ledger['to_drops'] += to_drop
                else:
                    to_film += to_drop
                if film is not None and to_film > 0:
                    film.add_volume(ev.x, ev.y, to_film, ev.max_radius)
                self.ledger['to_film'] += to_film       # (counted even without film: evaporates)
                ev.water_amount = 0.0
                self.finished.append((ev.duration, math.pi * ev.max_radius ** 2))
                continue
            keep.append(ev)
        self.events = keep

    # ---------------------------------------------------------------- rendering
    def peak_height(self, ev):
        """Size-dependent surface height, capped (bigger splash = thicker sheet)."""
        sc = self.cfg.splash
        return sc.height * min((ev.max_radius / sc.radius_ref_px) ** sc.height_size_exp, sc.height_cap)

    def rasterize(self, H, W, dt):
        """(height, alpha) of all events, H x W float32 each; separate quantities."""
        sc = self.cfg.splash
        h = np.zeros((H, W), np.float32)
        a = np.zeros((H, W), np.float32)
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
            inside = np.clip(1 - rho ** 4, 0, 1)
            # crown: the rim stands higher than the thin centre, so both flanks of the
            # rim catch light (highlight) and are steep (dark meniscus) -> visible splash
            rim = 0.35 + sc.crown * np.clip(rho, 0, 1) ** 3
            f = ev.fade(dt)
            hv = (self.peak_height(ev) * f * inside * rim).astype(np.float32)
            # graded opacity (thin at the edge, like the height): as the event fades its
            # visible footprint shrinks gradually instead of the whole disc dropping out
            av = (sc.alpha * f * np.sqrt(inside) * rim).astype(np.float32)
            h[y0:y1, x0:x1] = np.maximum(h[y0:y1, x0:x1], hv)
            a[y0:y1, x0:x1] = np.maximum(a[y0:y1, x0:x1], av)
        return h, a

    def area_px(self):
        sc = self.cfg.splash
        return [math.pi * e.radius(sc.tau_s) ** 2 for e in self.events]
