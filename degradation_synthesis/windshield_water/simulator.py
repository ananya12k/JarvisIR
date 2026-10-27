"""
Windshield-water simulator: time stepping and active-object management.

    sim = WindshieldWaterSimulator(W, H, 'heavy', seed=0)
    sim.warmup()                       # so a clip doesn't start on a dry windshield
    for frame in video:
        sim.step()                     # dt = 1 / fps
        maps = sim.render_maps()       # WaterMaps (height, normals, offsets, alpha, layers)
        out = renderer.apply_maps(wet_frame, maps)

The simulator never looks at image content; the renderer never decides physics.
Update order per step (water is counted once):
  rain field -> arrivals (new drops / impacts on existing drops / splashes) -> growth
  -> pinned / sliding motion (+ trail deposition) -> shape -> merging -> rivulets
  -> film -> splash lifecycles -> trail decay -> removal -> bookkeeping.
"""

import math
import time
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from .config import SimConfig, preset
from .droplet import Droplet
from .film import FilmField
from .interactions import merge_droplets, volume_error
from .rasterizer import WaterMaps, droplet_layer, height_to_maps
from .splashes import SplashSystem
from .trails import DepositionField, TrailSystem


class RainField:
    """Normalised rain intensity R(x, y) (mean ~1), smooth, slowly drifting."""

    def __init__(self, cfg, H, W, rng, cell=8):
        rf = cfg.rain_field
        self.cell, self.H, self.W = cell, H, W
        gh, gw = math.ceil(H / cell), math.ceil(W / cell)
        sigma = max(rf.scale_px / cell / 2, 0.5)
        pad = int(3 * sigma) + 1
        z = np.pad(rng.standard_normal((gh, gw)).astype(np.float32), pad, mode='wrap')
        z = cv2.GaussianBlur(z, (0, 0), sigma)[pad:-pad, pad:-pad]
        z = (z - z.mean()) / (z.std() + 1e-6)
        base = np.clip(1 + rf.contrast * z / 2, 0.05, None)
        self.base = (base / base.mean()).astype(np.float32)
        self.drift = np.array(rf.drift_px_s, np.float32) / cell
        self.shift = np.zeros(2, np.float32)
        self._gy, self._gx = np.mgrid[0:gh, 0:gw].astype(np.float32)
        self.grid = self.base.copy()
        self.rmax = float(self.base.max())

    def advance(self, dt):
        self.shift += self.drift * dt
        gh, gw = self.base.shape
        self.grid = cv2.remap(self.base, (self._gx - self.shift[0]) % gw, (self._gy - self.shift[1]) % gh,
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def at(self, x, y) -> float:
        gh, gw = self.grid.shape
        return float(self.grid[min(max(int(y / self.cell), 0), gh - 1), min(max(int(x / self.cell), 0), gw - 1)])

    def sample(self, rng):
        """A position with density proportional to R (rejection sampling)."""
        for _ in range(50):
            x, y = rng.uniform(0, self.W), rng.uniform(0, self.H)
            if rng.uniform(0, self.rmax) <= self.at(x, y):
                return x, y
        return x, y

    def resized(self, gh, gw):
        return cv2.resize(self.grid, (gw, gh), interpolation=cv2.INTER_LINEAR)


class WindshieldWaterSimulator:
    def __init__(self, width: int, height: int, config, seed: int = 0, device='cuda',
                 rain_intensity: float = 1.0):
        self.cfg: SimConfig = preset(config) if isinstance(config, str) else config.validate()
        self.W, self.H = width, height
        self.device = torch.device(device)
        self.dt = self.cfg.dt
        self.rain_intensity = rain_intensity
        self.rng = np.random.default_rng(seed)
        self.drops: Dict[int, Droplet] = {}
        self._next_id = 0
        self.t = 0.0
        self.frame = 0
        self.rain = RainField(self.cfg, height, width, self.rng)
        self.trails = TrailSystem(self.cfg)
        self.rivulets = DepositionField(self.cfg, height, width) if self.cfg.rivulets.enabled else None
        self.film = FilmField(self.cfg, height, width) if self.cfg.film.enabled else None
        self.splashes = SplashSystem(self.cfg)
        self.stats: dict = {}
        self.merge_events = []
        self._prev_drawn = {}
        self.last_maps: Optional[WaterMaps] = None

    # ------------------------------------------------------------------ objects
    def new_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    def _sample_radius(self) -> float:
        sp = self.cfg.spawn
        w = np.array([c.weight for c in sp.size_mixture], float)
        c = sp.size_mixture[int(self.rng.choice(len(w), p=w / w.sum()))]     # p(r) = sum w_c p_c(r)
        return float(np.clip(self.rng.lognormal(math.log(c.median_px), c.sigma), sp.r_min_px, sp.r_max_px))

    def make_drop(self, x, y, volume) -> Droplet:
        cfg = self.cfg
        K = cfg.shape.n_harmonics
        kk = np.arange(2, K + 2)
        r_slide = cfg.motion.slide_radius_px * float(self.rng.uniform(*cfg.motion.slide_jitter))
        d = Droplet(
            drop_id=self.new_id(), x=float(x), y=float(y), volume=float(volume), k=cfg.growth.volume_k,
            slide_volume=cfg.growth.volume_k * r_slide ** 3,
            shape_a=self.rng.normal(0, 1, K) * cfg.shape.irregularity / kk,
            shape_b=self.rng.normal(0, 1, K) * cfg.shape.irregularity / kk,
        )
        self.drops[d.drop_id] = d
        return d

    # ------------------------------------------------------------------ processes
    def _arrivals(self):
        """N_t ~ Poisson(lambda dt). An arrival inside an existing drop feeds it."""
        sp, k = self.cfg.spawn, self.cfg.growth.volume_k
        n = int(self.rng.poisson(sp.rate_per_s * self.rain_intensity * self.dt))
        impacts = 0
        # impacts are tested against the drops that existed before this step's arrivals
        ids = list(self.drops)
        arr = np.array([(d.x, d.y, d.radius) for d in self.drops.values()]) if ids else None
        for _ in range(n):
            x, y = self.rain.sample(self.rng) if sp.use_rain_field else (self.rng.uniform(0, self.W),
                                                                          self.rng.uniform(0, self.H))
            r = self._sample_radius()
            if arr is not None:
                dd = np.hypot(arr[:, 0] - x, arr[:, 1] - y)
                hit = np.nonzero(dd < arr[:, 2])[0]
                if hit.size:
                    tgt = self.drops.get(ids[int(hit[np.argmin(dd[hit])])])
                    if tgt is not None:
                        tgt.volume += k * r ** 3          # growth on impact, no new object
                        impacts += 1
                        continue
            self.make_drop(x, y, k * r ** 3)
        new_splash = self.splashes.spawn(self.rng, self.W, self.H, self.dt,
                                         self.rain.sample if sp.use_rain_field else None)
        self.stats.update(arrivals=n, impacts=impacts, splashes_new=new_splash)

    def _growth(self):
        """dV/dt = g_max R(x, y) r^2 (pinned drops collect rain) - evaporation V."""
        g = self.cfg.growth
        for d in self.drops.values():
            if d.state == 'pinned':
                d.volume += g.g_max * self.rain_intensity * self.rain.at(d.x, d.y) * d.radius ** 2 * self.dt
            d.volume = max(0.0, d.volume * (1 - g.evaporation_per_s * self.dt))

    def _motion(self):
        """Pinned until V >= V_slide; then v -> v_max phi(V) with limited acceleration,
        stick-slip pauses and lateral wobble. Sliding drops lay trails (and lose water)."""
        m, dt = self.cfg.motion, self.dt
        disp = []
        for d in self.drops.values():
            if d.state == 'pinned':
                if d.volume >= d.slide_volume and d.pause_left <= 0:
                    d.state = 'sliding'
                else:
                    d.pause_left = max(0.0, d.pause_left - dt)
                    continue
            r_slide = (d.slide_volume / d.k) ** (1 / 3)
            if d.pause_left > 0:                                  # stick-slip: decelerate, stay put
                d.pause_left -= dt
                tx, ty = 0.0, 0.0
            else:
                if self.rng.random() < 1 - math.exp(-m.pause_rate_per_s * dt):
                    d.pause_left = float(self.rng.uniform(*m.pause_s))
                phi = min(max(m.phi_min + (d.radius / r_slide - 1.0), m.phi_min), 1.0)
                ty = m.v_max_px_s * phi
                d.wobble += (-d.wobble / m.wobble_tau_s) * dt + m.wobble_px_s * math.sqrt(2 * dt / m.wobble_tau_s) * float(self.rng.normal())
                tx = m.wind_px_s + d.wobble
            # a = clip(target - v): no instant jump to max speed
            ddx, ddy = tx - d.vx, ty - d.vy
            mag = math.hypot(ddx, ddy)
            lim = m.accel_px_s2 * dt
            if mag > lim:
                ddx, ddy = ddx * lim / mag, ddy * lim / mag
            d.vx += ddx
            d.vy += ddy
            x0, y0 = d.x, d.y
            d.x += d.vx * dt                                      # p(t + dt) = p(t) + v dt
            d.y += d.vy * dt
            disp.append(math.hypot(d.x - x0, d.y - y0))
            d.volume -= self.trails.deposit(d, x0, y0, dt)
            if d.volume < m.repin_volume_frac * d.slide_volume:     # thinned out: pins again
                d.state, d.vx, d.vy, d.trail_id = 'pinned', 0.0, 0.0, -1
        self.stats['sliding'] = sum(d.state == 'sliding' for d in self.drops.values())
        self.stats['mean_slide_px_frame'] = float(np.mean(disp)) if disp else 0.0

    def _shape(self):
        """Stretch / orientation follow the motion smoothly; the boundary eps(phi, t)
        evolves as a slow OU process (never re-drawn per frame)."""
        sh, dt = self.cfg.shape, self.dt
        K = sh.n_harmonics
        kk = np.arange(2, K + 2)
        a_s = 1 - math.exp(-dt / sh.stretch_tau_s)
        a_d = 1 - math.exp(-dt / sh.direction_tau_s)
        theta = math.exp(-dt / sh.irregularity_tau_s)
        for d in self.drops.values():
            target = min(1.0 + sh.stretch_per_px_s * d.speed, sh.max_stretch)
            d.stretch += (target - d.stretch) * a_s
            if d.speed > 5.0:
                nx = d.dir_x + (d.vx / d.speed - d.dir_x) * a_d
                ny = d.dir_y + (d.vy / d.speed - d.dir_y) * a_d
                nn = math.hypot(nx, ny) or 1.0
                d.dir_x, d.dir_y = nx / nn, ny / nn
            amp = (sh.irregularity_sliding if d.state == 'sliding' else sh.irregularity) / kk
            noise = np.sqrt(1 - theta ** 2) * amp
            d.shape_a = d.shape_a * theta + noise * self.rng.normal(0, 1, K)
            d.shape_b = d.shape_b * theta + noise * self.rng.normal(0, 1, K)

    def _merge(self):
        self.merge_events = merge_droplets(self.drops, self.cfg, self.new_id, self.rng)
        self.stats['merges'] = len(self.merge_events)
        self.stats['max_volume_error'] = max((volume_error(e) for e in self.merge_events), default=0.0)
        self.stats['max_merge_shift_px'] = max((e['shift_px'] for e in self.merge_events), default=0.0)

    def _rivulets(self):
        if self.rivulets is None:
            self.trails.deposited = []
            return
        n = self.rivulets.update(self.dt, self.trails.deposited, self.drops, self.rng)
        self.trails.deposited = []
        self.stats.update(drops_feeding_rivulets=n, rivulet_coverage=float(self.rivulets.mask().mean()),
                          rivulet_components=self.rivulets.components())

    def _film(self):
        if self.film is None:
            return
        overflow = None
        if self.rivulets is not None:
            rc = self.cfg.rivulets
            ex = np.maximum(self.rivulets.F - 2 * rc.threshold, 0) * self.cfg.film.rivulet_overflow
            self.rivulets.F -= ex
            overflow = cv2.resize(ex, (self.film.gw, self.film.gh), interpolation=cv2.INTER_AREA) * 0.1
        self.film.update(self.dt, self.rain.resized(self.film.gh, self.film.gw) * self.rain_intensity, overflow)
        self.stats.update(film_coverage=self.film.coverage(), film_mean=float(self.film.F.mean()))

    def _splashes(self):
        self.splashes.update(self.dt, self.rng, lambda x, y, v: self.make_drop(x, y, v), self.film)
        self.stats['splashes_active'] = len(self.splashes.events)

    def _remove(self):
        g = self.cfg.growth
        before = len(self.drops)
        for i in [i for i, d in self.drops.items()
                  if d.radius < g.min_radius_px or d.y - 2 * d.radius > self.H
                  or d.x < -2 * d.radius or d.x > self.W + 2 * d.radius]:
            del self.drops[i]
        self.stats['removed'] = before - len(self.drops)

    def _bookkeeping(self):
        dt, sp = self.dt, self.cfg.spawn
        jumps, jerks = [], []
        drawn = {}
        for d in self.drops.values():
            d.age += dt
            d.opacity = min(1.0, d.age / max(sp.fade_in_s, 1e-6))
            if d.ease_left_s > 0:                              # merge easing of the drawn position
                f = min(1.0, dt / d.ease_left_s)
                d.rox -= d.rox * f
                d.roy -= d.roy * f
                d.ease_left_s -= dt
            drawn[d.drop_id] = (d.rx, d.ry)
            if d.drop_id in self._prev_drawn:
                px, py = self._prev_drawn[d.drop_id]
                jumps.append(math.hypot(d.rx - px, d.ry - py))
                jerks.append(math.hypot(d.rx - px - d.vx * dt, d.ry - py - d.vy * dt))
        self._prev_drawn = drawn
        rads = np.array([d.radius for d in self.drops.values()]) if self.drops else np.zeros(1)
        self.stats.update(
            active=len(self.drops), max_drawn_step_px=max(jumps, default=0.0),
            max_jerk_px=max(jerks, default=0.0),
            r_p10=float(np.percentile(rads, 10)), r_p50=float(np.percentile(rads, 50)),
            r_p90=float(np.percentile(rads, 90)),
            trails=len(self.trails.trails), trail_points=self.trails.total_points(),
            total_volume=float(sum(d.volume for d in self.drops.values())))
        self.t += dt
        self.frame += 1

    # ------------------------------------------------------------------ public
    def step(self):
        """Advance the simulation by exactly one frame (dt = 1 / fps)."""
        self.stats = {}
        self.rain.advance(self.dt)
        self._arrivals()
        self._growth()
        self._motion()
        self._shape()
        self._merge()
        self._rivulets()
        self._film()
        self._splashes()
        self.trails.update(self.dt)
        self._remove()
        self._bookkeeping()

    def warmup(self, seconds: Optional[float] = None):
        n = int(round((self.cfg.warmup_s if seconds is None else seconds) * self.cfg.fps))
        for _ in range(n):
            self.step()

    def render_maps(self) -> WaterMaps:
        """Rasterise the current state (does not change it)."""
        H, W, dev = self.H, self.W, self.device
        t0 = time.perf_counter()
        h_drop, a_drop = droplet_layer(list(self.drops.values()), H, W, self.cfg, dev)
        layers = {'drops': a_drop}
        height = h_drop
        alpha = a_drop

        def add(name, h_np, alpha_scale, ref):
            nonlocal height, alpha
            h = torch.from_numpy(h_np).to(dev)
            a = (h / max(ref, 1e-6)).clamp(0, 1) * alpha_scale
            layers[name] = a
            height = torch.maximum(height, h)
            alpha = torch.maximum(alpha, a)

        tc = self.cfg.trails
        if tc.enabled and self.trails.trails:
            add('trails', self.trails.rasterize(H, W), tc.alpha, tc.height)
        if self.rivulets is not None:
            rc = self.cfg.rivulets
            add('rivulets', self.rivulets.rasterize(H, W), rc.alpha, rc.height)
        if self.film is not None:
            fc = self.cfg.film
            add('film', self.film.rasterize(H, W), fc.max_alpha, fc.height)
        if self.splashes.events:
            add('splashes', self.splashes.rasterize(H, W), 1.0, self.cfg.splash.height)
        maps = height_to_maps(height, alpha, self.cfg, layers)
        self.stats['coverage'] = float((maps.alpha > 0.05).float().mean())
        for name, a in layers.items():                     # visible share of each water layer
            self.stats[f'alpha_cov_{name}'] = float((a > 0.05).float().mean())
        self.stats['render_ms'] = 1000 * (time.perf_counter() - t0)
        self.last_maps = maps
        return maps

    def snapshot(self) -> dict:
        """Debug snapshot of the full water state (not needed for normal runs)."""
        return {
            't': self.t, 'frame': self.frame,
            'drops': {i: dict(x=d.x, y=d.y, vx=d.vx, vy=d.vy, r=d.radius, V=d.volume, state=d.state,
                              age=d.age, stretch=d.stretch) for i, d in self.drops.items()},
            'trails': {i: [list(p) for p in t.points] for i, t in self.trails.trails.items()},
            'splashes': len(self.splashes.events),
            'rng_state': self.rng.bit_generator.state,
        }
