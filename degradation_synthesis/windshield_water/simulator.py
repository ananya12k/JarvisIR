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


class LiveIndex:
    """Spatial hash of drops that is updated as arrivals create / grow drops, so later
    arrivals in the same step see the current water (no stale snapshot)."""

    def __init__(self, drops, cell=32.0):
        self.cell = cell
        self.cells = {}
        self.rmax = 0.0
        for d in drops.values():
            self.insert(d)

    def _key(self, x, y):
        return int(x // self.cell), int(y // self.cell)

    def insert(self, d):
        self.cells.setdefault(self._key(d.x, d.y), []).append(d)
        self.rmax = max(self.rmax, d.radius)

    def grew(self, d):
        self.rmax = max(self.rmax, d.radius)

    def nearest_contact(self, x, y, r):
        """Drop whose footprint touches the arrival: d_ij <= r_i + r_j (closest centre)."""
        reach = r + self.rmax
        k = int(math.ceil(reach / self.cell))
        cx, cy = self._key(x, y)
        best, best_d = None, float('inf')
        for gx in range(cx - k, cx + k + 1):
            for gy in range(cy - k, cy + k + 1):
                for d in self.cells.get((gx, gy), ()):
                    dist = math.hypot(d.x - x, d.y - y)
                    if dist <= d.radius + r and dist < best_d:
                        best, best_d = d, dist
        return best


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
        self.water = {'arrived': 0.0, 'to_drops': 0.0, 'to_splash': 0.0}   # arrival ledger (volume)
        self.arrival_log = []
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
        """Rain ARRIVALS (water entering the windshield system), N_t ~ Poisson(lambda dt),
        lambda = base_rate x I x clip intensity. For each arrival, in order:
          1. sample position (rain field) and radius (size mixture)
          2. find an existing drop in contact, d_ij <= r_i + r_j (live index)
          3. hit  -> its water joins that drop;   miss -> a new drop
          4. the impact may trigger a splash AT THE IMPACT POINT, with probability
             impact_prob x strength x (dry_factor on dry glass .. 1 on water); the splash
             carries water_frac of the arriving volume (counted once - see ledger)
        Drop-drop COLLISIONS (two existing water bodies) are handled in _merge."""
        sp, sc, k = self.cfg.spawn, self.cfg.splash, self.cfg.growth.volume_k
        lam = sp.base_rate_per_s * self.cfg.intensity * self.rain_intensity
        n = int(self.rng.poisson(lam * self.dt))
        index = LiveIndex(self.drops)
        hits = impact_events = 0
        self.arrival_log = []                            # (x, y, r, hit_id or -1, splashed) this step
        for _ in range(n):
            x, y = self.rain.sample(self.rng) if sp.use_rain_field else (self.rng.uniform(0, self.W),
                                                                          self.rng.uniform(0, self.H))
            r = self._sample_radius()
            vol = k * r ** 3
            self.water['arrived'] += vol
            tgt = index.nearest_contact(x, y, r)
            # splash decision from the impact itself
            u = sc.impact_speed_ref * (r / sc.impact_radius_ref) ** sc.speed_exp
            strength = min(1.0, (r / sc.impact_radius_ref) ** 3 * (u / sc.impact_speed_ref) ** 2)
            wet = 1.0 if tgt is not None else (self.film.level_at(x, y) if self.film is not None else 0.0)
            p = sc.impact_prob * strength * (sc.dry_factor + (1 - sc.dry_factor) * min(wet, 1.0))
            to_splash = 0.0
            if sc.enabled and self.rng.random() < p:
                if self.splashes.trigger(self.rng, x, y, vol, u, sc.water_frac * vol) is not None:
                    to_splash = sc.water_frac * vol
                    impact_events += 1
            self.arrival_log.append((x, y, r, tgt.drop_id if tgt is not None else -1, to_splash > 0))
            rest = vol - to_splash
            self.water['to_splash'] += to_splash
            self.water['to_drops'] += rest
            if tgt is not None:
                tgt.volume += rest                       # arrival joins the drop it hit
                index.grew(tgt)
                hits += 1
            else:
                index.insert(self.make_drop(x, y, rest))
        ambient = self.splashes.spawn_ambient(self.rng, self.W, self.H, self.dt,
                                              self.rain.sample if sp.use_rain_field else None)
        self.stats.update(arrivals=n, drop_hits=hits, impact_events=impact_events,
                          splashes_new=impact_events + ambient)

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
        # spatial pattern and global intensity are passed separately (no cancellation)
        self.film.update(self.dt, self.rain.resized(self.film.gh, self.film.gw),
                         intensity=self.cfg.intensity * self.rain_intensity, overflow=overflow)
        self.stats.update(film_coverage=self.film.coverage(), film_mean=float(self.film.F.mean()),
                          film_saturation_fraction=float((self.film.F >= 0.99).mean()))

    def _splashes(self):
        self.splashes.update(self.dt, self.rng, lambda x, y, v: self.make_drop(x, y, v), self.film)
        areas = self.splashes.area_px()
        self.stats.update(splashes_active=len(self.splashes.events),
                          splash_area_mean=float(np.mean(areas)) if areas else 0.0,
                          splashes_ended=len(self.splashes.finished),
                          splash_duration_sum=float(sum(d for d, _ in self.splashes.finished)))

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

    def layer_maps(self):
        """Per-layer (height, alpha) as H x W tensors, before composition."""
        H, W, dev = self.H, self.W, self.device
        h_drop, a_drop = droplet_layer(list(self.drops.values()), H, W, self.cfg, dev)
        layers = {'drops': (h_drop, a_drop)}

        def thin(name, h_np, alpha_scale, ref):
            h = torch.from_numpy(h_np).to(dev)
            layers[name] = (h, (h / max(ref, 1e-6)).clamp(0, 1) * alpha_scale)

        tc = self.cfg.trails
        if tc.enabled and self.trails.trails:
            thin('trails', self.trails.rasterize(H, W), tc.alpha, tc.height)
        if self.rivulets is not None:
            rc = self.cfg.rivulets
            thin('rivulets', self.rivulets.rasterize(H, W), rc.alpha, rc.height)
        if self.film is not None:
            fc = self.cfg.film
            thin('film', self.film.rasterize(H, W), fc.max_alpha, fc.height)
        if self.splashes.events:
            h_s, a_s = self.splashes.rasterize(H, W, self.dt)
            layers['splashes'] = (torch.from_numpy(h_s).to(dev), torch.from_numpy(a_s).to(dev))
        return layers

    def compose(self, layers):
        """Combine layers into one height + alpha.
        'max'     : height = max_j h_j, alpha = max_j a_j            (baseline)
        'layered' : thin water on the glass  h_thin = film + max(rivulet, trail) + splash
                    (a rivulet sits on wet film; a trail inside a rivulet is the same water),
                    capped; drops sit ON TOP of it: h = h_drop + h_thin;
                    alpha = 1 - prod_j (1 - a_j)  (independent-layer approximation)"""
        zero = torch.zeros_like(layers['drops'][0])
        get = lambda k: layers[k] if k in layers else (zero, zero)
        if self.cfg.render.composition == 'max':
            height, alpha = zero.clone(), zero.clone()
            for h, a in layers.values():
                height, alpha = torch.maximum(height, h), torch.maximum(alpha, a)
            return height, alpha
        h_thin = get('film')[0] + torch.maximum(get('rivulets')[0], get('trails')[0]) + get('splashes')[0]
        h_thin = h_thin.clamp(max=self.cfg.render.thin_cap_px)
        height = get('drops')[0] + h_thin
        transmit = torch.ones_like(zero)
        for _, a in layers.values():
            transmit = transmit * (1 - a.clamp(0, 1))
        return height, 1 - transmit

    def render_maps(self) -> WaterMaps:
        """Rasterise the current state (does not change it)."""
        t0 = time.perf_counter()
        layers = self.layer_maps()
        height, alpha = self.compose(layers)
        maps = height_to_maps(height, alpha, self.cfg, {k: a for k, (h, a) in layers.items()})
        maps.layer_heights = {k: h for k, (h, a) in layers.items()}
        self.stats['coverage'] = float((maps.alpha > 0.05).float().mean())
        for name, (h, a) in layers.items():                # visible share / peak of each water layer
            self.stats[f'alpha_cov_{name}'] = float((a > 0.05).float().mean())
            self.stats[f'layer_height_max_{name}'] = float(h.max())
        off = maps.offset_px.norm(dim=0)[maps.alpha > 0.05]
        self.stats['offset_p99_px'] = float(torch.quantile(off.float()[:200000], 0.99)) if off.numel() else 0.0
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
