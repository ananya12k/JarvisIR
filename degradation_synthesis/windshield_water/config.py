"""
Scenario presets and parameter validation for the windshield-water simulator.

Units: pixels (at the native 848x480 GT resolution), seconds, and "volume units"
with V = k * r^3 (k = volume_k). Every rate is per second; the simulator converts
with dt = 1 / fps, so presets do not change meaning when the frame rate changes.

All values are hand-tuned starting points, NOT calibrated rainfall measurements.
Heavier rain changes the whole water system (arrivals, accumulation, motion,
merging, trails, rivulets, film, splashes), not just droplet size.
"""

import copy
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List, Tuple


@dataclass
class SizeComponent:
    weight: float           # mixture weight w (normalised over components)
    median_px: float        # lognormal median radius
    sigma: float            # lognormal sigma


@dataclass
class SpawnConfig:
    rate_per_s: float = 90.0                       # lambda: N_t ~ Poisson(lambda * dt)
    size_mixture: List[SizeComponent] = field(default_factory=lambda: [
        SizeComponent(0.80, 4.5, 0.50), SizeComponent(0.20, 11.0, 0.35)])
    r_min_px: float = 2.0
    r_max_px: float = 22.0
    fade_in_s: float = 0.13
    use_rain_field: bool = True                    # spawn density follows R(x, y)


@dataclass
class RainFieldConfig:
    """Normalised rain-intensity field R(x, y), mean 1: uneven accumulation."""
    scale_px: float = 200.0                        # size of denser / sparser regions
    contrast: float = 0.5                          # 0 = uniform, 1 = strongly uneven
    drift_px_s: Tuple[float, float] = (8.0, 15.0)  # (x, y) drift of the pattern


@dataclass
class GrowthConfig:
    g_max: float = 0.6                             # dV/dt = g_max * R(x, y) * r^2  (pinned drops)
    evaporation_per_s: float = 0.06                # dV/dt -= evaporation * V
    volume_k: float = 1.0                          # V = k r^3
    min_radius_px: float = 1.0                     # drops that shrink below this are removed


@dataclass
class MotionConfig:
    slide_radius_px: float = 13.5                  # V_slide = k * r_slide^3 (per-drop jitter below)
    slide_jitter: Tuple[float, float] = (0.6, 1.7)
    v_max_px_s: float = 120.0                      # speed of the largest drops
    phi_min: float = 0.15                          # phi(V): bounded, increasing with volume
    accel_px_s2: float = 600.0                     # |dv/dt| limit: no instant max speed
    pause_rate_per_s: float = 0.9                  # stick-slip: chance per second to re-pin briefly
    pause_s: Tuple[float, float] = (0.1, 0.4)
    wobble_px_s: float = 4.5                       # lateral Ornstein-Uhlenbeck noise
    wobble_tau_s: float = 0.15
    wind_px_s: float = 0.0
    repin_volume_frac: float = 0.22                # re-pin when V < frac * V_slide (lost water to trail)


@dataclass
class ShapeConfig:
    stretch_per_px_s: float = 0.0167               # target stretch s = 1 + gain * |v|
    max_stretch: float = 3.0
    stretch_tau_s: float = 0.13
    direction_tau_s: float = 0.15
    n_harmonics: int = 3                           # boundary modes k = 2 .. n+1
    irregularity: float = 0.06                     # epsilon amplitude, pinned
    irregularity_sliding: float = 0.12
    irregularity_tau_s: float = 1.5                # how slowly the boundary evolves
    taper: float = 0.6                             # rear (trailing) edge narrows when stretched
    cap_height: float = 0.6                        # spherical-cap height / footprint radius


@dataclass
class MergeConfig:
    enabled: bool = True
    contact_frac: float = 0.7                      # merge if |p_i - p_j| <= frac * (r_i + r_j)
    ease_s: float = 0.17                           # drawn position eases to the merged centre


@dataclass
class TrailConfig:
    enabled: bool = True
    point_spacing_px: float = 2.0
    width_frac: float = 0.35                       # trail width ~ frac * r
    width_speed_ref_px_s: float = 60.0             # faster drops deposit less -> thinner
    deposit_per_px: float = 0.12                   # volume laid per px moved, x r
    decay_per_s: float = 0.8                       # D(t + dt) = D(t) exp(-mu dt)
    min_strength: float = 0.03
    height: float = 0.25                           # trail height (relative to drop caps)
    alpha: float = 0.55


@dataclass
class RivuletConfig:
    enabled: bool = False
    grid_px: int = 4
    decay_per_s: float = 0.5                       # F = max(0, F - mu dt) + S
    deposit_gain: float = 0.08                     # trail water -> deposition field
    spread_per_s: float = 4.5                      # local spreading (no global blur)
    spread_sigma_cells: float = 1.0
    threshold: float = 0.6                         # connected rivulet where F > threshold
    flow_px_s: float = 60.0                        # downward transport, x clip(F, 0.3, 1.5)
    absorb_per_s: float = 0.5                      # drops on a rivulet transfer water into it
    min_component_cells: int = 6
    height: float = 0.35
    alpha: float = 0.6


@dataclass
class FilmConfig:
    enabled: bool = False
    grid_px: int = 4
    source_per_s: float = 0.06                     # S = source * R^gamma / mean(R^gamma) (+ splash / overflow)
    rain_exponent: float = 3.0                     # gamma > 1 concentrates film where rain is densest
    loss_per_s: float = 0.5                        # L = loss * F  -> patchy equilibrium F ~ S / L
    diffusion_cells2_s: float = 3.0                # D in div(D grad F): local spreading only
    advect_cells_s: float = 4.0                    # downward u, x F (thicker film flows faster)
    rivulet_overflow: float = 0.3                  # rivulet water above 2 x threshold spills into film
    visible: float = 0.15                          # film counted / drawn above this
    full_level: float = 0.6                        # film drawn at full height / alpha from this level
    height: float = 0.15
    max_alpha: float = 0.25


@dataclass
class SplashConfig:
    enabled: bool = True
    rate_per_s: float = 1.0
    min_spacing_px: float = 40.0
    refractory_s: float = 0.3
    r_max_px: Tuple[float, float] = (10.0, 25.0)
    r0_frac: float = 0.3                           # R(t) = R0 + (Rmax - R0)(1 - exp(-t / tau))
    tau_s: float = 0.08
    duration_s: float = 0.6
    relax_s: float = 0.25                          # irregularity relaxes with exp(-t / relax)
    fingers: Tuple[int, int] = (3, 7)
    finger_amp: float = 0.35
    satellites_mean: float = 1.5
    satellite_r_px: Tuple[float, float] = (1.8, 3.5)
    residual_drop_frac: float = 0.5                # water left as a droplet when the event ends
    residual_film_frac: float = 0.5                # ... and as film
    water_per_px2: float = 0.6                     # splash water ~ this x Rmax^2
    height: float = 0.3


@dataclass
class RenderConfig:
    refraction_px: float = 110.0                   # offset = -k * n_xy (px); ~DropModel lens strength
    edge_px: float = 1.0                           # anti-aliased drop edge
    max_patch: int = 161
    patch_buckets: Tuple[int, ...] = (25, 41, 65, 97, 129)


@dataclass
class SimConfig:
    name: str = 'steady'
    fps: float = 30.0
    warmup_s: float = 10.0
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    rain_field: RainFieldConfig = field(default_factory=RainFieldConfig)
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    trails: TrailConfig = field(default_factory=TrailConfig)
    rivulets: RivuletConfig = field(default_factory=RivuletConfig)
    film: FilmConfig = field(default_factory=FilmConfig)
    splash: SplashConfig = field(default_factory=SplashConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    @property
    def dt(self):
        return 1.0 / self.fps

    def validate(self):
        """Raise ValueError on parameters that would break the simulation."""
        def need(cond, msg):
            if not cond:
                raise ValueError(f'[{self.name}] {msg}')
        need(self.fps > 0, 'fps must be > 0')
        need(self.spawn.rate_per_s >= 0, 'spawn rate must be >= 0')
        need(len(self.spawn.size_mixture) > 0, 'size_mixture is empty')
        need(sum(c.weight for c in self.spawn.size_mixture) > 0, 'size_mixture weights sum to 0')
        need(all(c.median_px > 0 and c.sigma >= 0 for c in self.spawn.size_mixture), 'bad size component')
        need(0 < self.spawn.r_min_px < self.spawn.r_max_px, 'need 0 < r_min < r_max')
        need(0 <= self.rain_field.contrast <= 1, 'rain_field.contrast must be in [0, 1]')
        need(self.growth.volume_k > 0, 'volume_k must be > 0')
        need(self.motion.slide_jitter[0] > 0 and self.motion.slide_jitter[0] <= self.motion.slide_jitter[1],
             'bad slide_jitter')
        need(0 < self.motion.phi_min <= 1, 'phi_min must be in (0, 1]')
        need(self.shape.max_stretch >= 1, 'max_stretch must be >= 1')
        need(self.shape.irregularity_sliding < 0.5, 'irregularity too large (boundary may self-intersect)')
        need(0 < self.merge.contact_frac <= 1.0, 'merge.contact_frac must be in (0, 1]')
        need(self.trails.decay_per_s >= 0, 'trail decay must be >= 0')
        need(self.film.loss_per_s > 0 or not self.film.enabled, 'film needs loss > 0 to reach equilibrium')
        need(self.splash.r0_frac < 1, 'splash r0_frac must be < 1')
        need(max(self.render.patch_buckets) < self.render.max_patch, 'patch buckets must be < max_patch')
        return self


SCENARIOS = ('light', 'steady', 'heavy', 'torrential')


def preset(name: str) -> SimConfig:
    """Scenario preset as a config object (no scenario conditionals in the simulator)."""
    c = SimConfig(name=name)
    if name == 'light':
        # mostly stationary small beads, occasional big ones, few sliders, rare splashes;
        # growth / evaporation balance radius g / e = 1.2 px < r_min: beads slowly
        # evaporate instead of piling up
        c.warmup_s = 30.0
        c.spawn.rate_per_s = 30.0
        c.spawn.size_mixture = [SizeComponent(0.90, 4.0, 0.45), SizeComponent(0.10, 10.0, 0.30)]
        c.growth.g_max, c.growth.evaporation_per_s = 0.18, 0.15
        c.spawn.r_min_px = 2.2
        c.growth.min_radius_px = 2.2
        c.motion.slide_radius_px, c.motion.v_max_px_s = 16.0, 90.0
        c.trails.decay_per_s, c.trails.width_frac = 1.2, 0.30
        c.splash.rate_per_s = 0.3
    elif name == 'steady':
        # uneven sizes, regular sliding, visible trails, merging, local film patches
        c.film.enabled = True
        c.film.source_per_s, c.film.max_alpha = 0.06, 0.3
    elif name == 'heavy':
        # larger and faster drops, frequent impacts, growth, merging, rivulets, broad film
        c.spawn.rate_per_s = 160.0
        c.spawn.size_mixture = [SizeComponent(0.70, 5.0, 0.50), SizeComponent(0.30, 13.0, 0.35)]
        c.spawn.r_max_px = 26.0
        c.growth.g_max = 0.9
        c.motion.slide_radius_px, c.motion.v_max_px_s = 12.0, 180.0
        c.trails.decay_per_s, c.trails.width_frac = 1.0, 0.45
        c.rivulets.enabled = True
        c.rivulets.deposit_gain, c.rivulets.decay_per_s, c.rivulets.threshold = 0.4, 0.3, 0.45
        c.film.enabled = True
        c.film.source_per_s, c.film.max_alpha = 0.06, 0.4
        c.splash.rate_per_s, c.splash.satellites_mean = 3.0, 2.5
    elif name == 'torrential':
        # dense accumulation, connected rivulets, strong film, frequent splashes
        c.spawn.rate_per_s = 300.0
        c.spawn.size_mixture = [SizeComponent(0.65, 5.0, 0.50), SizeComponent(0.35, 14.0, 0.35)]
        c.spawn.r_max_px = 28.0
        c.growth.g_max = 1.2
        c.motion.slide_radius_px, c.motion.v_max_px_s = 11.0, 210.0
        c.trails.decay_per_s, c.trails.width_frac = 1.1, 0.5
        c.rivulets.enabled = True
        c.rivulets.threshold, c.rivulets.flow_px_s = 0.4, 80.0
        c.rivulets.deposit_gain, c.rivulets.decay_per_s = 0.6, 0.25
        c.film.enabled = True
        c.film.source_per_s, c.film.max_alpha = 0.09, 0.6
        c.film.advect_cells_s = 6.0
        c.splash.rate_per_s, c.splash.satellites_mean = 6.0, 3.0
        c.splash.r_max_px = (14.0, 32.0)
    else:
        raise ValueError(f'unknown scenario {name!r}; choose from {SCENARIOS}')
    return c.validate()


def to_dict(cfg: SimConfig) -> dict:
    return asdict(cfg)


def apply_overrides(cfg: SimConfig, over: dict) -> SimConfig:
    """Return a copy of cfg with a nested dict of overrides applied (e.g. from YAML)."""
    cfg = copy.deepcopy(cfg)

    def apply(obj, d):
        for k, v in d.items():
            if not hasattr(obj, k):
                raise ValueError(f'unknown config key {k!r} in {type(obj).__name__}')
            cur = getattr(obj, k)
            if is_dataclass(cur) and isinstance(v, dict):
                apply(cur, v)
            elif k == 'size_mixture':
                setattr(obj, k, [SizeComponent(**c) if isinstance(c, dict) else c for c in v])
            elif isinstance(cur, tuple):
                setattr(obj, k, tuple(v))
            else:
                setattr(obj, k, v)
    apply(cfg, over)
    return cfg.validate()
