"""
temporal_dynamic.py
Windshield-water simulator for video: individually tracked droplets that evolve
over time, rendered to the (v, mask) droplet map that DropModel.apply_drops()
consumes.

Simulation (CPU, numpy RNG, pixel / frame units) and rendering (GPU) are
separate:
  step()          spawn -> grow -> slide -> deform -> merge -> remove
  render_maps()   current droplets + trails -> v (1xHxWx3), mask (1xHxWx1)

Map convention (same as DropModel.get_normal_map):
  v[..., 0:2]  (row, col) unit-ish surface normal of the water
  v[..., 2]    thickness; refraction offset = -v[:2] * v[2] in normalised
               [-1, 1] image coordinates
  mask         water coverage / alpha in [0, 1]

Physics is a first-pass rule-based model, not a fluid solver; all parameters
live in temporal_dynamic.yaml and are tunable, not calibrated constants.
"""

import copy
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temporal_dynamic.yaml')
SCENARIOS = ('light', 'steady', 'heavy', 'torrential')


def _deep_merge(base, over):
    """Recursively overlay `over` onto `base` (dicts merge, everything else replaces)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_rain_config(path=DEFAULT_CONFIG, scenario='steady'):
    """Resolved simulator config for one scenario: `base` deep-merged with
    `scenarios[scenario]`. A legacy single-config file (top-level
    `temporal_dynamic:` key, e.g. presets/dynamic_v1.yaml) is returned as is."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if 'temporal_dynamic' in raw:
        return raw['temporal_dynamic']
    if scenario not in raw['scenarios']:
        raise ValueError(f'unknown scenario {scenario!r}; available: {list(raw["scenarios"])}')
    return _deep_merge(raw['base'], raw['scenarios'][scenario] or {})


@dataclass
class Droplet:
    id: int
    x: float
    y: float
    volume: float                       # radius = volume ** (1/3)  (pixels)
    template: int                       # index into the shape-template bank
    thickness: float                    # refraction strength (DropModel range 0.3-0.8)
    slide_radius: float                 # per-drop critical radius for sliding
    age: int = 0
    vx: float = 0.0
    vy: float = 0.0
    state: Literal['pinned', 'sliding'] = 'pinned'
    stretch: float = 1.0                # length / width of the bead
    dir_x: float = 0.0                  # smoothed motion direction (unit)
    dir_y: float = 1.0
    pause: int = 0                      # stick-slip: frames left stuck
    wobble: float = 0.0                 # lateral OU state (px / frame)
    fade_in_frames: int = 4
    trail_width_jitter: float = 1.0
    # R2 shape evolution: blend template -> template_b as shape_phase goes 0 -> pi
    template_b: int = 0
    shape_phase: float = 0.0
    shape_rate: float = 0.0
    asym: float = 0.0                   # left/right width asymmetry (OU, larger when sliding)
    # R3 smooth merges: rendered = physical + offset, easing to zero over ease_left frames
    rox: float = 0.0
    roy: float = 0.0
    vol_deficit: float = 0.0            # merged-in volume not yet shown (grows in visibly)
    ease_left: int = 0
    alpha_scale: float = 1.0            # < 1 only for fading merge ghosts
    ghost_target: int = -1              # merge ghost: id of the drop it is flowing into

    @property
    def radius(self):
        return self.volume ** (1.0 / 3.0)

    @property
    def opacity(self):
        return min(1.0, (self.age + 1) / max(1, self.fade_in_frames)) * self.alpha_scale

    # what the renderer draws (== physical state unless a merge is easing in)
    @property
    def rx(self):
        return self.x + self.rox

    @property
    def ry(self):
        return self.y + self.roy

    @property
    def render_radius(self):
        return max(self.volume - self.vol_deficit, 1e-3) ** (1.0 / 3.0)


@dataclass
class TrailSegment:
    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    strength: float
    age: int = 0
    parent: int = -1                    # id of the drop that laid it
    water: float = 0.0                  # volume it took from the drop


def make_shape_templates(num, res, shape_threshold, generator, device):
    """Procedural single-drop templates, same maths as DropModel.get_drop_stock()
    (which builds them at 256x256 and immediately pads them to full frames).

    Returns tmpl (num x 4 x res x res): channels (n_row, n_col, mask, field), plus each
    template's blob centre and radius in grid_sample's normalised [-1, 1] units.
    """
    coords = torch.stack(torch.meshgrid(torch.arange(res), torch.arange(res), indexing='ij'), -1)
    coords = coords.float().to(device) / res                                        # res x res x 2  (row, col)
    noise = torch.rand(num, 3, 2, 2, generator=generator, device=device)            # drop_size / noise_resize = 2
    noise = F.interpolate(noise, size=(res, res), mode='bilinear')
    n = noise[:, 2].unsqueeze(-1)                                                   # num x res x res x 1

    p = math.pi * 2 * (coords - 0.25) + (n - 0.5) * 2
    s = torch.sin(p)
    t = s[..., 0] + s[..., 1]
    k = torch.cos(p)
    interp = (0.2 * (1 - t - 0.5) + 2. * (t - 0.5)).unsqueeze(-1)
    v = -torch.cat((k, interp), -1)
    v = v / v.norm(dim=-1, keepdim=True)
    mask = (t > shape_threshold).float()
    # continuous field t - threshold, clamped and offset to [0, 2] so zero padding
    # outside the template reads as "far outside the drop" (-1) after subtracting 1
    field = (t - shape_threshold).clamp(-1, 1) + 1

    tmpl = torch.stack([v[..., 0], v[..., 1], mask, field], 1)                      # num x 4 x res x res

    area = mask.sum(dim=(1, 2)).clamp(min=1)
    rows = torch.arange(res, device=device).float().view(1, res, 1)
    cols = torch.arange(res, device=device).float().view(1, 1, res)
    cy = (mask * rows).sum(dim=(1, 2)) / area
    cx = (mask * cols).sum(dim=(1, 2)) / area
    half = (res - 1) / 2
    centre = torch.stack([cx / half - 1, cy / half - 1], -1)                        # (x, y) for grid_sample
    radius = torch.sqrt(area / math.pi) / half
    return tmpl, centre, radius


class TemporalDynamicRain:
    def __init__(self, width, height, cfg=None, rain_intensity=1.0, shape_threshold=0.6,
                 seed=None, device='cuda', scenario=None, config_path=None):
        """cfg: resolved config dict; or leave it None and pass `scenario`
        (light / steady / heavy / torrential) to load it from `config_path`."""
        if cfg is None:
            cfg = load_rain_config(config_path or DEFAULT_CONFIG, scenario or 'steady')
        self.scenario = scenario
        self.width = width
        self.height = height
        self.cfg = cfg
        self.rain_intensity = rain_intensity
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.droplets: List[Droplet] = []
        self.trails: List[TrailSegment] = []
        self.next_id = 0
        self.frame_index = 0
        self.stats = {}
        self._prev_pos = {}
        self._prev_rpos = {}
        self._merged_ids = set()
        self.ghosts: List[Droplet] = []
        self._new_segments: List[TrailSegment] = []
        wc = cfg['trails'].get('wetness', {})
        self.gs = wc.get('grid_scale', 4)
        self.wet = np.zeros((math.ceil(height / self.gs), math.ceil(width / self.gs)), np.float32)
        self._wet_gx = np.zeros_like(self.wet)
        self.profile = defaultdict(float) if cfg.get('profile', False) else None

        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(self.rng.integers(0, 2 ** 31 - 1)))
        sh = cfg['shape']
        self.tmpl, self.tmpl_centre, self.tmpl_radius = make_shape_templates(
            sh['num_templates'], sh['template_res'], shape_threshold, gen, self.device)

    # ------------------------------------------------------------------ simulation
    def _new_droplet(self, x, y, volume, state='pinned'):
        c = self.cfg
        d = Droplet(
            id=self.next_id, x=x, y=y, volume=volume,
            template=int(self.rng.integers(0, c['shape']['num_templates'])),
            thickness=float(self.rng.uniform(*c['shape']['thickness_range'])),
            slide_radius=c['motion']['slide_radius'] * float(self.rng.uniform(*c['motion'].get('slide_jitter', [0.8, 1.2]))),
            state=state, fade_in_frames=c['spawn']['fade_in_frames'],
            trail_width_jitter=float(self.rng.uniform(0.7, 1.3)),
        )
        if c['shape'].get('evolve', False):
            d.template_b = int(self.rng.integers(0, c['shape']['num_templates']))
            d.shape_rate = float(self.rng.uniform(*c['shape']['shape_rate']))
            d.shape_phase = float(self.rng.uniform(0, math.pi))
        self.next_id += 1
        return d

    def _sample_radius(self):
        """Initial radius: one lognormal, or a mixture (e.g. many small + some big drops)."""
        sp = self.cfg['spawn']
        mix = sp.get('size_mixture')
        if mix:
            w = np.array([m['weight'] for m in mix], float)
            m = mix[int(self.rng.choice(len(mix), p=w / w.sum()))]
            med, sig = m['median'], m['sigma']
        else:
            med, sig = sp['radius_median'], sp['radius_sigma']
        return float(np.clip(self.rng.lognormal(math.log(med), sig), sp['radius_min'], sp['radius_max']))

    def spawn_droplets(self):
        """N_t ~ Poisson(lambda * dt) arrivals, lambda = rate_per_second * intensity."""
        sp = self.cfg['spawn']
        lam = sp['rate_per_second'] * self.rain_intensity / self.cfg['fps']
        n = int(self.rng.poisson(lam))
        impact = self.cfg['merge'].get('impact_growth', False)
        impacts = 0
        for _ in range(n):
            r = self._sample_radius()
            x, y = float(self.rng.uniform(0, self.width)), float(self.rng.uniform(0, self.height))
            if impact and self.droplets:
                # a drop landing inside an existing bead feeds it instead of becoming a
                # separate drop + merge event (no position jump for the existing bead)
                xs = np.array([d.x for d in self.droplets])
                ys = np.array([d.y for d in self.droplets])
                rs = np.array([d.radius for d in self.droplets])
                dd = np.hypot(xs - x, ys - y)
                hit = np.nonzero(dd < rs)[0]
                if hit.size:
                    self.droplets[int(hit[np.argmin(dd[hit])])].volume += r ** 3
                    impacts += 1
                    continue
            self.droplets.append(self._new_droplet(x, y, r ** 3))
        self.stats['arrivals'] = n
        self.stats['impacts'] = impacts

    def update_growth(self):
        """Pinned drops collect mist in proportion to their area; all drops evaporate slowly."""
        g = self.cfg['growth']
        for d in self.droplets:
            if d.state == 'pinned':
                d.volume += g['rate'] * self.rain_intensity * d.radius ** 2
            d.volume *= (1.0 - g['evaporation'])

    def update_motion(self):
        """Pinned -> sliding above a per-drop critical size; sliding drops accelerate
        towards a size-dependent speed with inertia, stick-slip pauses, lateral
        wobble, lose volume into a trail and shed small beads."""
        m, tr = self.cfg['motion'], self.cfg['trails']
        wc = tr.get('wetness', {})
        wet_on = wc.get('enabled', False)
        new_beads = []
        disp = []
        self._new_segments = []
        for d in self.droplets:
            r = d.radius
            wet_here = self._wet_at(d.x, d.y) if wet_on else 0.0
            if d.state == 'pinned':
                # a pre-wetted track lowers the size needed to start sliding
                thr = d.slide_radius * (1 - wc.get('threshold_relief', 0.0) * min(wet_here, 1.0)) if wet_on else d.slide_radius
                if r > thr and self.rng.random() < m['activation_prob']:
                    d.state = 'sliding'
                else:
                    continue

            if d.pause > 0:
                d.pause -= 1
                d.vx *= 0.3
                d.vy *= 0.3
            else:
                if self.rng.random() < m['pause_prob']:
                    d.pause = int(self.rng.integers(*m['pause_frames']))
                frac = np.clip((r - 0.8 * d.slide_radius) / d.slide_radius, 0.15, 1.0)
                target_vy = m['max_speed'] * frac
                steer = 0.0
                if wet_on:
                    # wet tracks are faster, and drops are drawn sideways into them
                    target_vy *= 1 + wc['speed_boost'] * min(wet_here, 1.0)
                    steer = wc['steer_gain'] * self._wet_grad_x(d.x, d.y + wc['lookahead'] * r)
                d.vy += (target_vy - d.vy) * m['inertia']
                d.wobble += -0.2 * d.wobble + self.rng.normal(0, m['lateral_wobble'])
                d.vx += (d.wobble + m['wind_drift'] + steer - d.vx) * m['inertia']

            x0, y0 = d.x, d.y
            d.x += d.vx
            d.y += d.vy
            step = math.hypot(d.vx, d.vy)
            disp.append(step)

            if step > 0.05:
                # water left behind on the glass: a trail segment of its own shape
                loss = min(tr['volume_loss'] * r * step, 0.5 * d.volume)
                d.volume -= loss
                w = tr['width_frac'] * r * d.trail_width_jitter * float(self.rng.uniform(0.75, 1.25))
                seg = TrailSegment(x0, y0, d.x, d.y, w, tr['strength'], parent=d.id, water=loss)
                self.trails.append(seg)
                self._new_segments.append(seg)
                # occasionally a small bead pinches off at the tail
                if self.rng.random() < tr['bead_prob'] and d.volume > 8:
                    bv = d.volume * float(self.rng.uniform(0.03, 0.08))
                    d.volume -= bv
                    tail = 0.8 * r
                    if self.cfg['merge'].get('bead_clearance', False):
                        tail = 1.05 * self.cfg['merge']['overlap_frac'] * (d.radius + bv ** (1 / 3)) + 1.0
                    new_beads.append(self._new_droplet(x0 - d.dir_x * tail, y0 - d.dir_y * tail, bv))

            # a sliding drop that has thinned out stops (pins) again
            if d.radius < m['stop_radius_frac'] * d.slide_radius:
                d.state, d.vx, d.vy, d.pause = 'pinned', 0.0, 0.0, 0
        for b in new_beads:
            b.age = b.fade_in_frames   # already part of the water, no fade
        self.droplets += new_beads
        self.stats['sliding'] = sum(d.state == 'sliding' for d in self.droplets)
        self.stats['mean_slide_disp'] = float(np.mean(disp)) if disp else 0.0
        self.stats['beads'] = len(new_beads)

    def update_shapes(self):
        """Stretch and orientation follow speed / direction smoothly, never re-randomised."""
        sh = self.cfg['shape']
        evolve = sh.get('evolve', False)
        for d in self.droplets:
            if evolve:
                sliding = d.state == 'sliding'
                d.shape_phase += d.shape_rate * (sh['slide_shape_boost'] if sliding else 1.0)
                if d.shape_phase >= math.pi:          # fully at template_b: make it the base
                    d.shape_phase -= math.pi
                    d.template, d.template_b = d.template_b, int(self.rng.integers(0, sh['num_templates']))
                noise = sh['asym_noise'] * (sh['asym_sliding_mult'] if sliding else 1.0)
                d.asym = float(np.clip(d.asym * 0.95 + self.rng.normal(0, noise), -sh['asym_max'], sh['asym_max']))
            speed = math.hypot(d.vx, d.vy)
            target = 1.0 + min(sh['stretch_gain'] * speed, sh['max_stretch'] - 1.0)
            d.stretch += (target - d.stretch) * sh['stretch_smooth']
            if speed > 0.2:
                dx, dy = d.vx / speed, d.vy / speed
                a = sh['direction_smooth']
                nx, ny = d.dir_x + (dx - d.dir_x) * a, d.dir_y + (dy - d.dir_y) * a
                nn = math.hypot(nx, ny) or 1.0
                d.dir_x, d.dir_y = nx / nn, ny / nn

    def handle_merging(self):
        """Overlapping drops coalesce: volume conserved, position / velocity volume-weighted,
        the larger drop keeps its template (its look stays continuous)."""
        if len(self.droplets) < 2:
            self.stats['merges'] = 0
            self.stats['max_merge_shift'] = 0.0
            return
        xs = np.array([d.x for d in self.droplets])
        ys = np.array([d.y for d in self.droplets])
        rs = np.array([d.radius for d in self.droplets])
        dist = np.hypot(xs[:, None] - xs[None], ys[:, None] - ys[None])
        touch = dist < self.cfg['merge']['overlap_frac'] * (rs[:, None] + rs[None])
        ii, jj = np.nonzero(np.triu(touch, 1))
        order = np.argsort(dist[ii, jj])
        alive = np.ones(len(self.droplets), bool)
        merges = 0
        shifts = []
        for k in order:
            i, j = ii[k], jj[k]
            if not (alive[i] and alive[j]):
                continue
            a, b = self.droplets[i], self.droplets[j]
            big, small = (a, b) if a.volume >= b.volume else (b, a)
            v = a.volume + b.volume
            bx, by = big.x, big.y
            brx, bry = big.rx, big.ry          # where it is currently drawn
            big.x = (a.x * a.volume + b.x * b.volume) / v
            big.y = (a.y * a.volume + b.y * b.volume) / v
            big.vx = (a.vx * a.volume + b.vx * b.volume) / v
            big.vy = (a.vy * a.volume + b.vy * b.volume) / v
            big.volume = v
            big.age = max(big.age, small.age)
            if small.state == 'sliding' or big.radius > big.slide_radius:
                big.state = 'sliding'
            alive[j if big is a else i] = False
            merges += 1
            shifts.append(math.hypot(big.x - bx, big.y - by))
            ease = self.cfg['merge'].get('ease_frames', 0)
            if ease > 0:
                # physics jumps to the volume-weighted centre; the drawing eases there,
                # and the absorbed drop visibly flows into the survivor as a fading ghost
                big.rox, big.roy = brx - big.x, bry - big.y
                big.vol_deficit = big.vol_deficit + small.volume
                big.ease_left = ease
                small.rox, small.roy = small.rx - small.x, small.ry - small.y
                small.ghost_target, small.ease_left = big.id, ease
                self.ghosts.append(small)
            self._merged_ids.add(big.id)
        self.droplets = [d for d, keep in zip(self.droplets, alive) if keep]
        self.stats['merges'] = merges
        self.stats['max_merge_shift'] = float(max(shifts)) if shifts else 0.0

    def remove_expired(self):
        rm, tr = self.cfg['remove'], self.cfg['trails']
        before = len(self.droplets)
        self.droplets = [d for d in self.droplets
                         if d.radius >= rm['min_radius']
                         and -3 * d.radius < d.x < self.width + 3 * d.radius
                         and d.y < self.height + 3 * d.radius]
        wp = rm.get('wiper_period')
        if wp and self.frame_index > 0 and self.frame_index % wp == 0:
            self.droplets, self.trails, self.ghosts = [], [], []
            self.wet[:] = 0
        self.stats['removed'] = before - len(self.droplets)
        for s in self.trails:
            s.age += 1
        self.trails = [s for s in self.trails if s.age < tr['lifetime_frames']]

    def step(self):
        """Advance the simulation by exactly one frame."""
        self.stats = {}
        self._merged_ids = set()
        self.spawn_droplets()
        self.update_growth()
        self.update_motion()
        self.update_shapes()
        self.handle_merging()
        self.remove_expired()
        self.update_wetness()
        for d in self.droplets:
            d.age += 1
        self._update_easing()

        # track continuity: largest per-frame jump of any surviving id
        pos = {d.id: (d.x, d.y) for d in self.droplets}
        jumps = [math.hypot(x - self._prev_pos[i][0], y - self._prev_pos[i][1])
                 for i, (x, y) in pos.items() if i in self._prev_pos]
        self.stats['max_track_jump'] = float(max(jumps)) if jumps else 0.0
        motion = [math.hypot(x - self._prev_pos[i][0], y - self._prev_pos[i][1])
                  for i, (x, y) in pos.items() if i in self._prev_pos and i not in self._merged_ids]
        self.stats['max_motion_jump'] = float(max(motion)) if motion else 0.0
        # what the viewer sees: jump of the DRAWN position (includes merge easing)
        rpos = {d.id: (d.rx, d.ry) for d in self.droplets}
        rj = [math.hypot(x - self._prev_rpos[i][0], y - self._prev_rpos[i][1])
              for i, (x, y) in rpos.items() if i in self._prev_rpos]
        self.stats['max_render_jump'] = float(max(rj)) if rj else 0.0
        # jerk: drawn displacement NOT explained by the drop's own velocity this frame
        # (fast sliding is legitimate motion; merges / teleports show up here)
        jerk = [math.hypot(d.rx - self._prev_rpos[d.id][0] - d.vx, d.ry - self._prev_rpos[d.id][1] - d.vy)
                for d in self.droplets if d.id in self._prev_rpos]
        self.stats['max_render_jerk'] = float(max(jerk)) if jerk else 0.0
        self._prev_rpos = rpos
        self.stats['ghosts'] = len(self.ghosts)
        self.stats['persisting'] = len(jumps)
        rads = np.array([d.radius for d in self.droplets]) if self.droplets else np.zeros(1)
        self.stats['median_area'] = float(np.median(math.pi * rads ** 2))
        self.stats['p90_radius'] = float(np.percentile(rads, 90))
        self.stats['p10_radius'] = float(np.percentile(rads, 10))
        sr = [d.radius for d in self.droplets if d.state == 'sliding']
        self.stats['slide_r_p10'] = float(np.percentile(sr, 10)) if sr else 0.0
        self.stats['slide_r_p90'] = float(np.percentile(sr, 90)) if sr else 0.0
        self._prev_pos = pos
        self.stats['active'] = len(self.droplets)
        self.stats['trail_segments'] = len(self.trails)
        self.frame_index += 1

    def _cell(self, x, y):
        gh, gw = self.wet.shape
        return min(max(int(y / self.gs), 0), gh - 1), min(max(int(x / self.gs), 0), gw - 1)

    def _wet_at(self, x, y):
        return float(self.wet[self._cell(x, y)])

    def _wet_grad_x(self, x, y):
        """d(wetness)/dx in wetness-units per cell, from the smoothed field."""
        return float(self._wet_gx[self._cell(x, y)])

    def update_wetness(self):
        """Persistent wet-path memory on a coarse grid: this frame's trail water is
        deposited along the segments, everything drains slowly. Also measures how
        many new segments run along existing wet paths (trail reuse)."""
        wc = self.cfg['trails'].get('wetness', {})
        if not wc.get('enabled', False):
            return
        segs = self._new_segments
        # trail reuse: is the glass just AHEAD of a sliding drop (where its own trail
        # hasn't reached yet) already wet from earlier water?
        ahead = [self._wet_at(d.x + 2 * d.dir_x * d.radius, d.y + 2 * d.dir_y * d.radius) > wc['reuse_threshold']
                 for d in self.droplets if d.state == 'sliding']
        self.stats['trail_reuse'] = float(np.mean(ahead)) if ahead else 0.0
        self.wet *= (1.0 - wc['drain'])
        if segs:
            dep = np.zeros_like(self.wet)
            g = self.gs
            for sg in segs:
                cv2.line(dep, (int(sg.x0 / g), int(sg.y0 / g)), (int(sg.x1 / g), int(sg.y1 / g)),
                         float(min(1.0, sg.water * wc['deposit_gain'])),
                         thickness=max(1, int(round(sg.width / g))))
            self.wet = np.minimum(self.wet + dep, wc['max'])
        sm = cv2.GaussianBlur(self.wet, (0, 0), 1.0)
        self._wet_gx = np.gradient(sm, axis=1).astype(np.float32)
        self.stats['wet_coverage'] = float((self.wet > wc['reuse_threshold']).mean())

    def _update_easing(self):
        """Advance merge easing: drawn offsets / hidden volume of survivors shrink to
        zero; ghosts move onto their target's drawn position while shrinking and fading."""
        by_id = {d.id: d for d in self.droplets}
        for d in self.droplets:
            if d.ease_left > 0:
                k = 1.0 / d.ease_left
                d.rox -= d.rox * k
                d.roy -= d.roy * k
                d.vol_deficit -= d.vol_deficit * k
                d.ease_left -= 1
        keep = []
        for g in self.ghosts:
            if g.ease_left <= 0:
                continue
            k = 1.0 / g.ease_left
            t = by_id.get(g.ghost_target)
            if t is not None:                   # flow towards where the survivor is drawn
                g.rox += (t.rx - g.rx) * k
                g.roy += (t.ry - g.ry) * k
            g.volume *= (1.0 - k)
            g.alpha_scale *= (1.0 - k)
            g.ease_left -= 1
            if g.ease_left > 0:
                keep.append(g)
        self.ghosts = keep

    def warmup(self, frames):
        """Run the simulation before the first output frame so the clip doesn't start dry."""
        for _ in range(frames):
            self.step()

    # ------------------------------------------------------------------ rendering
    def _tick(self, key, t0):
        """Profiling: wall time of a render stage (GPU synchronised), only when cfg.profile."""
        if self.profile is None:
            return None
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        t = time.perf_counter()
        if t0 is not None:
            self.profile[key] += t - t0
        return t

    def _drop_patches(self, sel, P, r, st, ux, uy, xs, ys, th, op, idx, ev=None):
        """Sample the deformed template of drops `sel` into P x P patches centred on
        each drop. Returns coverage pm (n x P x P) and v pv (n x 3 x P x P).
        ev: optional R2 shape-evolution tensors (idx_b, w, asym, neck), per drop."""
        sh, dev = self.cfg['shape'], self.device
        r, st, ux, uy, xs, ys, th, op, idx = (t[sel] for t in (r, st, ux, uy, xs, ys, th, op, idx))
        if ev is not None:
            ev = {k: t[sel] for k, t in ev.items()}
        half = P // 2
        off = torch.arange(P, device=dev).float() - half
        cx_i, cy_i = xs.round(), ys.round()
        # pixel offsets from the true (sub-pixel) centre -> smooth sub-pixel motion
        dx = off.view(1, 1, P) - (xs - cx_i).view(-1, 1, 1)
        dy = off.view(1, P, 1) - (ys - cy_i).view(-1, 1, 1)

        # drop frame: 'along' points in the motion direction (down when sliding down)
        along = dx * ux.view(-1, 1, 1) + dy * uy.view(-1, 1, 1)
        across = -dx * uy.view(-1, 1, 1) + dy * ux.view(-1, 1, 1)
        a = st.sqrt().view(-1, 1, 1)
        rr = r.view(-1, 1, 1)
        along_n = along / (rr * a)                         # elongate along motion,
        across_n = across * a / rr                         # narrow across it (area kept)
        # teardrop: the trailing side (along < 0) tapers more the faster / longer it is
        taper_k = sh['taper'] if ev is None else sh['taper'] * ev['neck'].view(-1, 1, 1)
        taper = 1 + taper_k * (st.view(-1, 1, 1) - 1) * (-along_n).clamp(0, 1)
        across_n = across_n * taper
        if ev is not None:
            # asymmetric width: one flank narrower than the other (smooth across the axis)
            across_n = across_n * (1 + ev['asym'].view(-1, 1, 1) * torch.tanh(4 * across_n))

        def sample(ti):
            tc = self.tmpl_centre[ti].view(-1, 1, 1, 2)
            trad = self.tmpl_radius[ti].view(-1, 1, 1)
            # template frame: rows = along, cols = across
            grid = torch.stack([across_n * trad, along_n * trad], -1) + tc
            return F.grid_sample(self.tmpl[ti], grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        patch = sample(idx)
        if ev is None:
            n_along, n_across, pm = patch[:, 0], patch[:, 1], patch[:, 2]
        else:
            # blend the two shapes' continuous fields, THEN threshold -> crisp, gradually morphing outline
            pb = sample(ev['idx_b'])
            w = ev['w'].view(-1, 1, 1)
            n_along = (1 - w) * patch[:, 0] + w * pb[:, 0]
            n_across = (1 - w) * patch[:, 1] + w * pb[:, 1]
            field = (1 - w) * (patch[:, 3] - 1) + w * (pb[:, 3] - 1)
            # the sin+sin template field has diamond-like contours (kite shapes once stretched);
            # mixing in a round field (1 at centre, 0 at the drop edge, same taper / asymmetry
            # warp) gives a rounded front with a tapering tail
            rho = sh.get('roundness', 0.0)
            if rho > 0:
                round_f = (1 - torch.sqrt(across_n ** 2 + along_n ** 2)).clamp(-1, 1)
                field = (1 - rho) * field + rho * round_f
            pm = (field / sh['edge_band'] + 0.5).clamp(0, 1)
        # rotate normals back to image (row, col)
        n_col = n_across * (-uy.view(-1, 1, 1)) + n_along * ux.view(-1, 1, 1)
        n_row = n_across * ux.view(-1, 1, 1) + n_along * uy.view(-1, 1, 1)
        pm = pm * op.view(-1, 1, 1)
        pv = torch.stack([n_row, n_col, th.view(-1, 1, 1).expand_as(n_row)], 1)
        return pm, pv, cx_i.long() - half, cy_i.long() - half

    def _render_droplets(self, renderer=None):
        """Composite rule (both renderers): per pixel the drop with the highest coverage
        wins, ties go to the larger drop -- no averaging, so overlapping / merging water
        never looks optically weaker.
          'batched': drops bucketed by patch size, one scatter_reduce for the frame
          'loop':    original per-drop pasting (reference for equivalence tests)"""
        renderer = renderer or self.cfg['shape'].get('renderer', 'batched')
        t0 = self._tick(None, None)
        H, W, dev = self.height, self.width, self.device
        v_map = torch.zeros(3, H, W, device=dev)
        m_map = torch.zeros(1, H, W, device=dev)
        drops = sorted(self.droplets + self.ghosts, key=lambda d: d.volume)   # rank = volume order
        if not drops:
            return v_map, m_map
        sh = self.cfg['shape']
        f32 = dict(device=dev, dtype=torch.float32)
        r = torch.tensor([d.render_radius for d in drops], **f32)
        st = torch.tensor([d.stretch for d in drops], **f32)
        ux = torch.tensor([d.dir_x for d in drops], **f32)
        uy = torch.tensor([d.dir_y for d in drops], **f32)
        xs = torch.tensor([d.rx for d in drops], **f32)
        ys = torch.tensor([d.ry for d in drops], **f32)
        th = torch.tensor([d.thickness for d in drops], **f32)
        op = torch.tensor([d.opacity for d in drops], **f32)
        idx = torch.tensor([d.template for d in drops], device=dev)
        args = (r, st, ux, uy, xs, ys, th, op, idx)
        N = len(drops)
        ev = None
        grow = 1.0     # patch-extent factor; asymmetry can widen one flank by 1 / (1 - asym_max)
        if sh.get('evolve', False):
            grow = 1.0 / (1.0 - sh['asym_max'])
            ev = {
                'idx_b': torch.tensor([d.template_b for d in drops], device=dev),
                'w': torch.tensor([0.5 - 0.5 * math.cos(d.shape_phase) for d in drops], **f32),
                'asym': torch.tensor([d.asym for d in drops], **f32),
                # big sliding drops develop a stronger trailing neck
                'neck': torch.tensor([1 + sh['neck_gain'] * min(max(d.radius / d.slide_radius - 1, 0), 1)
                                      if d.state == 'sliding' else 1.0 for d in drops], **f32),
            }

        if renderer == 'loop':
            ext = (r * st.sqrt() * 1.3 * grow + 3).max().item()
            P = int(min(2 * math.ceil(ext) + 1, sh['max_patch']))
            pm, pv, x0s, y0s = self._drop_patches(torch.arange(N, device=dev), P, *args, ev=ev)
            t0 = self._tick('drops_patches', t0)
            x0s, y0s = x0s.tolist(), y0s.tolist()
            for k in range(N):
                x0, y0 = x0s[k], y0s[k]
                xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + P, W), min(y0 + P, H)
                if xa >= xb or ya >= yb:
                    continue
                pk = pm[k, ya - y0:yb - y0, xa - x0:xb - x0]
                reg_m = m_map[0, ya:yb, xa:xb]
                sel = (pk > 0.01) & (pk >= reg_m)
                reg_m[sel] = pk[sel]
                reg_v = v_map[:, ya:yb, xa:xb]
                reg_v[:, sel] = pv[k][:, ya - y0:yb - y0, xa - x0:xb - x0][:, sel]
            self._tick('drops_composite', t0)
            return v_map, m_map

        # ---- batched: bucket drops by the patch size they need ----
        ext = r * st.sqrt() * 1.3 * grow + 3
        need = (2 * torch.ceil(ext) + 1).clamp(max=sh['max_patch'])
        buckets = [b for b in sh.get('patch_buckets', [25, 41, 65, 97, 129]) if b < sh['max_patch']] + [sh['max_patch']]
        flats, keys, ms, vs = [], [], [], []
        lower = 0
        for P in buckets:
            sel = torch.nonzero((need > lower) & (need <= P)).flatten()
            lower = P
            if sel.numel() == 0:
                continue
            pm, pv, x0s, y0s = self._drop_patches(sel, P, *args, ev=ev)
            ar = torch.arange(P, device=dev)
            gy = y0s.view(-1, 1, 1) + ar.view(1, P, 1)
            gx = x0s.view(-1, 1, 1) + ar.view(1, 1, P)
            valid = (pm > 0.01) & (gy >= 0) & (gy < H) & (gx >= 0) & (gx < W)
            flat = (gy * W + gx).expand_as(pm)[valid]
            # integer key: coverage (quantised to 2^-32, below float32 resolution) first,
            # volume rank breaks exact ties -- same order as the loop's `pk >= reg_m`
            q = torch.round(pm[valid].double() * (1 << 32)).long()
            rank = sel.view(-1, 1, 1).expand_as(pm)[valid]
            flats.append(flat)
            keys.append(q * N + rank)
            ms.append(pm[valid])
            vs.append(pv.permute(1, 0, 2, 3)[:, valid])
        t0 = self._tick('drops_patches', t0)
        if flats:
            flat, key = torch.cat(flats), torch.cat(keys)
            best = torch.full((H * W,), -1, device=dev, dtype=torch.long)
            best.scatter_reduce_(0, flat, key, reduce='amax')
            win = key == best[flat]
            m_map.view(-1)[flat[win]] = torch.cat(ms)[win]
            v_map.view(3, -1)[:, flat[win]] = torch.cat(vs, 1)[:, win]
        self._tick('drops_composite', t0)
        return v_map, m_map

    def render_maps(self):
        """Current droplets + trails -> (v 1xHxWx3, mask 1xHxWx1) on self.device."""
        v_d, m_d = self._render_droplets()
        t0 = self._tick(None, None)
        v_t, m_t = self._trail_maps()
        self._tick('trails', t0)
        use_drop = m_d >= m_t
        v = torch.where(use_drop, v_d, v_t)
        m = torch.where(use_drop, m_d, m_t)
        self.stats['coverage'] = float((m > 0.05).float().mean())
        return v.permute(1, 2, 0).unsqueeze(0), m.permute(1, 2, 0).unsqueeze(0)

    def _trail_maps(self):
        """Trails as a separate thin water film: a height field drawn from the
        segments (fading with age), normals from its gradient, weak refraction."""
        tr = self.cfg['trails']
        H, W, dev = self.height, self.width, self.device
        h = np.zeros((H, W), np.float32)
        segs = sorted(self.trails, key=lambda s: s.strength * (1 - s.age / tr['lifetime_frames']))
        for s in segs:
            val = s.strength * (1.0 - s.age / tr['lifetime_frames'])
            cv2.line(h, (int(round(s.x0)), int(round(s.y0))), (int(round(s.x1)), int(round(s.y1))),
                     float(val), thickness=max(1, int(round(s.width))), lineType=cv2.LINE_AA)
        h = cv2.GaussianBlur(h, (0, 0), tr['blur_sigma'])
        gy, gx = np.gradient(h)
        g = torch.from_numpy(np.stack([gy, gx])).to(dev)
        hm = torch.from_numpy(h).to(dev)[None]
        gn = g.norm(dim=0, keepdim=True)
        # unit normal across the streak, fading to zero on the flat centre line
        n = g / (gn + 1e-6) * (gn * tr['normal_gain']).clamp(max=1.0)
        v = torch.cat([-n, torch.full_like(hm, tr['thickness'])], 0)
        m = (hm * tr['alpha']).clamp(0, 1)
        return v, m
