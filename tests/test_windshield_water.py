"""
Acceptance tests for degradation_synthesis/windshield_water, one per step of the plan.

    ~/miniconda3/envs/visia/bin/python -m pytest tests/test_windshield_water.py -q
"""

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'degradation_synthesis'))

from windshield_water import WindshieldWaterSimulator, preset, apply_overrides, SCENARIOS  # noqa: E402
from windshield_water.droplet import Droplet  # noqa: E402
from windshield_water.interactions import SpatialGrid, merge_droplets, volume_error  # noqa: E402
from windshield_water.splashes import SplashEvent  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
W, H = 320, 240


def quiet(name='steady', **over):
    """Preset with every process off unless switched back on by `over`."""
    base = {'spawn': {'base_rate_per_s': 0.0}, 'splash': {'enabled': False}, 'merge': {'enabled': False},
            'trails': {'enabled': False}, 'film': {'enabled': False}, 'rivulets': {'enabled': False},
            'growth': {'g_max': 0.0, 'evaporation_per_s': 0.0}}
    for k, v in over.items():
        base.setdefault(k, {}).update(v)
    return apply_overrides(preset(name), base)


def sim_with(cfg, seed=0):
    return WindshieldWaterSimulator(W, H, cfg, seed=seed, device=DEV)


# Step 2 -- simulation separated from rendering ------------------------------------------
def test_maps_do_not_depend_on_the_frame_and_are_reproducible():
    from windshield_water.renderer import WaterRenderer
    sim = sim_with(preset('steady'))
    sim.warmup(1.0)
    m1, m2 = sim.render_maps(), sim.render_maps()            # same state -> identical maps
    assert torch.equal(m1.height, m2.height) and torch.equal(m1.alpha, m2.alpha)
    ren = WaterRenderer(H, W, device=DEV)
    f1 = torch.rand(1, 3, H, W, device=DEV) * 2 - 1
    f2 = torch.rand(1, 3, H, W, device=DEV) * 2 - 1
    o1, o2 = ren.apply_maps(f1, m1), ren.apply_maps(f2, m1)
    assert not torch.equal(o1, o2)                            # different frames, same water
    # away from water (beyond the renderer's mask blur, sigma 3.5 px) the frame is untouched
    near = torch.nn.functional.max_pool2d((m1.alpha > 0).float()[None, None], 21, stride=1, padding=10)[0, 0]
    dry = near == 0
    assert dry.any() and torch.allclose(o1[0][:, dry], f1[0][:, dry], atol=1e-5)


# Step 3 -- persistent identity ---------------------------------------------------------
def test_droplet_id_persists_while_state_evolves():
    cfg = quiet(motion={'slide_radius_px': 5.0})
    sim = sim_with(cfg)
    d = sim.make_drop(100.0, 50.0, 9.0 ** 3)                  # big enough to slide
    trace = []
    for _ in range(30):
        sim.step()
        assert d.drop_id in sim.drops and sim.drops[d.drop_id] is d
        trace.append((d.x, d.y, d.age))
    ys, ages = [t[1] for t in trace], [t[2] for t in trace]
    assert ys[-1] > ys[0] and all(b > a for a, b in zip(ages, ages[1:]))


# Step 4 -- continuous Poisson arrivals, size mixture ------------------------------------
def test_arrivals_change_population_and_old_ids_survive():
    cfg = quiet(spawn={'base_rate_per_s': 90.0}, motion={'slide_radius_px': 1e6})   # no sliding / merging
    sim = sim_with(cfg)
    counts, early, arrivals = [], None, 0
    for f in range(100):
        sim.step()
        counts.append(len(sim.drops))
        arrivals += sim.stats['arrivals']
        if f == 20:
            early = set(sim.drops)
    assert counts[-1] > counts[10] > 0
    assert early <= set(sim.drops)                            # old drops keep their ids
    rate = arrivals / (100 / cfg.fps)
    assert 0.7 * 90 < rate < 1.3 * 90                         # E[N_t] = lambda dt


def test_size_mixture_has_small_and_large_drops():
    sim = sim_with(preset('heavy'))
    r = np.array([sim._sample_radius() for _ in range(4000)])
    assert np.percentile(r, 10) < 4.0 and np.percentile(r, 95) > 12.0


# Step 5 -- growth ---------------------------------------------------------------------
def test_growth_is_smooth_and_keeps_identity_and_position():
    cfg = quiet(growth={'g_max': 0.6}, motion={'slide_radius_px': 1e6})
    sim = sim_with(cfg)
    d = sim.make_drop(150.0, 100.0, 4.0 ** 3)
    rs, pos = [], (d.x, d.y)
    for _ in range(90):
        sim.step()
        rs.append(d.radius)
    steps = np.diff(rs)
    assert (steps > 0).all() and steps.max() < 0.1            # monotone, no jumps
    assert (d.x, d.y) == pos and d.drop_id in sim.drops


# Step 6 -- pinned / sliding -------------------------------------------------------------
def test_large_drops_slide_small_ones_stay_and_motion_is_continuous():
    cfg = quiet(motion={'pause_rate_per_s': 0.0})
    sim = sim_with(cfg)
    small = [sim.make_drop(20 + 30 * i, 20.0, 3.0 ** 3) for i in range(8)]
    big = [sim.make_drop(20 + 30 * i, 60.0, 22.0 ** 3) for i in range(8)]
    prev = {d.drop_id: (d.x, d.y) for d in small + big}
    vmax = cfg.motion.v_max_px_s
    for _ in range(30):
        sim.step()
        for d in big:
            if d.drop_id in sim.drops:
                assert math.hypot(d.x - prev[d.drop_id][0], d.y - prev[d.drop_id][1]) <= (vmax + 50) * cfg.dt
                prev[d.drop_id] = (d.x, d.y)
    assert all(d.state == 'pinned' and d.y == 20.0 for d in small)
    assert sum(d.state == 'sliding' for d in big) >= 6


# Step 7 -- shape --------------------------------------------------------------------
def test_moving_drops_elongate_and_shapes_evolve_slowly():
    cfg = quiet(motion={'slide_radius_px': 8.0, 'pause_rate_per_s': 0.0})
    sim = sim_with(cfg)
    mover = sim.make_drop(100.0, 10.0, 16.0 ** 3)
    still = sim.make_drop(250.0, 100.0, 5.0 ** 3)
    prev = still.shape_a.copy()
    max_change = 0.0
    for _ in range(30):
        sim.step()
        max_change = max(max_change, float(np.abs(still.shape_a - prev).max()))
        prev = still.shape_a.copy()
    assert mover.stretch > 1.3 and still.stretch < 1.05
    assert max_change < 0.02                                  # no per-frame re-randomisation


# Step 8 -- merging ------------------------------------------------------------------
def test_two_touching_drops_become_one_with_conserved_volume():
    cfg = quiet(merge={'enabled': True})
    sim = sim_with(cfg)
    a = sim.make_drop(100.0, 100.0, 6.0 ** 3)
    b = sim.make_drop(106.0, 100.0, 5.0 ** 3)
    v0 = a.volume + b.volume
    sim.step()
    assert len(sim.merge_events) == 1
    ev = sim.merge_events[0]
    assert set(ev['parents']) == {a.drop_id, b.drop_id}
    assert a.drop_id not in sim.drops and b.drop_id not in sim.drops          # parents retired
    child = sim.drops[ev['child']]
    assert child.drop_id not in (a.drop_id, b.drop_id)                        # new id
    assert volume_error(ev) < 1e-9 and abs(child.volume - v0) / v0 < 1e-9
    sim.step()
    assert len(sim.merge_events) == 0 and len(sim.drops) == 1                 # merges only once


def test_spatial_grid_finds_the_same_contacts_as_brute_force():
    rng = np.random.default_rng(0)
    drops = {i: Droplet(i, float(x), float(y), float(r) ** 3)
             for i, (x, y, r) in enumerate(zip(rng.uniform(0, 300, 300), rng.uniform(0, 200, 300),
                                               rng.uniform(2, 10, 300)))}
    rmax = max(d.radius for d in drops.values())
    cand = set(SpatialGrid(2 * rmax).candidate_pairs(drops))
    brute = {(i, j) for i in drops for j in drops if i < j and
             math.hypot(drops[i].x - drops[j].x, drops[i].y - drops[j].y) <= drops[i].radius + drops[j].radius}
    assert brute <= cand


# Step 9 -- persistent trails ------------------------------------------------------------
def test_trail_survives_its_drop_and_decays_on_its_own():
    cfg = quiet(trails={'enabled': True}, motion={'slide_radius_px': 8.0, 'pause_rate_per_s': 0.0})
    sim = sim_with(cfg)
    d = sim.make_drop(150.0, 10.0, 14.0 ** 3)
    for _ in range(20):
        sim.step()
    assert d.trail_id in sim.trails.trails
    tid = d.trail_id
    del sim.drops[d.drop_id]                                  # the drop is gone ...
    s0 = max(p[3] for p in sim.trails.trails[tid].points)
    for _ in range(15):
        sim.step()
    assert tid in sim.trails.trails                           # ... its trail is not
    s1 = max(p[3] for p in sim.trails.trails[tid].points)
    expect = s0 * math.exp(-cfg.trails.decay_per_s * 15 * cfg.dt)
    assert abs(s1 - expect) / expect < 1e-6


# Step 10 -- rivulets -----------------------------------------------------------------
def test_parallel_trails_join_into_a_connected_rivulet_without_global_haze():
    cfg = quiet(trails={'enabled': True}, rivulets={'enabled': True, 'flow_px_s': 0.0},
                motion={'slide_radius_px': 8.0, 'pause_rate_per_s': 0.0, 'wobble_px_s': 0.0})
    sim = sim_with(cfg)
    for k in range(60):
        if k % 4 == 0:
            for x in (150.0, 156.0, 162.0):                   # three close streams
                sim.make_drop(x, 5.0, 14.0 ** 3)
        sim.step()
    assert sim.rivulets.components() >= 1
    lbl_n, lbl = __import__('cv2').connectedComponents(sim.rivulets.mask().astype(np.uint8))
    cols = {c for c in np.unique(lbl[:, 150 // 4 - 1:162 // 4 + 2]) if c}
    assert len(cols) >= 1 and sim.rivulets.mask().mean() < 0.2  # joined, local, not a haze


# Step 11 -- film --------------------------------------------------------------------
def _film_cov(name, frames=120):
    sim = sim_with(preset(name), seed=3)
    sim.warmup(5)
    covs = []
    for _ in range(frames):
        sim.step()
        covs.append(sim.film.coverage() if sim.film else 0.0)
    return float(np.mean(covs)), (sim.film.F if sim.film else None)


def test_heavier_presets_have_more_film_but_not_a_uniform_layer():
    c_steady, _ = _film_cov('steady')
    c_heavy, _ = _film_cov('heavy')
    c_torr, F = _film_cov('torrential')
    assert c_steady < c_heavy < c_torr
    assert c_torr < 0.97 and F.std() > 0.05                   # patchy, not fog


def test_film_update_is_stable():
    sim = sim_with(apply_overrides(preset('torrential'), {'film': {'diffusion_cells2_s': 40.0}}))
    for _ in range(60):
        sim.step()
    assert np.isfinite(sim.film.F).all() and sim.film.F.min() >= 0 and sim.film.F.max() <= 1


# Step 12 -- splashes ------------------------------------------------------------------
def test_splash_spreads_relaxes_ends_and_leaves_water():
    cfg = quiet(splash={'enabled': True}, film={'enabled': True, 'source_per_s': 0.0})
    sim = sim_with(cfg)
    ev = SplashEvent(x=160.0, y=120.0, water_amount=300.0, age=0.0, duration=0.5,
                     initial_radius=4.0, strength=1.0, max_radius=16.0,
                     harm_a=np.full(3, 0.05), harm_b=np.zeros(3),
                     finger_angles=np.array([0.0, 2.0]), finger_widths=np.array([0.1, 0.1]))
    sim.splashes.events.append(ev)
    radii = []
    while sim.splashes.events:
        radii.append(ev.radius(cfg.splash.tau_s))
        sim.step()
    grow = radii[:int(0.7 * len(radii))]                                   # until the transfer stage
    assert all(b >= a for a, b in zip(grow, grow[1:])) and max(radii) > 0.9 * 16.0
    assert radii[-1] < max(radii)                                          # then recedes
    assert len(radii) <= int(0.5 * cfg.fps) + 2                         # temporary
    assert len(sim.drops) >= 1 or sim.film.F.max() > 0                   # leaves water behind


def test_splashes_respect_minimum_spacing():
    cfg = quiet(splash={'enabled': True, 'ambient_rate_per_s': 3000.0, 'min_spacing_px': 60.0})
    sim = sim_with(cfg)
    sim.step()
    ev = sim.splashes.events
    for i in range(len(ev)):
        for j in range(i + 1, len(ev)):
            assert math.hypot(ev[i].x - ev[j].x, ev[i].y - ev[j].y) >= 60.0


# Step 13 -- maps follow the state smoothly -------------------------------------------------
def test_small_state_change_gives_small_map_change():
    sim = sim_with(quiet())
    d = sim.make_drop(160.0, 120.0, 10.0 ** 3)
    d.opacity = 1.0
    m0 = sim.render_maps().height.clone()
    d.x += 0.25                                               # sub-pixel move
    m1 = sim.render_maps().height
    assert 0 < (m1 - m0).abs().max().item() < 0.25 * 10.0 * sim.cfg.shape.cap_height


# Step 15 -- presets: deterministic and different -------------------------------------------
def test_same_seed_same_simulation():
    a, b = sim_with(preset('heavy'), seed=7), sim_with(preset('heavy'), seed=7)
    for _ in range(40):
        a.step()
        b.step()
    sa, sb = a.snapshot(), b.snapshot()
    assert sa['drops'] == sb['drops'] and sa['trails'] == sb['trails']


@pytest.mark.parametrize('lo,hi', [('light', 'steady'), ('steady', 'heavy'), ('heavy', 'torrential')])
def test_heavier_presets_get_more_water(lo, hi):
    def cov(name):
        sim = sim_with(preset(name), seed=2)
        sim.warmup(5)
        vals = []
        for _ in range(30):
            sim.step()
            vals.append(float((sim.render_maps().alpha > 0.05).float().mean()))
        return np.mean(vals), sim.cfg.spawn.base_rate_per_s * sim.cfg.intensity
    (c_lo, r_lo), (c_hi, r_hi) = cov(lo), cov(hi)
    assert r_lo < r_hi and c_lo < c_hi


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        apply_overrides(preset('steady'), {'merge': {'contact_frac': 1.5}})
    with pytest.raises(ValueError):
        apply_overrides(preset('steady'), {'no_such_group': {}})


# ======================== fixes: impacts, splashes, film intensity, composition ==========
def test_impacts_use_both_radii_and_conserve_water():
    """Arrival touching a drop's edge (d <= r_i + r_j) is a hit; every arriving volume is
    accounted for exactly once; splashes sit at real impact points."""
    cfg = quiet(spawn={'base_rate_per_s': 3000.0}, splash={'enabled': True, 'impact_prob': 1.0, 'dry_factor': 1.0},
                motion={'slide_radius_px': 1e6})
    sim = sim_with(cfg)
    impacts = []
    for _ in range(100):
        before = len(sim.splashes.events)
        sim.step()
        new = sim.splashes.events[before:] if len(sim.splashes.events) > before else []
        log = sim.arrival_log
        assert sim.stats['arrivals'] == len(log)
        for ev in [e for e in sim.splashes.events if e.age <= sim.dt + 1e-9]:
            assert any(abs(ev.x - x) < 1e-9 and abs(ev.y - y) < 1e-9 and sp for x, y, _, _, sp in log)
        impacts.append((sim.stats['arrivals'], sim.stats['drop_hits'], sim.stats['impact_events']))
        w = sim.water
        assert abs(w['arrived'] - (w['to_drops'] + w['to_splash'])) <= 1e-9 * max(w['arrived'], 1)
    a, h, i = np.sum(impacts, 0)
    assert a > 0 and h > 0 and i > 0
    # splash water: budget = given to drops + film + still held by active events
    led = sim.splashes.ledger
    held = sum(e.water_amount for e in sim.splashes.events)
    assert abs(led['budget'] - (led['to_drops'] + led['to_film'] + held)) <= 1e-6 * max(led['budget'], 1)


def test_arrival_at_drop_edge_is_a_hit_and_same_step_arrivals_see_new_drops():
    cfg = quiet(spawn={'base_rate_per_s': 0.0}, motion={'slide_radius_px': 1e6})
    sim = sim_with(cfg)
    d = sim.make_drop(100.0, 100.0, 6.0 ** 3)
    # arrival of radius 3 at distance 8.5: outside r_i = 6 (old test missed it), inside r_i + r_j = 9
    seq = iter([(108.5, 100.0, 3.0), (200.0, 50.0, 4.0), (202.0, 50.0, 2.5)])
    sim.rain.sample = lambda rng: next(seq)[:2]
    radii = iter([3.0, 4.0, 2.5])
    sim._sample_radius = lambda: next(radii)
    class FixedCount:                                     # exactly 3 arrivals this step
        def __init__(self, rng):
            self._rng = rng

        def poisson(self, lam):
            return 3

        def __getattr__(self, name):
            return getattr(self._rng, name)
    sim.rng = FixedCount(sim.rng)
    sim.cfg.spawn.base_rate_per_s = 1.0
    v0 = d.volume
    sim._arrivals()
    log = sim.arrival_log
    assert log[0][3] == d.drop_id and abs(d.volume - (v0 + 27.0)) < 1e-9      # edge contact = hit
    assert log[1][3] == -1                                                     # new drop on dry glass
    assert log[2][3] != -1 and log[2][3] != d.drop_id                          # ... seen by the next arrival


def _single_impact(rmax_drop=8.0, frames=40, save_dir=None):
    cfg = quiet(splash={'enabled': True}, film={'enabled': True, 'source_per_s': 0.0})
    sim = sim_with(cfg)
    vol = rmax_drop ** 3
    ev = sim.splashes.trigger(sim.rng, 160.0, 120.0, vol, 1.0, 0.6 * vol)
    rows = []
    for f in range(frames):
        sim.step()                                        # same order as the simulator: update, then draw
        h, a = sim.splashes.rasterize(H, W, sim.dt) if sim.splashes.events else (np.zeros((H, W)), np.zeros((H, W)))
        rows.append((float((a > 0.05).sum()), float(h.max()), float(a.sum()), len(sim.splashes.events)))
        if save_dir is not None and f in (0, 2, 5, 10, 15, 20, 25):
            import cv2
            cv2.imwrite(os.path.join(save_dir, f'splash_t{f:02d}_alpha.png'), (np.clip(a, 0, 1) * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(save_dir, f'splash_t{f:02d}_height.png'),
                        (np.clip(h / max(h.max(), 1e-6), 0, 1) * 255).astype(np.uint8))
    return sim, ev, rows


def test_single_impact_expands_relaxes_and_leaves_water(tmp_path):
    sim, ev, rows = _single_impact(save_dir=str(tmp_path))
    vis = [r for r in rows if r[3]]
    areas, water = [r[0] for r in vis], [r[2] for r in vis]
    peak = int(np.argmax(areas))
    assert 0 < areas[0] < 0.35 * areas[peak]               # appears small (impact) ...
    assert all(b >= a - 1e-6 for a, b in zip(areas[:peak], areas[1:peak + 1]))   # ... expands
    assert areas[-1] < 0.2 * areas[peak]                   # nearly gone before removal: no pop
    wsteps = np.abs(np.diff(water[2:])) / max(max(water), 1e-6)
    assert wsteps.max() < 0.25                             # visible water changes smoothly
    assert not sim.splashes.events                         # temporary
    assert len(sim.drops) >= 1 or sim.film.F.sum() > 0      # residual water remains
    led = sim.splashes.ledger
    assert abs(led['budget'] - led['to_drops'] - led['to_film']) < 1e-6 * led['budget']
    assert len(list(tmp_path.glob('splash_t*_alpha.png'))) == 7


def test_splash_height_scales_with_size_but_is_capped():
    s_small, ev_s, _ = _single_impact(rmax_drop=4.0, frames=1)
    s_big, ev_b, _ = _single_impact(rmax_drop=12.0, frames=1)
    s_huge, ev_h, _ = _single_impact(rmax_drop=60.0, frames=1)
    hs, hb, hh = (s.splashes.peak_height(e) for s, e in ((s_small, ev_s), (s_big, ev_b), (s_huge, ev_h)))
    assert hs < hb <= hh <= s_huge.cfg.splash.height * s_huge.cfg.splash.height_cap + 1e-9


def test_film_source_follows_global_intensity():
    """Same initial film and rain pattern; only the global intensity differs."""
    from windshield_water.film import FilmField
    cfg = preset('steady')
    rain = sim_with(cfg).rain.resized(*FilmField(cfg, H, W).F.shape)
    runs = {}
    for I in (1.0, 2.0):
        f = FilmField(cfg, H, W)
        cov = []
        for _ in range(60):
            f.update(cfg.dt, rain, intensity=I)
            cov.append(f.coverage())
        runs[I] = (f.F.mean(), cov)
    assert runs[2.0][0] > 1.5 * runs[1.0][0]                       # accumulates faster
    assert runs[2.0][1][-1] > runs[1.0][1][-1]                     # reaches more coverage


def _composite(case, mode):
    cfg = quiet(render={'composition': mode}, trails={'enabled': True}, splash={'enabled': True},
                film={'enabled': True, 'source_per_s': 0.0}, rivulets={'enabled': True})
    sim = sim_with(cfg)
    d = sim.make_drop(160.0, 120.0, 12.0 ** 3)
    d.opacity = 1.0
    if case == 'trail':
        sim.trails.trails[0] = __import__('windshield_water.trails', fromlist=['Trail']).Trail(
            0, -1, [[160.0, 60.0, 4.0, 1.0], [160.0, 180.0, 4.0, 1.0]])
    if case == 'splash':
        sim.splashes.trigger(sim.rng, 165.0, 120.0, 10.0 ** 3, 1.0, 300.0)
        sim.splashes.events[0].age = 0.1
    if case == 'rivulet':
        sim.rivulets.F[:, 160 // 4 - 1:160 // 4 + 2] = 1.0
    if case == 'film':
        sim.film.F[:] = 0.5
        e = sim.make_drop(200.0, 120.0, 8.0 ** 3)
        e.opacity = 1.0
    return sim, sim.layer_maps(), sim.render_maps()


@pytest.mark.parametrize('case', ['trail', 'splash', 'rivulet', 'film'])
def test_layered_composition_keeps_the_layer_under_a_drop(case):
    _, layers_m, m_max = _composite(case, 'max')
    _, layers_l, m_lay = _composite(case, 'layered')
    drop_h = layers_l['drops'][0]
    inside = drop_h > 0.5 * drop_h.max()
    other = [k for k in layers_l if k != 'drops']
    thin = sum(layers_l[k][0] for k in other)
    under = inside & (thin > 1e-3)
    assert under.any()
    # max-composition hides the thin layer under the drop; layered keeps it
    assert torch.allclose(m_max.height[under], drop_h[under], atol=1e-5)
    assert (m_lay.height[under] > drop_h[under] + 1e-4).all()
    # opacity: 1 - prod(1 - a) is at least the strongest layer, more where layers overlap
    amax = torch.stack([a for _, a in layers_l.values()]).max(0).values
    assert (m_lay.alpha >= amax - 1e-6).all()


def test_isolated_drop_is_identical_under_both_compositions():
    cfg_m, cfg_l = quiet(render={'composition': 'max'}), quiet(render={'composition': 'layered'})
    out = []
    for cfg in (cfg_m, cfg_l):
        sim = sim_with(cfg)
        d = sim.make_drop(160.0, 120.0, 10.0 ** 3)
        d.opacity = 1.0
        out.append(sim.render_maps())
    assert torch.allclose(out[0].height, out[1].height) and torch.allclose(out[0].alpha, out[1].alpha)
