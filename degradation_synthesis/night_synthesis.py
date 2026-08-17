import os
import sys
import glob
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1. VDA WRAPPER  (identical to fog_synthesis.py)# ---------------------------------------------------------------------------
class VDAWrapper:
    def __init__(self, encoder='vits', vda_root_path=None, device_id=0):
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        sys.path.append(vda_root_path)
        from video_depth_anything.video_depth import VideoDepthAnything
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48,  96,  192,  384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        self.model = VideoDepthAnything(**model_configs[encoder])
        ckpt_path = os.path.join(vda_root_path, 'checkpoints',
                                 f'metric_video_depth_anything_{encoder}.pth')
        self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.model = self.model.to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def infer_batch(self, frame_batch_rgb):
        h_orig, w_orig = frame_batch_rgb[0].shape[:2]
        target_h, target_w = (h_orig // 14) * 14, (w_orig // 14) * 14
        frames_tensor = [
            torch.from_numpy(cv2.resize(img, (target_w, target_h)))
                 .permute(2, 0, 1).float() / 255.0
            for img in frame_batch_rgb
        ]
        batch_tensor = (torch.stack(frames_tensor).unsqueeze(0).to(self.device)
                        - self.mean) / self.std
        with torch.no_grad():
            depth_pred = self.model(batch_tensor)
            if isinstance(depth_pred, (list, tuple)):
                depth_pred = depth_pred[-1]
            depth_out = depth_pred.squeeze().cpu().numpy()
        if depth_out.ndim == 2:
            depth_out = depth_out[np.newaxis, ...]
        return np.array([cv2.resize(d, (w_orig, h_orig)) for d in depth_out])


# ---------------------------------------------------------------------------
# 2. ILLUMINANT GAINS
# ---------------------------------------------------------------------------
# Per-illuminant RGB gains (relative to daylight D65).
# Normalised so max channel = 1.0 to prevent clipping.
#   tungsten (3200K) → warm orange        sodium (2200K) → strong amber-orange
#   led      (6000K) → neutral/cool       moonlit        → faint grey-blue
# Calibrated against real Dark_Driving night footage (see calibration note
# in apply_night, step 4) — not a free knob, keep it tied to that measurement
# if retuned.
SENSOR_NOISE_GAIN = 0.04

ILLUMINANT_GAINS = {
    'tungsten':  np.array([1.30, 1.00, 0.60], dtype=np.float32),
    'sodium':    np.array([1.30, 1.04, 0.58], dtype=np.float32),   # toned down: amber, not orange
    'led':       np.array([0.88, 1.00, 1.20], dtype=np.float32),
    'moonlit':   np.array([0.90, 1.00, 1.14], dtype=np.float32),
    # Residual atmospheric warmth — not a headlight colour, an AMBIENT one:
    # right after sunset the sky/haze still carries leftover warm scatter
    # before it settles into the cool blue-black of full night. Much milder
    # swing than tungsten (this is atmosphere, not a light bulb).
    'dusk_glow': np.array([1.08, 1.02, 0.85], dtype=np.float32),
}


# ---------------------------------------------------------------------------
# 2b. DISCRETE LIGHT SOURCES (oncoming headlights, streetlamps, ...)
# ---------------------------------------------------------------------------
def _discrete_lights_map(h, w, lights):
    """
    Composite point/area light sources into an [H, W] illuminance map.

    Each light: {'x', 'y'} in [0,1] image-fraction coords, 'radius' in
    image-width fraction, 'strength' in the same rough units as headlight
    peak (~1). Positions/strengths are meant to be sampled ONCE per clip
    (see sample_variant_params) so a light stays put for the whole
    sequence — a streetlamp doesn't teleport frame to frame, and an
    oncoming car's glare persists over several consecutive frames.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx /= w
    yy /= h
    out = np.zeros((h, w), dtype=np.float32)
    for lt in lights:
        dx = (xx - lt['x']) / max(lt['radius'], 1e-3)
        dy = (yy - lt['y']) / max(lt['radius'] * 0.6, 1e-3)   # squashed ellipse
        out += lt['strength'] * np.exp(-0.5 * (dx ** 2 + dy ** 2))
    return out


# ---------------------------------------------------------------------------
# 3. NIGHT ILLUMINATION ENGINE
# ---------------------------------------------------------------------------
class NightEngine:
    """
    Physically-grounded ego-centric low-light synthesis.

    The governing equation mirrors fog's Koschmieder law but for illuminance:

        I_night(x) = ISP[ Poisson(η · L_total(x) · t_exp) + N_read ]

    where:
        L_total(x) = L_ambient + E_headlight(d(x), θ(x))   [replaces fog depth term]
        E_headlight ∝  beam_pattern(θ) / (1 + c·d²)        [inverse-square + beam lobe]

    The depth map from VDA feeds both:
      1. headlight falloff  → how bright each pixel is illuminated
      2. noise variance     → darker (deeper/off-beam) pixels are noisier
    """

    def __init__(self,
                 ambient       = 0.14,   # fraction of daylight from sky/street ambient
                 headlight_str = 0.25,   # peak headlight contribution on top of ambient
                 desaturation  = 0.15,   # scotopic colour wash-out (0=vivid, 1=grey)
                 haze_strength = 0.0,    # horizon glow (0=off)
                 illuminant    = 'tungsten',
                 beam_center   = 0.12,   # lobe offset from centre, frac of half-width
                 beam_width    = 0.28,   # lobe width, frac of half-width
                 beam_asym     = 0.40,   # SAE right/left beam asymmetry strength
                 max_range_m   = 80.0,   # depth clamp for headlight falloff
                 lights        = None,   # discrete sources: [{'x','y','radius','strength'}, ...]
                 ambient_warmth = 0.0,   # 0=full cool moonlit night, 1=strong residual dusk glow
                 bloom_strength = 0.0):  # lens-scatter halo ring around bright sources (0=off)

        self.ambient        = ambient
        self.headlight_str  = headlight_str
        self.desaturation   = desaturation
        self.haze_strength  = haze_strength
        self.illuminant     = illuminant
        self.beam_center     = beam_center
        self.beam_width      = beam_width
        self.beam_asym       = beam_asym
        self.max_range_m     = max_range_m
        self.lights          = lights or []
        self.ambient_warmth  = ambient_warmth
        self.bloom_strength  = bloom_strength
        gains               = ILLUMINANT_GAINS[illuminant]
        self.illum_gains    = gains / gains.max()   # normalise: max channel = 1.0

        moonlit_gain  = ILLUMINANT_GAINS['moonlit']   / ILLUMINANT_GAINS['moonlit'].max()
        dusk_gain     = ILLUMINANT_GAINS['dusk_glow'] / ILLUMINANT_GAINS['dusk_glow'].max()
        w             = np.clip(ambient_warmth, 0.0, 1.0)
        self.ambient_gain = (1.0 - w) * moonlit_gain + w * dusk_gain   # RGB order

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headlight_map(self, depth_m, h, w):
        """
        Per-pixel illuminance from ego headlights + any discrete light
        sources (oncoming glare, streetlamps).

        E(x,y) = B_h(x) · B_v(y) · F(d(x,y))  +  Σ discrete lights

        Two fixes vs. the original version:
          1. Beam geometry (center/width/asymmetry) is instance-level, not
             hardcoded — sample_variant_params() jitters it per clip so the
             mask isn't a literal constant array shared by every frame in
             the dataset.
          2. Range falloff is normalised by a FIXED reference value (its
             theoretical max at the 0.5 m clamp), not by this frame's own
             min/max. Per-frame min-max stretch was throwing away absolute
             depth scale — every frame got rescaled to look equally "lit",
             regardless of whether the scene actually had anything close.
        """
        x = (np.arange(w) - w * 0.5) / (w * 0.5)   # [-1, 1]  left→right

        # Two Gaussian lobes, offset/width jittered per clip
        beam_h = np.maximum(
            np.exp(-0.5 * ((x + self.beam_center) / self.beam_width) ** 2),
            np.exp(-0.5 * ((x - self.beam_center) / self.beam_width) ** 2),
        )
        # SAE asymmetry: right side brighter, left side (oncoming lane) dimmer
        asym   = np.clip(1.0 + self.beam_asym * x, 0.70, 1.35)
        beam_h = beam_h * asym                        # [W]

        # Vertical: horizon near 40% from top.  y=-0.6=sky, y=+1.0=near road
        y      = np.linspace(-0.6, 1.0, h).reshape(h, 1)
        beam_v = np.exp(-0.5 * (y / 0.55) ** 2)      # [H, 1]

        beam_pattern = beam_h * beam_v                # [H, W]

        # Range falloff: 1/(1 + c·d²), clamped to [0.5 m, max_range_m].
        # Normalised by its OWN fixed theoretical peak (value at d=0.5 m),
        # not by min/max of this frame — so absolute scene depth matters.
        d        = np.clip(depth_m, 0.5, self.max_range_m)
        falloff  = 1.0 / (1.0 + 0.06 * d ** 2)
        falloff  = falloff / (1.0 / (1.0 + 0.06 * 0.5 ** 2))

        head = beam_pattern * falloff

        # Discrete sources (oncoming glare / streetlamps) — content-independent
        # position but genuinely different from clip to clip.
        if self.lights:
            head = head + _discrete_lights_map(h, w, self.lights)

        return np.clip(head, 0.0, None).astype(np.float32)

    def _alpha_components(self, depth_m, h, w):
        """
        Illuminance split into its two sources, kept separate (not summed)
        so the caller can colour them differently:
          alpha_amb  — sky/street ambient glow: physically closer to neutral
                       (moonlight, sky glow, distant unfocused city lights),
                       NOT the warm colour of a halogen headlight.
          headlight  — ego headlight beam: genuinely warm (halogen/tungsten).

        Fix: ambient has a vertical sky gradient. In real urban night, sky
        glow (light pollution, moon) makes the upper sky ~35% brighter than
        road level, while headlights illuminate the lower (road) portion.
        """
        y_norm    = np.linspace(0.0, 1.0, h).reshape(h, 1)   # 0=top, 1=bottom
        sky_fac   = 1.0 + 0.35 * (1.0 - y_norm)              # sky 35% above floor
        alpha_amb = self.ambient * sky_fac                    # [H, 1]

        headlight = self._headlight_map(depth_m, h, w)       # [H, W]
        return alpha_amb, headlight

    def _alpha_map(self, depth_m, h, w):
        """Total scene illuminance α(x,y) ∈ [0,1] relative to daylight."""
        alpha_amb, headlight = self._alpha_components(depth_m, h, w)
        alpha = alpha_amb + self.headlight_str * headlight
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_night(self, frame_bgr, depth_m, rng=None):
        """
        Ego-centric night degradation with sensor simulation.

        Pipeline (all in linear light until step 4):
          1.  α(x,y)   sky_gradient·α_amb + headlight_str·E_head(d)
          2.  linearise sRGB ^ 2.2
          3.  darken    linear × α × g_illum
          4.  ISO noise spatially-correlated shot + read + (weak) chroma noise
              — applied in LINEAR domain before gamma (physically correct)
              — GT is the clean daylight frame; noisy input is what the model sees
          5.  display   noisy_linear ^ (1/2.2)
          6.  bloom     two-scale (wide scatter + narrow glare) with halo desaturation
          7.  haze      additive warm band at horizon (only if haze_strength > 0)
          8.  shadow    detail loss: heavy blur dominates deep shadow, faint residual grain
              — not just contrast reduction; genuine irreversible detail loss
          9.  purkinje  dark regions shift cyan-ward
        """
        if rng is None:
            rng = np.random.default_rng()

        h, w = frame_bgr.shape[:2]
        img  = frame_bgr.astype(np.float32) / 255.0

        # 1. Alpha map, split by source so colour can be applied per-source
        alpha_amb, headlight = self._alpha_components(depth_m, h, w)   # both broadcastable to (H, W)
        headlight_term = self.headlight_str * headlight
        alpha  = np.clip(alpha_amb + headlight_term, 0.0, 1.0).astype(np.float32)   # (H, W) total
        alpha3 = alpha[:, :, np.newaxis]                                             # (H, W, 1)

        # 2–3. Linearise → darken → illuminant colour shift
        #
        # Fix: the colour cast used to be a single flat gain applied to the
        # WHOLE frame regardless of what's actually lighting each pixel —
        # so sky and foliage that get zero headlight (pure ambient) were
        # still tinted the full tungsten-warm colour. At low ambient that's
        # invisible under the darkness, but once ambient increases
        # (moderate/low_light) it reads as "a colour filter over a normal
        # photo", not genuine low light. Real ambient night light (sky glow,
        # moonlight, unfocused distant lights) is close to neutral; only the
        # ego headlight beam is actually tungsten-warm. So: blend per pixel
        # between neutral (1,1,1) and illum_gains, weighted by how much of
        # that pixel's illumination actually comes from the headlight.
        img_lin = np.power(np.maximum(img, 1e-8), 2.2)

        # Ambient isn't pure neutral either — real night sky/ambient reads as
        # a subtle tone somewhere between cool (moonlight, full night) and
        # warm (residual dusk scatter, not long after sunset) — how far
        # along that axis is `ambient_warmth`, sampled per CLIP (see
        # sample_variant_params) so different sequences land at different
        # points instead of the whole dataset sharing one fixed sky colour.
        color_mix     = (headlight_term / (alpha_amb + headlight_term + 1e-6))[:, :, np.newaxis]  # (H, W, 1)
        ambient_gain  = self.ambient_gain[::-1]   # RGB → BGR
        pixel_gain    = (1.0 - color_mix) * ambient_gain + color_mix * self.illum_gains[::-1]      # (H, W, 3)

        img_dark = img_lin * alpha3 * pixel_gain

        # 4. ISO-calibrated sensor noise (in linear domain — physically correct)
        #    Camera auto-ISO boosts gain to compensate for dark scene.
        #
        #    Recalibrated, two changes from the original version:
        #      (a) magnitude — old sigmas were several times LARGER than the
        #          signal itself in these near-zero-ambient presets.
        #      (b) colour — noise used to be drawn fully independently per
        #          R/G/B channel, so every pixel got its own random
        #          saturated colour ("RGB confetti"). Real high-ISO sensor
        #          noise (and whatever chroma subsampling a video codec
        #          applies on top) is overwhelmingly LUMINANCE noise — one
        #          shared noisy value per pixel across channels, not three
        #          independent ones. So: generate ONE noise field from the
        #          luma-weighted signal and broadcast it to all channels,
        #          then add only a small independent per-channel residual
        #          for subtle colour jitter. That turns speckle into grain.
        scene_brightness = float(np.mean(alpha))
        iso_factor = np.clip(1.0 / (scene_brightness + 0.04), 1.0, 24.0)

        luma_w      = np.array([0.114, 0.587, 0.299], dtype=np.float32)[::-1]  # BGR
        signal_luma = np.sum(img_dark * luma_w, axis=2, keepdims=True)         # (H, W, 1)

        # Shot noise: σ² ∝ signal (Poisson), approx Gaussian for large counts
        gain            = max(3.0, 14.0 / iso_factor)
        shot_sigma_luma = np.sqrt(np.maximum(signal_luma, 0.0) / gain).astype(np.float32)
        shot_noise_luma = rng.normal(0.0, 1.0, shot_sigma_luma.shape).astype(np.float32) * shot_sigma_luma
        shot_noise      = np.repeat(shot_noise_luma, 3, axis=2)   # achromatic — shared across channels

        # Read noise: ISO-dependent Gaussian floor (independent of signal), also achromatic
        read_sigma      = 0.0025 * np.sqrt(iso_factor)
        read_noise_luma = rng.normal(0.0, read_sigma, shot_sigma_luma.shape).astype(np.float32)
        read_noise      = np.repeat(read_noise_luma, 3, axis=2)

        # Weak independent-per-channel residual — the only source of colour
        # in the noise, deliberately a small fraction of the luma term.
        chroma_sigma = 0.15 * shot_sigma_luma
        chroma_noise = rng.normal(0.0, 1.0, img_dark.shape).astype(np.float32) * chroma_sigma

        # SENSOR_NOISE_GAIN: measured against real Dark_Driving night footage
        # (high-freq residual std in flat dark regions ≈ 2-4 / 255 there —
        # real dashcams are near noise-free thanks to in-camera + video-codec
        # denoising). The "physically derived" sigmas above, taken at face
        # value, overshoot that by ~6-7x. This single calibrated scale pulls
        # the whole noise stack back down to match the real reference instead
        # of an idealised, un-denoised sensor.
        img_noisy = np.clip(img_dark + SENSOR_NOISE_GAIN * (shot_noise + read_noise + chroma_noise), 0.0, 1.0)

        # 5. Back to display gamma
        img_out    = np.power(np.clip(img_noisy, 1e-8, 1.0), 1.0 / 2.2)
        alpha_disp = np.power(np.clip(alpha3,    1e-6, 1.0), 1.0 / 2.2)

        # 6. Two-scale bloom: wide (lens scatter) + narrow (glare ring) — OFF by
        #    default (bloom_strength=0). This was the "halo ring" around bright
        #    sources — removed on request, but note it never touched alpha/
        #    darkening, so headlight/ambient intensity is unaffected either way.
        if self.bloom_strength > 0.0:
            bright_src   = np.maximum(img_out - 0.75, 0.0).astype(np.float32)
            bloom_wide   = cv2.GaussianBlur(bright_src * 3.0, (101, 101), 30)
            bloom_narrow = cv2.GaussianBlur(bright_src * 2.0, (21,  21),   6)
            bloom_total  = self.bloom_strength * (0.12 * bloom_wide + 0.10 * bloom_narrow)

            # Halo around bright sources is achromatic (lens scatter strips colour)
            bloom_luma = bloom_total.mean(axis=2, keepdims=True)
            img_out    = img_out + bloom_total * (1.0 - alpha_disp)
            # Desaturate the halo area
            img_gray   = np.sum(img_out * luma_w, axis=2, keepdims=True)
            desat_w    = np.clip(bloom_luma * 4.0, 0.0, 0.6)
            img_out    = img_out * (1.0 - desat_w) + img_gray * desat_w

        # 7. Horizon haze (only if haze_strength > 0 for this variant)
        if self.haze_strength > 0.0:
            y_norm_h  = np.arange(h, dtype=np.float32) / h
            y_dist    = (y_norm_h - 0.40) / 0.12
            haze_mask = np.exp(-0.5 * y_dist ** 2).reshape(h, 1, 1)
            haze_col  = np.array([0.18, 0.42, 0.78], dtype=np.float32)
            img_out   = img_out + self.haze_strength * haze_mask * haze_col

        # 8. Deep-shadow detail loss
        #    Two regimes:
        #      soft shadow (α_disp 0.1–0.3): mild contrast reduction
        #      deep shadow (α_disp < 0.1)  : real cameras' in-ISP noise
        #      reduction dominates here — heavy blur wins, with only a
        #      faint residual grain on top (not a fresh full-strength
        #      noise layer piled onto the already-noisy pixels, which is
        #      what the old 0.5/0.3/0.2 blend was doing — it kept half the
        #      raw speckle AND added more on top instead of cleaning it up).
        soft_w  = np.clip((0.30 - alpha_disp) / 0.20, 0.0, 1.0) * 0.30
        deep_w  = np.clip((0.10 - alpha_disp) / 0.10, 0.0, 1.0)

        blur_soft   = cv2.GaussianBlur(img_out.astype(np.float32), (9,  9), 3)
        blur_deep   = cv2.GaussianBlur(img_out.astype(np.float32), (21, 21), 8)
        grain_luma  = rng.normal(0.0, 0.010, img_out.shape[:2] + (1,)).astype(np.float32)
        grain       = np.repeat(grain_luma, 3, axis=2)   # achromatic, matches step 4

        img_out = (img_out  * (1.0 - soft_w) + blur_soft * soft_w)
        img_out = (img_out  * (1.0 - deep_w * 0.85)
                   + blur_deep * (deep_w * 0.85)
                   + grain     * (deep_w * 0.10))

        # 9. Purkinje hue shift (unchanged)
        δ            = self.desaturation * (1.0 - alpha_disp[..., 0])
        img_purkinje = img_out.copy()
        img_purkinje[..., 2] = img_out[..., 2] * (1.0 - 0.4  * δ)
        img_purkinje[..., 0] = img_out[..., 0] * (1.0 + 0.20 * δ)
        δ3      = δ[:, :, np.newaxis]
        img_out = (1.0 - δ3) * img_out + δ3 * img_purkinje

        return np.clip(img_out * 255.0, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. CONDITIONS
# ---------------------------------------------------------------------------
# All three are "hard, genuinely dark" night driving — same HEADLIGHT
# illuminant (tungsten, physically correct for halogen headlamps), varying
# only in SEVERITY: how dark, how far the headlights reach, and whether
# there's oncoming glare. Deliberately no discrete colour-temperature
# PRESET sweep (no golden_hour/dusk/twilight/sodium-streetlamp variants) —
# those were strongly different, classifiable personas a model could
# shortcut-learn. (The ambient — not headlight — tone does still get a
# continuous, mild per-clip nudge between cool/full-night and residual
# dusk warmth via `ambient_warmth`; that's real atmospheric variation, not
# a fixed branded look, and doesn't touch darkness level or headlight colour.)
#
#   extreme_unlit     — worst case: unlit rural road, weak headlights, nothing else lit
#   extreme_headlight — unlit road, normal headlights (reference case)
#   extreme_glare     — same darkness as reference + 1-2 oncoming headlight glare sources
EXTREME_VARIANTS = {
    'extreme_unlit':     {'ambient': 0.005, 'headlight_str': 0.14, 'illuminant': 'tungsten',
                           'desaturation': 0.42, 'haze_strength': 0.0, 'n_lights': 0},
    'extreme_headlight': {'ambient': 0.01,  'headlight_str': 0.16, 'illuminant': 'tungsten',
                           'desaturation': 0.38, 'haze_strength': 0.0, 'n_lights': 0},
    'extreme_glare':      {'ambient': 0.01,  'headlight_str': 0.16, 'illuminant': 'tungsten',
                           'desaturation': 0.38, 'haze_strength': 0.0, 'n_lights': 2,
                           'light_strength_range': (1.5, 3.0)},
}

# Two lighter severity tiers, same family (tungsten, no colour sweep) as
# EXTREME_VARIANTS — for comparison against it, not yet wired into
# create_night_data.py. Display-mean targets were calibrated empirically
# against real Dark_Driving night frames (measured mean ≈ 41-50 there):
#   extreme_headlight (existing) → mean ≈ 11   (deliberately harsher than real)
#   moderate           (new)     → mean ≈ 23   (roughly halfway)
#   low_light          (new)     → mean ≈ 44   (matches real Dark_Driving average)
MODERATE_VARIANTS = {
    'moderate':  {'ambient': 0.05, 'headlight_str': 0.30, 'illuminant': 'tungsten',
                  'desaturation': 0.28, 'haze_strength': 0.0, 'n_lights': 0},
    'low_light': {'ambient': 0.20, 'headlight_str': 0.40, 'illuminant': 'tungsten',
                  'desaturation': 0.18, 'haze_strength': 0.0, 'n_lights': 0},
}


def sample_variant_params(base, rng):
    """
    Sample ONE clip's worth of engine params from a base condition (an entry
    of EXTREME_VARIANTS or MODERATE_VARIANTS).

    Same "identity" (illuminant, difficulty tier) as `base`, but different
    exact numbers — ambient level, headlight strength, beam geometry, and
    (for extreme_glare) discrete light positions — each time it's called.
    That's what stops a model from shortcut-learning one fixed vignette:
    no two clips share an identical illumination mask.

    Call this ONCE PER CLIP, not per frame. Build a single NightEngine from
    the result and reuse it for every frame of that clip — lighting then
    stays temporally coherent across the sequence (a streetlamp doesn't
    move frame to frame), while only the per-frame sensor noise (passed
    separately via `rng` to apply_night) varies frame to frame.
    """
    p = dict(base)
    p['ambient']       = max(0.0, base['ambient']       * rng.uniform(0.7,  1.3))
    p['headlight_str'] = max(0.0, base['headlight_str'] * rng.uniform(0.75, 1.25))
    p['beam_center']   = 0.12 * rng.uniform(0.5, 1.5)
    p['beam_width']    = 0.28 * rng.uniform(0.8, 1.25)
    p['beam_asym']     = 0.40 * rng.uniform(0.7, 1.3)

    # How far past sunset this clip is: mostly full-night cool (low values),
    # occasionally more residual dusk warmth (higher values). This is NOT
    # the golden_hour/dusk/twilight preset sweep we deliberately removed —
    # those were discrete, strongly-different, headlight-affecting presets
    # a model could classify and shortcut. This only nudges the AMBIENT
    # (non-headlight) tone continuously and mildly per clip; the headlight
    # colour and the darkness level are unaffected.
    p['ambient_warmth'] = rng.uniform(0.0, 0.5) ** 1.5   # skewed toward cool/low

    n_lights       = p.pop('n_lights', 0)
    strength_range = p.pop('light_strength_range', (0.8, 2.2))
    lights = []
    for _ in range(rng.integers(0, n_lights + 1)):
        lights.append({
            'x':        rng.uniform(0.1, 0.9),
            'y':        rng.uniform(0.15, 0.55),
            'radius':   rng.uniform(0.03, 0.10),
            'strength': rng.uniform(*strength_range),
        })
    p['lights'] = lights
    return p


# Full realistic range the continuous sampler below draws from — floor
# matches extreme_unlit, ceiling matches low_light, so the continuum covers
# (and extends slightly past) every named preset above.
_AMBIENT_LOG_RANGE     = (np.log(0.004), np.log(0.28))
_HEADLIGHT_LOG_RANGE   = (np.log(0.10),  np.log(0.55))


def sample_night_clip(rng, illuminant='tungsten'):
    """
    Fully continuous per-clip severity sampling — the replacement for
    picking one of a handful of named presets (EXTREME_VARIANTS /
    MODERATE_VARIANTS) and jittering ±25-30% around it.

    A small number of discrete tiers, even jittered, are still only a small
    number of clusters a model can key off of. This instead draws ambient
    and headlight_str LOG-uniformly across the whole realistic range in one
    shot, so a generated dataset is a smooth spectrum from near-pitch-black
    unlit road to a reasonably-lit low-light street, not 3-5 clumps with
    gaps between them. Desaturation is derived from the sampled darkness
    (physically: scotopic colour wash-out increases as light drops) instead
    of being yet another fixed-per-tier constant. Discrete lights (oncoming
    glare / streetlamps) are present with some probability at ANY severity
    level, not only under one named "glare" condition — real traffic
    density doesn't correlate with how dark the road is.

    Call ONCE PER CLIP; build one NightEngine from the result and reuse it
    for every frame in that clip (temporal coherence) — same contract as
    sample_variant_params.
    """
    ambient       = float(np.exp(rng.uniform(*_AMBIENT_LOG_RANGE)))
    headlight_str = float(np.exp(rng.uniform(*_HEADLIGHT_LOG_RANGE)))

    beam_center = 0.12 * rng.uniform(0.5, 1.5)
    beam_width  = 0.28 * rng.uniform(0.8, 1.25)
    beam_asym   = 0.40 * rng.uniform(0.7, 1.3)

    # Desaturation scales with sampled darkness: darkest end ≈ 0.42
    # (extreme_unlit's old constant), brightest end ≈ 0.15 (low_light's).
    t = np.clip((np.log(ambient) - _AMBIENT_LOG_RANGE[0])
                / (_AMBIENT_LOG_RANGE[1] - _AMBIENT_LOG_RANGE[0]), 0.0, 1.0)
    desaturation = 0.42 - 0.27 * t

    # Discrete lights at any severity, ~45% of clips get 1-3.
    n_lights = int(rng.integers(1, 4)) if rng.random() > 0.55 else 0
    lights = [{
        'x':        rng.uniform(0.1, 0.9),
        'y':        rng.uniform(0.15, 0.55),
        'radius':   rng.uniform(0.03, 0.10),
        'strength': rng.uniform(0.8, 3.0),
    } for _ in range(n_lights)]

    ambient_warmth = rng.uniform(0.0, 0.5) ** 1.5

    return dict(
        ambient=ambient, headlight_str=headlight_str, illuminant=illuminant,
        desaturation=desaturation, haze_strength=0.0,
        beam_center=beam_center, beam_width=beam_width, beam_asym=beam_asym,
        lights=lights, ambient_warmth=ambient_warmth,
    )


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Ego-centric night synthesis (depth-driven)')
    parser.add_argument('--input',      required=True,  help='Video file or folder of images')
    parser.add_argument('--output',     required=True,  help='Output video file or folder')
    parser.add_argument('--intensity',  default='extreme_headlight', choices=list(EXTREME_VARIANTS.keys()))
    parser.add_argument('--ambient',    type=float, default=None,
                        help='Override ambient level (0.0–1.0)')
    parser.add_argument('--headlight',  type=float, default=None,
                        help='Override headlight strength (0.0–1.0)')
    parser.add_argument('--illuminant', default=None,
                        choices=list(ILLUMINANT_GAINS.keys()),
                        help='Override illuminant colour temperature')
    parser.add_argument('--seed',       type=int, default=None,
                        help='Seed for per-clip variant jitter (beam geometry, lights)')
    parser.add_argument('--encoder',    default='vits', choices=['vits', 'vitl'])
    parser.add_argument('--gpu_id',     type=int, default=0)
    args = parser.parse_args()

    vda_root = '/scratch/Ananya_Kulkarni/Video-Depth-Anything'
    imsize   = (480, 270)   # (W, H) — matches GT in Defog_balanced

    base   = EXTREME_VARIANTS[args.intensity]
    params = sample_variant_params(base, np.random.default_rng(args.seed))
    engine = NightEngine(
        ambient       = args.ambient    if args.ambient    is not None else params['ambient'],
        headlight_str = args.headlight  if args.headlight  is not None else params['headlight_str'],
        illuminant    = args.illuminant if args.illuminant is not None else params['illuminant'],
        desaturation  = params.get('desaturation', 0.15),
        haze_strength = params.get('haze_strength', 0.0),
        beam_center   = params['beam_center'],
        beam_width    = params['beam_width'],
        beam_asym     = params['beam_asym'],
        lights        = params['lights'],
    )

    pipe = VDAWrapper(encoder=args.encoder, vda_root_path=vda_root, device_id=args.gpu_id)

    is_folder = os.path.isdir(args.input)
    if is_folder:
        img_paths = sorted(
            glob.glob(os.path.join(args.input, '*.[jJ][pP][gG]')) +
            glob.glob(os.path.join(args.input, '*.[pP][nN][gG]'))
        )
        os.makedirs(args.output, exist_ok=True)
        total = len(img_paths)
    else:
        cap    = cv2.VideoCapture(args.input)
        fps    = cap.get(cv2.CAP_PROP_FPS)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_v  = cv2.VideoWriter(args.output,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, imsize)

    tag  = f"GPU {args.gpu_id} | {args.intensity} | ambient {engine.ambient:.3f} | head {engine.headlight_str:.2f}"
    pbar = tqdm(total=total, desc=tag)
    batch_size = 4

    for i in range(0, total, batch_size):
        batch_f, batch_n = [], []
        for j in range(i, min(i + batch_size, total)):
            if is_folder:
                img = cv2.imread(img_paths[j])
                batch_n.append(os.path.basename(img_paths[j]))
            else:
                ret, img = cap.read()
                if not ret:
                    break
            if img is not None:
                batch_f.append(cv2.resize(img, imsize))

        if not batch_f:
            break

        depths = pipe.infer_batch([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_f])

        for k in range(len(batch_f)):
            result = engine.apply_night(batch_f[k], depths[k])
            if is_folder:
                cv2.imwrite(os.path.join(args.output, batch_n[k]), result)
            else:
                out_v.write(result)
        pbar.update(len(batch_f))

    if not is_folder:
        cap.release()
        out_v.release()
    pbar.close()


if __name__ == '__main__':
    main()
