"""
Water film F(x, y, t) in [0, 1] on a coarse grid.

    F_{t+dt} = clamp(F_t + dt S_t - dt L_t + dt div(D grad F_t), 0, 1)
    optional directional transport:  dF/dt + div(F u) = S - L,  u = (0, advect * F)

S = s0 G(I) m R^gamma / mean(R^gamma) + (splash residue, rivulet overflow), L = loss F.
The spatial factor R^gamma / mean(R^gamma) only says WHERE rain is denser (gamma > 1
sharpens patches); the global intensity I enters separately through G(I) = I^kappa
(a uniform multiplier inside the normalised term would cancel). Residue / overflow
arrive as water volume (converted with depth_px) and are added once, not scaled by I.
Without transport the film settles at F ~ S / L: patchy, never a uniform layer. Diffusion is explicit with sub-steps so that
D dt_sub / dx^2 <= 0.2 (stable); transport is semi-Lagrangian (stable).
"""

import math

import cv2
import numpy as np


class FilmField:
    def __init__(self, cfg, H, W):
        self.cfg = cfg
        fc = cfg.film
        self.g = fc.grid_px
        self.gh, self.gw = math.ceil(H / self.g), math.ceil(W / self.g)
        self.F = np.zeros((self.gh, self.gw), np.float32)
        self._extra = np.zeros_like(self.F)
        self._gy, self._gx = np.mgrid[0:self.gh, 0:self.gw].astype(np.float32)

    def cell(self, x, y):
        return min(max(int(y / self.g), 0), self.gh - 1), min(max(int(x / self.g), 0), self.gw - 1)

    def add(self, x, y, amount, radius_px):
        """Raise the film level by `amount` over a disc (applied in the next update)."""
        cy, cx = self.cell(x, y)
        rr = max(1, int(round(radius_px / self.g)))
        splat = np.zeros_like(self.F)
        cv2.circle(splat, (cx, cy), rr, 1.0, -1)
        self._extra += splat * amount

    def add_volume(self, x, y, volume, radius_px):
        """Deposit a water VOLUME over a disc: level = volume / (area_px * depth_px)."""
        cy, cx = self.cell(x, y)
        rr = max(1, int(round(radius_px / self.g)))
        splat = np.zeros_like(self.F)
        cv2.circle(splat, (cx, cy), rr, 1.0, -1)
        area_px = max(float(splat.sum()), 1.0) * self.g ** 2
        self._extra += splat * (volume / (area_px * self.cfg.film.depth_px))

    def source(self, rain_grid, intensity):
        """S_rain = s0 G(I) m R^gamma / mean(R^gamma)  (per second)."""
        fc = self.cfg.film
        Rg = np.power(np.maximum(rain_grid, 0), fc.rain_exponent)
        G = max(intensity, 0.0) ** fc.intensity_exponent
        return fc.source_per_s * G * self.cfg.film_mult * Rg / max(float(Rg.mean()), 1e-6)

    @staticmethod
    def laplacian(F):
        P = np.pad(F, 1, mode='edge')                  # no-flux borders
        return P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:] - 4 * F

    def update(self, dt, rain_grid, intensity=1.0, overflow=None):
        """rain_grid: spatial pattern R (not scaled by intensity); intensity: global I;
        overflow: film-level increment this step (added once)."""
        fc = self.cfg.film
        S = self.source(rain_grid, intensity) + self._extra / dt
        if overflow is not None:
            S = S + overflow / dt
        self._extra[:] = 0
        L = fc.loss_per_s * self.F
        self.F = self.F + dt * (S - L)
        # div(D grad F) with constant D: explicit sub-steps (stable for D dt / dx^2 <= 0.2)
        D = fc.diffusion_cells2_s
        if D > 0:
            n = max(1, math.ceil(D * dt / 0.2))
            h = dt / n
            for _ in range(n):
                self.F = self.F + h * D * self.laplacian(self.F)
        # downward transport, thicker film faster (semi-Lagrangian: unconditionally stable)
        if fc.advect_cells_s > 0:
            v = (fc.advect_cells_s * dt * np.clip(self.F, 0, 1)).astype(np.float32)
            self.F = cv2.remap(self.F.astype(np.float32), self._gx, self._gy - v, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        self.F = np.clip(self.F, 0.0, 1.0).astype(np.float32)

    def level_at(self, x, y) -> float:
        return float(self.F[self.cell(x, y)])

    def coverage(self) -> float:
        return float((self.F > self.cfg.film.visible).mean())

    def rasterize(self, H, W) -> np.ndarray:
        fc = self.cfg.film
        lvl = np.clip((self.F - fc.visible) / (fc.full_level - fc.visible + 1e-6), 0, 1)
        h = cv2.resize(lvl.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        return np.clip(h, 0, 1) * fc.height
