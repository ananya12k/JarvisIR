# Rebuilt engine vs. original engine (2026-11-01)

Reference: `VISIA_old/inference_out/v6_supplementary_all/02_mor_sensitivity/edfog_mor_ladder`,
test scene 03 rendered at 6 configs by the original (purged) engine, with its clean frames
and per-clip meta.json. The rebuild was run with the same per-clip params and VDA vits depth.

All 2000 frames, rounding quantisation: PSNR 42.6-46.5 dB, MAE 0.6-0.8, no drift over time.
After switching to truncating quantisation (300 frames):

| cfg | MOR | A    | H_fog | PSNR | MAE | mean orig/rebuild |
|-----|-----|------|-------|------|-----|-------------------|
| 0   | 150 | 0.72 | 8     | 46.7 | 0.3 | 135.7 / 135.8     |
| 1   | 150 | 0.82 | 40    | 42.1 | 0.5 | 151.3 / 151.4     |
| 2   | 100 | 0.82 | 8     | 44.0 | 0.5 | 159.4 / 159.5     |
| 3   | 100 | 0.72 | 8     | 45.6 | 0.4 | 145.9 / 145.9     |
| 4   | 300 | 0.64 | 40    | 47.5 | 0.3 | 118.7 / 118.7     |
| 5   |  75 | 0.64 | 20    | 45.6 | 0.3 | 142.7 / 142.8     |

Patchy path (EDFog_twoterm train/00 cfg0, patchy MOR 50, scatter off; clean frames
re-derived by resizing Defog_Dataset to 480x270): PSNR 43.2 dB.

Residual error is concentrated on depth edges (guided filter / VDA differences).
The fBm heterogeneity contributes < ~0.5 grey levels in both engines at the paper's
parameters, so this comparison cannot validate the noise-field implementation itself.

Note: the original build ran at 480x270, not the source's native 848x480.
