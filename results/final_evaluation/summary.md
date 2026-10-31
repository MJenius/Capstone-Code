# Final Evaluation — Blind ANN vs Non-Blind vs Centre-Tile Baseline

Generated 2026-10-31T15:04:46 on **120 held-out test hosts** (host-level split; no test host or test watermark ID was seen in training/validation).

## Information available to each method

- **ANN blind** — BLIND: attacked image only (+ secret key). No host, no mask.
- **Non-blind (pixel α)** — NON-BLIND: original host + crop mask; pixel-wise alpha compensation (corrected).
- **Non-blind (legacy α)** — NON-BLIND: original host + crop mask; alpha_base compensation (legacy).
- **Centre-tile baseline (non-blind)** — NON-BLIND: original host; conventional single centre-tile embedding.

> The non-blind extractors subtract the ORIGINAL host (and use the known crop mask). Their numbers are an upper reference, not a like-for-like competitor of the blind ANN.

## Mean NC ± std (n per cell = number of test hosts)

| attack | ANN blind | Non-blind (pixel α) | Non-blind (legacy α) | Centre-tile baseline (non-blind) |
|---|---|---|---|---|
| clean | 0.957 ± 0.079 (n=120) | 1.000 ± 0.000 (n=120) | 1.000 ± 0.000 (n=120) | 1.000 ± 0.000 (n=120) |
| crop_10 | 0.952 ± 0.083 (n=120) | 1.000 ± 0.000 (n=120) | 1.000 ± 0.000 (n=120) | 0.801 ± 0.310 (n=120) |
| crop_25 | 0.943 ± 0.087 (n=120) | 1.000 ± 0.000 (n=120) | 1.000 ± 0.000 (n=120) | 0.174 ± 0.227 (n=120) |
| crop_50 | 0.924 ± 0.089 (n=120) | 1.000 ± 0.000 (n=120) | 1.000 ± 0.000 (n=120) | 0.000 ± 0.000 (n=120) |
| jpeg_70 | 0.480 ± 0.065 (n=120) | 0.564 ± 0.133 (n=120) | 0.593 ± 0.127 (n=120) | 0.735 ± 0.070 (n=120) |
| jpeg_50 | 0.333 ± 0.060 (n=120) | 0.401 ± 0.104 (n=120) | 0.420 ± 0.096 (n=120) | 0.623 ± 0.093 (n=120) |
| noise_05 | 0.608 ± 0.062 (n=120) | 0.759 ± 0.060 (n=120) | 0.787 ± 0.061 (n=120) | 0.558 ± 0.036 (n=120) |
| blur_3 | 0.579 ± 0.084 (n=120) | 0.393 ± 0.073 (n=120) | 0.348 ± 0.074 (n=120) | 0.558 ± 0.161 (n=120) |
| median_3 | 0.328 ± 0.070 (n=120) | 0.197 ± 0.063 (n=120) | 0.185 ± 0.065 (n=120) | 0.507 ± 0.109 (n=120) |
| collusion_2 | 0.518 ± 0.081 (n=120) | 0.529 ± 0.086 (n=120) | 0.529 ± 0.085 (n=120) | 0.525 ± 0.088 (n=120) |
| collusion_5 | 0.331 ± 0.121 (n=120) | 0.401 ± 0.155 (n=120) | 0.403 ± 0.156 (n=120) | 0.379 ± 0.141 (n=120) |
| collusion_10 | 0.260 ± 0.141 (n=120) | 0.327 ± 0.177 (n=120) | 0.330 ± 0.178 (n=120) | 0.305 ± 0.160 (n=120) |
| collusion_20 | 0.209 ± 0.154 (n=120) | 0.277 ± 0.195 (n=120) | 0.280 ± 0.197 (n=120) | 0.253 ± 0.174 (n=120) |
| collusion_50 | 0.182 ± 0.166 (n=120) | 0.244 ± 0.215 (n=120) | 0.247 ± 0.219 (n=120) | 0.217 ± 0.195 (n=120) |
| collusion_100 | 0.177 ± 0.166 (n=120) | 0.238 ± 0.216 (n=120) | 0.239 ± 0.218 (n=120) | 0.207 ± 0.192 (n=120) |
| rgb_png_roundtrip | 0.950 ± 0.083 (n=120) | 0.907 ± 0.278 (n=120) | — | — |

## Mean BER

| attack | ANN blind | Non-blind (pixel α) | Non-blind (legacy α) | Centre-tile baseline (non-blind) |
|---|---|---|---|---|
| clean | 0.0204 | 0.0000 | 0.0000 | 0.0000 |
| crop_10 | 0.0229 | 0.0000 | 0.0000 | 0.0867 |
| crop_25 | 0.0273 | 0.0000 | 0.0000 | 0.3641 |
| crop_50 | 0.0361 | 0.0000 | 0.0000 | 0.4052 |
| jpeg_70 | 0.2535 | 0.1988 | 0.1857 | 0.1227 |
| jpeg_50 | 0.3171 | 0.2663 | 0.2584 | 0.1761 |
| noise_05 | 0.2023 | 0.1132 | 0.0997 | 0.2127 |
| blur_3 | 0.2069 | 0.2715 | 0.2938 | 0.2080 |
| median_3 | 0.3116 | 0.3502 | 0.3579 | 0.2284 |
| collusion_2 | 0.2426 | 0.2259 | 0.2260 | 0.2277 |
| collusion_5 | 0.3323 | 0.2871 | 0.2861 | 0.2976 |
| collusion_10 | 0.3675 | 0.3223 | 0.3209 | 0.3330 |
| collusion_20 | 0.3922 | 0.3450 | 0.3433 | 0.3568 |
| collusion_50 | 0.4061 | 0.3602 | 0.3593 | 0.3729 |
| collusion_100 | 0.4082 | 0.3636 | 0.3631 | 0.3779 |
| rgb_png_roundtrip | 0.0240 | 0.0481 | — | — |

## Imperceptibility (clean, vs host I-channel)

| scheme | PSNR (dB) | SSIM |
|---|---|---|
| Hybrid mosaic (ANN & non-blind share this embedding) | 41.61 ± 0.89 | 0.9805 |
| Centre-tile baseline | 46.03 ± 0.08 | 0.9958 |

## Research metrics

CRR(p) = NC(crop p)/NC(clean). CoRI(N) = P(NC_victim > τ=0.125).

| method | CRR 10% | CRR 25% | CRR 50% | CoRI N=2 | N=5 | N=10 | N=20 | N=50 | N=100 | CoRI agg |
|---|---|---|---|---|---|---|---|---|---|---|
| ANN blind | 0.995 | 0.986 | 0.966 | 1.00 | 0.99 | 0.82 | 0.61 | 0.51 | 0.52 | 0.740 |
| Non-blind (pixel α) | 1.000 | 1.000 | 1.000 | 1.00 | 0.98 | 0.86 | 0.64 | 0.53 | 0.53 | 0.757 |
| Non-blind (legacy α) | 1.000 | 1.000 | 1.000 | 1.00 | 0.98 | 0.87 | 0.65 | 0.53 | 0.52 | 0.757 |
| Centre-tile baseline (non-blind) | 0.801 | 0.174 | 0.000 | 1.00 | 0.99 | 0.84 | 0.63 | 0.52 | 0.51 | 0.749 |

## Negative control (blind ANN)

Unwatermarked test hosts fed to the ANN: NC vs the host's test watermark = 0.001 ± 0.031; empirical FPR at τ: 0.000. Clean-prediction vs unrelated watermark FPR: 0.233.

## Honest assessment

- Attack conditions where the blind ANN mean NC < 0.5 (treated as failure): **collusion_10, collusion_100, collusion_20, collusion_5, collusion_50, jpeg_50, jpeg_70, median_3**.
- Averaging collusion is symmetric in the colluders; the victim's watermark can at best retain ~1/√N correlation for ANY extractor. Results above show measured behaviour, not complete collusion resistance.
- JPEG/blur attacks remove the high-frequency band that carries the low-amplitude (α≈0.012–0.036) mosaic signal; degradation there is a property of the embedding strength, shared by the non-blind extractor.