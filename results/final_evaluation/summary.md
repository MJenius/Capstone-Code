# Final Evaluation — Blind ANN vs Non-Blind vs Centre-Tile Baseline

Generated 2026-10-31T17:18:31 on **120 held-out test hosts** (host-level split; no test host or test watermark ID was seen in training/validation).

## Information available to each method

- **ANN blind** — BLIND: attacked image only (+ secret key). No host, no mask.
- **Non-blind (pixel α)** — NON-BLIND: original host + crop mask; pixel-wise alpha compensation (corrected).
- **Non-blind (legacy α)** — NON-BLIND: original host + crop mask; alpha_base compensation (legacy).
- **Centre-tile baseline (non-blind)** — NON-BLIND: original host; conventional single centre-tile embedding.

> The non-blind extractors subtract the ORIGINAL host (and use the known crop mask). Their numbers are an upper reference, not a like-for-like competitor of the blind ANN.

## Mean NC ± std (n per cell = number of test hosts)

| attack | ANN blind | Non-blind (pixel α) | Non-blind (legacy α) | Centre-tile baseline (non-blind) |
|---|---|---|---|---|
| clean | 0.975 ± 0.062 (n=120) | 1.000 ± 0.000 (n=120) | 1.000 ± 0.000 (n=120) | 1.000 ± 0.000 (n=120) |
| crop_10 | 0.973 ± 0.065 (n=120) | 1.000 ± 0.000 (n=120) | 1.000 ± 0.000 (n=120) | 0.796 ± 0.315 (n=120) |
| crop_25 | 0.967 ± 0.071 (n=120) | 1.000 ± 0.000 (n=120) | 1.000 ± 0.000 (n=120) | 0.174 ± 0.224 (n=120) |
| crop_50 | 0.950 ± 0.077 (n=120) | 1.000 ± 0.000 (n=120) | 1.000 ± 0.000 (n=120) | 0.000 ± 0.000 (n=120) |
| jpeg_70 | 0.495 ± 0.062 (n=120) | 0.561 ± 0.133 (n=120) | 0.592 ± 0.127 (n=120) | 0.743 ± 0.070 (n=120) |
| jpeg_50 | 0.344 ± 0.056 (n=120) | 0.399 ± 0.104 (n=120) | 0.416 ± 0.096 (n=120) | 0.624 ± 0.095 (n=120) |
| noise_05 | 0.632 ± 0.049 (n=120) | 0.756 ± 0.059 (n=120) | 0.785 ± 0.060 (n=120) | 0.561 ± 0.041 (n=120) |
| blur_3 | 0.718 ± 0.079 (n=120) | 0.392 ± 0.067 (n=120) | 0.345 ± 0.067 (n=120) | 0.564 ± 0.162 (n=120) |
| median_3 | 0.523 ± 0.089 (n=120) | 0.194 ± 0.057 (n=120) | 0.182 ± 0.059 (n=120) | 0.509 ± 0.111 (n=120) |
| collusion_2 | 0.523 ± 0.095 (n=120) | 0.530 ± 0.093 (n=120) | 0.528 ± 0.093 (n=120) | 0.525 ± 0.098 (n=120) |
| collusion_5 | 0.365 ± 0.132 (n=120) | 0.434 ± 0.152 (n=120) | 0.436 ± 0.153 (n=120) | 0.408 ± 0.141 (n=120) |
| collusion_10 | 0.272 ± 0.140 (n=120) | 0.345 ± 0.172 (n=120) | 0.348 ± 0.174 (n=120) | 0.319 ± 0.158 (n=120) |
| collusion_20 | 0.233 ± 0.151 (n=120) | 0.311 ± 0.193 (n=120) | 0.315 ± 0.195 (n=120) | 0.282 ± 0.176 (n=120) |
| collusion_50 | 0.206 ± 0.158 (n=120) | 0.283 ± 0.207 (n=120) | 0.285 ± 0.208 (n=120) | 0.248 ± 0.191 (n=120) |
| collusion_100 | 0.188 ± 0.156 (n=120) | 0.266 ± 0.211 (n=120) | 0.270 ± 0.213 (n=120) | 0.241 ± 0.192 (n=120) |
| rgb_png_roundtrip | 0.969 ± 0.070 (n=120) | 0.960 ± 0.185 (n=120) | — | — |

## Mean BER

| attack | ANN blind | Non-blind (pixel α) | Non-blind (legacy α) | Centre-tile baseline (non-blind) |
|---|---|---|---|---|
| clean | 0.0107 | 0.0000 | 0.0000 | 0.0000 |
| crop_10 | 0.0119 | 0.0000 | 0.0000 | 0.0874 |
| crop_25 | 0.0146 | 0.0000 | 0.0000 | 0.3440 |
| crop_50 | 0.0228 | 0.0000 | 0.0000 | 0.3875 |
| jpeg_70 | 0.2433 | 0.1966 | 0.1828 | 0.1182 |
| jpeg_50 | 0.3101 | 0.2622 | 0.2553 | 0.1740 |
| noise_05 | 0.1790 | 0.1142 | 0.1001 | 0.2103 |
| blur_3 | 0.1352 | 0.2683 | 0.2911 | 0.2035 |
| median_3 | 0.2244 | 0.3400 | 0.3465 | 0.2248 |
| collusion_2 | 0.2318 | 0.2244 | 0.2253 | 0.2261 |
| collusion_5 | 0.3018 | 0.2680 | 0.2671 | 0.2805 |
| collusion_10 | 0.3422 | 0.3094 | 0.3081 | 0.3222 |
| collusion_20 | 0.3590 | 0.3241 | 0.3224 | 0.3381 |
| collusion_50 | 0.3696 | 0.3371 | 0.3361 | 0.3533 |
| collusion_100 | 0.3768 | 0.3448 | 0.3434 | 0.3565 |
| rgb_png_roundtrip | 0.0137 | 0.0260 | — | — |

## Imperceptibility (clean, vs host I-channel)

| scheme | PSNR (dB) | SSIM |
|---|---|---|
| Hybrid mosaic (ANN & non-blind share this embedding) | 41.61 ± 0.89 | 0.9807 |
| Centre-tile baseline | 46.03 ± 0.08 | 0.9959 |

## Research metrics

CRR(p) = NC(crop p)/NC(clean). CoRI(N) = P(NC_victim > τ=0.125).

| method | CRR 10% | CRR 25% | CRR 50% | CoRI N=2 | N=5 | N=10 | N=20 | N=50 | N=100 | CoRI agg |
|---|---|---|---|---|---|---|---|---|---|---|
| ANN blind | 0.998 | 0.992 | 0.974 | 1.00 | 0.99 | 0.81 | 0.64 | 0.60 | 0.60 | 0.774 |
| Non-blind (pixel α) | 1.000 | 1.000 | 1.000 | 1.00 | 0.99 | 0.88 | 0.71 | 0.62 | 0.61 | 0.801 |
| Non-blind (legacy α) | 1.000 | 1.000 | 1.000 | 1.00 | 0.99 | 0.88 | 0.71 | 0.62 | 0.60 | 0.800 |
| Centre-tile baseline (non-blind) | 0.796 | 0.174 | 0.000 | 1.00 | 0.99 | 0.86 | 0.68 | 0.60 | 0.61 | 0.790 |

## Negative control & out-of-distribution generalization (blind ANN)

- **Unwatermarked test hosts fed to the ANN:** NC vs the host's test watermark = `0.0026 ± 0.0302`; empirical FPR at τ=0.125: `0.0000`.
- **Clean prediction vs unrelated watermark (mixed even/odd IDs):** FPR = `0.3583` (caused by fixed text-bounding-box overlap when both IDs are even, p=0.25).
- **Clean prediction vs unrelated odd-ID block watermark (zero structural bias):** mean NC = `0.0032`, FPR = `0.0083`.
- **Clean prediction vs unrelated i.i.d. Bernoulli(0.5) watermark:** mean NC = `0.0049`, FPR = `0.0000`.
- **Out-of-distribution unseen i.i.d. Bernoulli(0.5) watermark recovery:** mean NC = `0.9836`, mean BER = `0.0083`.
- **Out-of-distribution unseen bit-inverted watermark (`1 - wm`) recovery:** mean NC = `0.9337`, mean BER = `0.0345`.

## Honest assessment

- Attack conditions where the blind ANN mean NC < 0.5 (treated as failure): **collusion_10, collusion_100, collusion_20, collusion_5, collusion_50, jpeg_50, jpeg_70**.
- Averaging collusion is symmetric in the colluders; the victim's watermark can at best retain ~1/√N correlation for ANY extractor. Results above show measured behaviour, not complete collusion resistance.
- JPEG/blur attacks remove the high-frequency band that carries the low-amplitude (α≈0.012–0.036) mosaic signal; degradation there is a property of the embedding strength, shared by the non-blind extractor.