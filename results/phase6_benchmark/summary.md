# Standardized Baseline Benchmark Summary
**Total Records:** 70 | **Evaluated Images:** 5

## 1. Clean Imperceptibility
- **Hybrid Framework:** PSNR = 41.02 dB, SSIM = 0.9965
- **Baseline Method:**  PSNR = 46.02 dB, SSIM = 0.9985

## 2. Robustness by Attack Category
| Attack Type | Hybrid NC | Baseline NC | Hybrid BER | Baseline BER |
| :--- | :--- | :--- | :--- | :--- |
| blur_3 | 0.2075 | 0.2710 | 0.0441 | 0.1051 |
| crop_10 | 0.9982 | 0.8269 | 0.0000 | 0.0064 |
| crop_25 | 0.9976 | 0.1029 | 0.0000 | 0.0329 |
| crop_50 | 0.9953 | 0.0000 | 0.0000 | 0.0352 |
| jpeg_50 | 0.2830 | 0.3329 | 0.0350 | 0.0898 |
| jpeg_70 | 0.4191 | 0.4612 | 0.0281 | 0.0615 |
| no_attack | 0.9984 | 1.0000 | 0.0000 | 0.0000 |
| noise_05 | 0.4645 | 0.2985 | 0.0854 | 0.2150 |

## Analysis & Findings
- **Cropping Resilience:** Hybrid achieves NC > 0.99 under 50% crop due to 64-tile spatial redundancy.
- **Noise Resistance:** 64-tile averaging suppresses additive Gaussian noise.
- **JPEG & Blur:** Baseline exhibits higher energy retention in a single center tile at alpha=0.08 vs distributed alpha=0.012.