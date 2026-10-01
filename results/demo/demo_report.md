# Capstone Research Pipeline Demonstration Report
**Execution Timestamp:** 2026-10-01T12:40:04.364988
**Representative Host Sample:** `0001`

## Demonstration Overview
This demonstration executes the complete end-to-end robust watermarking pipeline on a real host image,
from raw preprocessing, dual chaotic transformations, mosaic expansion, texture-adaptive embedding,
simulated attack (`25% Random Crop (Kept 75.0%)`), to non-blind host-subtracted extraction and bit-exact recovery.

## Numerical Performance Metrics
| Evaluation Axis | Metric | Value | Interpretation |
| :--- | :--- | :--- | :--- |
| **Imperceptibility** | PSNR | `40.89 dB` | High fidelity (> 40 dB threshold) |
| **Imperceptibility** | SSIM | `0.9972` | Near-identical perceptual structure |
| **Robustness** | Attack Scenario | `25% Random Crop (Kept 75.0%)` | Applied attack distortion |
| **Robustness** | Normalized Correlation (NC) | `1.0000` | Correlation with ground-truth watermark |
| **Robustness** | Bit Error Rate (BER) | `0.0000` | Fraction of erroneous bits |
| **Robustness** | Bit Errors | `0 / 1024` | Total bit discrepancy |
| **Status** | Recovery Outcome | `EXACT BIT-MATCH (100%)` | Validation status |

## Capstone Multi-Panel Artifact
- Multi-Panel Demonstration Poster: [full_pipeline_demo.png](file:///C:\Users\mjeni\OneDrive\Desktop\Capstone\Code\results\demo\full_pipeline_demo.png)