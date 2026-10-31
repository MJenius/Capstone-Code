# Complete Hybrid Watermarking Pipeline — Visual Demonstration Report

**Executed:** `2026-10-31T17:27:01`  
**Held-Out Test Host:** `0014` (never seen during ANN training/validation)  
**Attack Applied:** `25% Random Crop (Kept 75.0%)`

## 1. Imperceptibility (Clean Embedding vs Host)

- **PSNR:** `43.42 dB`
- **SSIM:** `0.9683`
- **Adaptive Alpha Range:** `[0.0120, 0.0360]` (mean `0.0131`)

## 2. Blind ANN vs Non-Blind Extraction Results

| Extractor Mode | Host Image Used? | Crop Mask Used? | NC | BER | Bit Accuracy | F1 Score | Bit Errors |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Blind ANN (`BlindWatermarkExtractor`)** | **NO (Host-Free)** | **NO** | `1.0000` | `0.0000` | `100.00%` | `1.0000` | `0 / 1024` |
| **Non-Blind (`NonBlindExtractor`)** | **YES (Clean Host)** | **YES** | `1.0000` | `0.0000` | `100.00%` | `—` | `0 / 1024` |

## 3. Generated Stage Artifacts

- `full_pipeline.png`: `results/final_demo/full_pipeline.png`
- `stage_01_preprocessing.png`: `results/final_demo/stage_01_preprocessing.png`
- `stage_02_watermark.png`: `results/final_demo/stage_02_watermark.png`
- `stage_03_embedding.png`: `results/final_demo/stage_03_embedding.png`
- `stage_04_attack.png`: `results/final_demo/stage_04_attack.png`
- `stage_05_nonblind_extraction.png`: `results/final_demo/stage_05_nonblind_extraction.png`
- `stage_06_ann_blind_extraction.png`: `results/final_demo/stage_06_ann_blind_extraction.png`
- `stage_07_comparison.png`: `results/final_demo/stage_07_comparison.png`