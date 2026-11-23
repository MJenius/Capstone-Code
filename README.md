# Deep Learning-Based Image Watermarking Using Catalan Transform and Mosaic Generation

**Project ID:** `PW26_PAC_01`  
**Topic:** Deep Learning-Based Image Watermarking Using Catalan Transform and Mosaic Generation for Enhanced Resistance to Cropping and Collusion Attacks.

---

## Overview

This project implements a hybrid robust image watermarking pipeline that combines:
1. **YIQ Color Space Preprocessing**: Host images are resized to `256×256` and converted to YIQ; embedding is performed on the normalized **I-channel** to preserve perceptual quality.
2. **Keyed Watermark Scrambling**: A `32×32` binary watermark is scrambled with **Arnold Cat Map (ACM)** (`10` iterations) followed by **Catalan Transform** (`5` iterations, `catalan_key=7`).
3. **8×8 Mosaic Generation**: The scrambled `32×32` watermark is tiled into an `8×8` mosaic (`256×256`) for spatial redundancy against cropping.
4. **Adaptive Texture Embedding**: Uses a local texture mask (`alpha_base=0.012`, `sensitivity=2.0`) to embed stronger signals in textured regions and weaker signals in smooth regions.
5. **Attack Simulation**: Evaluates robustness against **Cropping** (`10%`, `25%`, `50%`), **JPEG Compression** (`Q=70`, `Q=50`), **Gaussian Noise**, **Gaussian/Median Blur**, and **Collusion** (`N=2..100` colluding copies).
6. **Blind ANN Extractor (`BlindWatermarkExtractor`)**: A PyTorch CNN that extracts the `32×32` watermark directly from an attacked `256×256` I-channel without requiring the original host image, using high-pass residual conditioning, multi-scale context fusion, mosaic-aware confidence-weighted tile aggregation, and a fixed keyed inverse permutation.

---

## Project Structure

```text
Code/
├── app.py                     # Streamlit interactive research & demo dashboard
├── run.py                     # Master CLI for all pipeline phases (0–9, demo, all)
├── main.py                    # Standalone Phase 1 preprocessing script
├── core/
│   ├── hybrid_system.py       # Unified HybridSystem wrapper (transforms, embedding, attacks)
│   ├── preprocessing.py       # YIQ conversion, resizing, normalization
│   ├── watermark.py           # Arnold + Catalan + 8x8 mosaic transformer
│   ├── keyed_permutation.py   # Vectorized forward/inverse index maps
│   ├── embedding.py           # Adaptive texture-aware watermark embedder
│   ├── extraction.py          # Non-blind extractor (host-assisted)
│   ├── attacks.py             # Unified attack runner
│   └── evaluation.py          # PSNR, SSIM, BER, NC, Bit Accuracy metrics
├── attacks/                   # Cropping, signal (JPEG/noise/blur), and collusion engines
├── models/
│   └── blind_extractor.py     # PyTorch BlindWatermarkExtractor (MosaicAwareExtractor)
├── training/
│   ├── dataset.py             # Host-split ANN dataset generator
│   ├── trainer.py             # Training loop & checkpointing (models/checkpoints/)
│   └── inference.py           # Blind ANN inference wrapper
├── pipeline/
│   ├── phase_runner.py        # Phases 0–7 & visual demo orchestration
│   └── ann_phases.py          # Phases 8–9 (ann-data, ann-train, ann-eval, comparison)
├── reporting/
│   └── visualizer.py          # Grids, demo posters, and Markdown reports
└── utils/                     # Low-level image loaders, Catalan/Arnold/mosaic helpers
```

---

## Quick Start

### 1. Launch the Interactive Streamlit App
```powershell
streamlit run app.py
```
The Streamlit dashboard provides an interactive stage-by-stage walkthrough: host selection/upload, YIQ preprocessing, Arnold + Catalan + Mosaic visualization, adaptive embedding, attack simulation, and live side-by-side comparison of Non-Blind vs. Blind ANN extraction.

### 2. Run the Complete Pipeline (CLI)
```powershell
python run.py --phase all
```

---

## Individual Pipeline Steps (`run.py`)

Run any phase individually from the project root:

| Step | Command | Description |
| :--- | :--- | :--- |
| **Phase 0: Environment** | `python run.py --phase environment` | Verify directories, dependencies, and environment setup. |
| **Phase 1: Preprocessing** | `python run.py --phase preprocessing` | Load/download host images, convert to YIQ, resize to `256×256`, normalize I-channel, create splits. |
| **Phase 2: Watermark** | `python run.py --phase watermark` | Generate `32×32` binary watermark, apply Arnold + Catalan + `8×8` mosaic. |
| **Phase 3: Embedding** | `python run.py --phase embedding` | Embed watermark into host I-channel using adaptive texture masking. |
| **Phase 4: Attacks** | `python run.py --phase attacks --attack crop` | Apply attack (`crop`, `jpeg`, `noise`, `blur`, or `collusion`). |
| **Phase 5: Extraction** | `python run.py --phase extraction` | Run non-blind watermark extraction and compute recovery metrics. |
| **Phase 6: Benchmark** | `python run.py --phase benchmark --quick` | Run benchmark suite across attacks (omit `--quick` for full run). |
| **Phase 7: Tests** | `python run.py --phase tests` | Run automated validation tests. |
| **Phase 8: ANN Dataset** | `python run.py --phase ann-data --quick` | Generate host-separated train/val/test dataset for the blind ANN. |
| **Phase 9a: ANN Train** | `python run.py --phase ann-train --epochs 10` | Train `BlindWatermarkExtractor` (supports `--quick` and `--resume`). |
| **Phase 9b: ANN Eval** | `python run.py --phase ann-eval` | Evaluate trained blind ANN on the held-out test split. |
| **Phase 9c: ANN Inference** | `python run.py --phase ann-inference --image <path> --model models/checkpoints/best_model.pth` | Run blind extraction on a single image. |
| **Phase 9d: Comparison** | `python run.py --phase comparison` | Run 3-way comparison: Baseline vs. Hybrid Non-Blind vs. Hybrid Blind ANN. |
| **Visual Demo** | `python run.py --phase demo` | Run complete visual demonstration mode and generate posters/reports in `results/`. |
