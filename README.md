# Deep Learning-Based Image Watermarking Using Catalan Transform and Mosaic Generation for Enhanced Resistance to Cropping and Collusion Attacks

**Capstone Project ID:** `PW26_PAC_01`  
**Status:** Complete End-to-End Implementation (Non-Blind + Trained Blind PyTorch ANN Extractor + 3-Way Held-Out Evaluation + 10-Tab Interactive Streamlit Workbench)

---

## 1. Overview & Research Objective

Digital watermarks embedded in a single spatial region are easily destroyed when an attacker crops out that region, while watermarks embedded with uniform strength across an image either degrade visual quality in smooth regions or are stripped by signal processing and multi-copy collusion attacks.

This repository implements and validates a **complete end-to-end hybrid spatial-neural watermarking framework** that combines:
1. **YIQ Chrominance-Channel Targeting:** Embeds the watermark into the normalized $I$-chrominance channel ($256\times 256$) where human visual sensitivity is lower than in luminance ($Y$).
2. **Dual-Security Keyed Scrambling:** Applies a 10-iteration **Arnold Cat Map** followed by a 5-iteration **Catalan Transform permutation** (`key=7`) to destroy spatial autocorrelation in the $32\times 32$ binary watermark (`1,024` bits) with bit-exact reversibility (`NC = 1.0000`, `BER = 0.0000`).
3. **$8\times 8$ Spatial Mosaic Redundancy:** Tiles the $32\times 32$ encrypted watermark across an $8\times 8$ grid ($64$ copies covering $256\times 256$) so that every $32\times 32$ block of the host carries a complete watermark copy.
4. **Perceptual Luminance-Texture Adaptive Embedding:** Modulates embedding strength per pixel via local $3\times 3$ variance $M(x,y) \in [0,1]$:
   $$\alpha(x,y) = \alpha_{\text{base}}\cdot(1 + s\cdot M(x,y)) \in [0.012, 0.036], \qquad I'(x,y) = \text{clip}\big(I(x,y) + \alpha(x,y)\cdot(W_{\text{mosaic}}(x,y) - 0.5),\, 0,\, 1\big)$$
   achieving **`41.61 ± 0.89 dB` PSNR** and **`0.9805` SSIM** on held-out DIV2K test images.
5. **Dual Extraction Modes (Both Fully Implemented and Benchmarked):**
   - **Mode A — Non-Blind Residual Extractor (`core/extraction.py`, `core/hybrid_system.py`):** Uses the clean original host $I$-channel and crop mask (`nonblind_pixel` with exact pixel-wise $\alpha(x,y)$ compensation and `nonblind_base` with scalar $\alpha_{\text{base}}$ division).
   - **Mode B — Blind ANN Extractor (`models/blind_extractor.py`, `training/inference.py`):** A trained PyTorch neural network (`BlindWatermarkExtractor`, `276,466` parameters) that recovers the $32\times 32$ binary watermark **directly from the attacked $256\times 256$ $I$-channel (`(B,1,256,256) -> (B,1,32,32)`) without access to the clean host image, embedding residual, or crop mask**.

---

## 2. Quick Start — Interactive Streamlit Demo & CLI Execution

### 2.1 Launch the 10-Tab Interactive Demonstration App
```powershell
streamlit run app.py
```
The Streamlit workbench (`app.py`) provides:
- **Single-button `RUN COMPLETE DEMO` in the sidebar** (executes all pipeline stages end-to-end and populates every tab automatically).
- **10 Independent Stage-by-Stage Tabs:**
  1. Project Overview & Architecture
  2. Host Image & YIQ Preprocessing
  3. Watermark Generation, Arnold + Catalan Scrambling & $8\times 8$ Mosaic
  4. Perceptual Texture-Adaptive Embedding (`PSNR`, `SSIM`, heatmaps)
  5. Attack Simulation Lab (Cropping, JPEG, Gaussian Noise, Gaussian Blur, Median Filter, Multi-Watermark Collusion)
  6. Non-Blind Extraction (Host-Subtracted Reference)
  7. **Blind ANN Extraction (Host-Free PyTorch Inference)**
  8. **Side-by-Side Comparison (Blind ANN vs Non-Blind vs Center-Tile Baseline)**
  9. **ANN Dataset & Training Inspector** (curves, checkpoint metadata, held-out benchmark explorer)
  10. **Automated Test & Validation Status** (`26/26` pytest runner)

### 2.2 Run Individual Phases via CLI (`run.py`)
```powershell
# Phase 0-7: Individual pipeline stages & automated tests
python run.py --phase environment
python run.py --phase preprocessing
python run.py --phase watermark
python run.py --phase embedding
python run.py --phase attacks --attack crop
python run.py --phase extraction
python run.py --phase benchmark
python run.py --phase tests

# Phase 8: Leakage-free ANN dataset generation (2,960 samples across 800 unique DIV2K hosts)
python run.py --phase ann-data

# Phase 9: Blind ANN training, held-out evaluation, and host-free single-image inference
python run.py --phase ann-train --epochs 5
python run.py --phase ann-eval
python run.py --phase ann-inference --image results/ann_inference/demo_attacked_crop25.npy --model models/checkpoints/best_model.pth --ground-truth results/ann_inference/demo_gt_wm60000.npy

# Full 3-way system comparison across 120 held-out test hosts & 16 attack conditions
python run.py --phase comparison

# Complete 16-panel visual pipeline poster + 7 stage figures in results/final_demo/
python run.py --phase demo

# Run all phases sequentially
python run.py --phase all --quick
```

---

## 3. Blind ANN Architecture (`models/blind_extractor.py`)

`BlindWatermarkExtractor` (`276,466` trainable parameters) is purpose-built for mosaic-tiled blind extraction:

```
Attacked I-Channel x ∈ (B, 1, 256, 256)   [NO clean host, NO crop mask]
   │
   ├─► 1. Parameter-Free Input Conditioning:
   │      • Per-image Z-standardization: (x - μ) / (σ + 10⁻⁴)
   │      • Scaled 5×5 box high-pass residual: (x - AvgPool5×5(x)) × 50
   │      ──► Concatenated (B, 2, 256, 256)
   │
   ├─► 2. Full-Resolution Stem + Residual Block ──► h ∈ (B, 16, 256, 256)
   │
   ├─► 3. Multi-Scale Context Branch (128×128):
   │      • Stride-2 Conv + 2 Residual Blocks (32 ch) + Bilinear Upsample
   │      • Estimates local host texture & adaptive α(x,y) regime
   │      ──► Fused with h via 1×1 Conv + ResBlock ──► f ∈ (B, 16, 256, 256)
   │
   ├─► 4. Mosaic-Aware 8×8 Tile Folding & Learned Confidence Pooling:
   │      • Reshape 256×256 into 64 tiles of 32×32: (B, 16, 32, 32, 64)
   │      • Learned confidence head + Softmax across 64 tiles down-weights cropped/corrupted tiles
   │      • Concatenates confidence-weighted sum & uniform mean ──► agg ∈ (B, 32, 32, 32)
   │
   ├─► 5. 32×32 Residual Decoder:
   │      • 1×1 Conv (32→64) + 3 Residual Blocks (64 ch) + 1×1 Head
   │      ──► Catalo-Arnold Domain Logits (B, 1, 32, 32)
   │
   └─► 6. Exact Keyed Inverse Permutation Buffer (inv_perm):
          • Non-trainable persistent index bijection undoing Catalan (5 iters, k=7) + Arnold (10 iters)
          ──► Output Logits / Sigmoid Probabilities of Original Watermark (B, 1, 32, 32)
```

---

## 4. Measured Experimental Results (120 Held-Out Test Hosts)

All evaluations in `results/final_evaluation/` are computed on **120 held-out DIV2K test hosts (`splits/host_level/test.txt`)** with **zero host overlap** (`0` shared hosts with training or validation) and disjoint test watermark IDs.

### 4.1 Three-Way Comparison: Mean NC ± Std & Mean BER ($n = 120$ Test Hosts per Cell)

| Attack Condition | **Proposed Hybrid + Blind ANN (BLIND: Attacked Image Only)** | **Proposed Hybrid + Non-Blind (pixel $\alpha$, Uses Host + Mask)** | **Proposed Hybrid + Non-Blind (legacy $\alpha$, Uses Host + Mask)** | **Center-Tile Baseline (Non-Blind, Uses Host)** |
| :--- | :--- | :--- | :--- | :--- |
| **`clean`** | **0.957 ± 0.079** (`BER 0.0204`) | 1.000 ± 0.000 (`BER 0.0000`) | 1.000 ± 0.000 (`BER 0.0000`) | 1.000 ± 0.000 (`BER 0.0000`) |
| **`rgb_png_roundtrip`** | **0.950 ± 0.083** (`BER 0.0240`) | 0.907 ± 0.278 (`BER 0.0481`) | — | — |
| **`crop_10`** (10% random crop) | **0.952 ± 0.083** (`BER 0.0229`) | 1.000 ± 0.000 (`BER 0.0000`) | 1.000 ± 0.000 (`BER 0.0000`) | 0.801 ± 0.310 (`BER 0.0867`) |
| **`crop_25`** (25% random crop) | **0.943 ± 0.087** (`BER 0.0273`) | 1.000 ± 0.000 (`BER 0.0000`) | 1.000 ± 0.000 (`BER 0.0000`) | 0.174 ± 0.227 (`BER 0.3641`) |
| **`crop_50`** (50% random crop) | **0.924 ± 0.089** (`BER 0.0361`) | 1.000 ± 0.000 (`BER 0.0000`) | 1.000 ± 0.000 (`BER 0.0000`) | **0.000 ± 0.000** (`BER 0.4052`) |
| **`noise_05`** ($\sigma = 0.05$) | **0.608 ± 0.062** (`BER 0.2023`) | 0.759 ± 0.060 (`BER 0.1132`) | 0.787 ± 0.061 (`BER 0.0997`) | 0.558 ± 0.036 (`BER 0.2127`) |
| **`blur_3`** ($3\times 3$ Gaussian) | **0.579 ± 0.084** (`BER 0.2069`) | 0.393 ± 0.073 (`BER 0.2715`) | 0.348 ± 0.074 (`BER 0.2938`) | 0.558 ± 0.161 (`BER 0.2080`) |
| **`jpeg_70`** ($Q = 70$) | 0.480 ± 0.065 (`BER 0.2535`) | 0.564 ± 0.133 (`BER 0.1988`) | 0.593 ± 0.127 (`BER 0.1857`) | 0.735 ± 0.070 (`BER 0.1227`) |
| **`jpeg_50`** ($Q = 50$) | 0.333 ± 0.060 (`BER 0.3171`) | 0.401 ± 0.104 (`BER 0.2663`) | 0.420 ± 0.096 (`BER 0.2584`) | 0.623 ± 0.093 (`BER 0.1761`) |
| **`median_3`** ($3\times 3$ median) | 0.328 ± 0.070 (`BER 0.3116`) | 0.197 ± 0.063 (`BER 0.3502`) | 0.185 ± 0.065 (`BER 0.3579`) | 0.507 ± 0.109 (`BER 0.2284`) |
| **`collusion_2`** ($N = 2$) | **0.518 ± 0.081** (`BER 0.2426`) | 0.529 ± 0.086 (`BER 0.2259`) | 0.529 ± 0.085 (`BER 0.2260`) | 0.525 ± 0.088 (`BER 0.2277`) |
| **`collusion_5`** ($N = 5$) | 0.331 ± 0.121 (`BER 0.3323`) | 0.401 ± 0.155 (`BER 0.2871`) | 0.403 ± 0.156 (`BER 0.2861`) | 0.379 ± 0.141 (`BER 0.2976`) |
| **`collusion_10`** ($N = 10$) | 0.260 ± 0.141 (`BER 0.3675`) | 0.327 ± 0.177 (`BER 0.3223`) | 0.330 ± 0.178 (`BER 0.3209`) | 0.305 ± 0.160 (`BER 0.3330`) |
| **`collusion_20`** ($N = 20$) | 0.209 ± 0.154 (`BER 0.3922`) | 0.277 ± 0.195 (`BER 0.3450`) | 0.280 ± 0.197 (`BER 0.3433`) | 0.253 ± 0.174 (`BER 0.3568`) |
| **`collusion_50`** ($N = 50$) | 0.182 ± 0.166 (`BER 0.4061`) | 0.244 ± 0.215 (`BER 0.3602`) | 0.247 ± 0.219 (`BER 0.3593`) | 0.217 ± 0.195 (`BER 0.3729`) |
| **`collusion_100`** ($N = 100$) | 0.177 ± 0.166 (`BER 0.4082`) | 0.238 ± 0.216 (`BER 0.3636`) | 0.239 ± 0.218 (`BER 0.3631`) | 0.207 ± 0.192 (`BER 0.3779`) |

### 4.2 Cropping Resistance Ratio (CRR) & Collusion Robustness Index (CoRI, $\tau = 0.125$)

| Method | CRR 10% | CRR 25% | CRR 50% | CoRI $N=2$ | CoRI $N=5$ | CoRI $N=10$ | CoRI $N=20$ | CoRI $N=50$ | CoRI $N=100$ | Aggregate CoRI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposed Hybrid + Blind ANN** | **0.995** | **0.986** | **0.966** | **1.00** | **0.99** | **0.82** | **0.61** | **0.51** | **0.52** | **0.740** |
| **Proposed Hybrid + Non-Blind (pixel $\alpha$)** | 1.000 | 1.000 | 1.000 | 1.00 | 0.98 | 0.86 | 0.64 | 0.53 | 0.53 | 0.757 |
| **Center-Tile Baseline (Non-Blind)** | 0.801 | 0.174 | **0.000** | 1.00 | 0.99 | 0.84 | 0.63 | 0.52 | 0.51 | 0.749 |

---

## 5. Repository Structure & Documentation Map

```
├── app.py                          # 10-tab interactive Streamlit application
├── run.py                          # Unified CLI phase runner
├── create_training_data.py         # Leakage-free ANN dataset CLI generator
├── benchmark.py                    # Legacy Phase-6 non-blind benchmark runner
├── generate_watermark.py           # Canonical 32x32 binary watermark generator
├── models/
│   ├── blind_extractor.py          # PyTorch BlindWatermarkExtractor (276,466 params)
│   └── checkpoints/
│       ├── best_model.pth          # Best validation checkpoint (Epoch 4, val NC=0.7333)
│       └── last_model.pth          # Final epoch checkpoint (Epoch 5)
├── training/
│   ├── dataset.py                  # Leakage-free ANN dataset builder & PyTorch Dataset
│   ├── trainer.py                  # Training loop, AdamW, ReduceLROnPlateau, early stopping
│   ├── losses.py                   # BCEWithLogits + 0.1 * (1 - softNC) composite loss
│   ├── metrics.py                  # NC, BER, bit accuracy, precision, recall, F1
│   ├── inference.py                # Host-free BlindExtractor inference engine
│   └── evaluator.py                # 3-way evaluator, CRR, CoRI, negative controls
├── core/
│   ├── keyed_permutation.py        # Cached O(1) index bijection for Arnold + Catalan
│   ├── data_sources.py             # 800-host DIV2K deduplicator & host-level split manager
│   ├── hybrid_system.py            # Unified embedding, attack, collusion & extraction wrapper
│   ├── preprocessing.py            # RGB -> YIQ & I-channel normalization
│   ├── watermark.py                # Forward/inverse Arnold + Catalan + 8x8 mosaic
│   ├── embedding.py                # Adaptive luminance-texture embedder wrapper
│   ├── attacks.py                  # Standardized attack runner
│   ├── extraction.py               # Non-blind host-subtracted extractor
│   └── ann_interface.py            # ConcreteBlindANNExtractor interface
├── pipeline/
│   ├── phase_runner.py             # PhaseRunner orchestrator (Phases 0-9)
│   └── ann_phases.py               # ANN data, train, eval, inference, comparison & final demo
├── tests/
│   ├── test_ann_and_system.py      # ANN, split leakage, collusion & synthetic roundtrip tests
│   ├── test_components.py          # Transform, mosaic, embedder & attack unit tests
│   ├── test_phases.py              # CLI phase integration tests
│   └── test_pipeline.py            # End-to-end pipeline roundtrip tests
├── docs/
│   ├── technical_audit.md          # Complete pre-implementation repository & report audit
│   ├── ann_architecture.md         # Layer-by-layer BlindWatermarkExtractor specification
│   ├── ann_training_methodology.md # Dataset construction, loss, splits & anti-leakage design
│   ├── ann_evaluation.md           # Held-out test metrics, 3-way comparison, CRR & CoRI
│   ├── final_system_methodology.md # Complete mathematical specification of the hybrid system
│   ├── implementation_diagram.md   # ASCII & Mermaid architecture diagrams
│   ├── live_demo_script.md         # Step-by-step faculty defense script & Q&A guide
│   ├── known_limitations.md        # Honest technical limitations (JPEG, 1/sqrt(N) collusion)
│   ├── results_guide.md            # Guide to all generated reports, plots, and JSONs
│   └── project_status.md           # Current status & headline metrics
└── results/
    ├── ann_training/               # Training curves, history CSV/JSON, training_report.md
    ├── ann_evaluation/             # Held-out test split report & per-attack NC bar chart
    ├── ann_inference/              # Single-image blind inference outputs & probability map
    ├── final_evaluation/           # 3-way comparison CSV/JSON/MD, CRR/CoRI curves, recovery grid
    ├── final_demo/                 # 16-panel full_pipeline.png + stage_01..stage_07 figures
    └── final_validation/           # Automated 26-test pytest reports & validation summary
```
