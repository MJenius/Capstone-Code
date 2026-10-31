# Final Validation Summary — Capstone Project (`PW26_PAC_01`)

**Project Title:** *Deep Learning-Based Image Watermarking Using Catalan Transform and Mosaic Generation for Enhanced Resistance to Cropping and Collusion Attacks*  
**Validation Timestamp:** `2026-10-31T15:09:23`

---

## 1. Validation Overview

All **26 automated unit and integration tests** (`tests/test_ann_and_system.py`, `tests/test_components.py`, `tests/test_phases.py`, `tests/test_pipeline.py`) executed and passed (`100%` pass rate, `0` failures, `0` skipped). Every CLI phase and interactive Streamlit workflow has been verified end-to-end on real data and synthetic fixtures.

## 2. Validated Subsystems & Invariants

1. **Image Preprocessing & YIQ Conversion (`core/preprocessing.py`):**
   - Verified RGB/BGR $\leftrightarrow$ YIQ conversion, $256\times 256$ bicubic resizing, and $I$-channel min-max normalization to $[0, 1]$.
2. **Exact Reversibility of Keyed Scrambling (`core/keyed_permutation.py`, `utils/scrambler.py`, `utils/catalan.py`):**
   - Verified that forward Arnold Cat Map ($10$ iterations) + Catalan transform ($5$ iterations, `key=7`) followed by inverse Catalan + inverse Arnold achieves bit-exact recovery (`NC = 1.0000`, `BER = 0.0000`, `0 / 1024` bit errors).
3. **Mosaic Generation & Adaptive Perceptual Embedding (`utils/mosaic.py`, `utils/adaptive_embedder.py`):**
   - Verified $32\times 32 \to 256\times 256$ ($8\times 8$ tile grid) expansion and local variance texture-adaptive embedding (`alpha_base=0.012`, `sensitivity=2.0`), achieving `41.61 ± 0.89 dB` PSNR and `0.9805` SSIM across the $120$ held-out test hosts.
4. **Scientifically Valid Attack Suite (`core/attacks.py`, `core/hybrid_system.py`):**
   - Verified deterministic reproducibility of cropping ($10\%, 25\%, 50\%$), JPEG compression ($Q=70, 50$), Gaussian noise ($\sigma=0.05$), Gaussian blur ($3\times 3$), median filtering ($3\times 3$), and multi-watermark collusion ($N\in\{2,5,10,20,50,100\}$ where each colluder embeds an independent binary watermark).
5. **Leakage-Free Dataset & Host-Level Splits (`core/data_sources.py`, `training/dataset.py`):**
   - Verified that the $1,100$ files in `preprocessed/I_channel/` are de-duplicated by 4-digit DIV2K host prefix into $800$ unique hosts (`560` train, `120` val, `120` test) with `0` host overlap (`train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅`) and disjoint watermark ID ranges (`train: [10000, 50000)`, `val: [50000, 60000)`, `test: [60000, 70000)`).
6. **Blind ANN Architecture, Training, Checkpoint Roundtrip & Host-Free Inference (`models/blind_extractor.py`, `training/`):**
   - Verified forward pass shape `(B, 1, 256, 256) -> (B, 1, 32, 32)` ($276,466$ trainable parameters).
   - Verified training loop, `best_model.pth` / `last_model.pth` serialization, checkpoint reloading, and strictly host-free blind inference (`BlindExtractor.predict` accepts only the attacked image).
7. **Self-Contained Synthetic Fixture Roundtrip:**
   - Verified that the entire pipeline (host synthesis, watermark generation, embedding, attacks, non-blind extraction, ANN smoke training, and blind inference) passes on a clean checkout without requiring external DIV2K downloads.
