# Final Validation Report — Capstone Research Pipeline Refactoring

**Date of Execution:** 2026-10-01  
**Project:** Hybrid Robust Digital Image Watermarking Framework (Pre-ANN Non-Blind Baseline)  
**Author / Identifier:** PW26_PAC_01  

---

## 1. Commands Executed & Pass / Fail Summary

| Step | Executed Command | Exit Code | Status | Key Output Generated |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `python run.py --phase demo` | `0` | **PASS** | `results/demo/full_pipeline_demo.png` |
| 2 | `python run.py --phase tests --verbose` | `0` | **PASS** | `results/phase7_tests/test_report.md` (20 passed) |
| 3 | `python run.py --phase reproduce --quick` | `0` | **PASS** | Environment, tests, benchmark & demo verified |
| 4 | `python run.py --phase preprocessing` | `0` | **PASS** | `results/phase1_preprocessing/phase1_preprocessing_grid.png` |
| 5 | `python run.py --phase watermark` | `0` | **PASS** | `results/phase2_watermark/phase2_watermark_pipeline.png` |
| 6 | `python run.py --phase embedding` | `0` | **PASS** | `results/phase3_embedding/phase3_embedding_pipeline.png` |
| 7 | `python run.py --phase attacks --attack crop` | `0` | **PASS** | `results/phase4_attacks/crop/attack_crop_visualization.png` |
| 8 | `python run.py --phase attacks --attack noise` | `0` | **PASS** | `results/phase4_attacks/noise/attack_noise_visualization.png` |
| 9 | `python run.py --phase attacks --attack jpeg` | `0` | **PASS** | `results/phase4_attacks/jpeg/attack_jpeg_visualization.png` |
| 10 | `python run.py --phase extraction` | `0` | **PASS** | `results/phase5_extraction/phase5_extraction_pipeline.png` |
| 11 | `python run.py --phase ann-data` | `0` | **PASS** | `results/phase8_ann_data/dataset_summary.json` |
| 12 | `python run.py --phase all --quick` | `0` | **PASS** | Complete pipeline execution (Phases 0–8 + Phase 9 status) |

---

## 2. Generated Artifacts Audit

Every major phase produces demonstrable image and structured data artifacts:
- **Demo Poster:** [full_pipeline_demo.png](file:///results/demo/full_pipeline_demo.png) — 12-panel capstone slide graphic.
- **Phase 0 Audit:** [report.json](file:///results/phase0_environment/report.json) & [summary.txt](file:///results/phase0_environment/summary.txt).
- **Phase 1 Decomposition:** [phase1_preprocessing_grid.png](file:///results/phase1_preprocessing/phase1_preprocessing_grid.png).
- **Phase 2 Dual Security:** [phase2_watermark_pipeline.png](file:///results/phase2_watermark/phase2_watermark_pipeline.png).
- **Phase 3 Perceptual Embedding:** [phase3_embedding_pipeline.png](file:///results/phase3_embedding/phase3_embedding_pipeline.png) & [difference_heatmap.png](file:///results/phase3_embedding/difference_heatmap.png).
- **Phase 4 Attack Evaluations:** Visualizations for `crop`, `jpeg`, `noise`, `blur`, and `collusion`.
- **Phase 5 Extraction:** [phase5_extraction_pipeline.png](file:///results/phase5_extraction/phase5_extraction_pipeline.png).
- **Phase 6 Benchmark:** [summary.md](file:///results/phase6_benchmark/summary.md), CSV, JSON, and 4 diagnostic plots (`psnr.png`, `nc_by_attack.png`, `ber_by_attack.png`, `collusion_curve.png`).
- **Phase 7 Testing:** [test_report.md](file:///results/phase7_tests/test_report.md) documenting 20/20 test passes.
- **Phase 8 ANN Preparation:** [metadata.csv](file:///results/phase8_ann_data/metadata.csv) and (input, label) pairs.
- **Phase 9 ANN Interface:** [ann_status.json](file:///results/phase9_ann/ann_status.json).

---

## 3. Empirical Benchmark Summary (Direct Code Execution)

All benchmark numbers are calculated directly by the running code over the 165-image DIV2K test split:
- **Clean Embedding Quality:**
  - Hybrid Framework: **41.57 dB PSNR**, **0.9965 SSIM**
  - Center Baseline: **46.02 dB PSNR**, **0.9983 SSIM**
- **Robustness Advantage (Hybrid):**
  - Cropping (10%): NC = **0.9977** (vs Baseline: 0.8238)
  - Cropping (25%): NC = **0.9972** (vs Baseline: 0.0979)
  - Cropping (50%): NC = **0.9954** (vs Baseline: -0.0006)
  - Gaussian Noise ($\sigma=0.05$): NC = **0.4325** (vs Baseline: 0.2932)
- **Trade-off / Weakness (Hybrid):**
  - JPEG Compression ($Q=70$): Hybrid NC = **0.4138** (vs Baseline: 0.4764)
  - JPEG Compression ($Q=50$): Hybrid NC = **0.2724** (vs Baseline: 0.3418)
  - Gaussian Blur ($k=3$): Hybrid NC = **0.2133** (vs Baseline: 0.2768)
  *Reason:* Baseline concentrates all energy at higher $\alpha=0.08$ in a single 32x32 tile, whereas Hybrid distributes energy at $\alpha_{base}=0.012$ across 64 tiles.
- **Collusion Attack Limitation:**
  - Averaging dilutes victim signal to $\sim 1/N$. Degrades to NC = 0.2554 at $N=5$, and NC = 0.0241 at $N=100$.

---

## 4. Known Technical Limitations & Honest Declarations

1. **Non-Blind Dependency:**
   Current extraction strictly requires the original host image (`(attacked - host) / alpha_base + 0.5`).
2. **ANN Blind Extractor Status:**
   Phase 9 is **PLANNED and NOT YET IMPLEMENTED**. No fabricated neural weights or blind metrics are claimed.
3. **Adaptive Embedding Inversion:**
   Extraction uses `alpha_base` scaling. In highly textured areas where local $\alpha > \alpha_{base}$, exact pixel-wise amplitude scaling is approximate, though spatial averaging and binary thresholding robustly recover binary watermark bits.
4. **Test Pass Rate vs. Coverage:**
   The test suite reports **100% test pass rate** (20 of 20 unit/integration tests passing), which measures behavioral correctness rather than line coverage.
