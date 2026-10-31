# Final Automated Test Execution Report

**Timestamp:** `2026-10-31T15:09:23`  
**Runner:** `pytest 9.1.1` (Python `3.14.7`)  
**Command:** `python -m pytest -v` / `python run.py --phase tests`

- **Total Tests Executed:** `26`
- **Passed:** `26`
- **Failed:** `0`
- **Skipped:** `0`
- **Test Pass Rate:** `100.0%` *(reflects assertion pass rate across all unit and integration suites, not line coverage)*

## Test Case Breakdown

| Test Identifier | Category | Status | Duration (s) |
| :--- | :--- | :--- | :--- |
| `tests/test_ann_and_system.py::test_synthetic_fixture_roundtrip_without_dataset` | Synthetic Fixture Roundtrip | ✅ PASS | 0.096 |
| `tests/test_ann_and_system.py::test_collusion_correctness_independent_watermarks` | Multi-Watermark Collusion | ✅ PASS | 0.067 |
| `tests/test_ann_and_system.py::test_host_level_split_zero_leakage` | Zero-Leakage Host Split | ✅ PASS | 0.008 |
| `tests/test_ann_and_system.py::test_ann_model_shapes_and_forward` | Blind ANN Forward Pass `(B,1,256,256)->(B,1,32,32)` | ✅ PASS | 0.175 |
| `tests/test_ann_and_system.py::test_ann_smoke_training_and_checkpoint_roundtrip` | ANN Training + Checkpoint Save/Load + Blind Inference | ✅ PASS | 2.985 |
| `tests/test_ann_and_system.py::test_streamlit_app_and_cli_imports` | Streamlit App & CLI Phase Entrypoints | ✅ PASS | 0.003 |
| `tests/test_components.py::test_catalan_reversibility` | Catalan Transform Bijection | ✅ PASS | 0.092 |
| `tests/test_components.py::test_scrambler_reversibility` | Arnold Cat Map Bijection | ✅ PASS | 0.002 |
| `tests/test_components.py::test_mosaic_generation` | $8\times 8$ Mosaic Tiling | ✅ PASS | 0.001 |
| `tests/test_components.py::test_adaptive_embedder_fidelity` | Adaptive Texture Embedder (`PSNR > 38 dB`) | ✅ PASS | 0.019 |
| `tests/test_components.py::test_baseline_embedder_fidelity` | Center-Tile Baseline Embedder | ✅ PASS | 0.002 |
| `tests/test_components.py::test_cropping_attack_and_mask_reproducibility` | Cropping Attack & Mask Reproducibility | ✅ PASS | 0.002 |
| `tests/test_components.py::test_signal_attack_reproducibility` | JPEG / Noise / Blur Reproducibility | ✅ PASS | 0.007 |
| `tests/test_components.py::test_collusion_attack_reproducibility` | Collusion Averaging Reproducibility | ✅ PASS | 0.007 |
| `tests/test_components.py::test_nc_and_ber_metrics` | NC & BER Mathematical Edge Cases | ✅ PASS | 0.001 |
| `tests/test_phases.py::test_config_loading` | YAML Configuration Loading | ✅ PASS | 0.007 |
| `tests/test_phases.py::test_phase0_environment` | Phase 0 Environment Audit | ✅ PASS | 0.087 |
| `tests/test_phases.py::test_phase1_preprocessing` | Phase 1 YIQ Preprocessing | ✅ PASS | 1.824 |
| `tests/test_phases.py::test_phase2_watermark_reversibility` | Phase 2 Watermark Reversibility | ✅ PASS | 0.622 |
| `tests/test_phases.py::test_phase3_embedding_fidelity` | Phase 3 Perceptual Embedding | ✅ PASS | 1.099 |
| `tests/test_phases.py::test_phase4_attacks_reproducibility` | Phase 4 Attack Suite | ✅ PASS | 1.116 |
| `tests/test_phases.py::test_phase5_extraction` | Phase 5 Non-Blind Extraction | ✅ PASS | 1.081 |
| `tests/test_phases.py::test_demo_execution` | Full Pipeline Demo Poster Generation | ✅ PASS | 8.476 |
| `tests/test_pipeline.py::test_full_pipeline_roundtrip` | End-to-End Clean Roundtrip | ✅ PASS | 0.046 |
| `tests/test_pipeline.py::test_mosaic_redundancy_under_cropping` | 50% Crop Mosaic Redundancy | ✅ PASS | 0.002 |
| `tests/test_pipeline.py::test_benchmark_smoke_test` | Benchmark Suite Smoke Test | ✅ PASS | 0.421 |
