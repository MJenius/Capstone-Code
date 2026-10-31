# Automated Test Execution Report
**Timestamp:** 2026-10-31T15:09:23.690889

- **Total Tests:** 26
- **Passed:** 26
- **Failed:** 0
- **Skipped:** 0
- **Test Pass Rate:** 100.0% *(Note: Pass rate reflects test assertions, not code coverage)*

## Test Case Details

| Test Identifier | Status | Duration (s) |
| :--- | :--- | :--- |
| `tests/test_ann_and_system.py::test_synthetic_fixture_roundtrip_without_dataset` | ✅ PASS | 0.096 |
| `tests/test_ann_and_system.py::test_collusion_correctness_independent_watermarks` | ✅ PASS | 0.067 |
| `tests/test_ann_and_system.py::test_host_level_split_zero_leakage` | ✅ PASS | 0.008 |
| `tests/test_ann_and_system.py::test_ann_model_shapes_and_forward` | ✅ PASS | 0.175 |
| `tests/test_ann_and_system.py::test_ann_smoke_training_and_checkpoint_roundtrip` | ✅ PASS | 2.985 |
| `tests/test_ann_and_system.py::test_streamlit_app_and_cli_imports` | ✅ PASS | 0.003 |
| `tests/test_components.py::test_catalan_reversibility` | ✅ PASS | 0.092 |
| `tests/test_components.py::test_scrambler_reversibility` | ✅ PASS | 0.002 |
| `tests/test_components.py::test_mosaic_generation` | ✅ PASS | 0.001 |
| `tests/test_components.py::test_adaptive_embedder_fidelity` | ✅ PASS | 0.019 |
| `tests/test_components.py::test_baseline_embedder_fidelity` | ✅ PASS | 0.002 |
| `tests/test_components.py::test_cropping_attack_and_mask_reproducibility` | ✅ PASS | 0.002 |
| `tests/test_components.py::test_signal_attack_reproducibility` | ✅ PASS | 0.007 |
| `tests/test_components.py::test_collusion_attack_reproducibility` | ✅ PASS | 0.007 |
| `tests/test_components.py::test_nc_and_ber_metrics` | ✅ PASS | 0.001 |
| `tests/test_phases.py::test_config_loading` | ✅ PASS | 0.007 |
| `tests/test_phases.py::test_phase0_environment` | ✅ PASS | 0.087 |
| `tests/test_phases.py::test_phase1_preprocessing` | ✅ PASS | 1.824 |
| `tests/test_phases.py::test_phase2_watermark_reversibility` | ✅ PASS | 0.622 |
| `tests/test_phases.py::test_phase3_embedding_fidelity` | ✅ PASS | 1.099 |
| `tests/test_phases.py::test_phase4_attacks_reproducibility` | ✅ PASS | 1.116 |
| `tests/test_phases.py::test_phase5_extraction` | ✅ PASS | 1.081 |
| `tests/test_phases.py::test_demo_execution` | ✅ PASS | 8.476 |
| `tests/test_pipeline.py::test_full_pipeline_roundtrip` | ✅ PASS | 0.046 |
| `tests/test_pipeline.py::test_mosaic_redundancy_under_cropping` | ✅ PASS | 0.002 |
| `tests/test_pipeline.py::test_benchmark_smoke_test` | ✅ PASS | 0.421 |