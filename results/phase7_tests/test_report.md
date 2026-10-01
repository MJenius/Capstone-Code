# Automated Test Execution Report
**Timestamp:** 2026-10-01T12:38:48.466358

- **Total Tests:** 20
- **Passed:** 20
- **Failed:** 0
- **Skipped:** 0
- **Test Pass Rate:** 100.0% *(Note: Pass rate reflects test assertions, not code coverage)*

## Test Case Details

| Test Identifier | Status | Duration (s) |
| :--- | :--- | :--- |
| `tests/test_components.py::test_catalan_reversibility` | ✅ PASS | 0.069 |
| `tests/test_components.py::test_scrambler_reversibility` | ✅ PASS | 0.009 |
| `tests/test_components.py::test_mosaic_generation` | ✅ PASS | 0.001 |
| `tests/test_components.py::test_adaptive_embedder_fidelity` | ✅ PASS | 0.014 |
| `tests/test_components.py::test_baseline_embedder_fidelity` | ✅ PASS | 0.001 |
| `tests/test_components.py::test_cropping_attack_and_mask_reproducibility` | ✅ PASS | 0.002 |
| `tests/test_components.py::test_signal_attack_reproducibility` | ✅ PASS | 0.005 |
| `tests/test_components.py::test_collusion_attack_reproducibility` | ✅ PASS | 0.005 |
| `tests/test_components.py::test_nc_and_ber_metrics` | ✅ PASS | 0.001 |
| `tests/test_phases.py::test_config_loading` | ✅ PASS | 0.005 |
| `tests/test_phases.py::test_phase0_environment` | ✅ PASS | 0.102 |
| `tests/test_phases.py::test_phase1_preprocessing` | ✅ PASS | 1.957 |
| `tests/test_phases.py::test_phase2_watermark_reversibility` | ✅ PASS | 0.880 |
| `tests/test_phases.py::test_phase3_embedding_fidelity` | ✅ PASS | 1.493 |
| `tests/test_phases.py::test_phase4_attacks_reproducibility` | ✅ PASS | 1.225 |
| `tests/test_phases.py::test_phase5_extraction` | ✅ PASS | 1.052 |
| `tests/test_phases.py::test_demo_execution` | ✅ PASS | 2.573 |
| `tests/test_pipeline.py::test_full_pipeline_roundtrip` | ✅ PASS | 0.098 |
| `tests/test_pipeline.py::test_mosaic_redundancy_under_cropping` | ✅ PASS | 0.003 |
| `tests/test_pipeline.py::test_benchmark_smoke_test` | ✅ PASS | 0.671 |