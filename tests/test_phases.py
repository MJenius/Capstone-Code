"""
Unit and integration tests for the Master CLI, Configuration Loader, and Phase Orchestration.
"""
from pathlib import Path
import numpy as np
import pytest

from config import load_config
from pipeline.phase_runner import PhaseRunner
from core.evaluation import calculate_psnr, calculate_ssim, calculate_nc, calculate_ber


def test_config_loading():
    """Verify that the central YAML configuration loads with expected keys."""
    cfg = load_config()
    assert "pipeline" in cfg
    assert "paths" in cfg
    assert "attacks" in cfg
    assert cfg["pipeline"]["acm_iterations"] == 10
    assert cfg["pipeline"]["catalan_iterations"] == 5
    assert cfg["pipeline"]["catalan_key"] == 7
    assert cfg["pipeline"]["alpha_base"] == 0.012


def test_phase0_environment(tmp_path):
    """Test Phase 0 audit execution."""
    runner = PhaseRunner()
    res = runner.run_phase0()
    assert res["phase"] == "Phase 0"
    assert "metrics" in res
    assert res["metrics"]["raw_images_found"]["val"] > 0


def test_phase1_preprocessing():
    """Test Phase 1 preprocessing execution."""
    runner = PhaseRunner()
    res = runner.run_phase1()
    assert res["phase"] == "Phase 1"
    assert "resized_shape" in res["metrics"]
    assert res["metrics"]["resized_shape"]["val"] == [256, 256, 3]


def test_phase2_watermark_reversibility():
    """Test Phase 2 exact watermark reversibility."""
    runner = PhaseRunner()
    res = runner.run_phase2()
    assert res["phase"] == "Phase 2"
    assert res["metrics"]["exact_reversibility"]["val"] is True
    assert res["metrics"]["ber"]["val"] == 0.0
    assert res["metrics"]["nc"]["val"] == pytest.approx(1.0, abs=1e-5)


def test_phase3_embedding_fidelity():
    """Test Phase 3 adaptive embedding fidelity."""
    runner = PhaseRunner()
    res = runner.run_phase3()
    assert res["phase"] == "Phase 3"
    assert res["metrics"]["psnr_db"]["val"] >= 40.0
    assert res["metrics"]["ssim"]["val"] >= 0.98


def test_phase4_attacks_reproducibility():
    """Test Phase 4 attack runner produces identical attacked images with fixed seed."""
    runner = PhaseRunner()
    res_crop1 = runner.run_phase4("crop")
    res_crop2 = runner.run_phase4("crop")
    assert res_crop1["attack_type"] == "crop"
    assert res_crop1["parameters"]["retained_pct"] == res_crop2["parameters"]["retained_pct"]


def test_phase5_extraction():
    """Test Phase 5 extraction recovers watermark under crop attack."""
    runner = PhaseRunner()
    res = runner.run_phase5()
    assert res["phase"] == "Phase 5"
    assert res["metrics"]["nc"]["val"] >= 0.98
    assert res["metrics"]["ber"]["val"] == 0.0


def test_demo_execution():
    """Test that master demo mode executes end-to-end and produces artifact."""
    runner = PhaseRunner()
    res = runner.run_demo(attack_name="crop")
    poster_path = Path(res["poster_path"])
    report_path = Path(res["report_path"])
    assert poster_path.exists()
    assert report_path.exists()
    assert res["metrics"]["psnr"] >= 40.0
    assert res["metrics"]["nc"] >= 0.98
