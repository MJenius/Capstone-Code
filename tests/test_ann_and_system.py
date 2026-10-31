"""
Comprehensive automated tests for the Blind ANN Watermark Extractor, Dataset Splits,
Smoke Training, Checkpointing, Host-Free Inference, Streamlit App Imports, and Demo Artifacts.
"""
from pathlib import Path
import json
import numpy as np
import pytest
import torch

from core.preprocessing import bgr_to_yiq, yiq_to_bgr, normalize_channel, denormalize_channel, resize_image
from core.watermark import WatermarkTransformer
from core.embedding import WatermarkEmbedder
from core.extraction import NonBlindExtractor
from core.data_sources import _synthetic_host, make_watermark, host_level_split, list_hosts
from core.hybrid_system import HybridSystem
from models.blind_extractor import BlindWatermarkExtractor, build_model, count_parameters
from training.dataset import ANN_DIR, WatermarkANNDataset, generate_dataset
from training.trainer import Trainer, save_checkpoint, load_checkpoint
from training.inference import BlindExtractor, image_to_i_channel, score_against_ground_truth
from training.metrics import bit_metrics


def test_synthetic_fixture_roundtrip_without_dataset():
    """Verify full preprocessing, transform, mosaic, adaptive embedding, and non-blind recovery on a synthetic fixture."""
    host_bgr = _synthetic_host(0, size=256)
    yiq = bgr_to_yiq(host_bgr)
    bgr_rec = yiq_to_bgr(yiq)
    assert bgr_rec.shape == (256, 256, 3)

    i_norm, i_min, i_max = normalize_channel(yiq[:, :, 1])
    assert i_norm.shape == (256, 256)
    assert 0.0 <= float(i_norm.min()) <= float(i_norm.max()) <= 1.0 + 1e-6

    wm = make_watermark(12345, size=32)
    wt = WatermarkTransformer(watermark_size=32, acm_iterations=10, catalan_iterations=5, catalan_key=7)
    rev = wt.verify_reversibility(wm)
    assert rev["exact_reversibility"] is True
    assert rev["watermark_mosaic"].shape == (256, 256)

    emb = WatermarkEmbedder(alpha_base=0.012, sensitivity=2.0).embed_adaptive(i_norm, rev["watermark_mosaic"])
    assert emb["embedded_i"].shape == (256, 256)
    assert emb["psnr"] > 35.0

    ext = NonBlindExtractor(alpha_base=0.012).extract(emb["embedded_i"], i_norm, ground_truth_binary=wm)
    assert ext["metrics"]["nc"] > 0.99
    assert ext["metrics"]["ber"] == 0.0


def test_collusion_correctness_independent_watermarks():
    """Verify collusion attack uses N distinct watermarks and dilutes victim correlation as N increases."""
    host_bgr = _synthetic_host(1, size=256)
    host_i = normalize_channel(bgr_to_yiq(host_bgr)[:, :, 1])[0]
    wm_victim = make_watermark(0)
    hs = HybridSystem()

    cp2 = hs.collusion_copies(host_i, wm_victim, n=2, seed=42)
    cp10 = hs.collusion_copies(host_i, wm_victim, n=10, seed=42)
    assert len(cp2["copies"]) == 2
    assert len(cp10["copies"]) == 10
    assert not np.allclose(cp2["copies"][0], cp2["copies"][1])

    atk2 = hs.apply_attack("collusion_2", cp2["copies"][0], seed=42, collusion_copies=cp2["copies"])
    atk10 = hs.apply_attack("collusion_10", cp10["copies"][0], seed=42, collusion_copies=cp10["copies"])
    nb2 = hs.extract_nonblind(atk2["image"], host_i)
    nb10 = hs.extract_nonblind(atk10["image"], host_i)
    nc2 = bit_metrics(nb2["recovered"], wm_victim)["nc"]
    nc10 = bit_metrics(nb10["recovered"], wm_victim)["nc"]
    assert nc2 > nc10


def test_host_level_split_zero_leakage():
    """Verify host-level splitting guarantees zero host overlap across train, val, and test."""
    hosts = list_hosts()
    splits = host_level_split(hosts, seed=42, save=False)
    tr_ids = {h.host_id for h in splits["train"]}
    va_ids = {h.host_id for h in splits["val"]}
    te_ids = {h.host_id for h in splits["test"]}
    assert len(tr_ids & va_ids) == 0
    assert len(tr_ids & te_ids) == 0
    assert len(va_ids & te_ids) == 0


def test_ann_model_shapes_and_forward():
    """Verify BlindWatermarkExtractor tensor shapes, parameter count, and probability range."""
    model = build_model()
    assert count_parameters(model) > 10_000
    x = torch.rand(2, 1, 256, 256)
    logits = model(x)
    probs = model.predict_proba(x)
    assert logits.shape == (2, 1, 32, 32)
    assert probs.shape == (2, 1, 32, 32)
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0


def test_ann_smoke_training_and_checkpoint_roundtrip(tmp_path):
    """Execute a real smoke training step on synthetic hosts, save checkpoint, load it, and run blind inference."""
    ds_dir = tmp_path / "ann_smoke"
    summary = generate_dataset(
        out_dir=ds_dir,
        samples_per_host={"train": 1, "val": 1, "test": 1},
        max_hosts={"train": 4, "val": 2, "test": 2},
        seed=42,
    )
    assert summary["data_validation_status"] == "PASS"
    assert summary["label_dimensions"] == [32, 32]

    tr_ds = WatermarkANNDataset("train", root=ds_dir)
    va_ds = WatermarkANNDataset("val", root=ds_dir)
    ckpt_dir = tmp_path / "ckpts"
    trainer = Trainer(tr_ds, va_ds, epochs=1, batch_size=2, lr=1e-3, ckpt_dir=ckpt_dir, device="cpu")
    res = trainer.fit()
    assert res["epochs_completed"] == 1
    assert (ckpt_dir / "best_model.pth").exists()
    assert (ckpt_dir / "last_model.pth").exists()

    # Load checkpoint and verify host-free blind inference
    ext = BlindExtractor(ckpt_dir / "best_model.pth", device="cpu")
    sample_x, sample_y, _ = va_ds[0]
    out = ext.predict(sample_x[0].numpy())
    assert out["probability"].shape == (32, 32)
    assert out["binary"].shape == (32, 32)
    metrics = score_against_ground_truth(out["binary"], sample_y[0].numpy())
    assert "nc" in metrics and "ber" in metrics


def test_streamlit_app_and_cli_imports():
    """Verify Streamlit app module and CLI modules import cleanly."""
    import run
    import pipeline.ann_phases as ap
    assert hasattr(run, "main")
    assert hasattr(ap, "phase_ann_train")
    assert hasattr(ap, "phase_comparison")
    assert hasattr(ap, "phase_final_demo")


def test_zero_host_and_watermark_overlap_in_generated_dataset():
    """Verify data/ann/metadata.csv has zero host overlap and zero non-canonical watermark-ID overlap across splits."""
    import pandas as pd
    meta_path = ANN_DIR / "metadata.csv"
    if not meta_path.exists():
        pytest.skip("data/ann/metadata.csv not yet generated")
    df = pd.read_csv(meta_path)
    splits = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    # Zero host overlap
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        hosts_a = set(splits[a]["host_id"].unique())
        hosts_b = set(splits[b]["host_id"].unique())
        assert len(hosts_a & hosts_b) == 0, f"Host overlap between {a} and {b}"
        # Zero watermark-ID overlap (excluding canonical ID 0 used only on 1 sample of host 0)
        wms_a = set(splits[a]["watermark_id"].unique()) - {0}
        wms_b = set(splits[b]["watermark_id"].unique()) - {0}
        assert len(wms_a & wms_b) == 0, f"Watermark ID overlap between {a} and {b}"
    # Verify test set has diverse watermark patterns
    assert len(set(splits["test"]["watermark_id"].unique())) >= 50


def test_collusion_rejects_missing_host_and_tracks_victim():
    """Verify AttackRunner.run_collusion forbids fabricating colluders on an already-watermarked image and tracks victim ID."""
    from core.attacks import AttackRunner
    host_bgr = _synthetic_host(2, size=256)
    host_i = normalize_channel(bgr_to_yiq(host_bgr)[:, :, 1])[0]
    wm = make_watermark(60001)
    hs = HybridSystem()
    wm_i = hs.embed(host_i, wm)
    runner = AttackRunner(target_size=256)

    # Calling run_collusion without host_image or independent_watermarked_list MUST raise ValueError
    with pytest.raises(ValueError, match="Scientifically valid collusion requires"):
        runner.run_collusion(wm_i, n_colluders=5, seed=42)

    # Calling with clean host_image succeeds and tracks victim and colluder IDs
    res = runner.run_collusion(wm_i, n_colluders=5, seed=42, host_image=host_i, victim_watermark=wm, victim_watermark_id=60001)
    assert res["attacked_image"].shape == (256, 256)
    assert res["victim_watermark_id"] == 60001
    assert len(res["colluder_watermark_ids"]) == 4


def test_blind_inference_strictly_single_image_api():
    """Verify BlindExtractor.predict signature accepts ONLY (self, image, threshold=0.5) and runs with a single image."""
    import inspect
    sig = inspect.signature(BlindExtractor.predict)
    params = list(sig.parameters.keys())
    assert params == ["self", "image", "threshold"], f"Unexpected parameters in BlindExtractor.predict: {params}"
    forbidden = {"host", "host_image", "host_i", "mask", "crop_mask", "residual", "ground_truth", "wm"}
    assert len(set(params) & forbidden) == 0

    ckpt = Path(__file__).resolve().parent.parent / "models" / "checkpoints" / "best_model.pth"
    if not ckpt.exists():
        pytest.skip("Trained checkpoint not found")
    ext = BlindExtractor(ckpt, device="cpu")
    host_bgr = _synthetic_host(3, size=256)
    host_i = normalize_channel(bgr_to_yiq(host_bgr)[:, :, 1])[0]
    wm = make_watermark(60003)
    hs = HybridSystem()
    wm_i = hs.embed(host_i, wm)
    atk_i = hs.apply_attack("crop_25", wm_i, seed=99)["image"]

    # Blind prediction receives ONLY atk_i
    pred = ext.predict(atk_i)
    assert pred["binary"].shape == (32, 32)
    # Ground truth is evaluated strictly after prediction
    post_eval = score_against_ground_truth(pred["binary"], wm)
    assert post_eval["nc"] > 0.75


def test_ood_watermark_generalization_not_memorization():
    """Verify the trained BlindExtractor recovers unseen i.i.d. Bernoulli(0.5) and bit-inverted watermarks."""
    ckpt = Path(__file__).resolve().parent.parent / "models" / "checkpoints" / "best_model.pth"
    if not ckpt.exists():
        pytest.skip("Trained checkpoint not found")
    ext = BlindExtractor(ckpt, device="cpu")
    host_bgr = _synthetic_host(4, size=256)
    host_i = normalize_channel(bgr_to_yiq(host_bgr)[:, :, 1])[0]
    hs = HybridSystem()

    rng = np.random.RandomState(2026)
    wm_iid = (rng.rand(32, 32) > 0.5).astype(np.uint8)
    wm_inv = (1 - wm_iid).astype(np.uint8)

    emb_iid = hs.embed(host_i, wm_iid)
    emb_inv = hs.embed(host_i, wm_inv)

    pred_iid = ext.predict(emb_iid)
    pred_inv = ext.predict(emb_inv)
    pred_unwm = ext.predict(host_i)

    assert score_against_ground_truth(pred_iid["binary"], wm_iid)["nc"] > 0.85
    assert score_against_ground_truth(pred_inv["binary"], wm_inv)["nc"] > 0.85
    assert abs(score_against_ground_truth(pred_unwm["binary"], wm_iid)["nc"]) < 0.15


def test_streamlit_app_headless_execution():
    """Run app.py via streamlit.testing.v1.AppTest across all 10 tabs and verify initialization without exceptions."""
    from streamlit.testing.v1 import AppTest
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=60)
    at.run()
    assert not at.exception, f"Streamlit app raised exception on initial render: {at.exception}"
    assert len(at.tabs) == 10
    # Confirm primary input controls and metrics render properly
    assert len(at.selectbox) >= 4
    assert len(at.sidebar.selectbox) >= 1
    assert any("RUN COMPLETE DEMO" in b.label for b in at.button)


def test_portable_relative_paths_in_reports():
    """Verify no local Windows user directory paths (Users\\mjeni or Users/mjeni) appear in generated JSON/MD reports."""
    results_dir = Path(__file__).resolve().parent.parent / "results"
    if not results_dir.exists():
        pytest.skip("results/ directory not found")
    offending = []
    for ext in ("*.json", "*.md", "*.txt", "*.csv"):
        for fp in results_dir.rglob(ext):
            txt = fp.read_text(encoding="utf-8", errors="ignore")
            if "mjeni" in txt or "C:\\Users" in txt or "C:/Users" in txt:
                offending.append(fp.relative_to(results_dir.parent).as_posix())
    assert len(offending) == 0, f"Found non-portable local paths in: {offending}"


