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
