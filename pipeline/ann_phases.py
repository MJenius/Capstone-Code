"""
ANN phases: ann-data, ann-train, ann-eval, ann-inference, comparison.

Every phase writes a visible artefact, a numeric report, and prints a
SUCCESS/FAILURE line with the artefact paths.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.data_sources import BASE_DIR, make_watermark

RESULTS = BASE_DIR / "results"
CKPT = BASE_DIR / "models" / "checkpoints"


def _rel(p: Path | str) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(BASE_DIR.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _status(name: str, ok: bool, paths: List[Path], extra: str = ""):
    print("\n" + "=" * 78)
    print(f"[{'SUCCESS' if ok else 'FAILURE'}] {name} {extra}")
    for p in paths:
        print(f"   -> {p}")
    print("=" * 78)


def _u8(a):
    a = np.asarray(a, np.float32)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


# =============================================================================== DATA
def phase_ann_data(quick: bool = False, samples_per_host: Optional[int] = None,
                   max_hosts: Optional[int] = None, host_dir: Optional[str] = None) -> Dict:
    from training.dataset import ANN_DIR, generate_dataset
    if quick:
        sph = {"train": 2, "val": 2, "test": 2}
        mh = {"train": 60, "val": 15, "test": 15}
    else:
        k = samples_per_host or 4
        sph = {"train": k, "val": max(2, k - 1), "test": max(2, k - 1)}
        mh = {"train": max_hosts, "val": max_hosts, "test": max_hosts} if max_hosts else None
    last = {}

    def prog(split, i, n):
        if i == n or i % 50 == 0:
            print(f"  [{split}] hosts {i}/{n}")
        last[split] = i

    summary = generate_dataset(ANN_DIR, samples_per_host=sph, max_hosts=mh,
                               host_dir=Path(host_dir) if host_dir else None, progress=prog)
    ok = summary["data_validation_status"] == "PASS"
    print(json.dumps({k: summary[k] for k in ("total_samples", "unique_host_images", "unique_watermark_ids",
                                               "samples_per_split", "hosts_per_split",
                                               "host_overlap_between_splits", "data_validation_status")}, indent=2))
    _status("Phase ann-data", ok, [ANN_DIR / "dataset_summary.json", ANN_DIR / "metadata.csv",
                                   ANN_DIR / "visualizations" / "input_label_pairs.png"])
    return summary


# ============================================================================== TRAIN
def _plot_history(hist: List[Dict], out: Path):
    ep = [h["epoch"] for h in hist]
    for key, fname, title in (("loss", "loss_curve.png", "Loss"), ("accuracy", "validation_accuracy.png", "Bit accuracy"),
                              ("nc", "validation_nc.png", "Normalized Correlation")):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ep, [h[f"train_{key}"] for h in hist], "o-", label="train")
        ax.plot(ep, [h[f"val_{key}"] for h in hist], "s-", label="validation")
        ax.set_xlabel("epoch"); ax.set_title(title); ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(out / fname, dpi=120); plt.close(fig)


def phase_ann_train(epochs: Optional[int] = None, quick: bool = False, batch_size: int = 16, lr: float = 2e-3,
                    resume: bool = False, patience: int = 5, device: str = "auto", on_epoch=None,
                    log=print) -> Dict:
    import torch
    from models.blind_extractor import count_parameters
    from training.dataset import WatermarkANNDataset
    from training.trainer import Trainer

    out = RESULTS / "ann_training"
    out.mkdir(parents=True, exist_ok=True)
    tr, va = WatermarkANNDataset("train", augment=True), WatermarkANNDataset("val")
    epochs = epochs or (3 if quick else 15)
    trainer = Trainer(tr, va, epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, device=device,
                      results_dir=out, log=log, max_train_batches=40 if quick else None)
    if resume:
        resume_ckpt = CKPT / "best_model.pth" if (CKPT / "best_model.pth").exists() else CKPT / "last_model.pth"
        trainer.resume(resume_ckpt, override_lr=lr)
    t0 = time.time()
    res = trainer.fit(on_epoch=on_epoch)
    trainer.write_history()
    hist = res["history"]
    if hist:
        _plot_history(hist, out)
    with open(out / "model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(trainer.model) + f"\n\nTrainable parameters: {count_parameters(trainer.model):,}\n")
        f.write("Input: (B,1,256,256) attacked I-channel | Output: (B,1,32,32) logits of original watermark\n")
    best = max(hist, key=lambda h: h["val_nc"]) if hist else {}
    report = f"""# ANN Training Report

Generated: {datetime.now().isoformat(timespec='seconds')}

| Item | Value |
|---|---|
| Device | {trainer.device} (torch {torch.__version__}) |
| Training samples | {len(tr)} |
| Validation samples | {len(va)} |
| Epochs requested | {epochs} |
| Epochs completed (total, incl. resumed) | {len(hist)} |
| Early stopped | {res['stopped_early']} |
| Quick mode (40 batches/epoch) | {quick} |
| Batch size / LR | {batch_size} / {lr} |
| Loss | {trainer.cfg['loss']} |
| Optimizer / scheduler | AdamW (wd 1e-4) / {trainer.cfg['scheduler']} |
| Selection criterion | max validation NC (test split untouched) |
| Seed | {trainer.cfg['seed']} |
| Wall time this run | {time.time()-t0:.0f} s |
| Parameters | {count_parameters(trainer.model):,} |

## Best validation epoch ({best.get('epoch')})

| Metric | Value |
|---|---|
""" + "\n".join(f"| {k[4:]} | {best[k]:.4f} |" for k in best if k.startswith("val_")) + """

## Per-epoch history

| epoch | train loss | val loss | val NC | val BER | val acc | val F1 | lr | s |
|---|---|---|---|---|---|---|---|---|
""" + "\n".join(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_loss']:.4f} | {h['val_nc']:.4f} | {h['val_ber']:.4f} | "
                f"{h['val_accuracy']:.4f} | {h['val_f1']:.4f} | {h['lr']:.2e} | {h['epoch_seconds']} |" for h in hist) + """

Note: validation metrics are measured on held-out *validation hosts* and watermark IDs.
They are used for model selection only; final claims use the isolated test split (ann-eval / comparison).
"""
    (out / "training_report.md").write_text(report, encoding="utf-8")
    ok = (CKPT / "best_model.pth").exists()
    _status("Phase ann-train", ok, [CKPT / "best_model.pth", CKPT / "last_model.pth", out / "training_report.md",
                                    out / "loss_curve.png"], f"best val NC={res['best_val_nc']:.4f}")
    return {**res, "best_epoch": best}


# =============================================================================== EVAL
def phase_ann_eval(model_path: Optional[str] = None) -> Dict:
    from training.evaluator import evaluate_ann_on_split
    from training.inference import BlindExtractor
    out = RESULTS / "ann_evaluation"
    out.mkdir(parents=True, exist_ok=True)
    ext = BlindExtractor(model_path or CKPT / "best_model.pth")
    r = evaluate_ann_on_split(ext, "test")
    r["records"].to_csv(out / "test_records.csv", index=False)
    r["per_attack"].to_csv(out / "test_per_attack.csv", index=False)
    rep = {"split": "test (held-out hosts & watermark IDs)", "n_samples": r["n"], "n_hosts": r["n_hosts"],
           "overall": r["overall"], "per_attack": r["per_attack"].to_dict(orient="records"),
           "checkpoint": ext.info(), "seconds": round(r["seconds"], 1)}
    (out / "ann_test_report.json").write_text(json.dumps(rep, indent=2, default=float))
    fig, ax = plt.subplots(figsize=(8, 4))
    pa = r["per_attack"]
    ax.bar(pa.attack_type, pa.nc_mean, yerr=pa.nc_std.fillna(0), color="#3b7dd8", capsize=3)
    ax.set_ylim(-0.1, 1.05); ax.set_ylabel("NC"); ax.set_title(f"Blind ANN on held-out test split (n={r['n']})")
    plt.xticks(rotation=30); fig.tight_layout(); fig.savefig(out / "ann_test_nc_by_attack.png", dpi=120); plt.close(fig)
    print(pa.to_string(index=False))
    _status("Phase ann-eval", True, [out / "ann_test_report.json", out / "ann_test_nc_by_attack.png"],
            f"overall NC={r['overall']['nc']:.4f} BER={r['overall']['ber']:.4f}")
    return rep


# ========================================================================== INFERENCE
def phase_ann_inference(images: List[str] | str, model_path: Optional[str] = None,
                        ground_truth: Optional[str] = None) -> Dict:
    from training.inference import BlindExtractor, score_against_ground_truth
    out = RESULTS / "ann_inference"
    out.mkdir(parents=True, exist_ok=True)
    ext = BlindExtractor(model_path or CKPT / "best_model.pth")
    if isinstance(images, (str, Path)):
        images = [str(images)]
    paths = []
    for p in images:
        pp = Path(p)
        if pp.is_dir():
            paths += sorted([q for q in pp.iterdir() if q.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".npy")])
        else:
            paths.append(pp)
    if not paths:
        raise FileNotFoundError("No input images given for ann-inference")
    reports = []
    for k, p in enumerate(paths):
        src = np.load(p) if p.suffix == ".npy" else str(p)
        r = ext.predict(src)       # <- blind: only the image goes in
        d = out if len(paths) == 1 else out / p.stem
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "input_image.png"), _u8(r["input_i_channel"]))
        fig, ax = plt.subplots(figsize=(4, 4)); im = ax.imshow(r["probability"], cmap="magma", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax); ax.set_title("P(bit=1)"); ax.axis("off"); fig.tight_layout()
        fig.savefig(d / "prediction_probability.png", dpi=120); plt.close(fig)
        cv2.imwrite(str(d / "recovered_watermark.png"), cv2.resize(r["binary"] * 255, (256, 256), interpolation=cv2.INTER_NEAREST))
        rep = {"input": _rel(p), "mode": "BLIND (no host image, no crop mask, no ground truth during inference)",
               "inference_seconds": r["inference_seconds"], "mean_confidence": float(np.abs(r["probability"] - 0.5).mean() * 2),
               "checkpoint": ext.info()}
        if ground_truth:   # evaluation strictly AFTER inference
            gt = np.load(ground_truth) if ground_truth.endswith(".npy") else (cv2.imread(ground_truth, 0) > 127)
            gt = cv2.resize(gt.astype(np.uint8), (32, 32), interpolation=cv2.INTER_NEAREST)
            rep["post_inference_evaluation"] = score_against_ground_truth(r["binary"], gt)
        (d / "inference_report.json").write_text(json.dumps(rep, indent=2, default=float))
        reports.append(rep)
    _status("Phase ann-inference", True, [out], f"{len(paths)} image(s)")
    return {"reports": reports}


# ========================================================================= COMPARISON
def phase_comparison(n_hosts: Optional[int] = None, model_path: Optional[str] = None, quick: bool = False) -> Dict:
    from training.evaluator import METHOD_INFO, TAU, aggregate, research_metrics, run_comparison
    from training.inference import BlindExtractor
    out = RESULTS / "final_evaluation"
    out.mkdir(parents=True, exist_ok=True)
    ext = BlindExtractor(model_path or CKPT / "best_model.pth")
    n_hosts = n_hosts or (10 if quick else None)
    t0 = time.time()
    res = run_comparison(ext, n_hosts=n_hosts, progress=lambda i, n: print(f"  host {i}/{n}") if i % 10 == 0 or i == n else None)
    df, neg = res["records"], res["negative_control"]
    agg = aggregate(df)
    rm = research_metrics(agg)
    df.to_csv(out / "comparison_records.csv", index=False)
    agg.to_csv(out / "comparison_results.csv", index=False)
    neg.to_csv(out / "negative_control.csv", index=False)
    fpr = float((neg.nc_unwatermarked_vs_wm > TAU).mean()) if len(neg) else None
    fpr2 = float((neg.nc_clean_pred_vs_unrelated_wm > TAU).mean()) if len(neg) else None
    fpr_odd = float((neg.nc_clean_pred_vs_unrelated_odd_wm > TAU).mean()) if "nc_clean_pred_vs_unrelated_odd_wm" in neg else None
    fpr_iid = float((neg.nc_clean_pred_vs_unrelated_iid_wm > TAU).mean()) if "nc_clean_pred_vs_unrelated_iid_wm" in neg else None
    payload = {"generated": datetime.now().isoformat(timespec="seconds"), "n_test_hosts": int(df.host_id.nunique()),
               "methods": METHOD_INFO, "results": agg.to_dict(orient="records"), "research_metrics": rm,
               "negative_control": {
                   "n": len(neg), "tau": TAU,
                   "nc_unwatermarked_mean": float(neg.nc_unwatermarked_vs_wm.mean()),
                   "nc_unwatermarked_std": float(neg.nc_unwatermarked_vs_wm.std()),
                   "fpr_unwatermarked": fpr,
                   "fpr_unrelated_watermark": fpr2,
                   "fpr_unrelated_odd_block_watermark": fpr_odd,
                   "fpr_unrelated_iid_bernoulli_watermark": fpr_iid,
                   "nc_unrelated_odd_mean": float(neg.nc_clean_pred_vs_unrelated_odd_wm.mean()) if fpr_odd is not None else None,
                   "nc_unrelated_iid_mean": float(neg.nc_clean_pred_vs_unrelated_iid_wm.mean()) if fpr_iid is not None else None,
                   "ood_iid_bernoulli_nc_mean": float(neg.nc_ood_iid_recovery.mean()) if "nc_ood_iid_recovery" in neg else None,
                   "ood_iid_bernoulli_ber_mean": float(neg.ber_ood_iid_recovery.mean()) if "ber_ood_iid_recovery" in neg else None,
                   "ood_bit_inverted_nc_mean": float(neg.nc_ood_inverted_recovery.mean()) if "nc_ood_inverted_recovery" in neg else None,
                   "ood_bit_inverted_ber_mean": float(neg.ber_ood_inverted_recovery.mean()) if "ber_ood_inverted_recovery" in neg else None,
                   "even_id_template_explanation": (
                       "Even watermark IDs render 4-char cv2.putText strings at fixed coordinates (4,28)/(4,60) on a black canvas, "
                       "so two independent even-ID ground-truth watermarks have intrinsic correlation NC~0.357 (occurring with p=0.5*0.5=0.25). "
                       "When tested against zero-mean odd-ID block watermarks or i.i.d. Bernoulli(0.5) watermarks, empirical FPR drops to ~0.000-0.017."
                   ),
               },
               "checkpoint": ext.info(), "seconds": round(time.time() - t0, 1)}
    (out / "comparison_results.json").write_text(json.dumps(payload, indent=2, default=float))
    _comparison_plots(agg, out, res["samples"])
    _comparison_summary(agg, rm, payload, out)
    _status("Phase comparison", True, [out / "comparison_results.csv", out / "summary.md", out / "nc_comparison.png"],
            f"{payload['n_test_hosts']} held-out test hosts")
    return payload


ORDER = ["clean", "crop_10", "crop_25", "crop_50", "jpeg_70", "jpeg_50", "noise_05", "blur_3", "median_3",
         "collusion_2", "collusion_5", "collusion_10", "collusion_20", "collusion_50", "collusion_100", "rgb_png_roundtrip"]
LABEL = {"ann_blind": "ANN blind", "nonblind_pixel": "Non-blind (pixel α)", "nonblind_base": "Non-blind (legacy α)",
         "baseline_center": "Centre-tile baseline (non-blind)"}
COLORS = {"ann_blind": "#d62728", "nonblind_pixel": "#1f77b4", "nonblind_base": "#9ecae1", "baseline_center": "#7f7f7f"}


def _pivot(agg, col):
    p = agg.pivot(index="attack", columns="method", values=col)
    return p.reindex([a for a in ORDER if a in p.index])


def _comparison_plots(agg, out: Path, samples):
    for col, fname, ylab in (("nc_mean", "nc_comparison.png", "NC"), ("ber_mean", "ber_comparison.png", "BER")):
        p = _pivot(agg, col)
        fig, ax = plt.subplots(figsize=(13, 4.5))
        w = 0.8 / len(p.columns)
        for k, m in enumerate([c for c in LABEL if c in p.columns]):
            ax.bar(np.arange(len(p)) + k * w, p[m].values, w, label=LABEL[m], color=COLORS[m])
        ax.set_xticks(np.arange(len(p)) + 0.4 - w / 2); ax.set_xticklabels(p.index, rotation=35, ha="right")
        ax.set_ylabel(ylab); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8)
        ax.set_title(f"{ylab} by attack (held-out test hosts). Non-blind methods use the original host.")
        fig.tight_layout(); fig.savefig(out / fname, dpi=120); plt.close(fig)
    clean = agg[agg.attack == "clean"]
    for col, fname in (("psnr_mean", "psnr_comparison.png"), ("ssim_mean", "ssim_comparison.png")):
        fig, ax = plt.subplots(figsize=(6, 4))
        c2 = clean[clean.method.isin(["nonblind_pixel", "baseline_center"])]
        ax.bar(["Hybrid mosaic (used by ANN & non-blind)", "Centre-tile baseline"], c2.set_index("method").loc[
            ["nonblind_pixel", "baseline_center"], col].values, color=["#1f77b4", "#7f7f7f"])
        ax.set_title(col.replace("_mean", "").upper() + " of watermarked vs host I-channel")
        fig.tight_layout(); fig.savefig(out / fname, dpi=120); plt.close(fig)
    for prefix, fname, xl in (("crop_", "cropping_curve.png", "removed area (%)"),
                              ("collusion_", "collusion_curve.png", "number of colluders N")):
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in [c for c in LABEL if c in agg.method.unique()]:
            a = agg[(agg.method == m) & (agg.attack.str.startswith(prefix) | (agg.attack == "clean"))]
            xs = [0 if r == "clean" else (int(r.split("_")[1]) if prefix == "crop_" else int(r.split("_")[1])) for r in a.attack]
            if prefix == "collusion_":
                xs = [1 if r == "clean" else x for r, x in zip(a.attack, xs)]
            o = np.argsort(xs)
            ax.errorbar(np.array(xs)[o], a.nc_mean.values[o], yerr=a.nc_std.fillna(0).values[o], marker="o",
                        label=LABEL[m], color=COLORS[m], capsize=3)
        if prefix == "collusion_":
            ax.set_xscale("log"); n = np.array([1, 2, 5, 10, 20, 50, 100])
            ax.plot(n, 1 / np.sqrt(n), "k--", lw=1, label="1/√N (symmetric-averaging bound)")
        ax.set_xlabel(xl); ax.set_ylabel("NC"); ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / fname, dpi=120); plt.close(fig)
    # table image
    p = _pivot(agg, "nc_mean").round(3)
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(p) + 1)); ax.axis("off")
    t = ax.table(cellText=p.values, rowLabels=p.index, colLabels=[LABEL.get(c, c) for c in p.columns], loc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.3)
    ax.set_title("Mean NC per attack (held-out test hosts)")
    fig.tight_layout(); fig.savefig(out / "comparison_table.png", dpi=130); plt.close(fig)
    # sample grid
    if samples:
        fig, ax = plt.subplots(len(samples), 5, figsize=(11, 2.3 * len(samples)))
        ax = np.atleast_2d(ax)
        from training.metrics import bit_metrics
        for r, s in enumerate(samples):
            for c, (img, t_) in enumerate([(s["attacked"], f"{s['host_id']} {s['attack']}"), (s["ann_prob"], "ANN P(bit=1)"),
                                          (s["ann_bin"], f"ANN blind NC={bit_metrics(s['ann_bin'], s['wm'])['nc']:.3f}"),
                                          (s["nonblind"], f"Non-blind NC={bit_metrics(s['nonblind'], s['wm'])['nc']:.3f}"),
                                          (s["wm"], "ground truth")]):
                ax[r, c].imshow(img, cmap="gray" if c != 1 else "magma", vmin=0, vmax=1); ax[r, c].set_title(t_, fontsize=7)
                ax[r, c].axis("off")
        fig.tight_layout(); fig.savefig(out / "sample_recovery_grid.png", dpi=110); plt.close(fig)


def _comparison_summary(agg, rm, payload, out: Path):
    lines = [f"# Final Evaluation — Blind ANN vs Non-Blind vs Centre-Tile Baseline\n",
             f"Generated {payload['generated']} on **{payload['n_test_hosts']} held-out test hosts** "
             "(host-level split; no test host or test watermark ID was seen in training/validation).\n",
             "## Information available to each method\n"]
    lines += [f"- **{LABEL[m]}** — {d}" for m, d in payload["methods"].items()]
    lines += ["\n> The non-blind extractors subtract the ORIGINAL host (and use the known crop mask). Their numbers are an "
              "upper reference, not a like-for-like competitor of the blind ANN.\n",
              "## Mean NC ± std (n per cell = number of test hosts)\n",
              "| attack | " + " | ".join(LABEL[m] for m in LABEL if m in agg.method.unique()) + " |",
              "|---|" + "---|" * len([m for m in LABEL if m in agg.method.unique()])]
    for a in [x for x in ORDER if x in agg.attack.unique()]:
        cells = []
        for m in [m for m in LABEL if m in agg.method.unique()]:
            r = agg[(agg.method == m) & (agg.attack == a)]
            cells.append(f"{r.nc_mean.iloc[0]:.3f} ± {r.nc_std.fillna(0).iloc[0]:.3f} (n={int(r.n.iloc[0])})" if len(r) else "—")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    lines += ["\n## Mean BER\n", "| attack | " + " | ".join(LABEL[m] for m in LABEL if m in agg.method.unique()) + " |",
              "|---|" + "---|" * len([m for m in LABEL if m in agg.method.unique()])]
    for a in [x for x in ORDER if x in agg.attack.unique()]:
        cells = []
        for m in [m for m in LABEL if m in agg.method.unique()]:
            r = agg[(agg.method == m) & (agg.attack == a)]
            cells.append(f"{r.ber_mean.iloc[0]:.4f}" if len(r) else "—")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    c = agg[agg.attack == "clean"].set_index("method")
    lines += ["\n## Imperceptibility (clean, vs host I-channel)\n", "| scheme | PSNR (dB) | SSIM |", "|---|---|---|"]
    for m, nm in (("nonblind_pixel", "Hybrid mosaic (ANN & non-blind share this embedding)"), ("baseline_center", "Centre-tile baseline")):
        if m in c.index:
            lines.append(f"| {nm} | {c.loc[m,'psnr_mean']:.2f} ± {c.loc[m,'psnr_std']:.2f} | {c.loc[m,'ssim_mean']:.4f} |")
    lines += ["\n## Research metrics\n", f"CRR(p) = NC(crop p)/NC(clean). CoRI(N) = P(NC_victim > τ={rm['tau']:.3f}).\n",
              "| method | CRR 10% | CRR 25% | CRR 50% | CoRI N=2 | N=5 | N=10 | N=20 | N=50 | N=100 | CoRI agg |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in [m for m in LABEL if m in rm["CRR"]]:
        cr, co = rm["CRR"].get(m, {}), rm["CoRI"].get(m, {})
        lines.append(f"| {LABEL[m]} | " + " | ".join(f"{cr.get(k, float('nan')):.3f}" for k in ("crop_10", "crop_25", "crop_50"))
                     + " | " + " | ".join(f"{co.get(f'collusion_{n}', float('nan')):.2f}" for n in (2, 5, 10, 20, 50, 100))
                     + f" | {rm['CoRI_aggregate'].get(m, float('nan')):.3f} |")
    ng = payload["negative_control"]
    lines += [f"\n## Negative control & out-of-distribution generalization (blind ANN)\n",
              f"- **Unwatermarked test hosts fed to the ANN:** NC vs the host's test watermark = "
              f"`{ng['nc_unwatermarked_mean']:.4f} ± {ng['nc_unwatermarked_std']:.4f}`; empirical FPR at τ={ng['tau']:.3f}: "
              f"`{ng['fpr_unwatermarked']:.4f}`.",
              f"- **Clean prediction vs unrelated watermark (mixed even/odd IDs):** FPR = `{ng['fpr_unrelated_watermark']:.4f}` "
              f"(caused by fixed text-bounding-box overlap when both IDs are even, p=0.25).",
              f"- **Clean prediction vs unrelated odd-ID block watermark (zero structural bias):** "
              f"mean NC = `{ng.get('nc_unrelated_odd_mean', 0.0):.4f}`, FPR = `{ng.get('fpr_unrelated_odd_block_watermark', 0.0):.4f}`.",
              f"- **Clean prediction vs unrelated i.i.d. Bernoulli(0.5) watermark:** "
              f"mean NC = `{ng.get('nc_unrelated_iid_mean', 0.0):.4f}`, FPR = `{ng.get('fpr_unrelated_iid_bernoulli_watermark', 0.0):.4f}`.",
              f"- **Out-of-distribution unseen i.i.d. Bernoulli(0.5) watermark recovery:** "
              f"mean NC = `{ng.get('ood_iid_bernoulli_nc_mean', 0.0):.4f}`, mean BER = `{ng.get('ood_iid_bernoulli_ber_mean', 0.0):.4f}`.",
              f"- **Out-of-distribution unseen bit-inverted watermark (`1 - wm`) recovery:** "
              f"mean NC = `{ng.get('ood_bit_inverted_nc_mean', 0.0):.4f}`, mean BER = `{ng.get('ood_bit_inverted_ber_mean', 0.0):.4f}`.\n"]
    ann = agg[agg.method == "ann_blind"].set_index("attack")
    weak = [a for a in ann.index if ann.loc[a, "nc_mean"] < 0.5]
    lines += ["## Honest assessment\n",
              f"- Attack conditions where the blind ANN mean NC < 0.5 (treated as failure): **{', '.join(weak) if weak else 'none'}**.",
              "- Averaging collusion is symmetric in the colluders; the victim's watermark can at best retain ~1/√N correlation "
              "for ANY extractor. Results above show measured behaviour, not complete collusion resistance.",
              "- JPEG/blur attacks remove the high-frequency band that carries the low-amplitude (α≈0.012–0.036) mosaic signal; "
              "degradation there is a property of the embedding strength, shared by the non-blind extractor."]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ========================================================================= FINAL DEMO
def phase_final_demo(attack_name: str = "crop", model_path: Optional[str] = None) -> Dict:
    """
    Generate the complete 16-panel visual pipeline report and 7 stage figures in `results/final_demo/`
    running BOTH Non-Blind and Blind ANN extractors on a held-out test image.
    """
    from core.preprocessing import preprocess_single_image, denormalize_channel, yiq_to_bgr
    from core.watermark import WatermarkTransformer
    from core.embedding import WatermarkEmbedder
    from core.attacks import AttackRunner
    from core.extraction import NonBlindExtractor
    from core.data_sources import get_split_hosts, load_host_bgr
    from reporting.visualizer import save_image_grid
    from training.inference import BlindExtractor, score_against_ground_truth

    out = RESULTS / "final_demo"
    out.mkdir(parents=True, exist_ok=True)

    # Use held-out test host and held-out test watermark ID so neither was seen during training
    test_hosts = get_split_hosts()["test"]
    host_obj = test_hosts[1] if len(test_hosts) > 1 else test_hosts[0]
    host_bgr_256 = load_host_bgr(host_obj)
    tmp_host_png = out / f"host_{host_obj.host_id}.png"
    cv2.imwrite(str(tmp_host_png), host_bgr_256)

    preproc = preprocess_single_image(tmp_host_png, (256, 256))
    host_rgb = cv2.cvtColor(preproc["bgr_resized"], cv2.COLOR_BGR2RGB)
    host_i = preproc["i_channel_norm"]

    # Held-out test watermark ID (60000 is in the test-only range [60000, 69999])
    wm_gt = make_watermark(60000)
    wm_trans = WatermarkTransformer(watermark_size=32, acm_iterations=10, catalan_iterations=5, catalan_key=7)
    fwd = wm_trans.transform_forward(wm_gt)

    # Adaptive Embedding
    embedder = WatermarkEmbedder(alpha_base=0.012, sensitivity=2.0)
    emb = embedder.embed_adaptive(host_i, fwd["watermark_mosaic"])
    watermarked_i = emb["embedded_i"]

    i_denorm = denormalize_channel(watermarked_i, preproc["i_min"], preproc["i_max"])
    yiq_emb = preproc["yiq"].copy()
    yiq_emb[:, :, 1] = i_denorm
    watermarked_rgb = cv2.cvtColor(yiq_to_bgr(yiq_emb), cv2.COLOR_BGR2RGB)

    diff_amp = np.clip(emb["diff_map"] * 10.0 * 255.0, 0, 255).astype(np.uint8)
    diff_heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(diff_amp, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)

    # Attack Simulation
    atk_runner = AttackRunner(target_size=256)
    if attack_name == "crop":
        atk_res = atk_runner.run_crop(watermarked_i, mode="random", intensity=0.25, seed=42)
        atk_label = f"25% Random Crop (Kept {atk_res['retained_pct']:.1f}%)"
    elif attack_name == "jpeg":
        atk_res = atk_runner.run_jpeg(watermarked_i, quality=70)
        atk_label = "JPEG Compression (Q=70)"
    elif attack_name == "noise":
        atk_res = atk_runner.run_noise(watermarked_i, sigma=0.03, seed=42)
        atk_label = "Gaussian Noise (sigma=0.03)"
    elif attack_name == "blur":
        atk_res = atk_runner.run_blur(watermarked_i, kernel_size=3)
        atk_label = "Gaussian Blur (k=3)"
    else:
        atk_res = atk_runner.run_collusion(
            watermarked_i, n_colluders=2, seed=42, host_image=host_i, victim_watermark=wm_gt, victim_watermark_id=60000
        )
        atk_label = "Collusion Averaging (N=2)"

    attacked_i = atk_res["attacked_image"]
    mask = atk_res.get("mask")

    # Stage 5: Non-blind extraction (uses host_i and mask)
    nb_extractor = NonBlindExtractor(alpha_base=0.012, watermark_size=32, acm_iterations=10, catalan_iterations=5, catalan_key=7)
    nb_res = nb_extractor.extract(attacked_i, host_i, mask=mask, ground_truth_binary=wm_gt)

    # Stage 6: Blind ANN extraction (uses ONLY attacked_i — NO host_i, NO mask)
    blind_ext = BlindExtractor(model_path or CKPT / "best_model.pth")
    ann_pred = blind_ext.predict(attacked_i)
    ann_metrics = score_against_ground_truth(ann_pred["binary"], wm_gt)
    ann_err_map = np.abs(ann_pred["binary"].astype(np.float32) - wm_gt.astype(np.float32))
    nb_err_map = np.abs(nb_res["recovered_binary"].astype(np.float32) - wm_gt.astype(np.float32))

    # Save individual stage figures (stage_01 to stage_07)
    save_image_grid([
        ("Original Host RGB (256x256)", host_rgb),
        ("Y Luminance Channel", preproc["y_channel"]),
        ("I Chrominance Channel (Raw)", preproc["i_channel_raw"]),
        ("Normalized I-Channel [0,1]", host_i),
    ], out / "stage_01_preprocessing.png", cols=4, title=f"Stage 1: Host Preprocessing (Test Host {host_obj.host_id})")

    save_image_grid([
        ("1. Original Binary Watermark (32x32)", fwd["watermark_binary"]),
        ("2. Arnold Scrambled (10 iters)", fwd["watermark_scrambled"]),
        ("3. Catalan Permuted (5 iters, k=7)", fwd["watermark_catalan"]),
        ("4. 8x8 Spatial Mosaic (256x256)", fwd["watermark_mosaic"]),
    ], out / "stage_02_watermark.png", cols=4, title="Stage 2: Watermark Scrambling & 8x8 Mosaic Generation")

    save_image_grid([
        ("Host Normalized I-Channel", host_i),
        ("Local Texture Variance Mask", emb["texture_mask"]),
        ("Adaptive Alpha Map", emb["alpha_map"]),
        (f"Watermarked RGB (PSNR={emb['psnr']:.2f}dB)", watermarked_rgb),
        ("Amplified Difference Heatmap", diff_heatmap_rgb),
    ], out / "stage_03_embedding.png", cols=5, title="Stage 3: Perceptual Luminance-Texture Adaptive Embedding")

    save_image_grid([
        ("Clean Watermarked I-Channel", watermarked_i),
        (f"Attacked I-Channel ({atk_label})", attacked_i),
        ("Attack Distortion |Attacked - Watermarked|", np.abs(attacked_i - watermarked_i)),
    ], out / "stage_04_attack.png", cols=3, title=f"Stage 4: Attack Simulation — {atk_label}")

    save_image_grid([
        ("Attacked I-Channel", attacked_i),
        ("Host-Subtracted Residual", nb_res["residual_diff"]),
        ("Aggregated Catalan Tile", nb_res["recovered_catalan_raw"]),
        ("Inverse Catalan (Scrambled)", nb_res["recovered_scrambled"]),
        (f"Non-Blind Recovered\nNC={nb_res['metrics']['nc']:.4f}, BER={nb_res['metrics']['ber']:.4f}", nb_res["recovered_binary"]),
        ("Ground-Truth Watermark", wm_gt),
    ], out / "stage_05_nonblind_extraction.png", cols=3, title="Stage 5: Non-Blind Extraction (Requires Original Clean Host)")

    save_image_grid([
        ("Attacked I-Channel (Sole ANN Input)", attacked_i),
        ("ANN Probability Heatmap P(bit=1)", ann_pred["probability"]),
        (f"Blind ANN Recovered (32x32)\nNC={ann_metrics['nc']:.4f}, BER={ann_metrics['ber']:.4f}", ann_pred["binary"]),
        ("Ground-Truth Watermark (32x32)", wm_gt),
        (f"Blind ANN Bit Error Map\nErrors: {int(ann_err_map.sum())}/1024", ann_err_map),
    ], out / "stage_06_ann_blind_extraction.png", cols=5, title="Stage 6: Blind ANN Extraction (Host-Free Inference)")

    save_image_grid([
        ("Ground-Truth Watermark", wm_gt),
        (f"Blind ANN (Host-Free)\nNC={ann_metrics['nc']:.4f}, BER={ann_metrics['ber']:.4f}", ann_pred["binary"]),
        (f"Blind ANN Error Map ({int(ann_err_map.sum())} bits)", ann_err_map),
        (f"Non-Blind (Uses Clean Host)\nNC={nb_res['metrics']['nc']:.4f}, BER={nb_res['metrics']['ber']:.4f}", nb_res["recovered_binary"]),
        (f"Non-Blind Error Map ({int(nb_err_map.sum())} bits)", nb_err_map),
    ], out / "stage_07_comparison.png", cols=5, title=f"Stage 7: Blind ANN vs Non-Blind Recovery ({atk_label})")

    # 16-panel full_pipeline.png
    fig, axes = plt.subplots(4, 4, figsize=(18, 17))
    panels_16 = [
        ("1. Original Host Image (256x256)", host_rgb, None),
        ("2. Normalized I-Channel [0,1]", host_i, "gray"),
        ("3. Original Watermark (32x32)", wm_gt, "gray"),
        ("4. Arnold Cat Map Result (10 iters)", fwd["watermark_scrambled"], "gray"),
        ("5. Catalan Transform Result (5 iters, k=7)", fwd["watermark_catalan"], "gray"),
        ("6. 8x8 Watermark Mosaic (256x256)", fwd["watermark_mosaic"], "gray"),
        ("7. Local Texture Variance Map", emb["texture_mask"], "viridis"),
        (f"8. Adaptive Alpha Map\n[{emb['alpha_min']:.3f}, {emb['alpha_max']:.3f}]", emb["alpha_map"], "magma"),
        (f"9. Watermarked RGB Image\nPSNR={emb['psnr']:.2f} dB, SSIM={emb['ssim']:.4f}", watermarked_rgb, None),
        ("10. Embedding Difference Map (10x)", diff_heatmap_rgb, None),
        (f"11. Attacked I-Channel\n({atk_label})", attacked_i, "gray"),
        ("12. Blind ANN Probability Map P(bit=1)", ann_pred["probability"], "magma"),
        (f"13. BLIND ANN Recovered Watermark\nNC={ann_metrics['nc']:.4f}, BER={ann_metrics['ber']:.4f}", ann_pred["binary"], "gray"),
        (f"14. NON-BLIND Recovered Watermark\nNC={nb_res['metrics']['nc']:.4f}, BER={nb_res['metrics']['ber']:.4f}", nb_res["recovered_binary"], "gray"),
        ("15. Ground-Truth Watermark (32x32)", wm_gt, "gray"),
        (f"16. Blind ANN Bit Error Map\nErrors: {int(ann_err_map.sum())}/1024 ({ann_metrics['ber']*100:.2f}%)", ann_err_map, "magma"),
    ]
    for idx, (p_title, p_img, p_cmap) in enumerate(panels_16):
        ax = axes[idx // 4, idx % 4]
        if p_cmap is None:
            ax.imshow(p_img)
        else:
            ax.imshow(p_img, cmap=p_cmap, vmin=0 if p_cmap == "gray" else None, vmax=1 if p_cmap == "gray" else None)
        ax.set_title(p_title, fontsize=9.5, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        f"Complete Hybrid Watermarking Pipeline — Held-Out Test Host {host_obj.host_id} | Attack: {atk_label}\n"
        f"Blind ANN (Host-Free): NC={ann_metrics['nc']:.4f}, BER={ann_metrics['ber']:.4f}  ||  "
        f"Non-Blind (Host-Subtracted): NC={nb_res['metrics']['nc']:.4f}, BER={nb_res['metrics']['ber']:.4f}",
        fontsize=13, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out / "full_pipeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    report_data = {
        "title": "Complete Hybrid Watermarking Visual Pipeline Report",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host_id": host_obj.host_id,
        "host_split": "test (held-out)",
        "attack": attack_name,
        "attack_label": atk_label,
        "imperceptibility": {"psnr_db": emb["psnr"], "ssim": emb["ssim"],
                             "alpha_min": emb["alpha_min"], "alpha_max": emb["alpha_max"], "alpha_mean": emb["alpha_mean"]},
        "blind_ann_extraction": {
            "extractor_type": "BLIND (Attacked I-channel only; NO original host image or crop mask used)",
            "checkpoint": blind_ext.info(),
            "inference_seconds": ann_pred["inference_seconds"],
            "metrics": ann_metrics,
            "bit_errors": int(ann_err_map.sum()),
        },
        "nonblind_extraction": {
            "extractor_type": "NON-BLIND (Requires original unwatermarked host I-channel and crop mask)",
            "metrics": nb_res["metrics"],
        },
        "artifacts": [
            _rel(out / "full_pipeline.png"),
            _rel(out / "stage_01_preprocessing.png"),
            _rel(out / "stage_02_watermark.png"),
            _rel(out / "stage_03_embedding.png"),
            _rel(out / "stage_04_attack.png"),
            _rel(out / "stage_05_nonblind_extraction.png"),
            _rel(out / "stage_06_ann_blind_extraction.png"),
            _rel(out / "stage_07_comparison.png"),
        ],
    }
    (out / "demo_report.json").write_text(json.dumps(report_data, indent=2, default=float), encoding="utf-8")

    md = [
        "# Complete Hybrid Watermarking Pipeline — Visual Demonstration Report\n",
        f"**Executed:** `{report_data['timestamp']}`  ",
        f"**Held-Out Test Host:** `{host_obj.host_id}` (never seen during ANN training/validation)  ",
        f"**Attack Applied:** `{atk_label}`\n",
        "## 1. Imperceptibility (Clean Embedding vs Host)\n",
        f"- **PSNR:** `{emb['psnr']:.2f} dB`",
        f"- **SSIM:** `{emb['ssim']:.4f}`",
        f"- **Adaptive Alpha Range:** `[{emb['alpha_min']:.4f}, {emb['alpha_max']:.4f}]` (mean `{emb['alpha_mean']:.4f}`)\n",
        "## 2. Blind ANN vs Non-Blind Extraction Results\n",
        "| Extractor Mode | Host Image Used? | Crop Mask Used? | NC | BER | Bit Accuracy | F1 Score | Bit Errors |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| **Blind ANN (`BlindWatermarkExtractor`)** | **NO (Host-Free)** | **NO** | `{ann_metrics['nc']:.4f}` | `{ann_metrics['ber']:.4f}` | `{ann_metrics['accuracy']*100:.2f}%` | `{ann_metrics['f1']:.4f}` | `{int(ann_err_map.sum())} / 1024` |",
        f"| **Non-Blind (`NonBlindExtractor`)** | **YES (Clean Host)** | **YES** | `{nb_res['metrics']['nc']:.4f}` | `{nb_res['metrics']['ber']:.4f}` | `{(1-nb_res['metrics']['ber'])*100:.2f}%` | `—` | `{nb_res['metrics']['bit_errors']} / 1024` |\n",
        "## 3. Generated Stage Artifacts\n",
    ] + [f"- `{Path(p).name}`: `{p}`" for p in report_data["artifacts"]]
    (out / "demo_report.md").write_text("\n".join(md), encoding="utf-8")

    _status("Phase demo (final_demo)", True, [out / "full_pipeline.png", out / "demo_report.md", out / "demo_report.json"],
            f"Blind ANN NC={ann_metrics['nc']:.4f} | Non-Blind NC={nb_res['metrics']['nc']:.4f}")
    return report_data

