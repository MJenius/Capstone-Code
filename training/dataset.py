"""
ANN dataset generation and loading.

Supervised target (resolves the legacy 256x256-residual inconsistency):

    input  x : attacked, watermarked, normalised I-channel   (256, 256) float16 on disk
    label  y : ORIGINAL 32x32 binary watermark                (32, 32)  uint8

Generation per sample: host (host-level split) -> watermark id (split-disjoint
range) -> Arnold -> Catalan -> 8x8 mosaic -> adaptive embedding -> random attack
-> store (x, y, metadata).  The host, residual, alpha map and crop mask are NOT
stored as model inputs.

Layout::

    data/ann/{train,val,test}/inputs.npy  (N,256,256) float16
    data/ann/{train,val,test}/labels.npy  (N,32,32)   uint8
    data/ann/metadata.csv
    data/ann/dataset_summary.json
    data/ann/visualizations/*.png
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from core.data_sources import (BASE_DIR, get_split_hosts, load_host_i, make_watermark,
                               sample_watermark_id)
from core.hybrid_system import HybridSystem, apply_training_attack, sample_training_attack

ANN_DIR = BASE_DIR / "data" / "ann"
SPLIT_SEEDS = {"train": 1_000, "val": 2_000, "test": 3_000}


def generate_dataset(out_dir: Path = ANN_DIR, samples_per_host: Optional[Dict[str, int]] = None,
                     max_hosts: Optional[Dict[str, int]] = None, seed: int = 42,
                     host_dir: Optional[Path] = None, system: Optional[HybridSystem] = None,
                     progress=None) -> Dict:
    samples_per_host = samples_per_host or {"train": 4, "val": 3, "test": 3}
    system = system or HybridSystem()
    splits = get_split_hosts(host_dir, seed=seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    t0 = time.time()
    for split in ("train", "val", "test"):
        hosts = splits[split]
        if max_hosts and max_hosts.get(split) is not None:
            hosts = hosts[: max_hosts[split]]
        k = samples_per_host[split]
        n = len(hosts) * k
        sd = out_dir / split
        sd.mkdir(parents=True, exist_ok=True)
        X = np.lib.format.open_memmap(sd / "inputs.npy", mode="w+", dtype=np.float16, shape=(n, 256, 256))
        Y = np.zeros((n, 32, 32), np.uint8)
        rng = np.random.RandomState(seed + SPLIT_SEEDS[split])
        idx = 0
        for hi, host in enumerate(hosts):
            host_i = load_host_i(host)
            for j in range(k):
                wm_id = 0 if (split == "test" and j == 0) else sample_watermark_id(split, rng)
                wm = make_watermark(wm_id)
                wmk = system.embed(host_i, wm)
                spec = sample_training_attack(rng)
                a_seed = int(rng.randint(0, 2**31 - 1))
                att = apply_training_attack(system, wmk, spec, a_seed)
                X[idx] = att.astype(np.float16)
                Y[idx] = wm
                rows.append({"split": split, "index": idx, "host_id": host.host_id, "host_file": host.file_id,
                             "host_source": host.source, "watermark_id": wm_id, "attack_type": spec["type"],
                             "attack_params": json.dumps({kk: v for kk, v in spec.items() if kk != "type"}),
                             "attack_seed": a_seed,
                             "psnr_attacked_vs_watermarked": float(10 * np.log10(1.0 / max(np.mean((att - wmk) ** 2), 1e-12)))})
                idx += 1
            if progress:
                progress(split, hi + 1, len(hosts))
        X.flush(); del X
        np.save(sd / "labels.npy", Y)
    meta = pd.DataFrame(rows)
    meta.to_csv(out_dir / "metadata.csv", index=False)
    summary = summarize(meta, out_dir, seed, samples_per_host, time.time() - t0)
    make_visualizations(out_dir, meta)
    return summary


def summarize(meta: pd.DataFrame, out_dir: Path, seed: int, sph: Dict, elapsed: float) -> Dict:
    per_split_hosts = {s: sorted(meta[meta.split == s].host_id.unique().tolist()) for s in ("train", "val", "test")}
    leak = {f"{a}&{b}": len(set(per_split_hosts[a]) & set(per_split_hosts[b]))
            for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    wm_ids = {s: sorted(meta[meta.split == s].watermark_id.unique().tolist()) for s in ("train", "val", "test")}
    wm_overlap = len(set(wm_ids["train"]) & (set(wm_ids["val"]) | set(wm_ids["test"])))
    checks = {}
    for s in ("train", "val", "test"):
        X = np.load(out_dir / s / "inputs.npy", mmap_mode="r")
        Y = np.load(out_dir / s / "labels.npy")
        checks[s] = {"inputs_shape": list(X.shape), "labels_shape": list(Y.shape),
                     "inputs_dtype": str(X.dtype), "labels_unique": [int(v) for v in np.unique(Y)],
                     "rows_match_metadata": int(X.shape[0]) == int((meta.split == s).sum()),
                     "inputs_finite": bool(np.isfinite(np.asarray(X[: min(64, len(X))], np.float32)).all())}
    valid = (all(v == 0 for v in leak.values()) and wm_overlap == 0
             and all(c["rows_match_metadata"] and c["inputs_finite"] and c["labels_shape"][1:] == [32, 32]
                     for c in checks.values()))
    summary = {
        "total_samples": int(len(meta)),
        "unique_host_images": int(meta.host_id.nunique()),
        "unique_watermark_ids": int(meta.watermark_id.nunique()),
        "samples_per_split": {s: int((meta.split == s).sum()) for s in ("train", "val", "test")},
        "hosts_per_split": {s: len(v) for s, v in per_split_hosts.items()},
        "per_split_host_ids": per_split_hosts,
        "host_overlap_between_splits": leak,
        "watermark_id_overlap_train_vs_heldout": wm_overlap,
        "canonical_watermark_in_test_only": bool((meta[(meta.watermark_id == 0)].split == "test").all()),
        "attack_distribution": {s: dict(Counter(meta[meta.split == s].attack_type)) for s in ("train", "val", "test")},
        "attack_parameter_ranges": {
            "crop_random_intensity": [0.05, 0.55], "crop_center_intensity": [0.1, 0.25, 0.5],
            "jpeg_quality": [45, 95], "noise_sigma": [0.005, 0.06], "gaussian_blur": "k=3, sigma in [0,1]",
            "median": "k=3", "combined": "random crop [0.05,0.4] + (jpeg 55-95 | noise 0.005-0.04 | blur k=3)",
            "collusion": "excluded from training (ill-posed victim label); evaluated separately"},
        "image_dimensions": [256, 256], "label_dimensions": [32, 32],
        "input_definition": "attacked watermarked normalised I-channel (no host, no mask)",
        "label_definition": "original 32x32 binary watermark (before Arnold/Catalan)",
        "samples_per_host": sph, "seed": seed, "split_seed": seed, "split_method": "by unique host id",
        "host_source": str(meta.host_source.iloc[0]) if len(meta) else None,
        "generation_seconds": round(elapsed, 1), "array_checks": checks,
        "data_validation_status": "PASS" if valid else "FAIL",
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def make_visualizations(out_dir: Path, meta: pd.DataFrame, n: int = 6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vis = out_dir / "visualizations"
    vis.mkdir(exist_ok=True)
    X = np.load(out_dir / "train" / "inputs.npy", mmap_mode="r")
    Y = np.load(out_dir / "train" / "labels.npy")
    m = meta[meta.split == "train"].reset_index(drop=True)
    picks = []
    for t in m.attack_type.unique():
        picks.append(int(m.index[m.attack_type == t][0]))
    picks = picks[: max(n, 1)]
    fig, ax = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.4))
    ax = np.atleast_2d(ax).reshape(2, -1)
    for c, i in enumerate(picks):
        ax[0, c].imshow(np.asarray(X[i], np.float32), cmap="gray", vmin=0, vmax=1)
        ax[0, c].set_title(f"input #{i}\n{m.attack_type[i]}", fontsize=8)
        ax[1, c].imshow(Y[i], cmap="gray", vmin=0, vmax=1)
        ax[1, c].set_title(f"label wm_id={m.watermark_id[i]}", fontsize=8)
        for a in ax[:, c]:
            a.axis("off")
    fig.suptitle("ANN training pairs: attacked 256x256 I-channel -> original 32x32 watermark")
    fig.tight_layout()
    fig.savefig(vis / "input_label_pairs.png", dpi=120)
    plt.close(fig)


class WatermarkANNDataset(Dataset):
    """Loads a stored split. Returns (x[1,256,256] float32, y[1,32,32] float32, index)."""

    def __init__(self, split: str, root: Path = ANN_DIR, augment: bool = False, in_memory: bool = True):
        self.root = Path(root)
        p = self.root / split / "inputs.npy"
        if not p.exists():
            raise FileNotFoundError(f"ANN dataset split '{split}' not found at {p}. Run: python run.py --phase ann-data")
        X = np.load(p, mmap_mode="r")
        self.X = np.asarray(X) if in_memory else X
        self.Y = np.load(self.root / split / "labels.npy")
        self.augment = augment

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(self.X[i], np.float32))[None]
        if self.augment:
            # label-preserving photometric jitter (affine change of I range, as from re-normalisation)
            g = 1.0 + (torch.rand(1) - 0.5) * 0.1
            x = (x - 0.5) * g + 0.5 + (torch.rand(1) - 0.5) * 0.04
        y = torch.from_numpy(self.Y[i].astype(np.float32))[None]
        return x, y, i
