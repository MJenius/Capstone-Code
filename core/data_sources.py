"""
Host-image and watermark sources with leakage-free, host-level splitting.

Host images
-----------
The repository's ``preprocessed/I_channel`` folder contains 1100 files but only
800 distinct DIV2K source images: files are named ``<hostid>_<counter>.npy`` and
several counters point at the *same* source image (identical arrays).  The legacy
``splits/*.txt`` files were built over the 1100 file IDs and therefore place
copies of the same host in different splits (measured: 59 hosts shared between
train and test).  This module de-duplicates by host ID (the 4-digit prefix) and
creates a new deterministic host-level split under ``splits/host_level/``.

If no preprocessed data exists (clean checkout), a user-supplied folder of images
can be used (``--host-dir``) or a small deterministic *synthetic* host set is
generated so that every phase and test can still run.

Watermarks
----------
ID 0 is the canonical project watermark (``data/watermark/watermark_binary.npy``,
text "PW26_PAC_01").  All other IDs are deterministic pseudo-random 32x32 binary
watermarks (mix of rendered text codes and random block patterns).  ID ranges are
disjoint per split so that the ANN cannot memorise test watermarks:

    train: 10000-49999   val: 50000-59999   test: 60000-69999 (+ canonical ID 0)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.preprocessing import bgr_to_yiq, normalize_channel, resize_image

BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = BASE_DIR / "splits" / "host_level"
WM_RANGES = {"train": (10000, 50000), "val": (50000, 60000), "test": (60000, 70000)}


@dataclass
class Host:
    host_id: str          # unique source-image id (e.g. "0002" or "synthetic_0007")
    file_id: str          # representative file id
    source: str           # "div2k_preprocessed" | "image_dir" | "synthetic"
    path: Optional[str] = None


# ----------------------------------------------------------------------------- hosts
def _synthetic_host(index: int, size: int = 256) -> np.ndarray:
    """Deterministic textured BGR image (gradients + blobs + stripes + noise)."""
    rng = np.random.RandomState(10_000 + index)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.zeros((size, size, 3), np.float32)
    for c in range(3):
        a, b, f = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(2, 12)
        img[..., c] = 0.5 + 0.25 * (a * xx + b * yy) + 0.15 * np.sin(2 * np.pi * f * (xx * a + yy * b))
    for _ in range(rng.randint(3, 8)):
        cx, cy, r = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0.05, 0.3)
        m = ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
        img[m] = rng.uniform(0, 1, 3)
    img += rng.normal(0, 0.04, img.shape)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def list_hosts(host_dir: Optional[Path] = None, allow_synthetic: bool = True,
               n_synthetic: int = 60) -> List[Host]:
    """Return de-duplicated hosts from the best available source."""
    if host_dir is not None:
        files = sorted([p for p in Path(host_dir).iterdir()
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")])
        if files:
            return [Host(p.stem, p.stem, "image_dir", str(p)) for p in files]
    i_dir = BASE_DIR / "preprocessed" / "I_channel"
    if i_dir.exists():
        seen: Dict[str, Host] = {}
        for p in sorted(i_dir.glob("*.npy")):
            hid = p.stem.split("_")[0]
            if hid not in seen:
                seen[hid] = Host(hid, p.stem, "div2k_preprocessed", str(p))
        if seen:
            return list(seen.values())
    if not allow_synthetic:
        raise FileNotFoundError("No host images found (preprocessed/I_channel or --host-dir).")
    return [Host(f"synthetic_{i:04d}", f"synthetic_{i:04d}", "synthetic") for i in range(n_synthetic)]


def load_host_bgr(host: Host) -> np.ndarray:
    """256x256 BGR uint8 host image."""
    if host.source == "synthetic":
        return _synthetic_host(int(host.host_id.split("_")[1]))
    if host.source == "div2k_preprocessed":
        rgb = BASE_DIR / "preprocessed" / "rgb_256" / f"{host.file_id}.png"
        if rgb.exists():
            return cv2.imread(str(rgb), cv2.IMREAD_COLOR)
    img = cv2.imread(str(host.path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read host image {host.path}")
    return resize_image(img, (256, 256))


def load_host_i(host: Host) -> np.ndarray:
    """Normalised [0,1] I-channel (256x256 float32) — identical to Phase-1 output."""
    if host.source == "div2k_preprocessed":
        return np.load(host.path).astype(np.float32)
    yiq = bgr_to_yiq(load_host_bgr(host))
    return normalize_channel(yiq[:, :, 1])[0]


def host_level_split(hosts: List[Host], seed: int = 42, ratios=(0.70, 0.15, 0.15),
                     save: bool = True) -> Dict[str, List[Host]]:
    """Deterministic split by unique host id. No host appears in two splits."""
    ids = sorted(h.host_id for h in hosts)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ids))
    n_tr = int(round(ratios[0] * len(ids)))
    n_va = int(round(ratios[1] * len(ids)))
    groups = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
    by_id = {h.host_id: h for h in hosts}
    out = {k: [by_id[ids[i]] for i in sorted(v)] for k, v in groups.items()}
    assert not (set(h.host_id for h in out["train"]) & set(h.host_id for h in out["test"]))
    assert not (set(h.host_id for h in out["train"]) & set(h.host_id for h in out["val"]))
    assert not (set(h.host_id for h in out["val"]) & set(h.host_id for h in out["test"]))
    if save and hosts and hosts[0].source != "synthetic":
        SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        for k, v in out.items():
            (SPLIT_DIR / f"{k}.txt").write_text("\n".join(h.host_id for h in v) + "\n")
        (SPLIT_DIR / "split_info.json").write_text(json.dumps({
            "seed": seed, "ratios": ratios, "source": hosts[0].source,
            "counts": {k: len(v) for k, v in out.items()},
            "note": "Split by unique host id (4-digit DIV2K prefix); duplicates collapsed.",
        }, indent=2))
    return out


def get_split_hosts(host_dir: Optional[Path] = None, seed: int = 42) -> Dict[str, List[Host]]:
    return host_level_split(list_hosts(host_dir), seed=seed)


# ------------------------------------------------------------------------ watermarks
def canonical_watermark() -> np.ndarray:
    p = BASE_DIR / "data" / "watermark" / "watermark_binary.npy"
    if p.exists():
        return (np.load(p) > 0.5).astype(np.uint8)
    from generate_watermark import generate_binary_watermark
    return (generate_binary_watermark(output_path=str(p.parent)) > 0.5).astype(np.uint8)


def make_watermark(wm_id: int, size: int = 32) -> np.ndarray:
    """Deterministic binary watermark for an integer id (0 = canonical)."""
    if wm_id == 0:
        return canonical_watermark()
    rng = np.random.RandomState(wm_id)
    if wm_id % 2 == 0:
        # rendered 4-character code on a 32x32 canvas, plus a random border pattern
        canvas = np.zeros((64, 64), np.uint8)
        code = "".join(rng.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ23456789"), 4))
        cv2.putText(canvas, code[:2], (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 3)
        cv2.putText(canvas, code[2:], (4, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 3)
        wm = (cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA) > 100).astype(np.uint8)
        border = (rng.rand(size, size) > 0.5).astype(np.uint8)
        frame = np.zeros_like(wm, bool)
        frame[:2, :] = frame[-2:, :] = frame[:, :2] = frame[:, -2:] = True
        wm[frame] = border[frame]
    else:
        blocks = (rng.rand(size // 4, size // 4) > 0.5).astype(np.uint8)
        wm = np.kron(blocks, np.ones((4, 4), np.uint8))
        flip = rng.rand(size, size) < 0.15
        wm = np.where(flip, 1 - wm, wm).astype(np.uint8)
    return wm


def sample_watermark_id(split: str, rng: np.random.RandomState) -> int:
    lo, hi = WM_RANGES[split]
    return int(rng.randint(lo, hi))
