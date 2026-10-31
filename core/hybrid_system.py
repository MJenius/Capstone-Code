"""
Unified hybrid watermarking system used by the ANN dataset generator, the final
comparison, the CLI demo and the Streamlit application.

It wraps the *existing* components (Arnold + Catalan + 8x8 mosaic, adaptive
texture embedder, centre-tile baseline, attack engines) so that every consumer
uses exactly the same embedding and attack realisations.

Named attack conditions
-----------------------
``clean, crop_10, crop_25, crop_50, crop_center_25, jpeg_70, jpeg_50, noise_05,
blur_3, median_3, combined_crop25_jpeg70, collusion_<N>`` plus parametric
variants produced by :func:`sample_training_attack`.

Collusion model (scientifically valid construction)
---------------------------------------------------
``N`` copies of the SAME host are embedded with N DIFFERENT binary watermarks
(victim watermark = copy 0, the other N-1 are independent fingerprints, each
passed through the full Arnold -> Catalan -> mosaic -> adaptive embedding chain).
The copies are averaged and slight Gaussian noise (sigma=0.01) is added.  The
ground truth for evaluation is the victim's watermark; the IDs of all colluders
are recorded.  Averaging is symmetric in the colluders, so no extractor (blind or
non-blind) can single out the victim beyond a correlation of about 1/sqrt(N);
collusion is therefore treated as an *evaluation* condition, not as a training
target (see docs/ann_training_methodology.md).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from attacks.collusion import CollusionAttack
from attacks.cropping import CroppingAttack
from attacks.signal import SignalAttack
from core import keyed_permutation as kp
from utils.adaptive_embedder import AdaptiveEmbedder
from utils.baseline import NormalEmbedder

EVAL_ATTACKS: List[str] = [
    "clean", "crop_10", "crop_25", "crop_50", "jpeg_70", "jpeg_50", "noise_05",
    "blur_3", "median_3", "collusion_2", "collusion_5", "collusion_10",
    "collusion_20", "collusion_50", "collusion_100",
]


class HybridSystem:
    def __init__(self, alpha_base: float = 0.012, sensitivity: float = 2.0, baseline_alpha: float = 0.08,
                 acm_iterations: int = 10, catalan_iterations: int = 5, catalan_key: int = 7,
                 watermark_size: int = 32, grid: int = 8):
        self.alpha_base, self.sensitivity, self.baseline_alpha = alpha_base, sensitivity, baseline_alpha
        self.key = dict(acm_iterations=acm_iterations, catalan_iterations=catalan_iterations,
                        catalan_key=catalan_key)
        self.ws, self.grid = watermark_size, grid
        self.size = watermark_size * grid
        self.adaptive = AdaptiveEmbedder(alpha_base=alpha_base, sensitivity=sensitivity)
        self.baseline = NormalEmbedder(alpha=baseline_alpha)
        self.cropper = CroppingAttack(target_size=self.size)
        self.signal = SignalAttack()
        self.colluder = CollusionAttack()

    # ------------------------------------------------------------------ transforms
    def transform(self, wm: np.ndarray) -> np.ndarray:
        """Arnold -> Catalan (vectorised via the verified index map)."""
        return kp.apply_forward((wm > 0.5).astype(np.uint8), **self.key)

    def inverse_transform(self, t: np.ndarray) -> np.ndarray:
        return kp.apply_inverse(t, **self.key)

    def mosaic(self, t: np.ndarray) -> np.ndarray:
        return np.tile(t, (self.grid, self.grid))

    # ------------------------------------------------------------------- embedding
    def alpha_map(self, host_i: np.ndarray) -> np.ndarray:
        mask = self.adaptive.get_texture_mask(host_i.astype(np.float32))
        return self.alpha_base * (1.0 + self.sensitivity * mask)

    def embed(self, host_i: np.ndarray, wm: np.ndarray) -> np.ndarray:
        """Hybrid embedding (identical formula to utils.adaptive_embedder.AdaptiveEmbedder.embed)."""
        return self.adaptive.embed(host_i.astype(np.float32), self.mosaic(self.transform(wm)).astype(np.float32))

    def embed_baseline(self, host_i: np.ndarray, wm: np.ndarray) -> np.ndarray:
        return self.baseline.embed(host_i.astype(np.float32), wm.astype(np.float32), visible=False)

    # --------------------------------------------------------------------- attacks
    def apply_attack(self, name: str, img: np.ndarray, seed: int, host_i: Optional[np.ndarray] = None,
                     collusion_copies: Optional[Sequence[np.ndarray]] = None) -> Dict[str, Any]:
        """Apply a named attack. Returns {'image', 'mask', 'params'}.

        For collusion, ``collusion_copies`` must contain the N differently
        watermarked copies (victim first); build them with :meth:`collusion_copies`.
        """
        mask = None
        if name == "clean":
            out, params = img.copy(), {}
        elif name.startswith("crop_center_"):
            p = int(name.split("_")[-1]) / 100
            out = self.cropper.apply_attack(img, mode="center", intensity=p)
            mask = self.cropper.get_mask("center", p)
            params = {"mode": "center", "intensity": p}
        elif name.startswith("crop_"):
            p = int(name.split("_")[1]) / 100
            out = self.cropper.apply_attack(img, mode="random", intensity=p, seed=seed)
            mask = self.cropper.get_mask("random", p, seed=seed)
            params = {"mode": "random", "intensity": p, "seed": seed}
        elif name.startswith("jpeg_"):
            q = int(name.split("_")[1])
            out, params = self.signal.apply_jpeg(img, quality=q), {"quality": q}
        elif name.startswith("noise_"):
            s = int(name.split("_")[1]) / 100
            out, params = self.signal.apply_gaussian_noise(img, sigma=s, seed=seed), {"sigma": s, "seed": seed}
        elif name.startswith("blur_"):
            k = int(name.split("_")[1])
            out, params = self.signal.apply_gaussian_blur(img, kernel_size=k), {"kernel": k}
        elif name.startswith("median_"):
            k = int(name.split("_")[1])
            out, params = self.signal.apply_median_blur(img, kernel_size=k), {"kernel": k}
        elif name == "combined_crop25_jpeg70":
            c = self.apply_attack("crop_25", img, seed)
            out, mask = self.signal.apply_jpeg(c["image"], quality=70), c["mask"]
            params = {"crop": c["params"], "jpeg_quality": 70}
        elif name.startswith("collusion_"):
            n = int(name.split("_")[1])
            if collusion_copies is None or len(collusion_copies) < n:
                raise ValueError("Collusion requires N differently-watermarked copies (use collusion_copies()).")
            out = self.colluder.simulate_collusion(list(collusion_copies[:n]), noise_std=0.01, seed=seed)
            params = {"n_colluders": n, "noise_std": 0.01, "seed": seed}
        else:
            raise ValueError(f"Unknown attack '{name}'")
        return {"image": out.astype(np.float32), "mask": mask, "params": params}

    def collusion_copies(self, host_i: np.ndarray, victim_wm: np.ndarray, n: int, seed: int,
                         baseline: bool = False) -> Dict[str, Any]:
        """N copies of the same host carrying different watermarks (victim first)."""
        rng = np.random.RandomState(seed)
        ids = [int(x) for x in rng.randint(70000, 2**31 - 1, size=max(n - 1, 0))]
        from core.data_sources import make_watermark
        wms = [victim_wm] + [make_watermark(i) for i in ids]
        f = self.embed_baseline if baseline else self.embed
        return {"copies": [f(host_i, w) for w in wms], "colluder_wm_ids": ids}

    # ----------------------------------------------------------------- extraction
    def extract_nonblind(self, attacked: np.ndarray, host_i: np.ndarray, mask: Optional[np.ndarray] = None,
                         alpha_mode: str = "pixel") -> Dict[str, np.ndarray]:
        """Non-blind (host-subtracted) extraction.

        alpha_mode='base'  : legacy — residual divided by alpha_base (approximation)
        alpha_mode='pixel' : corrected — residual divided by the pixel-wise alpha map
                             recomputed from the (available) host.
        """
        res = attacked.astype(np.float32) - host_i.astype(np.float32)
        denom = self.alpha_base if alpha_mode == "base" else self.alpha_map(host_i)
        diff = res / denom + 0.5
        tiles = diff.reshape(self.grid, self.ws, self.grid, self.ws).transpose(0, 2, 1, 3).reshape(-1, self.ws, self.ws)
        if mask is not None:
            tm = mask.reshape(self.grid, self.ws, self.grid, self.ws).transpose(0, 2, 1, 3).reshape(-1, self.ws, self.ws)
            w = tm.sum(0)
            soft = np.where(w > 0, (tiles * tm).sum(0) / (w + 1e-8), 0.5)
        else:
            soft = tiles.mean(0)
        t_bin = (soft > 0.5).astype(np.uint8)
        return {"residual": diff, "tile_soft": soft, "tile_bin": t_bin,
                "recovered": self.inverse_transform(t_bin)}

    def extract_baseline(self, attacked: np.ndarray, host_i: np.ndarray) -> np.ndarray:
        raw = (attacked - host_i)[112:144, 112:144] / self.baseline_alpha + 0.5
        return (raw > 0.5).astype(np.uint8)


# ----------------------------------------------------------------- training attacks
def sample_training_attack(rng: np.random.RandomState) -> Dict[str, Any]:
    """Random attack spec for ANN training data (collusion excluded, see docstring)."""
    r = rng.rand()
    if r < 0.12:
        return {"type": "clean"}
    if r < 0.27:
        return {"type": "crop_random", "intensity": float(rng.uniform(0.05, 0.55))}
    if r < 0.35:
        return {"type": "crop_center", "intensity": float(rng.choice([0.1, 0.25, 0.5]))}
    if r < 0.52:
        return {"type": "jpeg", "quality": int(rng.randint(45, 96))}
    if r < 0.65:
        return {"type": "noise", "sigma": float(rng.uniform(0.005, 0.06))}
    if r < 0.74:
        return {"type": "blur", "kernel": 3, "sigma": float(rng.uniform(0.0, 1.0))}
    if r < 0.82:
        return {"type": "median", "kernel": 3}
    return {"type": "combined", "intensity": float(rng.uniform(0.05, 0.4)),
            "second": str(rng.choice(["jpeg", "noise", "blur"])),
            "quality": int(rng.randint(55, 96)), "sigma": float(rng.uniform(0.005, 0.04))}


def apply_training_attack(sys_: HybridSystem, img: np.ndarray, spec: Dict[str, Any], seed: int) -> np.ndarray:
    t = spec["type"]
    s = sys_.signal
    if t == "clean":
        return img.copy()
    if t == "crop_random":
        return sys_.cropper.apply_attack(img, mode="random", intensity=spec["intensity"], seed=seed)
    if t == "crop_center":
        return sys_.cropper.apply_attack(img, mode="center", intensity=spec["intensity"])
    if t == "jpeg":
        return s.apply_jpeg(img, quality=spec["quality"])
    if t == "noise":
        return s.apply_gaussian_noise(img, sigma=spec["sigma"], seed=seed)
    if t == "blur":
        return np.clip(cv2.GaussianBlur(img, (3, 3), spec.get("sigma", 0.0)), 0, 1)
    if t == "median":
        return s.apply_median_blur(img, kernel_size=3)
    if t == "combined":
        out = sys_.cropper.apply_attack(img, mode="random", intensity=spec["intensity"], seed=seed)
        if spec["second"] == "jpeg":
            return s.apply_jpeg(out, quality=spec["quality"])
        if spec["second"] == "noise":
            return s.apply_gaussian_noise(out, sigma=spec["sigma"], seed=seed + 1)
        return s.apply_gaussian_blur(out, kernel_size=3)
    raise ValueError(t)
