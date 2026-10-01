"""
Core non-blind watermark extraction module.
Performs host subtraction, tile extraction, mask-weighted aggregation,
and inverse Catalan + inverse Arnold reconstruction.
"""
from typing import Any, Dict, Optional
import numpy as np

from utils.catalan import CatalanTransform
from utils.scrambler import WatermarkScrambler
from core.evaluation import evaluate_watermark_recovery


class NonBlindExtractor:
    """
    Non-Blind (Host-Subtracted) Watermark Extractor for the Hybrid Mosaic framework.
    """

    def __init__(
        self,
        alpha_base: float = 0.012,
        watermark_size: int = 32,
        acm_iterations: int = 10,
        catalan_iterations: int = 5,
        catalan_key: int = 7,
    ):
        self.alpha_base = alpha_base
        self.watermark_size = watermark_size
        self.acm_iterations = acm_iterations
        self.catalan_iterations = catalan_iterations
        self.catalan_key = catalan_key

        self.scrambler = WatermarkScrambler(default_size=(watermark_size, watermark_size))
        self.catalan = CatalanTransform()

    def extract(
        self,
        attacked_i: np.ndarray,
        host_i: np.ndarray,
        mask: Optional[np.ndarray] = None,
        ground_truth_binary: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Complete non-blind extraction pipeline:
        1. Residual host subtraction: diff = (attacked - host) / alpha_base + 0.5
        2. Mosaic tile decomposition into 64 blocks of 32x32
        3. Tile aggregation (mask-weighted if cropped)
        4. Binary thresholding (> 0.5)
        5. Inverse Catalan permutation
        6. Inverse Arnold Cat Map
        7. Recovery metric evaluation
        """
        diff = (attacked_i - host_i) / self.alpha_base + 0.5
        tiles_diff = diff.reshape(8, 32, 8, 32).transpose(0, 2, 1, 3).reshape(64, 32, 32)

        if mask is not None:
            tiles_mask = mask.reshape(8, 32, 8, 32).transpose(0, 2, 1, 3).reshape(64, 32, 32)
            sum_diff = np.sum(tiles_diff * tiles_mask, axis=0)
            sum_weights = np.sum(tiles_mask, axis=0)
            recovered_catalan_raw = np.where(
                sum_weights > 0, sum_diff / (sum_weights + 1e-8), 0.5
            )
        else:
            recovered_catalan_raw = np.mean(tiles_diff, axis=0)

        # Threshold to binary
        recovered_catalan_bin = (recovered_catalan_raw > 0.5).astype(np.uint8)

        # Invert Catalan
        recovered_scrambled = self.catalan.inverse_catalan_transform(
            recovered_catalan_bin,
            iterations=self.catalan_iterations,
            key=self.catalan_key,
        )

        # Invert Arnold
        recovered_binary = self.scrambler.inverse_arnold_cat_map(
            recovered_scrambled, iterations=self.acm_iterations
        )

        metrics = {}
        diff_with_gt = None
        if ground_truth_binary is not None:
            metrics = evaluate_watermark_recovery(ground_truth_binary, recovered_binary)
            diff_with_gt = np.abs(
                recovered_binary.astype(np.float32) - ground_truth_binary.astype(np.float32)
            )

        return {
            "residual_diff": diff,
            "recovered_catalan_raw": recovered_catalan_raw,
            "recovered_catalan_bin": recovered_catalan_bin,
            "recovered_scrambled": recovered_scrambled,
            "recovered_binary": recovered_binary,
            "metrics": metrics,
            "diff_with_gt": diff_with_gt,
        }
