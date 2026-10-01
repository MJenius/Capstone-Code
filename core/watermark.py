"""
Core watermark transformation module.
Wraps Arnold Cat Map, Catalan permutation, and 8x8 Mosaic expansion with reversible verification.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import cv2

from utils.scrambler import WatermarkScrambler
from utils.catalan import CatalanTransform
from utils.mosaic import MosaicGenerator
from core.evaluation import evaluate_watermark_recovery


class WatermarkTransformer:
    """
    Orchestrates:
    Binary Watermark -> Arnold Cat Map -> Catalan Permutation -> 8x8 Mosaic
    and their exact mathematical inverses.
    """

    def __init__(
        self,
        watermark_size: int = 32,
        acm_iterations: int = 10,
        catalan_iterations: int = 5,
        catalan_key: int = 7,
        mosaic_grid: Tuple[int, int] = (8, 8),
    ):
        self.watermark_size = watermark_size
        self.acm_iterations = acm_iterations
        self.catalan_iterations = catalan_iterations
        self.catalan_key = catalan_key
        self.mosaic_grid = mosaic_grid

        self.scrambler = WatermarkScrambler(default_size=(watermark_size, watermark_size))
        self.catalan = CatalanTransform()
        self.mosaic_gen = MosaicGenerator()

    def transform_forward(self, binary_watermark: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward transformation:
        1. Validate/resize to (N, N)
        2. Arnold Cat Map scrambling
        3. Catalan Permutation
        4. 8x8 Mosaic tiling
        """
        wm = (binary_watermark > 0.5).astype(np.uint8)
        if wm.shape != (self.watermark_size, self.watermark_size):
            wm = cv2.resize(
                wm, (self.watermark_size, self.watermark_size), interpolation=cv2.INTER_NEAREST
            )

        scrambled = self.scrambler.arnold_cat_map(wm, iterations=self.acm_iterations)
        catalan_perm = self.catalan.catalan_transform(
            scrambled, iterations=self.catalan_iterations, key=self.catalan_key
        )
        target_shape = (
            self.watermark_size * self.mosaic_grid[0],
            self.watermark_size * self.mosaic_grid[1],
        )
        mosaic = self.mosaic_gen.create_tiled_mosaic(catalan_perm, target_shape=target_shape)

        return {
            "watermark_binary": wm,
            "watermark_scrambled": scrambled,
            "watermark_catalan": catalan_perm,
            "watermark_mosaic": mosaic,
        }

    def transform_inverse(self, permuted_watermark: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Inverse transformation:
        1. Inverse Catalan Permutation
        2. Inverse Arnold Cat Map
        """
        # Threshold to binary if needed
        perm_bin = (permuted_watermark > 0.5).astype(np.uint8)
        inv_catalan = self.catalan.inverse_catalan_transform(
            perm_bin, iterations=self.catalan_iterations, key=self.catalan_key
        )
        recovered_binary = self.scrambler.inverse_arnold_cat_map(
            inv_catalan, iterations=self.acm_iterations
        )

        return {
            "recovered_scrambled": inv_catalan,
            "recovered_binary": recovered_binary,
        }

    def verify_reversibility(self, binary_watermark: np.ndarray) -> Dict[str, any]:
        """
        Runs full forward and inverse transforms and tests bit-exact reversibility.
        """
        fwd = self.transform_forward(binary_watermark)
        inv = self.transform_inverse(fwd["watermark_catalan"])
        eval_metrics = evaluate_watermark_recovery(
            fwd["watermark_binary"], inv["recovered_binary"]
        )
        return {
            **fwd,
            **inv,
            "reversibility_metrics": eval_metrics,
            "exact_reversibility": eval_metrics["exact_match"],
        }
