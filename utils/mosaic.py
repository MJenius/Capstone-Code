"""
Mosaic generation module for watermark tiling.
"""
import logging
from typing import Optional, Tuple

import numpy as np


class MosaicGenerator:
    """
    Generates a full-size watermark mosaic from a smaller 2D watermark.
    """

    def create_tiled_mosaic(
        self,
        watermark: np.ndarray,
        target_shape: Tuple[int, int] = (256, 256)
    ) -> Optional[np.ndarray]:
        """
        Tile a watermark to fill the target shape.

        Args:
            watermark: Input 2D watermark
            target_shape: (height, width) for output mosaic

        Returns:
            Tiled mosaic array, or None on error
        """
        try:
            if watermark is None or watermark.size == 0:
                logging.error("Invalid watermark: empty or None")
                return None

            if len(watermark.shape) != 2:
                logging.error(f"Watermark must be 2D, got shape: {watermark.shape}")
                return None

            target_h, target_w = target_shape
            wm_h, wm_w = watermark.shape

            if target_h % wm_h != 0 or target_w % wm_w != 0:
                logging.error(
                    "Target shape %s must be divisible by watermark shape %s",
                    target_shape,
                    watermark.shape,
                )
                return None

            tile_y = target_h // wm_h
            tile_x = target_w // wm_w

            mosaic = np.tile(watermark, (tile_y, tile_x))

            logging.info(
                "Generated mosaic: watermark=%sx%s, target=%sx%s, grid=%sx%s",
                wm_h,
                wm_w,
                target_h,
                target_w,
                tile_y,
                tile_x,
            )
            return mosaic

        except Exception as exc:
            logging.error(f"Error creating watermark mosaic: {str(exc)}")
            return None
