"""
Watermark embedding module for normalized I-channel data.
"""
import logging
from typing import Optional

import numpy as np


class WatermarkEmbedder:
    """
    Embeds a watermark mosaic into a normalized I-channel using additive blending.
    """

    def __init__(self, alpha: float = 0.08):
        """
        Initialize embedder.

        Args:
            alpha: Embedding strength in normalized domain
        """
        self.alpha = float(alpha)

    def embed(
        self,
        i_channel: np.ndarray,
        watermark_mosaic: np.ndarray,
        alpha: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """
        Embed watermark mosaic into an I-channel.

        Args:
            i_channel: Normalized host channel in range [0, 1]
            watermark_mosaic: Mosaic aligned with i_channel shape
            alpha: Optional override for embedding strength

        Returns:
            Embedded I-channel in [0, 1], or None on error
        """
        try:
            if i_channel is None or i_channel.size == 0:
                logging.error("Invalid i_channel: empty or None")
                return None

            if watermark_mosaic is None or watermark_mosaic.size == 0:
                logging.error("Invalid watermark_mosaic: empty or None")
                return None

            if i_channel.shape != watermark_mosaic.shape:
                logging.error(
                    "Shape mismatch: i_channel=%s vs watermark_mosaic=%s",
                    i_channel.shape,
                    watermark_mosaic.shape,
                )
                return None

            strength = self.alpha if alpha is None else float(alpha)
            if strength < 0.0:
                logging.error("Embedding alpha must be >= 0")
                return None

            host = i_channel.astype(np.float32)
            wm = watermark_mosaic.astype(np.float32)

            # Normalize watermark to approximately zero-mean in [-0.5, 0.5].
            if wm.max() > 1.0 or wm.min() < 0.0:
                wm = (wm - wm.min()) / (wm.max() - wm.min() + 1e-8)
            wm_centered = wm - 0.5

            embedded = host + (strength * wm_centered)
            embedded = np.clip(embedded, 0.0, 1.0)

            return embedded

        except Exception as exc:
            logging.error(f"Error embedding watermark: {str(exc)}")
            return None
