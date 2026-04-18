"""
Baseline watermarking module.
"""
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional

class NormalEmbedder:
    """
    Standard additive embedder that places a single watermark in the center.
    """
    def __init__(self, alpha: float = 0.08):
        self.alpha = float(alpha)

    def embed(
        self, 
        i_channel: np.ndarray, 
        watermark: np.ndarray,
        visible: bool = True,
        pos: Optional[tuple] = None
    ) -> Optional[np.ndarray]:
        """
        Embed watermark in the I-channel.
        
        Args:
            i_channel: 256x256 host channel [0, 1]
            watermark: 32x32 binary watermark
            visible: Whether to use visible alpha blending (default: True)
            pos: (y0, x0) top-left corner. Defaults to center (112, 112).
            
        Returns:
            Watermarked image
        """
        if i_channel.shape != (256, 256):
            logging.error(f"Host shape mismatch: {i_channel.shape}")
            return None
            
        if watermark.shape != (32, 32):
            logging.error(f"Watermark shape mismatch: {watermark.shape}")
            return None
            
        # Target area
        if pos is None:
            y0, x0 = 112, 112
        else:
            y0, x0 = pos
            
        y1, x1 = y0 + 32, x0 + 32
        
        # Ensure within bounds
        y0, y1 = max(0, y0), min(256, y1)
        x0, x1 = max(0, x0), min(256, x1)
        
        embedded = i_channel.copy().astype(np.float32)
        wm = watermark.astype(np.float32)
        if wm.max() > 1.0 or wm.min() < 0.0:
            wm = (wm - wm.min()) / (wm.max() - wm.min() + 1e-8)
        
        # Crop watermark if area is smaller due to bounds
        h_target, w_target = y1 - y0, x1 - x0
        wm_crop = cv2.resize(wm, (w_target, h_target))
        
        if visible:
            alpha_vis = 0.4
            embedded[y0:y1, x0:x1] = (1 - alpha_vis) * embedded[y0:y1, x0:x1] + alpha_vis * wm_crop
        else:
            embedded[y0:y1, x0:x1] += self.alpha * (wm_crop - 0.5)
        
        return np.clip(embedded, 0.0, 1.0)
