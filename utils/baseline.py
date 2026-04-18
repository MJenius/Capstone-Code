"""
Baseline watermarking module.
"""
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
        watermark: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Embed watermark in the center of the I-channel.
        
        Args:
            i_channel: 256x256 host channel [0, 1]
            watermark: 32x32 binary watermark
            
        Returns:
            Watermarked image
        """
        if i_channel.shape != (256, 256):
            logging.error(f"Host shape mismatch: {i_channel.shape}")
            return None
            
        if watermark.shape != (32, 32):
            logging.error(f"Watermark shape mismatch: {watermark.shape}")
            return None
            
        # Target: center 32x32 block
        # Start at (128-16) = 112
        y0, x0 = 112, 112
        y1, x1 = 144, 144
        
        embedded = i_channel.copy().astype(np.float32)
        
        # Center-normalize watermark to [-0.5, 0.5]
        wm = watermark.astype(np.float32)
        if wm.max() > 1.0 or wm.min() < 0.0:
            wm = (wm - wm.min()) / (wm.max() - wm.min() + 1e-8)
        wm_centered = wm - 0.5
        
        embedded[y0:y1, x0:x1] += self.alpha * wm_centered
        
        return np.clip(embedded, 0.0, 1.0)
