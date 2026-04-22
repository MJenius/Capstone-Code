import cv2
import numpy as np
import logging
from typing import Optional

class AdaptiveEmbedder:
    """
    Implements Luminance-Texture Masking to optimize imperceptibility (PSNR > 40dB).
    Higher strength in textured/edge areas, lower in smooth/flat areas.
    """
    
    def __init__(self, alpha_base: float = 0.05, sensitivity: float = 2.0):
        """
        Args:
            alpha_base: Minimum base embedding strength for smooth areas.
            sensitivity: How much texture increases the strength.
        """
        self.alpha_base = alpha_base
        self.sensitivity = sensitivity

    def get_texture_mask(self, i_channel: np.ndarray, win_size: int = 7) -> np.ndarray:
        """
        Calculate local variance based texture mask.
        """
        # E[X]
        mean = cv2.boxFilter(i_channel, -1, (win_size, win_size))
        # E[X^2]
        mean_sq = cv2.boxFilter(i_channel**2, -1, (win_size, win_size))
        # Var = E[X^2] - E[X]^2
        variance = mean_sq - mean**2
        variance = np.maximum(variance, 0)
        
        # Standardize mask to [0, 1]
        std = np.sqrt(variance)
        mask = (std - std.min()) / (std.max() - std.min() + 1e-8)
        return mask

    def embed(
        self, 
        i_channel: np.ndarray, 
        watermark_mosaic: np.ndarray
    ) -> np.ndarray:
        """
        Perceptually adaptive embedding.
        """
        # 1. Get texture mask
        mask = self.get_texture_mask(i_channel)
        
        # 1b. Create a spatial weight map (Gaussian) that emphasizes the center
        h, w = i_channel.shape
        y, x = np.ogrid[-h/2:h/2, -w/2:w/2]
        # Gaussian with sigma = h/2, amplitude = 1 at center, ~0.6 at edge
        spatial_weight = np.exp(-(x**2 + y**2) / (2 * (h/2)**2))
        # Scale to [1.0, 1.5]
        spatial_weight = 1.0 + 0.5 * spatial_weight
        
        # 2. Calculate pixel-wise alpha
        # Smooth areas (mask~0) get alpha_base
        # Textured areas (mask~1) get alpha_base * (1 + sensitivity)
        # Then multiply by the spatial weight to bump the center
        alpha_pixel = self.alpha_base * (1.0 + self.sensitivity * mask) * spatial_weight
        
        # 3. Embed
        host = i_channel.astype(np.float32)
        wm = watermark_mosaic.astype(np.float32)
        if wm.max() > 1.0 or wm.min() < 0.0:
            wm = (wm - wm.min()) / (wm.max() - wm.min() + 1e-8)
        wm_centered = wm - 0.5
        
        embedded = host + (alpha_pixel * wm_centered)
        return np.clip(embedded, 0.0, 1.0)
