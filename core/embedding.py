"""
Core embedding module for perceptual adaptive watermark embedding.
Uses local variance luminance-texture masking to modulate embedding strength.
"""
from typing import Dict, Tuple
import cv2
import numpy as np

from utils.adaptive_embedder import AdaptiveEmbedder
from core.evaluation import calculate_psnr, calculate_ssim


class WatermarkEmbedder:
    """
    Manages watermark embedding into the host I-channel.
    Supports both:
    1. Perceptually Adaptive Luminance-Texture Masking (Hybrid Framework)
    2. Center-Tile Normal Embedding (Baseline Method)
    """

    def __init__(self, alpha_base: float = 0.012, sensitivity: float = 2.0):
        self.alpha_base = alpha_base
        self.sensitivity = sensitivity
        self.adaptive_embedder = AdaptiveEmbedder(
            alpha_base=alpha_base, sensitivity=sensitivity
        )

    def embed_adaptive(
        self, host_i: np.ndarray, watermark_mosaic: np.ndarray
    ) -> Dict[str, any]:
        """
        Embed watermark mosaic into host I-channel adaptively.
        Returns:
            embedded_i: 2D watermarked I-channel in [0, 1]
            texture_mask: normalized local variance map
            alpha_map: pixel-wise embedding strength
            diff_map: absolute difference |embedded - host|
            psnr: clean embedding PSNR (dB)
            ssim: clean embedding SSIM
        """
        mask = self.adaptive_embedder.get_texture_mask(host_i)
        alpha_map = self.alpha_base * (1.0 + self.sensitivity * mask)

        host_f = host_i.astype(np.float32)
        wm_f = watermark_mosaic.astype(np.float32)
        if wm_f.max() > 1.0 or wm_f.min() < 0.0:
            wm_f = (wm_f - wm_f.min()) / (wm_f.max() - wm_f.min() + 1e-8)
        wm_centered = wm_f - 0.5

        embedded_i = np.clip(host_f + alpha_map * wm_centered, 0.0, 1.0)
        diff_map = np.abs(embedded_i - host_f)

        psnr_val = calculate_psnr(host_f, embedded_i, data_range=1.0)
        ssim_val = calculate_ssim(host_f, embedded_i, data_range=1.0)

        return {
            "embedded_i": embedded_i,
            "texture_mask": mask,
            "alpha_map": alpha_map,
            "diff_map": diff_map,
            "psnr": float(psnr_val),
            "ssim": float(ssim_val),
            "alpha_min": float(alpha_map.min()),
            "alpha_max": float(alpha_map.max()),
            "alpha_mean": float(alpha_map.mean()),
            "diff_min": float(diff_map.min()),
            "diff_max": float(diff_map.max()),
            "diff_mean": float(diff_map.mean()),
        }
