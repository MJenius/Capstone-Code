"""
Core evaluation metrics module for imperceptibility and robustness.
Provides PSNR, SSIM, NC, and BER calculations.
"""
from typing import Dict
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn


def calculate_psnr(image_true: np.ndarray, image_test: np.ndarray, data_range: float = 1.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio (dB)."""
    return float(psnr_fn(image_true, image_test, data_range=data_range))


def calculate_ssim(image_true: np.ndarray, image_test: np.ndarray, data_range: float = 1.0) -> float:
    """Calculate Structural Similarity Index (SSIM)."""
    return float(ssim_fn(image_true, image_test, data_range=data_range))


def calculate_nc(w1: np.ndarray, w2: np.ndarray) -> float:
    """
    Calculate Normalized Correlation (NC) between ground-truth and recovered watermarks.
    Uses zero-mean normalized cross-correlation for fair invariant alignment.
    """
    w1_f = w1.astype(np.float32).flatten()
    w2_f = w2.astype(np.float32).flatten()
    w1_m = w1_f - np.mean(w1_f)
    w2_m = w2_f - np.mean(w2_f)
    denom = np.sqrt(np.sum(w1_m ** 2) * np.sum(w2_m ** 2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(w1_m * w2_m) / (denom + 1e-8))


def calculate_ber(w_orig: np.ndarray, w_extr: np.ndarray) -> float:
    """
    Calculate Bit Error Rate (BER) after 0.5 thresholding.
    """
    w1 = (w_orig > 0.5).astype(np.int32)
    w2 = (w_extr > 0.5).astype(np.int32)
    return float(np.sum(w1 != w2) / w1.size)


def evaluate_watermark_recovery(w_true: np.ndarray, w_recovered: np.ndarray) -> Dict[str, float]:
    """
    Comprehensive watermark extraction evaluation.
    """
    nc = calculate_nc(w_true, w_recovered)
    ber = calculate_ber(w_true, w_recovered)
    bit_errors = int(np.sum((w_true > 0.5) != (w_recovered > 0.5)))
    total_bits = w_true.size
    return {
        "nc": float(nc),
        "ber": float(ber),
        "bit_errors": int(bit_errors),
        "total_bits": int(total_bits),
        "exact_match": bool(bit_errors == 0)
    }
