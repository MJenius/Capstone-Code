"""
Core preprocessing module for host images.
Handles resizing, YIQ color transformation, I-channel normalization, and preview generation.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2
import numpy as np


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Resize image to target size using bicubic interpolation."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)


def bgr_to_yiq(bgr_image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to YIQ color space using the standard NTSC transform matrix.
    Input: uint8 or float32 image in [0, 255]
    Output: float32 YIQ image
    """
    img = bgr_image.astype(np.float32)
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r - 0.274 * g - 0.322 * b
    q = 0.211 * r - 0.523 * g + 0.312 * b

    return np.stack([y, i, q], axis=2)


def yiq_to_bgr(yiq_image: np.ndarray) -> np.ndarray:
    """
    Convert YIQ image back to BGR color space using inverse NTSC transform.
    Output: uint8 image in [0, 255]
    """
    y = yiq_image[:, :, 0].astype(np.float32)
    i = yiq_image[:, :, 1].astype(np.float32)
    q = yiq_image[:, :, 2].astype(np.float32)

    r = y + 0.956 * i + 0.621 * q
    g = y - 0.272 * i - 0.647 * q
    b = y - 1.106 * i + 1.703 * q

    bgr = np.stack([b, g, r], axis=2)
    return np.clip(bgr, 0, 255).astype(np.uint8)


def normalize_channel(channel: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Min-max normalize a 2D channel to [0, 1] range.
    Returns: (normalized_channel, min_val, max_val)
    """
    min_val = float(np.min(channel))
    max_val = float(np.max(channel))
    denom = max_val - min_val
    if denom < 1e-8:
        denom = 1e-8
    norm = (channel - min_val) / denom
    return np.clip(norm, 0.0, 1.0).astype(np.float32), min_val, max_val


def denormalize_channel(normalized: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Denormalize a channel back to its original physical range."""
    return normalized * (max_val - min_val + 1e-8) + min_val


def preprocess_single_image(
    image_path: Path, target_size: Tuple[int, int] = (256, 256)
) -> Dict[str, np.ndarray]:
    """
    Complete preprocessing pipeline for a single image file.
    Returns dictionary with raw BGR, resized BGR, Y, I, Q, normalized I, and channel statistics.
    """
    bgr_raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr_raw is None:
        raise ValueError(f"Could not read image: {image_path}")

    bgr_resized = resize_image(bgr_raw, target_size)
    yiq = bgr_to_yiq(bgr_resized)
    y_chan = yiq[:, :, 0]
    i_chan = yiq[:, :, 1]
    q_chan = yiq[:, :, 2]

    i_norm, i_min, i_max = normalize_channel(i_chan)

    return {
        "bgr_raw": bgr_raw,
        "bgr_resized": bgr_resized,
        "yiq": yiq,
        "y_channel": y_chan,
        "i_channel_raw": i_chan,
        "q_channel": q_chan,
        "i_channel_norm": i_norm,
        "i_min": i_min,
        "i_max": i_max,
        "original_shape": bgr_raw.shape,
        "resized_shape": bgr_resized.shape,
    }
