"""
Blind inference: attacked/watermarked image -> recovered 32x32 watermark.

The public API deliberately has NO parameter for a host image, residual or crop
mask.  Ground truth is only accepted by :func:`score_against_ground_truth`, which
is called AFTER prediction.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Sequence, Union

import cv2
import numpy as np
import torch

from core.preprocessing import bgr_to_yiq, normalize_channel, resize_image
from training.metrics import bit_metrics
from training.trainer import get_device, load_checkpoint

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = BASE_DIR / "models" / "checkpoints" / "best_model.pth"


def _rel_path(p: Union[str, Path]) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(BASE_DIR.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def image_to_i_channel(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Same preprocessing as Phase 1: resize 256x256 (bicubic) -> YIQ -> I -> min-max [0,1].

    A 2-D float array in [0,1] is treated as an already-extracted I-channel.
    """
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot read image: {image}")
        image = bgr
    arr = np.asarray(image)
    if arr.ndim == 2:
        if arr.dtype != np.float32 and arr.max() > 1.0:
            arr = arr.astype(np.float32) / 255.0
        if arr.shape != (256, 256):
            arr = cv2.resize(arr.astype(np.float32), (256, 256), interpolation=cv2.INTER_CUBIC)
        return np.clip(arr.astype(np.float32), 0, 1)
    if arr.shape[:2] != (256, 256):
        arr = resize_image(arr, (256, 256))
    return normalize_channel(bgr_to_yiq(arr)[:, :, 1])[0]


class BlindExtractor:
    def __init__(self, checkpoint: Union[str, Path] = DEFAULT_CKPT, device: str = "auto"):
        self.device = get_device(device)
        self.checkpoint = Path(checkpoint)
        self.model, self.ckpt = load_checkpoint(self.checkpoint, self.device)

    @torch.no_grad()
    def predict_batch(self, i_channels: Sequence[np.ndarray], threshold: float = 0.5) -> List[Dict]:
        x = torch.from_numpy(np.stack([np.asarray(i, np.float32) for i in i_channels]))[:, None].to(self.device)
        t0 = time.perf_counter()
        prob = torch.sigmoid(self.model(x)).cpu().numpy()[:, 0]
        dt = (time.perf_counter() - t0) / len(i_channels)
        return [{"probability": p, "binary": (p > threshold).astype(np.uint8), "inference_seconds": dt}
                for p in prob]

    def predict(self, image: Union[str, Path, np.ndarray], threshold: float = 0.5) -> Dict:
        """Blind extraction from an image path, BGR array or I-channel. No host involved."""
        i = image_to_i_channel(image)
        out = self.predict_batch([i], threshold)[0]
        out["input_i_channel"] = i
        return out

    def info(self) -> Dict:
        return {"checkpoint": _rel_path(self.checkpoint), "epoch": self.ckpt.get("epoch"),
                "best_val_nc": self.ckpt.get("best_metric"), "model_config": self.ckpt.get("model_config"),
                "saved_at": self.ckpt.get("saved_at")}


def score_against_ground_truth(pred_binary: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Post-inference evaluation only."""
    return bit_metrics(pred_binary, ground_truth)
