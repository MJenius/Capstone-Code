"""
Interfaces and wrapper for the Blind ANN Watermark Extractor.

STATUS: IMPLEMENTED & TRAINABLE (PyTorch)
Provides BaseANNWatermarkExtractor contract and ConcreteBlindANNExtractor
wrapping models.blind_extractor.BlindWatermarkExtractor and training.inference.BlindExtractor.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent


def _rel_path(p: Optional[Path | str]) -> Optional[str]:
    if p is None:
        return None
    p = Path(p).resolve()
    try:
        return p.relative_to(BASE_DIR.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


class BaseANNWatermarkExtractor(ABC):
    """
    Abstract Base Class for Deep Learning Blind Watermark Extractors.
    
    Contract:
    - Input: Attacked / Watermarked Image or I-Channel of shape (B, 1, 256, 256)
             WITHOUT access to the clean host image (Blind Extraction).
    - Output: Predicted watermark array of shape (B, 1, 32, 32) in range [0, 1].
    """

    def __init__(self, model_name: str = "BlindWatermarkExtractor"):
        self.model_name = model_name
        self.is_trained = False

    @abstractmethod
    def forward(self, input_tensor: Any) -> Any:
        """Forward pass through the neural network."""
        pass

    @abstractmethod
    def extract_blind(self, watermarked_channel: np.ndarray) -> np.ndarray:
        """
        Inference interface:
        Given an unknown watermarked 256x256 image channel without host,
        predict and output the 32x32 binary watermark.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save network weights to disk."""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load network weights from disk."""
        pass


class ConcreteBlindANNExtractor(BaseANNWatermarkExtractor):
    """
    Production PyTorch Blind ANN Watermark Extractor fulfilling BaseANNWatermarkExtractor.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "auto"):
        super().__init__(model_name="BlindWatermarkExtractor")
        import torch
        from models.blind_extractor import build_model
        from training.trainer import get_device, load_checkpoint, save_checkpoint, CKPT_DIR

        self.device = get_device(device)
        self._save_fn = save_checkpoint
        self._load_fn = load_checkpoint
        default_ckpt = CKPT_DIR / "best_model.pth"
        ckpt_to_use = Path(checkpoint_path) if checkpoint_path else default_ckpt
        if ckpt_to_use.exists():
            self.model, self.ckpt_meta = load_checkpoint(ckpt_to_use, self.device)
            self.is_trained = True
            self.checkpoint_path = _rel_path(ckpt_to_use)
        else:
            self.model = build_model().to(self.device)
            self.model.eval()
            self.ckpt_meta = {}
            self.is_trained = False
            self.checkpoint_path = None

    def forward(self, input_tensor: Any) -> Any:
        import torch
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.as_tensor(input_tensor, dtype=torch.float32, device=self.device)
        if input_tensor.ndim == 2:
            input_tensor = input_tensor[None, None]
        elif input_tensor.ndim == 3:
            input_tensor = input_tensor[:, None]
        return torch.sigmoid(self.model(input_tensor.to(self.device)))

    def extract_blind(self, watermarked_channel: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        import torch
        from training.inference import image_to_i_channel
        i_chan = image_to_i_channel(watermarked_channel)
        with torch.no_grad():
            x = torch.from_numpy(i_chan.astype(np.float32))[None, None].to(self.device)
            prob = torch.sigmoid(self.model(x)).cpu().numpy()[0, 0]
        binary = (prob > threshold).astype(np.uint8)
        return {
            "probability": prob,
            "binary": binary,
            "input_i_channel": i_chan,
            "is_trained": self.is_trained,
            "checkpoint": self.checkpoint_path,
        }

    def save_checkpoint(self, path: str) -> None:
        self._save_fn(Path(path), self.model)

    def load_checkpoint(self, path: str) -> None:
        self.model, self.ckpt_meta = self._load_fn(Path(path), self.device)
        self.is_trained = True
        self.checkpoint_path = _rel_path(path)


# Backwards-compatible alias for any external scripts importing PlannedANNModelPlaceholder
PlannedANNModelPlaceholder = ConcreteBlindANNExtractor


def get_ann_model_info() -> Dict[str, Any]:
    """Return implementation status and architecture metadata for reports."""
    from models.blind_extractor import build_model, count_parameters
    from training.trainer import CKPT_DIR
    best_ckpt = CKPT_DIR / "best_model.pth"
    m = build_model()
    info = {
        "status": "TRAINED" if best_ckpt.exists() else "IMPLEMENTED (UNTRAINED)",
        "model_class": "BlindWatermarkExtractor",
        "description": "Mosaic-Aware Multi-Scale Residual CNN Blind Watermark Extractor (Host-Free)",
        "input_shape": [1, 256, 256],
        "output_shape": [1, 32, 32],
        "trainable_parameters": count_parameters(m),
        "loss_function": "BCEWithLogitsLoss + 0.1 * (1 - soft_NC)",
        "checkpoint_path": _rel_path(best_ckpt) if best_ckpt.exists() else None,
    }
    if best_ckpt.exists():
        import torch
        ck = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        info["epoch"] = ck.get("epoch")
        info["best_val_nc"] = ck.get("best_metric")
        info["saved_at"] = ck.get("saved_at")
    return info
