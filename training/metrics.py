"""
Watermark-recovery metrics (numpy + torch batch versions).

NC is the zero-mean normalised cross-correlation used throughout the repository
(core.evaluation.calculate_nc), computed on binarised predictions.  Precision /
recall / F1 treat bit value 1 as the positive class.
"""
from typing import Dict

import numpy as np
import torch


def bit_metrics(pred_bin: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    p = (np.asarray(pred_bin) > 0.5).astype(np.float64).ravel()
    t = (np.asarray(target) > 0.5).astype(np.float64).ravel()
    tp = float((p * t).sum()); fp = float((p * (1 - t)).sum()); fn = float(((1 - p) * t).sum())
    acc = float((p == t).mean())
    pm, tm = p - p.mean(), t - t.mean()
    den = np.sqrt((pm ** 2).sum() * (tm ** 2).sum())
    nc = float((pm * tm).sum() / den) if den > 1e-12 else 0.0
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"nc": nc, "ber": 1.0 - acc, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


@torch.no_grad()
def batch_metrics(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-sample metrics for a batch; returns tensors of shape (B,)."""
    B = logits.shape[0]
    p = (logits > 0).float().view(B, -1)
    t = (target > 0.5).float().view(B, -1)
    tp = (p * t).sum(1); fp = (p * (1 - t)).sum(1); fn = ((1 - p) * t).sum(1)
    acc = (p == t).float().mean(1)
    pm, tm = p - p.mean(1, keepdim=True), t - t.mean(1, keepdim=True)
    den = torch.sqrt((pm ** 2).sum(1) * (tm ** 2).sum(1))
    nc = torch.where(den > 1e-12, (pm * tm).sum(1) / den.clamp_min(1e-12), torch.zeros_like(den))
    prec = torch.where(tp + fp > 0, tp / (tp + fp).clamp_min(1), torch.zeros_like(tp))
    rec = torch.where(tp + fn > 0, tp / (tp + fn).clamp_min(1), torch.zeros_like(tp))
    f1 = torch.where(prec + rec > 0, 2 * prec * rec / (prec + rec).clamp_min(1e-12), torch.zeros_like(tp))
    return {"nc": nc, "ber": 1 - acc, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
