"""
Loss functions.

Primary: BCEWithLogitsLoss on the 1024 watermark bits (numerically stable
sigmoid + binary cross-entropy; the bits are a per-pixel binary classification).

Auxiliary (weight ``nc_weight``, default 0.1): 1 - soft NC, where soft NC is the
zero-mean normalised correlation between sigmoid probabilities and the target.
Justification: NC is the primary evaluation metric of the project; the term
penalises globally inverted / low-contrast predictions which per-bit BCE alone
weights only weakly.  It is differentiable and bounded in [0, 2].
"""
import torch
import torch.nn as nn


def soft_nc(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    B = logits.shape[0]
    p = torch.sigmoid(logits).view(B, -1)
    t = target.view(B, -1).float()
    pm, tm = p - p.mean(1, keepdim=True), t - t.mean(1, keepdim=True)
    return (pm * tm).sum(1) / (torch.sqrt((pm ** 2).sum(1) * (tm ** 2).sum(1)) + 1e-6)


class WatermarkLoss(nn.Module):
    def __init__(self, nc_weight: float = 0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.nc_weight = nc_weight

    def forward(self, logits, target):
        loss = self.bce(logits, target.float())
        if self.nc_weight > 0:
            loss = loss + self.nc_weight * (1 - soft_nc(logits, target)).mean()
        return loss
