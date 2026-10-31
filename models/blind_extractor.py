"""
Blind ANN watermark extractor (PyTorch).

Input : attacked / watermarked normalised I-channel, shape (B, 1, 256, 256)
Output: logits of the ORIGINAL 32x32 binary watermark, shape (B, 1, 32, 32)
        (``predict_proba`` applies the final sigmoid).

The network never receives the host image, the embedding residual, a crop mask or
the ground-truth watermark.  Architecture (MosaicAwareExtractor):

1. Input conditioning (parameter-free, input-only): per-image standardisation and
   a high-pass residual channel x - box5(x) that suppresses smooth host content.
2. Full-resolution encoder: conv stem + residual block (local noise-residual features).
3. Multi-scale context branch: stride-2 downsampling + 2 residual blocks at
   128x128, bilinearly upsampled and fused with the full-resolution features
   (estimates local host structure / texture, i.e. the adaptive alpha regime).
4. Mosaic-aware tile aggregation: the 256x256 feature map is folded into the
   8x8 grid of 32x32 tiles (the embedding period).  A learned per-pixel
   confidence map is soft-maxed across the 64 tile copies, so tiles destroyed by
   cropping / saturation receive low weight.  Weighted and plain means are
   concatenated.
5. 32x32 decoder: 1x1 projection + 3 residual blocks + 1x1 head -> logits in the
   transformed (Arnold+Catalan) domain.
6. Keyed inverse permutation: a FIXED, non-trainable index gather that undoes
   Catalan then Arnold using the secret key, producing logits of the original
   watermark.  The key is part of the watermarking secret, exactly as in the
   non-blind extractor; no host information is involved.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core import keyed_permutation as kp


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        return F.relu(x + y, inplace=True)


def conv_bn(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, stride=s, padding=k // 2, bias=False), nn.BatchNorm2d(o),
                         nn.ReLU(inplace=True))


class BlindWatermarkExtractor(nn.Module):
    def __init__(self, base_channels: int = 16, decoder_channels: int = 64, grid: int = 8, tile: int = 32,
                 acm_iterations: int = 10, catalan_iterations: int = 5, catalan_key: int = 7,
                 use_key_permutation: bool = True):
        super().__init__()
        self.config = dict(base_channels=base_channels, decoder_channels=decoder_channels, grid=grid, tile=tile,
                           acm_iterations=acm_iterations, catalan_iterations=catalan_iterations,
                           catalan_key=catalan_key, use_key_permutation=use_key_permutation)
        b, d = base_channels, decoder_channels
        self.grid, self.tile = grid, tile
        self.stem = conv_bn(2, b)
        self.full = ResBlock(b)
        self.down = conv_bn(b, 2 * b, s=2)
        self.ctx = nn.Sequential(ResBlock(2 * b), ResBlock(2 * b))
        self.fuse = nn.Sequential(conv_bn(3 * b, b, k=1), ResBlock(b))
        self.conf = nn.Conv2d(b, 1, 1)
        self.dec_in = conv_bn(2 * b, d, k=1)
        self.dec = nn.Sequential(ResBlock(d), ResBlock(d), ResBlock(d))
        self.head = nn.Conv2d(d, 1, 1)
        if use_key_permutation:
            inv = kp.inverse_index(tile, acm_iterations, catalan_iterations, catalan_key)
            self.register_buffer("inv_perm", torch.as_tensor(inv, dtype=torch.long), persistent=True)
        else:
            self.inv_perm = None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # zero-init last BN of each residual block -> identity at start (stable training)
        for m in self.modules():
            if isinstance(m, ResBlock):
                nn.init.zeros_(m.b2.weight)
        nn.init.zeros_(self.conf.weight)
        nn.init.zeros_(self.conf.bias)

    @staticmethod
    def condition(x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=(2, 3), keepdim=True)
        sd = x.std(dim=(2, 3), keepdim=True) + 1e-4
        xs = (x - mu) / sd
        hp = (x - F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), 5, stride=1)) * 50.0
        return torch.cat([xs, hp], 1)

    def features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.full(self.stem(self.condition(x)))
        c = self.ctx(self.down(h))
        c = F.interpolate(c, size=h.shape[-2:], mode="bilinear", align_corners=False)
        f = self.fuse(torch.cat([h, c], 1))
        B, C, H, W = f.shape
        g, t = self.grid, self.tile
        conf = self.conf(f).view(B, 1, g, t, g, t)
        w = torch.softmax(conf.permute(0, 1, 3, 5, 2, 4).reshape(B, 1, t, t, g * g), dim=-1)
        ft = f.view(B, C, g, t, g, t).permute(0, 1, 3, 5, 2, 4).reshape(B, C, t, t, g * g)
        agg = torch.cat([(ft * w).sum(-1), ft.mean(-1)], 1)
        tile_logits = self.head(self.dec(self.dec_in(agg)))
        return {"tile_logits": tile_logits, "tile_weights": w.view(B, t, t, g, g)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits of the original 32x32 watermark, shape (B,1,32,32)."""
        t = self.features(x)["tile_logits"]
        if self.inv_perm is None:
            return t
        B = t.shape[0]
        return t.view(B, -1)[:, self.inv_perm].view(B, 1, self.tile, self.tile)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: Optional[dict] = None) -> BlindWatermarkExtractor:
    return BlindWatermarkExtractor(**(cfg or {}))
