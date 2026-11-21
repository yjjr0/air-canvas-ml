from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gestures import FEATURES, LABELS


class GestureNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int = FEATURES,
        num_classes: int = len(LABELS),
        emb_dim: int = 32,
        width: int = 64,
        dropout: float = 0.2,
        recent: int = 4,
        logit_scale: float = 8.0,
    ) -> None:
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(1, in_channels, 1))
        self.register_buffer("feat_std", torch.ones(1, in_channels, 1))
        self.register_buffer("logit_scale", torch.tensor(float(logit_scale)))

        def block(cin: int, cout: int, dilation: int) -> list[nn.Module]:
            return [
                nn.Conv1d(cin, cout, kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(cout),
                nn.ReLU(),
            ]

        self.backbone = nn.Sequential(
            *block(in_channels, width, 1),
            nn.Dropout(dropout),
            *block(width, width, 2),
            *block(width, width, 4),
        )
        self.recent = recent
        self.proj = nn.Linear(2 * width, emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feat_mean.copy_(mean.reshape(1, -1, 1))
        self.feat_std.copy_(std.clamp_min(1e-2).reshape(1, -1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (x - self.feat_mean) / self.feat_std
        h = self.backbone(x)
        pooled = torch.cat([h.mean(dim=-1), h[..., -self.recent :].mean(dim=-1)], dim=-1)
        emb = F.normalize(self.proj(pooled), p=2.0, dim=-1)
        logits = self.head(emb * self.logit_scale)
        return logits, emb


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
