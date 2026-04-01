from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as tvm

BackboneName = Literal[
    "resnet18", "resnet34", "resnet50", "convnext_tiny", "mobilenet_v3_large"
]


@dataclass
class ModelConfig:
    backbone: BackboneName = "convnext_tiny"
    pretrained: bool = True
    dropout: float = 0.0


class GaugeRegressor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.backbone == "resnet18":
            weights = tvm.ResNet18_Weights.DEFAULT if cfg.pretrained else None
            backbone = tvm.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif cfg.backbone == "resnet34":
            weights = tvm.ResNet34_Weights.DEFAULT if cfg.pretrained else None
            backbone = tvm.resnet34(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif cfg.backbone == "resnet50":
            weights = tvm.ResNet50_Weights.DEFAULT if cfg.pretrained else None
            backbone = tvm.resnet50(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif cfg.backbone == "convnext_tiny":
            weights = tvm.ConvNeXt_Tiny_Weights.DEFAULT if cfg.pretrained else None
            backbone = tvm.convnext_tiny(weights=weights)
            in_features = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
        elif cfg.backbone == "mobilenet_v3_large":
            weights = tvm.MobileNet_V3_Large_Weights.DEFAULT if cfg.pretrained else None
            backbone = tvm.mobilenet_v3_large(weights=weights)
            in_features = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {cfg.backbone}")

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            nn.Linear(in_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        y = self.head(feats)
        return y.squeeze(1)
