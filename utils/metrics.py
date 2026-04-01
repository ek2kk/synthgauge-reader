from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class RegressionMeter:
    """Streaming regression metrics: MAE, RMSE, R2, and optional DRR@tol."""

    tol: Optional[float] = None
    _n: int = 0
    _sum_abs: float = 0.0
    _sum_sq: float = 0.0
    _sum_y: float = 0.0
    _sum_y2: float = 0.0
    _sum_correct: int = 0

    def reset(self) -> None:
        self._n = 0
        self._sum_abs = 0.0
        self._sum_sq = 0.0
        self._sum_y = 0.0
        self._sum_y2 = 0.0
        self._sum_correct = 0

    @torch.no_grad()
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        pred = y_pred.detach().float().view(-1)
        true = y_true.detach().float().view(-1)
        if pred.numel() != true.numel():
            raise ValueError(
                f"y_pred and y_true must have same numel, got {pred.numel()} vs {true.numel()}"
            )

        diff = pred - true
        batch_size = int(true.numel())

        self._n += batch_size
        self._sum_abs += float(diff.abs().sum().item())
        self._sum_sq += float((diff * diff).sum().item())
        self._sum_y += float(true.sum().item())
        self._sum_y2 += float((true * true).sum().item())

        if self.tol is not None:
            self._sum_correct += int((diff.abs() <= self.tol).sum().item())

    def compute(self) -> Dict[str, float]:
        if self._n == 0:
            drr_key = f"drr@{self.tol:.2f}" if self.tol is not None else None
            return {
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                **({"acc@tol": float("nan")} if self.tol is not None else {}),
                **({drr_key: float("nan")} if drr_key is not None else {}),
            }

        mae = self._sum_abs / self._n
        rmse = (self._sum_sq / self._n) ** 0.5
        mean_y = self._sum_y / self._n
        sst = self._sum_y2 - self._n * (mean_y * mean_y)
        sse = self._sum_sq
        r2 = 1.0 - (sse / sst) if sst > 1e-12 else float("nan")

        out = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}
        if self.tol is not None:
            acc = float(self._sum_correct / self._n)
            out["acc@tol"] = acc
            out[f"drr@{self.tol:.2f}"] = acc
        return out


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    items = []
    for key in sorted(metrics.keys()):
        value = metrics[key]
        if value != value:
            items.append(f"{prefix}{key}=nan")
        else:
            items.append(f"{prefix}{key}={value:.6f}")
    return " ".join(items)
