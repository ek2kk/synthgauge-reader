from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class GaugeValueDataset(Dataset):
    """Regression dataset backed by JSONL index with fields image_path/value."""

    def __init__(self, index_jsonl: str | Path, transform: Optional[Callable] = None):
        self.index_path = Path(index_jsonl)
        self.transform = transform

        self.samples: list[tuple[str, float]] = []
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                self.samples.append((record["image_path"], float(record["value"])))

        if not self.samples:
            raise ValueError(f"Index is empty: {self.index_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, value = self.samples[idx]
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")

        if self.transform is not None:
            x = self.transform(rgb)
        else:
            arr = np.asarray(rgb, dtype=np.uint8)
            x = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)

        y = torch.tensor(value, dtype=torch.float32)
        return x, y
