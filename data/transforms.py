from __future__ import annotations

from typing import Any, Callable, Dict

from torchvision import transforms as T


def _get_mean_std(cfg: Dict[str, Any]) -> tuple[list[float], list[float]]:
    mean = cfg.get("mean", [0.485, 0.456, 0.406])
    std = cfg.get("std", [0.229, 0.224, 0.225])
    return mean, std


def build_transforms(cfg: Dict[str, Any], split: str) -> Callable:
    """
    Build torchvision transforms for regression images.

    `split` should be one of: train/val/test.
    """
    tcfg = cfg.get("transforms", cfg.get("transforms_reg", {}))
    img_size = int(tcfg.get("img_size", 256))
    normalize = bool(tcfg.get("normalize", True))
    mean, std = _get_mean_std(tcfg)

    base = [
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ]

    if split == "train":
        train_cfg = tcfg.get("train", {})
        aug = []

        if float(train_cfg.get("color_jitter_p", 0.0)) > 0:
            b, c, s, h = train_cfg.get("color_jitter", [0.2, 0.2, 0.2, 0.05])
            aug.append(
                T.RandomApply(
                    [T.ColorJitter(b, c, s, h)],
                    p=float(train_cfg["color_jitter_p"]),
                )
            )

        if float(train_cfg.get("random_affine_p", 0.0)) > 0:
            degrees = float(train_cfg.get("degrees", 7))
            translate = float(train_cfg.get("translate", 0.02))
            scale = train_cfg.get("scale", [0.95, 1.05])
            aug.append(
                T.RandomApply(
                    [
                        T.RandomAffine(
                            degrees=degrees,
                            translate=(translate, translate),
                            scale=(float(scale[0]), float(scale[1])),
                            interpolation=T.InterpolationMode.BILINEAR,
                            fill=0,
                        )
                    ],
                    p=float(train_cfg["random_affine_p"]),
                )
            )

        if float(train_cfg.get("blur_p", 0.0)) > 0:
            aug.append(
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                    p=float(train_cfg["blur_p"]),
                )
            )

        hflip_p = float(train_cfg.get("hflip_p", 0.0))
        if hflip_p > 0:
            aug.append(T.RandomHorizontalFlip(p=hflip_p))

        pipeline = T.Compose(aug + base)
    else:
        pipeline = T.Compose(base)

    if normalize:
        pipeline = T.Compose([pipeline, T.Normalize(mean=mean, std=std)])
    return pipeline
