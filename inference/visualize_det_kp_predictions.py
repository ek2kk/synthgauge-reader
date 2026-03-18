from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from torchvision import transforms as T
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Visualize det+kps model predictions on random samples."
    )
    ap.add_argument("--config", type=str, default="configs/config_det_kp.yaml")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--score-thr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="from-config")
    ap.add_argument("--save", type=str, default=None)
    return ap.parse_args()


def _resolve_device(requested: str, cfg_device: str) -> torch.device:
    mode = cfg_device if requested == "from-config" else requested
    mode = str(mode).lower()

    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if mode == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    raise ValueError(f"Unknown device mode: {mode}")


def _log_device_info(device: torch.device) -> None:
    if device.type != "cuda":
        print(f"[INFO] device={device}")
        return

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    total_mem_gb = props.total_memory / (1024**3)
    print(
        "[INFO] cuda_device="
        f"index={idx} "
        f"name={torch.cuda.get_device_name(idx)} "
        f"capability={props.major}.{props.minor} "
        f"vram={total_mem_gb:.2f}GB"
    )


def _build_model(num_classes: int, num_keypoints: int) -> torch.nn.Module:
    model = keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_kp = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
        in_features_kp, num_keypoints
    )
    return model


def _resolve_weights_path(cfg: Dict[str, Any], weights_arg: Optional[str]) -> Path:
    if weights_arg:
        p = Path(weights_arg).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Weights not found: {p}")
        return p

    weights_dir = Path(
        cfg.get("paths", {}).get("weights_dir_det_kp", "models/weights/det_kp")
    ).resolve()
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    raise FileNotFoundError(
        f"No weights found in {weights_dir}. Expected best.pt or last.pt."
    )


def _load_model(
    cfg: Dict[str, Any],
    weights_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    num_keypoints = int(cfg.get("keypoints_target", {}).get("num_keypoints", 4))
    num_obj_classes = int(cfg.get("detector_target", {}).get("num_classes", 1))
    num_classes = num_obj_classes + 1

    model = _build_model(num_classes=num_classes, num_keypoints=num_keypoints)
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    if not items:
        raise RuntimeError(f"Index file is empty: {path}")
    return items


def _resolve_index_path(cfg: Dict[str, Any], split: str) -> Path:
    key = "train_det_kp_output_json" if split == "train" else "val_det_kp_output_json"
    p = Path(cfg["paths"][key]).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Index not found: {p}")
    return p


def _prepare_input_and_vis(
    img: Image.Image,
    img_size: int,
    mean: List[float],
    std: List[float],
) -> Tuple[torch.Tensor, np.ndarray]:
    model_tf = T.Compose(
        [
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    vis_img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    vis_np = np.asarray(vis_img, dtype=np.uint8)
    x = model_tf(img)
    return x, vis_np


def _scale_gt_to_vis(
    rec: Dict[str, Any], img_w: int, img_h: int, vis_size: int
) -> Tuple[List[float], List[List[float]]]:
    sx = vis_size / max(1.0, float(img_w))
    sy = vis_size / max(1.0, float(img_h))

    b = rec["bbox"]
    gt_box = [float(b[0]) * sx, float(b[1]) * sy, float(b[2]) * sx, float(b[3]) * sy]

    gt_kps: List[List[float]] = []
    for kp in rec["keypoints"]:
        gt_kps.append([float(kp[0]) * sx, float(kp[1]) * sy, float(kp[2])])
    return gt_box, gt_kps


def _select_prediction(
    out: Dict[str, torch.Tensor], score_thr: float
) -> Tuple[Optional[List[float]], Optional[np.ndarray], Optional[float]]:
    boxes = out.get("boxes")
    scores = out.get("scores")
    keypoints = out.get("keypoints")
    if boxes is None or keypoints is None or boxes.numel() == 0:
        return None, None, None

    if scores is None or scores.numel() == 0:
        idx = 0
    else:
        keep = torch.nonzero(scores >= float(score_thr), as_tuple=False).view(-1)
        if keep.numel() == 0:
            return None, None, None
        idx = int(keep[torch.argmax(scores[keep])].item())

    box = boxes[idx].detach().cpu().float().tolist()
    kps = keypoints[idx].detach().cpu().float().numpy()
    score = float(scores[idx].item()) if scores is not None and scores.numel() else None
    return box, kps, score


def _draw_keypoints(
    ax: plt.Axes, kps: List[List[float]] | np.ndarray, names: List[str], color: str
) -> None:
    for i, kp in enumerate(kps):
        x = float(kp[0])
        y = float(kp[1])
        v = float(kp[2]) if len(kp) > 2 else 2.0
        if v <= 0:
            continue
        ax.scatter([x], [y], s=20, c=color)
        if i < len(names):
            ax.text(x + 2, y + 2, names[i], color=color, fontsize=7)


def _draw_box(ax: plt.Axes, box: List[float], color: str, label: str) -> None:
    x1, y1, x2, y2 = box
    rect = Rectangle(
        (x1, y1),
        max(1.0, x2 - x1),
        max(1.0, y2 - y1),
        linewidth=2,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.text(x1, max(8.0, y1 - 2), label, color=color, fontsize=8)


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    cfg = load_config(args.config)
    tcfg = cfg.get("training_det_kp", {})
    score_thr = (
        float(args.score_thr)
        if args.score_thr is not None
        else float(tcfg.get("score_thr", 0.3))
    )

    device = _resolve_device(args.device, str(tcfg.get("device", "auto")))
    _log_device_info(device)

    index_path = _resolve_index_path(cfg, args.split)
    records = _read_jsonl(index_path)
    chosen = random.sample(records, k=min(args.num_samples, len(records)))

    weights_path = _resolve_weights_path(cfg, args.weights)
    model = _load_model(cfg, weights_path, device)

    tf_cfg = cfg.get("transforms_reg", {})
    mean = list(tf_cfg.get("mean", [0.485, 0.456, 0.406]))
    std = list(tf_cfg.get("std", [0.229, 0.224, 0.225]))
    img_size = int(cfg.get("transforms_det_kp", {}).get("img_size", 256))
    kp_names = list(cfg.get("keypoints_target", {}).get("names", []))

    cols = 5
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 4.0))
    axes_flat = np.array(axes, ndmin=1).reshape(-1)

    with torch.no_grad():
        for ax, rec in zip(axes_flat, chosen):
            img_path = Path(rec["image_path"]).resolve()
            with Image.open(img_path) as im:
                img = im.convert("RGB")
            w, h = img.size

            x, vis_np = _prepare_input_and_vis(img, img_size, mean, std)
            out = model([x.to(device, non_blocking=True)])[0]
            out_cpu = {k: v.detach().cpu() for k, v in out.items()}

            gt_box, gt_kps = _scale_gt_to_vis(rec, w, h, img_size)
            pred_box, pred_kps, pred_score = _select_prediction(out_cpu, score_thr)

            ax.imshow(vis_np)
            _draw_box(ax, gt_box, color="lime", label="GT")
            _draw_keypoints(ax, gt_kps, kp_names, color="lime")

            if pred_box is not None and pred_kps is not None:
                score_text = f"PRED s={pred_score:.2f}" if pred_score is not None else "PRED"
                _draw_box(ax, pred_box, color="red", label=score_text)
                _draw_keypoints(ax, pred_kps, kp_names, color="red")
            else:
                ax.text(4, 14, "PRED: none", color="red", fontsize=9)

            ax.set_title(f"id={rec.get('image_id', '?')} | {img_path.name}", fontsize=9)
            ax.axis("off")

    for ax in axes_flat[len(chosen) :]:
        ax.axis("off")

    fig.suptitle(
        f"Det+KPs predictions ({args.split}, n={len(chosen)}, thr={score_thr:.2f})",
        fontsize=14,
    )
    plt.tight_layout()

    if args.save:
        out_path = Path(args.save).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Saved figure: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
