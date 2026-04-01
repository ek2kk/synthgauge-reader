from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RegressionLayout:
    train_coco: Path
    val_coco: Path
    test_coco: Optional[Path]
    images_root: Path


@dataclass(frozen=True)
class CropConfig:
    enabled: bool
    pad_ratio: float
    out_root: Path


def _required_files_from_cfg(paths_cfg: Dict[str, Any]) -> Tuple[str, str]:
    return str(paths_cfg["train_inst_coco"]), str(paths_cfg["val_inst_coco"])


def _optional_test_file_from_cfg(paths_cfg: Dict[str, Any]) -> Optional[str]:
    rel = paths_cfg.get("test_inst_coco")
    return str(rel) if rel is not None else None


def _infer_test_relpaths(train_rel: str, val_rel: str) -> List[str]:
    candidates: List[str] = []
    for src, old in [(train_rel, "train"), (val_rel, "val")]:
        if old in src:
            repl = src.replace(old, "test", 1)
            if repl not in candidates:
                candidates.append(repl)
    return candidates


def _resolve_raw_root(cfg: Dict[str, Any], raw_root_arg: Optional[str]) -> Path:
    if raw_root_arg:
        return Path(raw_root_arg).resolve()
    return Path(str(cfg["paths"].get("raw_ds_path", ""))).resolve()


def _resolve_regression_layout(
    dataset_root: Path,
    required_files: Tuple[str, str],
    optional_test_file: Optional[str],
) -> RegressionLayout:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    cfg_train_rel, cfg_val_rel = required_files
    cfg_images_root = dataset_root
    if cfg_train_rel.replace("\\", "/").lower().startswith("annotations/"):
        candidate_images_root = dataset_root / "images"
        if candidate_images_root.exists():
            cfg_images_root = candidate_images_root

    cfg_test_path: Optional[Path] = None
    if optional_test_file:
        candidate = dataset_root / optional_test_file
        if candidate.exists():
            cfg_test_path = candidate
    else:
        for rel in _infer_test_relpaths(cfg_train_rel, cfg_val_rel):
            candidate = dataset_root / rel
            if candidate.exists():
                cfg_test_path = candidate
                break

    cfg_layout = RegressionLayout(
        train_coco=dataset_root / cfg_train_rel,
        val_coco=dataset_root / cfg_val_rel,
        test_coco=cfg_test_path,
        images_root=cfg_images_root,
    )
    if cfg_layout.train_coco.exists() and cfg_layout.val_coco.exists():
        return cfg_layout

    # HuggingFace synthetic-analog-gauges layout fallback.
    hf_train = dataset_root / "annotations" / "instances_train.json"
    hf_val = dataset_root / "annotations" / "instances_val.json"
    hf_test = dataset_root / "annotations" / "instances_test.json"
    hf_layout = RegressionLayout(
        train_coco=hf_train,
        val_coco=hf_val,
        test_coco=hf_test if hf_test.exists() else None,
        images_root=dataset_root / "images",
    )
    if hf_layout.train_coco.exists() and hf_layout.val_coco.exists():
        return hf_layout

    raise FileNotFoundError(
        "Could not resolve regression COCO layout.\n"
        f"Expected either configured files under {dataset_root}:\n"
        f"  - {cfg_layout.train_coco}\n"
        f"  - {cfg_layout.val_coco}\n"
        "or HuggingFace synthetic layout:\n"
        f"  - {hf_train}\n"
        f"  - {hf_val}"
    )


def _extract_numeric_value(ann: Dict[str, Any], value_key: str) -> Optional[float]:
    direct = ann.get(value_key)
    if isinstance(direct, (int, float)):
        return float(direct)

    attrs = ann.get("attributes")
    if isinstance(attrs, dict):
        nested = attrs.get(value_key)
        if isinstance(nested, (int, float)):
            return float(nested)
    return None


def _find_category_id(coco: Dict[str, Any], category_name: str) -> Optional[int]:
    for cat in coco.get("categories", []):
        if cat.get("name") == category_name and isinstance(cat.get("id"), int):
            return int(cat["id"])
    return None


def _annotation_matches_category(
    ann: Dict[str, Any],
    category_name: str,
    category_id: Optional[int],
) -> bool:
    if ann.get("category_name") == category_name:
        return True
    if category_id is not None and ann.get("category_id") == category_id:
        return True
    return False


def _bbox_xywh_to_xyxy(bbox: List[float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox]
    return x, y, x + w, y + h


def _crop_box_from_bbox(
    bbox_xywh: List[float],
    img_w: int,
    img_h: int,
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = _bbox_xywh_to_xyxy(bbox_xywh)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(bw, bh) * (1.0 + 2.0 * max(0.0, pad_ratio))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    rx1 = int(max(0, min(img_w - 1, round(cx - side / 2.0))))
    ry1 = int(max(0, min(img_h - 1, round(cy - side / 2.0))))
    rx2 = int(max(rx1 + 1, min(img_w, round(cx + side / 2.0))))
    ry2 = int(max(ry1 + 1, min(img_h, round(cy + side / 2.0))))
    return rx1, ry1, rx2, ry2


def _safe_crop_save(
    src_image_path: Path,
    dst_image_path: Path,
    bbox_xywh: List[float],
    pad_ratio: float,
) -> Optional[Path]:
    if not src_image_path.exists():
        return None
    with Image.open(src_image_path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        if w <= 1 or h <= 1:
            return None
        x1, y1, x2, y2 = _crop_box_from_bbox(bbox_xywh, w, h, pad_ratio=pad_ratio)
        crop = rgb.crop((x1, y1, x2, y2))
        dst_image_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(dst_image_path)
    return dst_image_path.resolve()


def _default_output_paths(paths_cfg: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    train_out = Path(str(paths_cfg["train_reg_output_json"])).resolve()
    val_out = Path(str(paths_cfg["val_reg_output_json"])).resolve()
    test_rel = paths_cfg.get("test_reg_output_json")
    if test_rel is not None:
        test_out = Path(str(test_rel)).resolve()
    else:
        test_out = val_out.with_name(val_out.name.replace("val", "test", 1))
    return train_out, val_out, test_out


def _resolve_crop_cfg(cfg: Dict[str, Any], train_out: Path) -> CropConfig:
    crop_section = cfg.get("dial_crop", cfg.get("crop", {}))
    enabled = bool(crop_section.get("enabled", True))
    pad_ratio = float(crop_section.get("pad_ratio", 0.08))
    out_root_raw = crop_section.get("out_root")
    if out_root_raw:
        out_root = Path(str(out_root_raw)).resolve()
    else:
        out_root = (train_out.parent / "crops").resolve()
    return CropConfig(enabled=enabled, pad_ratio=pad_ratio, out_root=out_root)


def build_pairs_from_coco(
    coco: Dict[str, Any],
    images_root: Path,
    category_name: str,
    value_key: str,
    *,
    split_name: str,
    crop_cfg: CropConfig,
) -> List[Tuple[str, float]]:
    id_to_file: Dict[int, str] = {}
    for img in coco.get("images", []):
        img_id = img.get("id")
        fn = img.get("file_name")
        if isinstance(img_id, int) and isinstance(fn, str) and fn:
            id_to_file[img_id] = fn

    target_category_id = _find_category_id(coco, category_name)

    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        if not _annotation_matches_category(ann, category_name, target_category_id):
            continue
        img_id = ann.get("image_id")
        value = _extract_numeric_value(ann, value_key)
        bbox = ann.get("bbox")
        if not isinstance(img_id, int) or value is None:
            continue
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        bw = float(bbox[2])
        bh = float(bbox[3])
        if bw <= 0 or bh <= 0:
            continue
        anns_by_img.setdefault(img_id, []).append(
            {
                "value": float(value),
                "bbox": [float(bbox[0]), float(bbox[1]), bw, bh],
                "area": float(ann.get("area", bw * bh)),
            }
        )

    pairs: List[Tuple[str, float]] = []
    missing_target = 0
    missing_file = 0
    cropped = 0

    for img_id, fn in id_to_file.items():
        anns = anns_by_img.get(img_id)
        if not anns:
            missing_target += 1
            continue

        ann = max(anns, key=lambda a: float(a.get("area", 0.0)))
        value = float(ann["value"])
        bbox = [float(v) for v in ann["bbox"]]

        rel = Path(fn)
        img_path = rel if rel.is_absolute() else (images_root / rel).resolve()
        if not img_path.exists():
            missing_file += 1
            continue

        if crop_cfg.enabled:
            rel_crop = Path(img_path.name) if rel.is_absolute() else rel
            if rel_crop.parts and rel_crop.parts[0] in {"train", "val", "test"}:
                rel_crop = Path(*rel_crop.parts[1:])
            rel_crop = rel_crop.with_suffix(rel_crop.suffix or ".jpg")
            crop_path = (crop_cfg.out_root / split_name / rel_crop).resolve()
            saved = _safe_crop_save(
                src_image_path=img_path,
                dst_image_path=crop_path,
                bbox_xywh=bbox,
                pad_ratio=crop_cfg.pad_ratio,
            )
            if saved is None:
                missing_file += 1
                continue
            pairs.append((str(saved), value))
            cropped += 1
        else:
            pairs.append((str(img_path), value))

    if missing_target:
        print(
            f"[WARN] Missing target for {missing_target} images (no '{category_name}' with '{value_key}')."
        )
    if missing_file:
        print("[WARN] Some images referenced in COCO were not found under images_root.")
        print(f"       images_root={images_root}")
    if crop_cfg.enabled:
        print(f"[INFO] Cropped samples for split='{split_name}': {cropped}")

    return pairs


def write_jsonl(pairs: List[Tuple[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for img_path, value in pairs:
            f.write(json.dumps({"image_path": img_path, "value": value}, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build regression JSONL index from synthetic-analog-gauges COCO annotations."
    )
    ap.add_argument("--config", type=str, default="configs/config_regression.yaml")
    ap.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help="Path to synthetic-analog-gauges dataset root.",
    )
    ap.add_argument(
        "--category-name",
        type=str,
        default=None,
        help="Override regression_target.category_name from config.",
    )
    ap.add_argument(
        "--value-key",
        type=str,
        default=None,
        help="Override regression_target.value_key from config.",
    )
    ap.add_argument("--out-train", type=str, default=None)
    ap.add_argument("--out-val", type=str, default=None)
    ap.add_argument("--out-test", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = cfg["paths"]

    required_files = _required_files_from_cfg(paths_cfg)
    optional_test_file = _optional_test_file_from_cfg(paths_cfg)
    dataset_root = _resolve_raw_root(cfg, args.raw_root)
    layout = _resolve_regression_layout(dataset_root, required_files, optional_test_file)

    if args.out_train:
        train_out = Path(args.out_train).resolve()
    else:
        train_out, _, _ = _default_output_paths(paths_cfg)
    if args.out_val:
        val_out = Path(args.out_val).resolve()
    else:
        _, val_out, _ = _default_output_paths(paths_cfg)
    if args.out_test:
        test_out = Path(args.out_test).resolve()
    else:
        _, _, test_out = _default_output_paths(paths_cfg)

    crop_cfg = _resolve_crop_cfg(cfg, train_out)
    if crop_cfg.enabled and crop_cfg.out_root.exists():
        shutil.rmtree(crop_cfg.out_root)
    if crop_cfg.enabled:
        crop_cfg.out_root.mkdir(parents=True, exist_ok=True)

    category_name = args.category_name or cfg["regression_target"]["category_name"]
    value_key = args.value_key or cfg["regression_target"]["value_key"]

    print(f"[INFO] dataset_root: {dataset_root}")
    print(f"[INFO] train_coco:   {layout.train_coco}")
    print(f"[INFO] val_coco:     {layout.val_coco}")
    if layout.test_coco is not None:
        print(f"[INFO] test_coco:    {layout.test_coco}")
    print(f"[INFO] images_root:  {layout.images_root}")
    print(f"[INFO] target:       category_name='{category_name}', value_key='{value_key}'")
    print(f"[INFO] dial_crop:    enabled={crop_cfg.enabled} pad_ratio={crop_cfg.pad_ratio}")
    if crop_cfg.enabled:
        print(f"[INFO] crops_root:   {crop_cfg.out_root}")
    print(f"[INFO] out_train:    {train_out}")
    print(f"[INFO] out_val:      {val_out}")
    print(f"[INFO] out_test:     {test_out}")

    train_coco = _read_json(layout.train_coco)
    train_pairs = build_pairs_from_coco(
        train_coco,
        layout.images_root,
        category_name,
        value_key,
        split_name="train",
        crop_cfg=crop_cfg,
    )
    val_coco = _read_json(layout.val_coco)
    val_pairs = build_pairs_from_coco(
        val_coco,
        layout.images_root,
        category_name,
        value_key,
        split_name="val",
        crop_cfg=crop_cfg,
    )

    write_jsonl(train_pairs, train_out)
    write_jsonl(val_pairs, val_out)
    print(f"[OK] wrote {len(train_pairs)} samples -> {train_out}")
    print(f"[OK] wrote {len(val_pairs)} samples -> {val_out}")

    if layout.test_coco is not None:
        test_coco = _read_json(layout.test_coco)
        test_pairs = build_pairs_from_coco(
            test_coco,
            layout.images_root,
            category_name,
            value_key,
            split_name="test",
            crop_cfg=crop_cfg,
        )
        write_jsonl(test_pairs, test_out)
        print(f"[OK] wrote {len(test_pairs)} samples -> {test_out}")
    else:
        print("[INFO] test output not written (no test COCO split found).")


if __name__ == "__main__":
    main()
