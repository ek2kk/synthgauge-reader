from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _required_files_from_cfg(paths_cfg: Dict[str, Any]) -> Tuple[str, str]:
    return (
        str(paths_cfg["train_inst_coco"]),
        str(paths_cfg["val_inst_coco"]),
    )


def _is_dataset_dir(path: Path, required_files: Tuple[str, str]) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return all((path / name).exists() for name in required_files)


def _dataset_sort_key(path: Path) -> Tuple[int, ...]:
    m = re.search(r"(\d+(?:\.\d+)*)", path.name)
    if not m:
        return (-1,)
    return tuple(int(p) for p in m.group(1).split("."))


def _resolve_raw_base(cfg: Dict[str, Any], raw_root_arg: str | None) -> Path:
    if raw_root_arg:
        return Path(raw_root_arg).resolve()

    cfg_raw = Path(str(cfg["paths"].get("raw_ds_path", ""))).resolve()
    if cfg_raw.exists():
        return cfg_raw

    fallback = Path("data/raw").resolve()
    if fallback.exists():
        return fallback

    return cfg_raw


def _discover_dataset_dirs(raw_base: Path, required_files: Tuple[str, str]) -> List[Path]:
    if _is_dataset_dir(raw_base, required_files):
        return [raw_base]

    if not raw_base.exists():
        raise FileNotFoundError(
            f"Raw dataset path not found: {raw_base}\n"
            "Hint: set paths.raw_ds_path to an existing dataset directory "
            "or pass --raw-root data/raw"
        )

    candidates: List[Path] = []
    for child in raw_base.iterdir():
        if _is_dataset_dir(child, required_files):
            candidates.append(child.resolve())

    candidates.sort(key=_dataset_sort_key)
    return candidates


def _select_dataset_dirs(candidates: List[Path], dataset_arg: str) -> List[Path]:
    if not candidates:
        raise FileNotFoundError(
            "No dataset directories with required COCO files were found."
        )

    mode = dataset_arg.strip()
    if mode == "all":
        return candidates
    if mode == "auto":
        return [candidates[-1]]

    selected = [d for d in candidates if d.name == mode]
    if not selected:
        available = ", ".join(d.name for d in candidates)
        raise ValueError(
            f"Dataset '{mode}' not found. Available: {available}. "
            "Use --dataset all|auto|<folder_name>."
        )
    return selected


def build_pairs_from_coco(
    coco: Dict[str, Any],
    images_root: Path,
    category_name: str,
    value_key: str,
) -> List[Tuple[str, float]]:
    """
    Делает пары (image_path, value) из Endava COCO:
    - image_path: images_root / images[*].file_name
    - value: берётся из annotations[*][value_key] для annotations[*].category_name == category_name
    """
    # image_id -> file_name
    id_to_file: Dict[int, str] = {}
    for img in coco.get("images", []):
        img_id = img.get("id")
        fn = img.get("file_name")
        if isinstance(img_id, int) and isinstance(fn, str) and fn:
            id_to_file[img_id] = fn

    # collect values per image_id from target category annotations
    values_by_img: Dict[int, List[float]] = {}
    for ann in coco.get("annotations", []):
        if ann.get("category_name") != category_name:
            continue

        img_id = ann.get("image_id")
        v = ann.get(value_key)

        if isinstance(img_id, int) and isinstance(v, (int, float)):
            values_by_img.setdefault(img_id, []).append(float(v))

    pairs: List[Tuple[str, float]] = []
    missing_target = 0
    missing_file = 0
    ambiguous = 0

    for img_id, fn in id_to_file.items():
        vals = values_by_img.get(img_id)
        if not vals:
            missing_target += 1
            continue

        v0 = vals[0]
        if any(abs(v - v0) > 1e-6 for v in vals[1:]):
            ambiguous += 1  # берём первый

        img_path = (images_root / fn).resolve()
        if not img_path.exists():
            missing_file += 1
            continue

        pairs.append((str(img_path), float(v0)))

    if missing_target:
        print(
            f"[WARN] Missing target for {missing_target} images (no '{category_name}' with '{value_key}')."
        )
    if missing_file:
        print(
            f"[WARN] {missing_file} images referenced in COCO were not found under images_root."
        )
        print(f"       images_root={images_root}")
    if ambiguous:
        print(f"[WARN] Ambiguous target values for {ambiguous} images (took first).")

    return pairs


def write_jsonl(pairs: List[Tuple[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for img_path, value in pairs:
            f.write(
                json.dumps({"image_path": img_path, "value": value}, ensure_ascii=False)
                + "\n"
            )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build regression JSONL index from one or more COCO dataset folders."
    )
    ap.add_argument("--config", type=str, default="configs/config_regression.yaml")
    ap.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help="Path to dataset folder OR parent folder containing multiple dataset folders.",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset selection mode: all | auto | <folder_name>.",
    )
    ap.add_argument("--out-train", type=str, default=None)
    ap.add_argument("--out-val", type=str, default=None)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    paths_cfg = cfg["paths"]

    required_files = _required_files_from_cfg(paths_cfg)
    raw_base = _resolve_raw_base(cfg, args.raw_root)
    candidates = _discover_dataset_dirs(raw_base, required_files)
    selected_dirs = _select_dataset_dirs(candidates, args.dataset)

    train_out = (
        Path(args.out_train).resolve()
        if args.out_train
        else Path(paths_cfg["train_reg_output_json"]).resolve()
    )
    val_out = (
        Path(args.out_val).resolve()
        if args.out_val
        else Path(paths_cfg["val_reg_output_json"]).resolve()
    )

    category_name = cfg["regression_target"]["category_name"]
    value_key = cfg["regression_target"]["value_key"]

    print(f"[INFO] raw_base:     {raw_base}")
    print(f"[INFO] selected ds: {', '.join(d.name for d in selected_dirs)}")
    print(
        f"[INFO] target:       category_name='{category_name}', value_key='{value_key}'"
    )
    print(f"[INFO] out_train:    {train_out}")
    print(f"[INFO] out_val:      {val_out}")

    all_train_pairs: List[Tuple[str, float]] = []
    all_val_pairs: List[Tuple[str, float]] = []

    for ds_dir in selected_dirs:
        images_root = ds_dir
        train_coco_path = ds_dir / paths_cfg["train_inst_coco"]
        val_coco_path = ds_dir / paths_cfg["val_inst_coco"]

        if not train_coco_path.exists():
            raise FileNotFoundError(f"train COCO not found: {train_coco_path}")
        if not val_coco_path.exists():
            raise FileNotFoundError(f"val COCO not found: {val_coco_path}")

        print(f"[INFO] processing:  {ds_dir}")
        print(f"[INFO] train_coco:   {train_coco_path}")
        print(f"[INFO] val_coco:     {val_coco_path}")

        train_coco = _read_json(train_coco_path)
        train_pairs = build_pairs_from_coco(
            train_coco, images_root, category_name, value_key
        )
        all_train_pairs.extend(train_pairs)
        print(f"[OK] train pairs from {ds_dir.name}: {len(train_pairs)}")

        val_coco = _read_json(val_coco_path)
        val_pairs = build_pairs_from_coco(
            val_coco, images_root, category_name, value_key
        )
        all_val_pairs.extend(val_pairs)
        print(f"[OK] val pairs from {ds_dir.name}:   {len(val_pairs)}")

    write_jsonl(all_train_pairs, train_out)
    write_jsonl(all_val_pairs, val_out)
    print(f"[OK] wrote {len(all_train_pairs)} samples -> {train_out}")
    print(f"[OK] wrote {len(all_val_pairs)} samples -> {val_out}")


if __name__ == "__main__":
    main()
