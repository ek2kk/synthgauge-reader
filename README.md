# Analog Gauge Reading Pipeline

Gauge reading pipeline on `synthetic-analog-gauges`.

## Run From Project Root

Always run scripts from the repository root via `uv run`.

## Tasks

- Detection: YOLOv8n (`configs/config_detection.yaml`)
- Keypoint detection: YOLO11s-pose (`configs/config_keypoints.yaml`)
- Regression: ResNet-18 (`configs/config_regression.yaml`)

## Metrics

- Detection: `precision`, `recall`, `mAP@0.5`, `mAP@0.5:0.95` (saved to `data/processed/detection_metrics.json`)
- Keypoint detection: `pose_precision`, `pose_recall`, `pose_mAP@0.5`, `pose_mAP@0.5:0.95`, `PCK@0.05`, `PCK@0.10` (saved to `data/processed/keypoints_metrics.json`)
- Regression: `mae`, `drr@0.02` (Dial Recognition Rate), plus `rmse` and `r2` (logged during validation)

## 1) Prepare Datasets

First download the source dataset from Hugging Face using your token:

```bash
uv run ./data/download_hf_dataset.py --token <your_hf_token>
```

By default, the dataset will be saved to `data/raw/synthetic-analog-gauges`.

Regression index (JSONL):

```bash
uv run ./data/build_regression_from_coco.py --raw-root data/raw/synthetic-analog-gauges --category-name gauge --value-key reading_normalized
```

Detection labels/data yaml (YOLO bbox):

```bash
uv run ./data/build_det_yolo_from_coco.py --config configs/config_detection.yaml
```

Keypoint labels/data yaml (YOLO pose):

```bash
uv run ./data/build_kp_yolo_pose_from_coco.py --config configs/config_keypoints.yaml
```

Detection and keypoint pipelines use separate YOLO dataset roots (`paths.yolo_dataset_root`), so labels do not overwrite each other.

## 2) Train Models

YOLOv8n detection:

```bash
uv run ./training/train_detection_yolo.py --config configs/config_detection.yaml
```

YOLO11s-pose keypoint detection:

```bash
uv run ./training/train_keypoints_yolo_pose.py --config configs/config_keypoints.yaml
```

Regression (ResNet-18, `reading_normalized`):

```bash
uv run ./training/train_regression.py --config configs/config_regression.yaml
```

Unified entrypoint:

```bash
uv run ./training/train.py --task detection
uv run ./training/train.py --task keypoints
uv run ./training/train.py --task regression
```

Weights are saved to:

```bash
models/weights/{dataset_name}/{det|kp|reg}_{model_name}
```

Example:

```bash
models/weights/synthetic-analog-gauges/det_yolov8n
models/weights/synthetic-analog-gauges/kp_yolo11s-pose
models/weights/synthetic-analog-gauges/reg_resnet18
```

## 3) Evaluate Models

Detection metrics:

```bash
uv run ./training/eval_detection_yolo.py --config configs/config_detection.yaml --split test
```

Keypoint metrics:

```bash
uv run ./training/eval_keypoints_yolo_pose.py --config configs/config_keypoints.yaml --split test
```

Regression metrics:

```bash
uv run ./training/eval_regression.py --config configs/config_regression.yaml --split test
```

Unified entrypoint:

```bash
uv run ./training/eval.py --task detection --split test
uv run ./training/eval.py --task keypoints --split test
uv run ./training/eval.py --task regression --split test
```

Saved metric files:

```bash
data/processed/detection_metrics.json
data/processed/keypoints_metrics.json
data/processed/regression_metrics.json
```

## 4) Inference Visualizations

Detection (bbox pred vs gt):

```bash
uv run ./inference/visualize_detection_predictions.py --config configs/config_detection.yaml --split val --num-samples 10 --save data/processed/det_pred_samples.png
```

Keypoints (pred vs gt):

```bash
uv run ./inference/visualize_keypoints_predictions.py --config configs/config_keypoints.yaml --split val --num-samples 10 --save data/processed/kp_pred_samples.png
```

Regression (pred value vs gt):

```bash
uv run ./inference/visualize_regression_predictions.py --config configs/config_regression.yaml --split val --num-samples 12 --save data/processed/reg_pred_samples.png
```

## 5) Unified Predict

Single image:

```bash
uv run ./inference/predict.py --task detection --image path/to/image.jpg
uv run ./inference/predict.py --task keypoints --image path/to/image.jpg
uv run ./inference/predict.py --task regression --image path/to/image.jpg
```

Dataset sample:

```bash
uv run ./inference/predict.py --task detection --split val --num-samples 10 --out data/processed/pred_det_val.json
uv run ./inference/predict.py --task keypoints --split val --num-samples 10 --out data/processed/pred_kp_val.json
uv run ./inference/predict.py --task regression --split val --num-samples 10 --out data/processed/pred_reg_val.json
```
