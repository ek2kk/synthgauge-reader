# SynthGauge: A Diverse Synthetic Dataset for Analog Gauge Reading with Domain Randomization

---

## Abstract

Automatic reading of analog gauges is essential for industrial monitoring, yet training robust computer vision models remains difficult because annotated data are scarce and existing public resources target only a subset of the problem. We present **SynthGauge**, a synthetic dataset of 9,000 photorealistic images of round analog gauges covering 10+ instrument types and 42 scale configurations. Generated procedurally in Blender using Domain Randomization across 97 parameters, the dataset spans diverse lighting conditions, camera viewpoints, materials, and weathering effects representative of industrial environments. Each image includes a COCO-format bounding box, four reading-specific keypoints (`center`, `needle_tip`, `scale_start`, `scale_end`), a normalized reading target, and per-image DR metadata, enabling detection, keypoint localization, and direct reading regression. We also describe reproducible baseline benchmarks and a public release package for the dataset, generation pipeline, and baseline training code.

---

## 1. Introduction

Analog gauges - pressure meters, voltmeters, ammeters, thermometers, and tachometers - remain prevalent in industrial facilities worldwide. Manual reading of these instruments is labor-intensive, error-prone, and potentially hazardous in hostile environments. Automated gauge reading through computer vision can enable continuous monitoring, reduce human exposure to dangerous conditions, and eliminate transcription errors.

Deep learning has achieved remarkable success in related tasks such as optical character recognition and object detection. However, applying these methods to analog gauge reading still faces a fundamental obstacle: the lack of sufficiently diverse annotated datasets. Existing public resources either focus on utility dial meters, a narrow family of circular gauges, or only a single downstream task. Collecting real images requires physical access to operational equipment, while annotation demands expertise to correctly identify pointer geometry and scale parameters.

Synthetic data generation offers a solution to this scarcity. Procedural generation with Domain Randomization (DR) can produce unlimited labeled samples at negligible marginal cost. By randomizing visual parameters during rendering - lighting, textures, camera pose, materials - the resulting models learn features robust to any specific appearance, including real-world conditions.

In this work, we introduce **SynthGauge**, a procedurally generated dataset designed for gauge detection, keypoint localization, and reading estimation. Our contributions are:

1. **A diverse synthetic dataset** of 9,000 photorealistic images covering 42 scale configurations across 10+ instrument types (manometers, voltmeters, ammeters, thermometers, tachometers, speedometers), with unified annotations for detection, keypoints, and normalized needle readings.

2. **Reading-oriented annotation schema** comprising COCO-format bounding boxes, ordered reading-specific keypoints, normalized readings, and per-image DR metadata, enabling both geometry-aware pipelines and direct end-to-end regression.

3. **Comprehensive Domain Randomization** spanning 97 parameters across seven categories: lighting (including industrial IES profiles), camera pose, PBR materials, gauge geometry, procedural weathering, post-processing effects, and position variation.

4. **A reproducible generation and benchmarking pipeline** based on Blender 4.2 LTS and released training/evaluation configurations for gauge detection, keypoint detection, and reading estimation on official SynthGauge splits.

---

**[FIGURE 1 - TEASER IMAGE HERE]**

**Figure 1.** Sample images from SynthGauge showing diversity of instrument types and imaging conditions. From left to right: manometer with color zones (0–150 bar), tachometer (0–8000 RPM), compound pressure gauge (−15 to +60 PSI), thermometer (0–250°C), tachometer with safety zones (0–6000 RPM). All images generated procedurally with Domain Randomization.

---

## 2. Related Work

### 2.1 Gauge Reading Datasets

Public datasets for automatic gauge reading remain limited and heterogeneous in scope. UFPR-ADMR-v2 [1] contains 5,000 field images of electricity dial meters (22,410 annotated dials) collected in unconstrained conditions, with counter corners, dial boxes, and reading annotations. It is an important benchmark for utility metering, but it focuses on multi-dial counters rather than round industrial gauges.

The closest prior work to our setting is Howells et al. [2], who released two datasets for circular single-pointer gauges. Their SyntheticGauges split contains 10,000 training and 1,000 test renderings at 1024x1024 resolution, accompanied by COCO-format JSON labels for the gauge bounding box, perspective points, scale minimum and maximum, and pointer center and tip. Their RealGauges benchmark is built from six real gauges and evaluates gauge detection, pose recovery, and reading.

Leon-Alcazar et al. [3] further showed that synthetic data can drive competitive real-world gauge reading performance. Their WACV 2024 pipeline uses a Blender-generated synthetic dataset of 12,000 images and is validated on two real datasets: 4,813 landmark-labeled images and 59 images with semantic segmentation masks. This work demonstrates successful sim-to-real transfer, but it is centered on a task-specific two-stage reading pipeline rather than a reusable multi-task benchmark across multiple industrial instrument families.

Community datasets also exist. The Roboflow Pointer instrument dataset [4] provides about 1.5k real images for one-class object detection. It is useful as a lightweight real-image detection benchmark, but it does not provide gauge readings or reading-specific keypoints.

Table 1 compares these resources. Relative to prior datasets, SynthGauge emphasizes industrial diversity and unified multi-task annotations rather than raw image count alone.

**Table 1.** Comparison with representative gauge datasets and benchmarks.

| Dataset | Images | Domain | Gauge scope | Detection | Geometry labels | Reading |
|---------|--------|--------|-------------|-----------|-----------------|---------|
| UFPR-ADMR-v2 [1] | 5,000 | Real | Utility dial meters | Dial boxes | No | Yes |
| SyntheticGauges [2] | 11,000 | Synthetic | Circular single-pointer gauges | Yes | Yes | Yes |
| RealGauges [2] | 6 physical gauges / task-specific benchmark | Real | Circular single-pointer gauges | Yes | Yes | Yes |
| Roboflow Pointer instrument [4] | ~1,500 | Real | Mixed pointer instruments | Yes | No | No |
| **SynthGauge (Ours)** | **9,000** | **Synthetic** | **Industrial round gauges, 10+ types** | **Yes** | **Yes** | **Yes** |

### 2.2 Synthetic Data and Domain Randomization

Domain Randomization was popularized by Tobin et al. [5], who showed that aggressively varying textures, lighting, camera parameters, and distractors in simulation can enable detectors trained purely on synthetic data to transfer to real images. Subsequent industrial robotics work by Horvath et al. [6] demonstrated that physically plausible sim-to-real randomization can substantially reduce the annotation burden for object detection in manufacturing settings. BlenderProc [7] generalized this idea into a reusable, modular pipeline for procedural scene generation, photorealistic rendering, and automatic annotation across tasks such as segmentation, depth, optical flow, and pose estimation.

Our work follows this line but specializes it to analog instrumentation. Instead of generic object-centric annotations, we model gauge geometry, scale semantics, weathering, and reading-specific keypoints required for downstream gauge transcription.

### 2.3 Gauge Reading Methods

Classical analog gauge readers typically decompose the problem into gauge localization, geometric rectification, pointer extraction, and scale interpretation. Representative recent systems still rely heavily on hand-crafted geometry, including scale-mark reasoning under perspective and illumination distortions [8] and hybrid deep-learning plus Hough-transform pipelines for pointer extraction and OCR-assisted reading [9].

Recent learning-based systems increasingly regress structured intermediate representations rather than only the final scalar value. Howells et al. [2] learn pointer layout and gauge-face homography from synthetic data for mobile inference on circular gauges. Leon-Alcazar et al. [3] train a two-stage pipeline that combines semantic segmentation with angular landmark regression and validate it on more than 4,800 real images. Reitsma et al. [10] propose an interpretable pipeline for reading analog gauges in the wild that estimates scale marks, units, and readings without prior knowledge of the gauge range, reporting relative reading errors below 2% in real-world robotic inspection settings.

SynthGauge is designed to support both families of methods. Its annotations can be used for detection-based or geometry-aware pipelines, while the normalized reading target also supports direct end-to-end regression from the image to the displayed value.

---

## 3. Dataset

SynthGauge supports three computer vision tasks: (1) gauge face detection via bounding box localization, (2) keypoint detection (`center`, `needle_tip`, `scale_start`, `scale_end`), and (3) needle reading estimation via regression. We describe the generation pipeline, randomization parameters, and dataset statistics.

### 3.1 Generation Pipeline

Figure 2 illustrates the generation architecture comprising six components:

**Configuration.** YAML files define 97 DR parameter ranges and 42 scale configurations covering manometers (bar, kPa, PSI), voltmeters, ammeters, thermometers (°C), tachometers (RPM), speedometers (km/h), and specialized instruments (vacuum, compound, differential pressure gauges).

**Domain Randomization.** For each image, parameters are sampled from configured distributions across seven groups: lighting, camera, materials, geometry, weathering, post-processing, and position.

**3D Gauge Model.** Parametric geometry generated in Blender includes housing, bezel, glass cover, dial face with procedural scale markings, and needle. Dial textures support configurable tick marks, numeric labels, color zones, and brand-agnostic manufacturer-style markings.

**Scene Setup.** Camera pose, lighting sources (point, sun, area, IES profiles), and HDRI environment maps selected from 484 industrial backgrounds (Poly Haven, CC0).

**Rendering.** Blender Cycles path tracer at 128 samples produces photorealistic output at 640×640 pixels.

**Post-processing.** Camera simulation effects including film grain, Gaussian blur, vignette, chromatic aberration, barrel distortion, bloom, and JPEG compression.

**Annotation.** COCO JSON with bounding boxes and custom fields: `reading_normalized` ∈ [0,1], `needle_angle_cw_deg`, scale parameters, and complete DR metadata per image.

---

**[FIGURE 2 - PIPELINE DIAGRAM HERE: pipeline.png]**

**Figure 2.** SynthGauge generation pipeline. Configuration files define 97 DR parameters and 42 scale configurations. Domain Randomization samples parameters for each image. The 3D gauge model and scene setup feed into Cycles rendering, followed by post-processing effects. COCO JSON annotations include bounding boxes, normalized readings, and full DR metadata.

---

### 3.2 Domain Randomization Parameters

We randomize 97 parameters across seven groups. Table 2 summarizes key parameters; complete specifications are available in the dataset documentation.

**Table 2.** Domain Randomization parameter groups.

| Group | Params | Key Ranges |
|-------|--------|------------|
| Lighting | 9 | 2–4 sources, 2700–6500K, 484 HDRIs, IES profiles (30%), dim mode (5%) |
| Camera | 6 | 35–85mm focal, 0.4–0.55m distance, ±15° tilt/azimuth |
| Materials | 18 | 11 housing materials, 4 needle shapes, 8 colors, glass types |
| Geometry | 6 | 42 scale configs, 0–270° needle angle, body shapes |
| Weathering | 4 | Edge wear, scratches, oil stains, glass smudges |
| Post-process | 12 | Noise, JPEG, blur, vignette, chromatic aberration, barrel distortion |
| Position | 2 | 30% off-center, up to 25% offset |

**Lighting** encompasses 2–4 sources with randomized type, intensity (1.1–1.25), and color temperature (2700–6500K). A dim lighting mode (5% of images) simulates poorly-lit industrial environments. Industrial IES luminaire profiles (30% of images) add realistic factory lighting patterns. HDRI environment maps provide contextual reflections and backgrounds.

**Materials** define visual appearance of gauge components. Housing materials include stainless steel variants, plastics, aluminum, brass, and rusty metal with procedural roughness. Glass covers are clean (70%) or scratched (30%). Needle shapes follow industrial standards: knife (70%), lollipop (15%), line (10%), spade (5%).

**Geometry** encompasses 42 scale configurations spanning pressure (bar, kPa, PSI), electrical (V, A), temperature (°C), and speed (RPM, km/h) measurements. Sweep angles range from 90° to 320°; most use 270° clockwise with counter-clockwise variants. Color zones (35%) and dual scales (15%) add visual diversity.

**Weathering** applies procedural aging via shader nodes: edge wear revealing bare metal, surface scratches, oil stains, and glass fingerprints simulate real industrial equipment in service.

**Post-processing** simulates camera characteristics: film grain (ISO 200–1600 equivalent), JPEG compression (quality 60–95), blur, vignette, chromatic aberration, and barrel distortion (40% of images).

---

**[FIGURE 3 - SAMPLE GRID HERE: figure2_sample_grid.jpg]**

**Figure 3.** Sample images from SynthGauge demonstrating diversity across instrument types (manometers, tachometers, thermometers, ammeters), scale configurations, materials (stainless steel, plastic, brass), dial colors, needle shapes, color zones, backgrounds (industrial, warehouse, outdoor), and imaging conditions (lighting, angles, weathering effects).

---

### 3.3 Annotation Format

Annotations follow COCO JSON format with custom extensions:

```json
{
  "annotations": [{
    "bbox": [x, y, width, height],
    "keypoints": [
      center_x, center_y, center_v,
      needle_tip_x, needle_tip_y, needle_tip_v,
      scale_start_x, scale_start_y, scale_start_v,
      scale_end_x, scale_end_y, scale_end_v
    ],
    "reading_normalized": 0.472,
    "attributes": {
      "needle_angle_cw_deg": 127.5,
      "scale_min": 0, "scale_max": 10,
      "scale_unit": "bar"
    }
  }]
}
```

Keypoints are ordered as `center -> needle_tip -> scale_start -> scale_end` and stored in standard COCO `[x, y, v]` triplets.

The `scale_start` and `scale_end` keypoints denote the endpoints of the **primary** reading scale used to define `reading_normalized`. For dials with additional decorative or secondary scales, the current release still defines a single supervisory target per image so that all tasks remain well-posed.

The `reading_normalized` field provides the primary regression target, with actual readings computed as: `v = scale_min + reading_normalized * (scale_max - scale_min)`. The auxiliary field `needle_angle_cw_deg` records the pointer angle around the annotated center in image coordinates and can be used for custom geometry-based evaluators.

Per-image metadata records all 97 DR parameter values, enabling fine-grained ablation studies and dataset filtering by specific conditions.

### 3.4 Dataset Statistics

SynthGauge contains 9,000 images split 7,000/1,000/1,000 for train/val/test. All images are 640×640 JPEG.

**Instrument distribution:** Manometers 34%, voltmeters 30%, ammeters 30%, thermometers 2.5%, tachometers 1.5%, speedometers 1%, other 1%.

**Bounding box statistics:** Mean width 445px, mean height 449px, gauge coverage 55–75% of frame.

**Reading distribution:** Mean 0.50, range [0.02, 0.98], approximately uniform.

The category distribution is intentionally skewed toward pressure and electrical gauges because these dominate the target industrial use cases. As a result, aggregate benchmark metrics should be interpreted together with future per-category analyses rather than as a uniform measure of performance over all gauge families.

---

**[FIGURE 4 - DR VARIATIONS HERE: figure3_dr_variations.jpg]**

**Figure 4.** Examples showing Domain Randomization effects. Different dial colors (cream, white, dark), lighting conditions, backgrounds, and units (bar, RPM, kPa) demonstrate the visual diversity achieved through procedural generation.

---

## 4. Experiments

We evaluate SynthGauge on three benchmark tasks that correspond to the provided annotations: (1) gauge face detection, (2) keypoint localization, and (3) direct reading regression.

### 4.1 Experimental Protocol

All experiments use the official split files corresponding to 7,000/1,000/1,000 images. Detection is trained on full images converted from COCO annotations to Ultralytics YOLO detection format. Keypoint estimation is trained on cropped dial images converted to Ultralytics YOLO pose format; the crop is centered on the gauge bounding box and uses `crop_pad_ratio=0.08`. Regression is also trained on dial crops extracted from the gauge bounding boxes with square padding ratio 0.08 and stored as JSONL image-value pairs.

Default random seed is 42 in all released configurations. Detection training uses the default Ultralytics augmentation pipeline with no task-specific overrides from the repository config. Keypoint training uses explicit augmentation settings from `config_keypoints.yaml`: `hsv_h=0.005`, `hsv_s=0.2`, `hsv_v=0.2`, `degrees=2.0`, `translate=0.02`, `scale=0.05`, while `shear`, `perspective`, `flipud`, `fliplr`, `mosaic`, `mixup`, `copy_paste`, and `erasing` are disabled. The regression baseline applies color jitter (`p=0.3`), mild affine perturbations (`degrees=7`, `translate=0.02`, `scale=[0.95, 1.05]`, `p=0.3`), Gaussian blur (`p=0.2`), and no horizontal flipping. No model ensembling or test-time augmentation is used in the reported evaluation scripts.

We report metrics using the repository evaluation scripts. When no explicit checkpoint path is provided, the evaluation scripts default to the saved `best.pt` checkpoint if it exists, otherwise `last.pt`. For regression, checkpoint selection during training is based on the lowest validation `MAE`. We report task-specific metrics standard in object detection, pose estimation, and scalar regression.

### 4.2 Gauge Detection (Bounding Box)

**Model:** YOLOv8n, COCO-pretrained.

**Training setup:** 100 epochs, batch size 16, image size 640, 4 dataloader workers, YOLOv8n pretrained weights (`yolov8n.pt`), AdamW optimizer, initial learning rate `1e-3`, weight decay `1e-4`, cosine learning-rate schedule, CUDA device selection from config, and seed 42.

**Evaluation metrics:** Precision, Recall, `mAP@0.5`, and `mAP@0.5:0.95`.

**Table 3.** Detection results on SynthGauge test split.

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|--------------|
| YOLOv8n | 0.9864 | 0.9630 | 0.9848 | 0.7803 |

These results indicate that single-gauge localization is nearly saturated under in-domain synthetic evaluation. They should therefore be interpreted mainly as a sanity check for annotation consistency and train/test split difficulty, rather than as evidence that real-world detection is solved.

### 4.3 Gauge Keypoint Detection (Pose)

**Model:** YOLO11s-pose, pretrained initialization.

**Training setup:** 100 epochs, batch size 8, image size 960, 4 dataloader workers, cropped dial inputs with `crop_pad_ratio=0.08`, YOLO11s-pose pretrained weights (`yolo11s-pose.pt`), AdamW optimizer, initial learning rate `1e-3`, weight decay `1e-4`, cosine learning-rate schedule, CUDA device selection from config, and seed 42.

**Keypoint schema:** ground-truth keypoints are ordered as:
`center -> needle_tip -> scale_start -> scale_end`.

**Metrics:**

- Pose detection quality: `pose_precision`, `pose_recall`, `pose_mAP@0.5`, `pose_mAP@0.5:0.95`
- Keypoint localization quality: `PCK@0.05`, `PCK@0.10`

`PCK` is computed by a custom evaluator with confidence threshold 0.25. For each image, the predicted instance is matched to ground truth by maximum IoU; PCK thresholds are normalized by `max(bbox_width, bbox_height)`.

**Table 4.** Keypoint detection results on SynthGauge test split.

| Model | pose_precision | pose_recall | pose_mAP@0.5 | pose_mAP@0.5:0.95 | PCK@0.05 | PCK@0.10 |
|-------|----------------|-------------|--------------|-------------------|----------|----------|
| YOLO11s-pose | 0.9999 | 1.0000 | 0.9950 | 0.6200 | 0.6048 | 0.6665 |

### 4.4 Reading Estimation (Regression)

**Model:** ResNet-18 regressor (ImageNet-pretrained backbone, dropout 0.1) predicting `reading_normalized` from cropped dial images.

**Training setup:** 50 epochs, batch size 32, 4 dataloader workers, dial crops with `pad_ratio=0.08`, resized to 256×256, ImageNet normalization, ResNet-18 pretrained weights, dropout 0.1, MSE loss, AdamW optimizer (`lr=3e-4`, `weight_decay=1e-4`), automatic mixed precision on CUDA, and seed 42. Model selection is based on validation `MAE`.

**Evaluation metrics:** `MAE` (normalized), `DRR@0.02` (Dial Recognition Rate; percentage of predictions within 2% absolute error), with `RMSE` and `R2` as auxiliary statistics.

**Table 5.** Reading estimation results on SynthGauge test split.

| Model | MAE | DRR@0.02 | RMSE | R2 |
|-------|-----|----------|------|----|
| ResNet-18 | 0.0182 | 0.6970 | 0.0372 | 0.9819 |

The strong in-domain regression results show that the rendered dial crops contain sufficient signal for direct reading estimation without explicit geometric post-processing at inference time. At the same time, this task remains the most sensitive to the sim-to-real gap because small shifts in pointer appearance, glass reflections, or unseen dial layouts can induce disproportionately large reading errors.

### 4.5 Scope Alignment

The current benchmark focuses on controlled in-domain evaluation (synthetic train/val/test), which measures learnability and annotation consistency but does not by itself establish deployment readiness. Cross-domain sim-to-real transfer, ablations over individual DR factors, robustness under severe occlusions, and per-category error analysis are important extensions and should accompany future versions of the benchmark.

## 5. Limitations and Future Work

SynthGauge focuses on round analog gauges with single needles. Multi-needle instruments, digital displays, non-circular gauges, and gauges embedded in complex panels are out of scope. The 42 scale configurations are diverse but not exhaustive, and the dataset is intentionally biased toward pressure and electrical gauges.

The current annotation design defines one primary supervisory scale per image. This keeps detection, keypoint, and regression tasks unambiguous, but it also means that secondary scales, auxiliary markings, and text semantics are not yet evaluated as first-class targets.

Despite extensive DR, systematic sim-to-real differences persist for factors not fully modeled: severe motion blur, extreme contamination, strong specular reflections, partial occlusion, hand-held camera artifacts, and non-standard field modifications. In addition, the current paper emphasizes in-domain synthetic benchmarks; without dedicated real-image evaluation, claims about downstream transfer should be interpreted cautiously.

Future directions include expanding to additional instrument types, adding real-image benchmark subsets, reporting per-category metrics, incorporating richer text and unit annotations, and studying targeted DR ablations or adaptation protocols for rapid deployment to new gauge families.

---

## 6. Conclusion

We introduced SynthGauge, a synthetic dataset of 9,000 analog gauge images with comprehensive Domain Randomization across 97 parameters. The dataset covers 42 scale configurations spanning 10+ instrument types and provides unified annotations for detection, reading-specific keypoints, and normalized readings. We position SynthGauge as a reproducible benchmark for studying multi-task gauge understanding under controlled synthetic variability, and as a foundation for future work on sim-to-real transfer in industrial computer vision.

---

## References

[1] G. Salomon, R. Laroca, D. Menotti. "Image-based Automatic Dial Meter Reading in Unconstrained Scenarios." Measurement, vol. 204, 112025, 2022.

[2] B. Howells, J. Charles, R. Cipolla. "Real-Time Analogue Gauge Transcription on Mobile Phone." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 2369-2377, 2021.

[3] J. Leon-Alcazar, Y. Alnumay, C. Zheng, H. Trigui, S. Patel, B. Ghanem. "Learning to Read Analog Gauges from Synthetic Data." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024.

[4] MnZn Li. "Pointer instrument Dataset." Roboflow Universe, 2024. https://universe.roboflow.com/mnzn-li-j1jzu/pointer-instrument-7afwm

[5] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, P. Abbeel. "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World." 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 23-30, 2017.

[6] D. Horvath, G. Erdos, Z. Istenes, T. Horvath, S. Foldi. "Object Detection Using Sim2Real Domain Randomization for Robotic Applications." IEEE Transactions on Robotics, vol. 39, no. 2, pp. 1225-1243, 2023.

[7] M. Denninger, M. Sundermeyer, D. Winkelbauer, D. Olefir, T. Hodan, Y. Zidan, M. Elbadrawy, M. Knauer, H. Katam, A. Lodhi. "BlenderProc: Reducing the Reality Gap with Photorealistic Rendering." RSS 2020 Workshop on Closing the Reality Gap in Sim2Real Transfer for Robotics, 2020.

[8] C.-H. Wang, K.-K. Huang, R.-I. Chang, C.-K. Huang. "Scale-Mark-Based Gauge Reading for Gauge Sensors in Real Environments with Light and Perspective Distortions." Sensors, vol. 22, no. 19, 7490, 2022.

[9] C. Zhang, L. Shi, D. Zhang, T. Ke, J. Li. "Pointer Meter Recognition Method Based on Yolov7 and Hough Transform." Applied Sciences, vol. 13, no. 15, 8722, 2023.

[10] M. Reitsma, J. Keller, K. Blomqvist, R. Siegwart. "Under pressure: learning-based analog gauge reading in the wild." 2024 IEEE International Conference on Robotics and Automation, pp. 14-20, 2024.

---

## Dataset Availability, Licensing, and Reproducibility

The intended public release includes: (i) rendered images and official train/val/test split files; (ii) COCO annotations with custom reading fields and per-image DR metadata; (iii) configuration files for data generation, training, and evaluation; and (iv) baseline scripts for detection, keypoint estimation, and reading regression.

All experiments reported in this paper use fixed configuration files and seed 42. The release package is designed to make the benchmark reproducible end-to-end, from split construction and annotation conversion to training and evaluation.

Because SynthGauge is fully synthetic, it does not contain personal data. The final public release will document separate licenses for dataset content, source code, and any third-party assets. Only assets with compatible redistribution terms will be bundled directly; otherwise, the release will provide attribution and instructions for obtaining them from their original sources.

**Dataset:** https://huggingface.co/datasets/[anonymous]/synthetic-analog-gauges

**Code:** https://github.com/[anonymous]/synthgauge

**Licensing details:** to be specified in the public release package

---

*Generated with Blender 4.2.1 LTS on NVIDIA RTX 3060. Total generation time: ~12.5 hours for 9,000 images.*
