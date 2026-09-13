# Retail Checkout Product Detector

A YOLO-based single-class ("product") object detector for retail checkout
scenes, trained on **synthetically composited multi-object images** built
from a single-object product dataset (RPC — Retail Product Checkout).

## Problem

The source dataset's training images each contain exactly **one** product
per image (clean, exemplar-style photos). Real checkout scenes, however,
contain multiple overlapping products at once. Training directly on the
single-object images would not teach the model to handle clutter or
occlusion — so this project builds a **copy-paste synthetic data pipeline**
that composites multiple cropped product images onto a blank canvas to
simulate realistic multi-item checkout scenes.

## Pipeline

```
COCO annotations ─▶ YOLO label conversion ─▶ per-object crops ─▶
synthetic multi-object composites ─▶ YOLO training ─▶ deployment pipeline
```

Deployment pipeline:

```
Camera → YOLO Detection → Tracking (ByteTrack) → Alerts (item counting) → Render → Screen
```

## Project structure

```
retail-checkout-detector/
├── data/
│   └── prepare_dataset.py     # COCO -> YOLO label conversion
├── synth/
│   ├── augmentations.py       # Albumentations pipelines (camera-style augmentation)
│   ├── compositing.py         # object cropping, pasting, blending, occlusion handling
│   └── generate_dataset.py    # builds the synthetic multi-object dataset + data.yaml
├── train.py                   # YOLO training entrypoint
├── pipeline/
│   └── run_pipeline.py        # live camera -> detect -> track -> alert -> render -> screen
├── notebooks/
│   └── exploration.ipynb      # original EDA / sanity-check notebook
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Convert the source COCO annotations to YOLO format:**
```bash
python data/prepare_dataset.py \
    --base-path /path/to/retail-product-checkout-dataset \
    --output-dir ./dataset \
    --num-samples 8000
```

**2. Generate the synthetic multi-object training set:**
```bash
python synth/generate_dataset.py \
    --dataset-dir ./dataset \
    --output-dir ./synth_dataset \
    --num-train 2000 \
    --num-val 400 \
    --max-pool-size 3000
```

**3. Train:**
```bash
python train.py --data ./synth_dataset/data.yaml --epochs 50 --model yolov8n.pt
```

**4. Run the live detection/counting pipeline:**
```bash
python pipeline/run_pipeline.py --weights runs/detect/train/weights/best.pt --source 0
```

## Notable engineering issues found and fixed along the way

- **Bounding-box scale bug** — the original COCO→YOLO conversion normalized
  every box using a hardcoded `640x640` placeholder size instead of each
  image's real dimensions (source images are `2592x1944`). This silently
  corrupted every label and, downstream, caused most object crops to be
  filtered out during pool-building (94 valid crops recovered out of
  thousands of images). Fixed by reading each image's true width/height
  from the COCO metadata (or PIL as a fallback).
- **RAM blowup** — after the bbox fix, the object pool went from 94 crops to
  several thousand full-resolution (2592x1944) arrays held in memory at
  once, crashing the session. Fixed by capping each crop's longer side to
  400px and capping the total pool size, with early stopping once the cap
  is reached.
- **Unused augmentation pipelines** — `camera_sim`/`multi_camera_aug` were
  defined but never actually called during compositing. Wired them into
  the object-pasting step so each pasted object gets a randomized
  "camera style" (phone / CCTV / overhead) applied before placement.
- **Invalid bbox passed to Albumentations** — a placeholder full-frame box
  was passed as `[0.0, 0.0, 1.0, 1.0]` in YOLO format (center-x, center-y,
  w, h), which describes a box centered at the corner — invalid range.
  Corrected to `[0.5, 0.5, 1.0, 1.0]` (centered, full-size).

## Results

_Add your final mAP50 / mAP50-95 numbers here once training completes,
ideally with a before/after annotated image from real (non-synthetic)
checkout photos._

## License / data source

Dataset: [Retail Product Checkout (RPC) Dataset](https://rpc-dataset.github.io/).
Follow the dataset's own license terms for redistribution of any raw images.
