"""
Convert the RPC (Retail Product Checkout) COCO-format annotations into
single-class YOLO label files, sampling a subset of images and splitting
into train/val.

IMPORTANT FIX vs. the original notebook: bounding boxes are normalized
using each image's REAL width/height (from COCO metadata, falling back to
opening the file with PIL) instead of a hardcoded 640x640 placeholder.
Using a fixed placeholder silently corrupts every label whenever the real
image size differs from the placeholder (RPC images are 2592x1944).
"""

import argparse
import json
import os
import random
import shutil

from PIL import Image
from tqdm import tqdm


def convert_bbox(size, box):
    """COCO [x_min, y_min, width, height] -> YOLO [x_center, y_center, w, h], normalized."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    return (x * dw, y * dh, w * dw, h * dh)


def process(ids, img_map, img_size_map, ann_map, images_dir, img_out, lbl_out):
    for img_id in tqdm(ids, desc=f"Processing -> {img_out}"):
        if img_id not in img_map:
            continue

        file_name = img_map[img_id]
        src_img = os.path.join(images_dir, file_name)

        if not os.path.exists(src_img):
            continue

        shutil.copy(src_img, os.path.join(img_out, file_name))

        anns = ann_map.get(img_id, [])

        # Use each image's REAL width/height, never a hardcoded placeholder.
        w, h = img_size_map.get(img_id, (None, None))
        if not w or not h:
            with Image.open(src_img) as im:
                w, h = im.size

        label_file = os.path.splitext(file_name)[0] + ".txt"

        with open(os.path.join(lbl_out, label_file), "w") as f:
            for ann in anns:
                bbox = ann["bbox"]
                x, y, bw, bh = convert_bbox((w, h), bbox)
                # single class = 0 ("product")
                f.write(f"0 {x} {y} {bw} {bh}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", required=True,
                        help="Path to the RPC dataset root (contains the COCO json and image folder).")
    parser.add_argument("--json-name", default="instances_train2019.json")
    parser.add_argument("--images-subdir", default="train2019")
    parser.add_argument("--output-dir", required=True, help="Where to write images/ and labels/.")
    parser.add_argument("--num-samples", type=int, default=8000,
                        help="Number of images to sample from the full annotation set.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    json_path = os.path.join(args.base_path, args.json_name)
    images_dir = os.path.join(args.base_path, args.images_subdir)

    img_train_dir = os.path.join(args.output_dir, "images/train")
    img_val_dir = os.path.join(args.output_dir, "images/val")
    lbl_train_dir = os.path.join(args.output_dir, "labels/train")
    lbl_val_dir = os.path.join(args.output_dir, "labels/val")

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        os.makedirs(d, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    img_map = {img["id"]: img["file_name"] for img in images}
    img_size_map = {img["id"]: (img.get("width"), img.get("height")) for img in images}

    ann_map = {}
    for ann in annotations:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    all_ids = list(img_map.keys())
    sample_ids = random.sample(all_ids, min(args.num_samples, len(all_ids)))

    split = int((1 - args.val_fraction) * len(sample_ids))
    train_ids = sample_ids[:split]
    val_ids = sample_ids[split:]

    process(train_ids, img_map, img_size_map, ann_map, images_dir, img_train_dir, lbl_train_dir)
    process(val_ids, img_map, img_size_map, ann_map, images_dir, img_val_dir, lbl_val_dir)

    print(f"Train images: {len(os.listdir(img_train_dir))}")
    print(f"Val images:   {len(os.listdir(img_val_dir))}")


if __name__ == "__main__":
    main()
