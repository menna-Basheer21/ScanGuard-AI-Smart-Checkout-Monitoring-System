"""Build a synthetic multi-object checkout dataset by compositing cropped
single-object images onto blank canvases, and write the YOLO data.yaml
pointing at it."""

import argparse
import os

from tqdm import tqdm

from synth.compositing import (
    build_object_pool,
    create_canvas,
    place_objects,
    place_with_overlap,
    save_sample,
)


def generate_split(n, object_pool, img_out, lbl_out, prefix, max_objects):
    import random

    for i in tqdm(range(n), desc=f"Generating {prefix}"):
        canvas = create_canvas()

        if random.random() < 0.5:
            canvas, bboxes = place_objects(canvas, object_pool, max_objects=max_objects)
        else:
            canvas, bboxes = place_with_overlap(canvas, object_pool)

        if not bboxes:
            continue

        img_path = os.path.join(img_out, f"{prefix}_{i:05d}.jpg")
        label_path = os.path.join(lbl_out, f"{prefix}_{i:05d}.txt")
        save_sample(canvas, bboxes, img_path, label_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True,
                        help="Output dir from prepare_dataset.py (contains images/train, labels/train, ...).")
    parser.add_argument("--output-dir", required=True, help="Where to write the synthetic dataset.")
    parser.add_argument("--num-train", type=int, default=2000)
    parser.add_argument("--num-val", type=int, default=400)
    parser.add_argument("--max-objects", type=int, default=5)
    parser.add_argument("--max-obj-dim", type=int, default=400,
                        help="Cap each object crop's longer side (px) to control memory use.")
    parser.add_argument("--max-pool-size", type=int, default=3000,
                        help="Cap the total number of object crops held in memory.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src_images_train = os.path.join(args.dataset_dir, "images/train")
    src_labels_train = os.path.join(args.dataset_dir, "labels/train")

    img_train = os.path.join(args.output_dir, "images/train")
    img_val = os.path.join(args.output_dir, "images/val")
    lbl_train = os.path.join(args.output_dir, "labels/train")
    lbl_val = os.path.join(args.output_dir, "labels/val")

    for d in [img_train, img_val, lbl_train, lbl_val]:
        os.makedirs(d, exist_ok=True)

    print("Building object pool...")
    object_pool = build_object_pool(
        src_images_train,
        src_labels_train,
        max_obj_dim=args.max_obj_dim,
        max_pool_size=args.max_pool_size,
        seed=args.seed,
    )
    print(f"Total objects in pool: {len(object_pool)}")

    if not object_pool:
        raise RuntimeError(
            "Object pool is empty - check that prepare_dataset.py produced valid "
            "labels and that --dataset-dir points at the right folder."
        )

    generate_split(args.num_train, object_pool, img_train, lbl_train, "train", args.max_objects)
    generate_split(args.num_val, object_pool, img_val, lbl_val, "val", args.max_objects)

    print(f"Train images: {len(os.listdir(img_train))}")
    print(f"Val images:   {len(os.listdir(img_val))}")

    # write data.yaml for YOLO training
    data_yaml = f"""path: {os.path.abspath(args.output_dir)}
train: images/train
val: images/val

nc: 1
names: ['product']
"""
    yaml_path = os.path.join(args.output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(data_yaml)

    print(f"Saved data.yaml to {yaml_path}")


if __name__ == "__main__":
    main()
