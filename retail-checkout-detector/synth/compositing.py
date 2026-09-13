"""Core compositing logic: crop objects from single-object source images,
build a pool of reusable object crops, and paste them onto blank canvases
to build synthetic multi-object scenes with occlusion handling and
camera-style augmentation."""

import os
import random

import cv2
import numpy as np

from synth.augmentations import multi_camera_aug


def load_object(img_path, label_path):
    """Crop every labeled object out of a single source image."""
    img = cv2.imread(img_path)
    if img is None:
        return []

    h, w, _ = img.shape
    objects = []

    with open(label_path) as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.split())

            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)

            crop = img[y1:y2, x1:x2]
            objects.append(crop)

    return objects


def resize_cap(img, max_dim):
    """Resize so the longer side is at most max_dim, preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    return img


def build_object_pool(images_dir, labels_dir, max_obj_dim=400, max_pool_size=3000, seed=None):
    """Build a memory-capped pool of object crops from a directory of
    single-object images + YOLO labels.

    Caps applied to avoid the RAM blowup seen with uncapped full-resolution
    crops: each crop's longer side is capped to `max_obj_dim`, and the pool
    stops growing once it reaches `max_pool_size`.
    """
    if seed is not None:
        random.seed(seed)

    object_pool = []
    img_names = os.listdir(images_dir)
    random.shuffle(img_names)  # so an early cutoff is still representative

    for img_name in img_names:
        if len(object_pool) >= max_pool_size:
            break

        img_path = os.path.join(images_dir, img_name)
        name = os.path.splitext(img_name)[0]
        label_path = os.path.join(labels_dir, name + ".txt")

        if not os.path.exists(label_path):
            continue

        for obj in load_object(img_path, label_path):
            if obj is None or obj.size == 0:
                continue
            if obj.shape[0] < 5 or obj.shape[1] < 5:
                continue

            object_pool.append(resize_cap(obj, max_obj_dim))

            if len(object_pool) >= max_pool_size:
                break

    return object_pool


def create_canvas(size=640):
    return np.ones((size, size, 3), dtype=np.uint8) * 255


def iou_xywh_frac(b1, b2):
    """Fraction of box b1's area covered by box b2. Both normalized [cx, cy, w, h]."""

    def to_xyxy(b):
        cx, cy, w, h = b
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    x1a, y1a, x2a, y2a = to_xyxy(b1)
    x1b, y1b, x2b, y2b = to_xyxy(b2)

    inter_w = max(0, min(x2a, x2b) - max(x1a, x1b))
    inter_h = max(0, min(y2a, y2b) - max(y1a, y1b))
    inter = inter_w * inter_h

    area_a = (x2a - x1a) * (y2a - y1a)
    if area_a <= 0:
        return 0
    return inter / area_a


def augment_object(obj):
    """Apply one of the camera-style augmentations to a single cropped object.

    NOTE: the full-frame placeholder box is [0.5, 0.5, 1.0, 1.0] in YOLO
    format (center of frame, full width/height) - NOT [0.0, 0.0, 1.0, 1.0],
    which describes an invalid box centered at the corner.
    """
    aug = multi_camera_aug(
        image=obj,
        bboxes=[[0.5, 0.5, 1.0, 1.0]],
        class_labels=[0],
    )
    return aug["image"]


def paste_object(canvas, obj, occlusion_thresh=0.6):
    """Augment, resize, and blend one object onto the canvas.
    Returns (canvas, bbox) where bbox is None if the object couldn't be placed."""
    h, w, _ = canvas.shape

    obj = augment_object(obj)

    scale = random.uniform(0.3, 0.8)
    oh, ow = obj.shape[:2]
    new_w, new_h = max(1, int(ow * scale)), max(1, int(oh * scale))
    obj = cv2.resize(obj, (new_w, new_h))

    oh, ow = obj.shape[:2]
    if ow >= w or oh >= h:
        return canvas, None

    x = random.randint(0, w - ow)
    y = random.randint(0, h - oh)

    # blend instead of a hard paste, to avoid a visible cut-paste seam
    mask = 255 * np.ones(obj.shape, obj.dtype)
    center = (x + ow // 2, y + oh // 2)
    try:
        canvas = cv2.seamlessClone(obj, canvas, mask, center, cv2.NORMAL_CLONE)
    except cv2.error:
        # seamlessClone can fail on degenerate/edge cases - fall back to a hard paste
        canvas[y:y + oh, x:x + ow] = obj

    cx = (x + ow / 2) / w
    cy = (y + oh / 2) / h
    bw = ow / w
    bh = oh / h

    return canvas, [0, cx, cy, bw, bh]


def place_objects(canvas, object_list, max_objects=5, occlusion_thresh=0.6):
    """Paste 2..max_objects objects onto the canvas, dropping any earlier
    box that ends up mostly covered by a later one."""
    bboxes = []
    num_objects = random.randint(2, max_objects)

    for _ in range(num_objects):
        obj = random.choice(object_list)
        canvas, new_box = paste_object(canvas, obj, occlusion_thresh)
        if new_box is None:
            continue

        bboxes = [
            old_box for old_box in bboxes
            if iou_xywh_frac(old_box[1:], new_box[1:]) < occlusion_thresh
        ]
        bboxes.append(new_box)

    return canvas, bboxes


def place_with_overlap(canvas, object_list, occlusion_thresh=0.6):
    """Same as place_objects, but forces one extra object on top of the
    others with 70% probability, simulating a partially hidden item."""
    canvas, bboxes = place_objects(canvas, object_list, occlusion_thresh=occlusion_thresh)

    if random.random() < 0.7:
        obj = random.choice(object_list)
        canvas, new_box = paste_object(canvas, obj, occlusion_thresh)
        if new_box is not None:
            bboxes = [
                old_box for old_box in bboxes
                if iou_xywh_frac(old_box[1:], new_box[1:]) < occlusion_thresh
            ]
            bboxes.append(new_box)

    return canvas, bboxes


def save_sample(img, bboxes, img_path, label_path):
    cv2.imwrite(img_path, img)
    with open(label_path, "w") as f:
        for box in bboxes:
            f.write(" ".join(map(str, box)) + "\n")
