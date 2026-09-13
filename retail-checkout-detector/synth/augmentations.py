"""Albumentations pipelines used to vary the visual style of pasted objects
before they're composited into a synthetic scene, and to simulate different
"camera" conditions (phone, CCTV, overhead)."""

import random

import albumentations as A
import cv2

_BBOX_PARAMS = A.BboxParams(format="yolo", label_fields=["class_labels"])

aspect_ratio_aug = A.Compose(
    [
        A.LongestMaxSize(max_size=640),
        A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT),
        A.RandomScale(scale_limit=0.3, p=0.5),
        A.Resize(640, 640),
    ],
    bbox_params=_BBOX_PARAMS,
)

camera_sim = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=25, p=0.7),
        A.Perspective(scale=(0.05, 0.1), p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussianBlur(p=0.2),
    ],
    bbox_params=_BBOX_PARAMS,
)


def multi_camera_aug(image, bboxes, class_labels):
    """Randomly pick one of three "camera style" augmentation pipelines and
    apply it. bboxes must be in YOLO format: [x_center, y_center, w, h]."""

    mode = random.choice(["phone", "cctv", "overhead"])

    if mode == "phone":
        transform = A.Compose(
            [
                A.Rotate(limit=30, p=0.7),
                A.ColorJitter(p=0.5),
                A.RandomShadow(p=0.3),
            ],
            bbox_params=_BBOX_PARAMS,
        )
    elif mode == "cctv":
        transform = A.Compose(
            [
                A.GaussNoise(p=0.6),
                A.MotionBlur(p=0.4),
                A.RandomBrightnessContrast(0.2, 0.3, p=0.7),
            ],
            bbox_params=_BBOX_PARAMS,
        )
    else:  # overhead
        transform = A.Compose(
            [
                A.Perspective(scale=(0.08, 0.12), p=0.7),
                A.Rotate(limit=90, p=0.5),
                A.Resize(640, 640),
            ],
            bbox_params=_BBOX_PARAMS,
        )

    return transform(image=image, bboxes=bboxes, class_labels=class_labels)
