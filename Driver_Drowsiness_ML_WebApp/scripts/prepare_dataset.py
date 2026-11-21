from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from PIL import Image

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RANDOM_SEED

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test dataset using MediaPipe Tasks Face Landmarker."
    )
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--face-model", type=Path, default=PROJECT_ROOT / "models" / "face_landmarker.task")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--crop-face", action="store_true")
    parser.add_argument("--margin", type=float, default=0.18)

    # New behavior
    parser.add_argument(
        "--fallback-original",
        action="store_true",
        default=True,
        help="If face crop fails, keep the original image instead of skipping it.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float) -> None:
    if not (0 < train_ratio < 1):
        raise ValueError("train-ratio must be between 0 and 1.")
    if not (0 < val_ratio < 1):
        raise ValueError("val-ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train-ratio + val-ratio must be less than 1.")


def get_class_names(root: Path) -> list[str]:
    classes = [p.name for p in sorted(root.iterdir()) if p.is_dir()]
    if not classes:
        raise FileNotFoundError(f"No class folders found in: {root}")
    return classes


def valid_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTS:
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            files.append(path)
        except Exception:
            print(f"[SKIP] Invalid image: {path}")
    return files


def split_counts(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    test_n = n - train_n - val_n
    return train_n, val_n, test_n


def crop_face_with_tasks(image_bgr, landmarker, margin: float = 0.18):
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    x_min = max(0.0, min(xs) - margin)
    y_min = max(0.0, min(ys) - margin)
    x_max = min(1.0, max(xs) + margin)
    y_max = min(1.0, max(ys) + margin)

    x1 = int(x_min * w)
    y1 = int(y_min * h)
    x2 = int(x_max * w)
    y2 = int(y_max * h)

    if x2 <= x1 or y2 <= y1:
        return None

    return image_bgr[y1:y2, x1:x2]


def save_image(
    src: Path,
    dst: Path,
    crop_face: bool,
    landmarker,
    margin: float,
    fallback_original: bool,
) -> tuple[bool, str]:
    """
    Returns:
        (success, mode)
        mode in {"copied", "cropped", "fallback_original", "skip"}
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not crop_face:
        shutil.copy2(src, dst)
        return True, "copied"

    image = cv2.imread(str(src))
    if image is None:
        print(f"[SKIP] Cannot read image: {src}")
        return False, "skip"

    cropped = crop_face_with_tasks(image, landmarker, margin=margin)
    if cropped is not None:
        ok = cv2.imwrite(str(dst), cropped)
        if not ok:
            print(f"[SKIP] Cannot write cropped image: {dst}")
            return False, "skip"
        return True, "cropped"

    if fallback_original:
        shutil.copy2(src, dst)
        print(f"[FALLBACK] No face detected, kept original image: {src}")
        return True, "fallback_original"

    print(f"[SKIP] No face detected: {src}")
    return False, "skip"


def print_split_counts(split_map: dict[str, list[Path]]) -> None:
    for split_name in ["train", "val", "test"]:
        print(f"  - {split_name}: {len(split_map.get(split_name, []))}")


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio)
    random.seed(args.seed)

    if not args.input.exists():
        raise FileNotFoundError(f"Input folder not found: {args.input}")
    if not args.face_model.exists():
        raise FileNotFoundError(f"Face Landmarker model not found: {args.face_model}")

    if args.output.exists():
        print(f"[INFO] Clearing existing processed data: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    class_names = get_class_names(args.input)

    print("=" * 60)
    print("PREPARE DATASET")
    print(f"input            : {args.input}")
    print(f"output           : {args.output}")
    print(f"crop_face        : {args.crop_face}")
    print(f"fallback_original: {args.fallback_original}")
    print(f"margin           : {args.margin}")
    print(f"train_ratio      : {args.train_ratio}")
    print(f"val_ratio        : {args.val_ratio}")
    print(f"seed             : {args.seed}")
    print(f"classes          : {class_names}")
    print("=" * 60)

    BaseOptions = python.BaseOptions
    FaceLandmarker = vision.FaceLandmarker
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.face_model)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    total_valid = 0
    total_saved = 0
    total_copied = 0
    total_cropped = 0
    total_fallback = 0
    total_skipped = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
        for cls in class_names:
            cls_dir = args.input / cls
            files = valid_images(cls_dir)
            random.shuffle(files)

            train_n, val_n, test_n = split_counts(len(files), args.train_ratio, args.val_ratio)

            splits = {
                "train": files[:train_n],
                "val": files[train_n: train_n + val_n],
                "test": files[train_n + val_n:],
            }

            cls_valid = len(files)
            cls_saved = 0
            cls_copied = 0
            cls_cropped = 0
            cls_fallback = 0
            cls_skipped = 0

            for split_name, split_files in splits.items():
                for src in split_files:
                    dst = args.output / split_name / cls / src.name
                    ok, mode = save_image(
                        src=src,
                        dst=dst,
                        crop_face=args.crop_face,
                        landmarker=landmarker,
                        margin=args.margin,
                        fallback_original=args.fallback_original,
                    )

                    if ok:
                        cls_saved += 1
                        if mode == "copied":
                            cls_copied += 1
                        elif mode == "cropped":
                            cls_cropped += 1
                        elif mode == "fallback_original":
                            cls_fallback += 1
                    else:
                        cls_skipped += 1

            total_valid += cls_valid
            total_saved += cls_saved
            total_copied += cls_copied
            total_cropped += cls_cropped
            total_fallback += cls_fallback
            total_skipped += cls_skipped

            print(f"\n[CLASS] {cls}")
            print(f"  valid images : {cls_valid}")
            print_split_counts(splits)
            print(f"  saved        : {cls_saved}")
            print(f"  copied       : {cls_copied}")
            print(f"  cropped      : {cls_cropped}")
            print(f"  fallback     : {cls_fallback}")
            print(f"  skipped      : {cls_skipped}")

    print("\n" + "=" * 60)
    print(f"[READY] Processed dataset created at: {args.output}")
    print(f"[SUMMARY] valid={total_valid}")
    print(f"[SUMMARY] saved={total_saved}")
    print(f"[SUMMARY] copied={total_copied}")
    print(f"[SUMMARY] cropped={total_cropped}")
    print(f"[SUMMARY] fallback_original={total_fallback}")
    print(f"[SUMMARY] skipped={total_skipped}")
    print("=" * 60)


if __name__ == "__main__":
    main()