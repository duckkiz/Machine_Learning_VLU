import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BASE_MODEL_PATH,
    BEST_MODEL_PATH,
    DEFAULT_BATCH,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_IMGSZ,
    FIGURES_DIR,
    REPORTS_DIR,
    RUNS_DIR,
)


EXPECTED_CLASSES = {"awake", "drowsy", "yawning", "looking_away"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11 classification model.")
    parser.add_argument("--model", type=Path, default=BASE_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--project", type=Path, default=RUNS_DIR)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--name", type=str, default="classify_remake")
    parser.add_argument("--run-eval", action="store_true", help="Run evaluation on the test set after training.")
    return parser.parse_args()


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[DONE] Copied: {src} -> {dst}")


def _read_results_csv(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_key(keys: list[str], candidates: list[str]) -> str | None:
    lower_map = {k.lower(): k for k in keys}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for k in keys:
        kl = k.lower()
        for cand in candidates:
            if cand.lower() in kl:
                return k
    return None


def _extract_series(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        raw = row.get(key, "")
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return values


def _count_images_in_class_dir(class_dir: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def inspect_dataset(data_dir: Path) -> None:
    print("=" * 60)
    print("DATASET INSPECTION")
    print(f"root: {data_dir}")

    split_classes: dict[str, list[str]] = {}

    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"[WARN] Missing split folder: {split_dir}")
            continue

        class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        class_names = [p.name for p in class_dirs]
        split_classes[split] = class_names

        print(f"\n[{split.upper()}]")
        if not class_dirs:
            print("  [WARN] No class folders found.")
            continue

        for cls_dir in class_dirs:
            n = _count_images_in_class_dir(cls_dir)
            print(f"  - {cls_dir.name}: {n}")

    train_classes = set(split_classes.get("train", []))
    if train_classes:
        print("\n[INFO] Train classes:", sorted(train_classes))

        missing_expected = EXPECTED_CLASSES - train_classes
        extra_classes = train_classes - EXPECTED_CLASSES

        if missing_expected:
            print(f"[WARN] Missing expected classes in train: {sorted(missing_expected)}")
        if extra_classes:
            print(f"[WARN] Unexpected extra classes in train: {sorted(extra_classes)}")

    for split in ["val", "test"]:
        split_set = set(split_classes.get(split, []))
        if split_set and train_classes and split_set != train_classes:
            print(
                f"[WARN] Class mismatch between train and {split}: "
                f"train={sorted(train_classes)} vs {split}={sorted(split_set)}"
            )

    print("=" * 60)


def export_training_reports(run_dir: Path) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results_png = run_dir / "results.png"
    _copy_if_exists(results_png, FIGURES_DIR / "results_train.png")

    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        print(f"[WARN] results.csv not found in {run_dir}")
        return

    _copy_if_exists(results_csv, REPORTS_DIR / "results.csv")
    rows = _read_results_csv(results_csv)
    if not rows:
        print("[WARN] results.csv is empty")
        return

    keys = list(rows[0].keys())
    epoch_key = _pick_key(keys, ["epoch"])
    train_loss_key = _pick_key(keys, ["train/loss", "train_loss", "loss"])
    val_loss_key = _pick_key(keys, ["val/loss", "val_loss", "loss/val"])
    top1_key = _pick_key(keys, ["metrics/accuracy_top1", "accuracy_top1", "top1", "top1_acc"])
    top5_key = _pick_key(keys, ["metrics/accuracy_top5", "accuracy_top5", "top5", "top5_acc"])

    epochs = _extract_series(rows, epoch_key) if epoch_key else list(range(1, len(rows) + 1))
    train_loss = _extract_series(rows, train_loss_key) if train_loss_key else []
    val_loss = _extract_series(rows, val_loss_key) if val_loss_key else []
    top1 = _extract_series(rows, top1_key) if top1_key else []
    top5 = _extract_series(rows, top5_key) if top5_key else []

    if train_loss or val_loss:
        fig, ax = plt.subplots(figsize=(7, 5))
        if train_loss:
            ax.plot(epochs, train_loss, label="Train loss")
        if val_loss:
            ax.plot(epochs, val_loss, label="Val loss")
        ax.set_title("Loss over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        loss_path = FIGURES_DIR / "loss_curve.png"
        plt.savefig(loss_path, dpi=200)
        plt.close(fig)
        print(f"[DONE] Saved loss curve to: {loss_path}")

    if top1 or top5:
        fig, ax = plt.subplots(figsize=(7, 5))
        if top1:
            ax.plot(epochs, top1, label="Top-1 accuracy")
        if top5:
            ax.plot(epochs, top5, label="Top-5 accuracy")
        ax.set_title("Accuracy over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        acc_path = FIGURES_DIR / "accuracy_curve.png"
        plt.savefig(acc_path, dpi=200)
        plt.close(fig)
        print(f"[DONE] Saved accuracy curve to: {acc_path}")

    best_top1 = max(top1) if top1 else float("nan")
    best_val_loss = min(val_loss) if val_loss else float("nan")
    final_train_loss = train_loss[-1] if train_loss else float("nan")
    final_val_loss = val_loss[-1] if val_loss else float("nan")
    final_top1 = top1[-1] if top1 else float("nan")
    final_top5 = top5[-1] if top5 else float("nan")

    summary_lines = [
        "TRAIN METRICS SUMMARY",
        f"run_dir: {run_dir}",
        f"num_epochs_logged: {len(rows)}",
        f"best_top1_acc: {best_top1:.4f}" if top1 else "best_top1_acc: N/A",
        f"best_val_loss: {best_val_loss:.4f}" if val_loss else "best_val_loss: N/A",
        f"final_train_loss: {final_train_loss:.4f}" if train_loss else "final_train_loss: N/A",
        f"final_val_loss: {final_val_loss:.4f}" if val_loss else "final_val_loss: N/A",
        f"final_top1_acc: {final_top1:.4f}" if top1 else "final_top1_acc: N/A",
        f"final_top5_acc: {final_top5:.4f}" if top5 else "final_top5_acc: N/A",
    ]
    summary_path = REPORTS_DIR / "train_metrics_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[DONE] Saved train metrics summary to: {summary_path}")


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Base model not found: {args.model}")
    if not (args.data / "train").exists() or not (args.data / "val").exists():
        raise FileNotFoundError(f"Expected train/val folders inside: {args.data}")

    inspect_dataset(args.data)

    print("=" * 60)
    print("TRAIN CONFIG")
    print(f"model   : {args.model}")
    print(f"data    : {args.data}")
    print(f"epochs  : {args.epochs}")
    print(f"imgsz   : {args.imgsz}")
    print(f"batch   : {args.batch}")
    print(f"device  : {args.device}")
    print(f"project : {args.project}")
    print(f"name    : {args.name}")
    print(f"patience: {args.patience}")
    print(f"run_eval: {args.run_eval}")
    print("=" * 60)

    model = YOLO(str(args.model))
    print(f"[INFO] Loaded base model: {args.model}")

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        patience=args.patience,
        exist_ok=False,
    )

    run_dir = Path(results.save_dir)
    print(f"[DONE] Training run saved to: {run_dir}")

    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    if best_weights.exists():
        BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, BEST_MODEL_PATH)
        print(f"[DONE] Copied best weights to: {BEST_MODEL_PATH}")
    else:
        print(f"[WARN] best.pt not found in: {best_weights}")

    if last_weights.exists():
        print(f"[INFO] Last weights available at: {last_weights}")

    export_training_reports(run_dir)

    if args.run_eval:
        test_dir = args.data / "test"
        if test_dir.exists():
            eval_weights = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else best_weights
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate.py"),
                "--weights",
                str(eval_weights),
                "--data",
                str(test_dir),
                "--imgsz",
                str(args.imgsz),
                "--save-dir",
                str(run_dir),
            ]
            print(f"[INFO] Running evaluation: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        else:
            print(f"[WARN] Test folder not found, skip evaluation: {test_dir}")


if __name__ == "__main__":
    main()