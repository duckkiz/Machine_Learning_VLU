import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from ultralytics import YOLO

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BEST_MODEL_PATH, CLASSES, FIGURES_DIR, REPORTS_DIR


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained classifier on test set.")
    parser.add_argument("--weights", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "processed" / "test")
    parser.add_argument("--imgsz", type=int, default=224)
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    items = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            items.append(p)
    return sorted(items)


def save_classification_report_table(report_dict: dict, save_path: Path) -> None:
    metric_names = ["precision", "recall", "f1-score", "support"]
    rows = []
    row_labels = []

    for cls in CLASSES:
        cls_metrics = report_dict.get(cls, {})
        rows.append([
            f"{cls_metrics.get('precision', 0.0):.3f}",
            f"{cls_metrics.get('recall', 0.0):.3f}",
            f"{cls_metrics.get('f1-score', 0.0):.3f}",
            f"{int(cls_metrics.get('support', 0))}",
        ])
        row_labels.append(cls)

    # Add summary rows
    for name in ["macro avg", "weighted avg"]:
        metrics = report_dict.get(name, {})
        rows.append([
            f"{metrics.get('precision', 0.0):.3f}",
            f"{metrics.get('recall', 0.0):.3f}",
            f"{metrics.get('f1-score', 0.0):.3f}",
            f"{int(metrics.get('support', 0))}",
        ])
        row_labels.append(name)

    accuracy_value = report_dict.get("accuracy", 0.0)
    rows.append(["-", "-", f"{accuracy_value:.3f}", f"{sum(report_dict.get(c, {}).get('support', 0) for c in CLASSES)}"])
    row_labels.append("accuracy")

    fig_h = max(3.2, 0.55 * (len(rows) + 1))
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=metric_names,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    ax.set_title("Classification Report on Test Set", fontsize=13, pad=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Saved classification report table to: {save_path}")


def save_per_class_metrics(report_dict: dict, save_path: Path) -> None:
    precision = [report_dict.get(cls, {}).get("precision", 0.0) for cls in CLASSES]
    recall = [report_dict.get(cls, {}).get("recall", 0.0) for cls in CLASSES]
    f1 = [report_dict.get(cls, {}).get("f1-score", 0.0) for cls in CLASSES]

    x = np.arange(len(CLASSES))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1-score")

    ax.set_title("Per-class Metrics on Test Set")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Saved per-class metrics chart to: {save_path}")


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.data.exists():
        raise FileNotFoundError(f"Test folder not found: {args.data}")

    model = YOLO(str(args.weights))
    y_true, y_pred = [], []

    for cls in CLASSES:
        cls_dir = args.data / cls
        if not cls_dir.exists():
            print(f"[WARN] missing class dir: {cls_dir}")
            continue
        for img_path in list_images(cls_dir):
            result = model.predict(source=str(img_path), imgsz=args.imgsz, verbose=False)[0]
            pred_idx = int(result.probs.top1)
            pred_name = model.names[pred_idx]
            y_true.append(cls)
            y_pred.append(pred_name)

    if not y_true:
        raise RuntimeError("No test images found for evaluation.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_dict = classification_report(y_true, y_pred, labels=CLASSES, zero_division=0, output_dict=True)
    report_text = classification_report(y_true, y_pred, labels=CLASSES, zero_division=0)

    report_txt_path = REPORTS_DIR / "classification_report_test.txt"
    report_json_path = REPORTS_DIR / "classification_report_test.json"
    report_txt_path.write_text(report_text, encoding="utf-8")
    report_json_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    summary_path = REPORTS_DIR / "test_metrics_summary.txt"
    summary_lines = [
        "TEST METRICS SUMMARY",
        f"num_samples: {len(y_true)}",
        f"accuracy: {accuracy:.4f}",
        f"macro_precision: {report_dict['macro avg']['precision']:.4f}",
        f"macro_recall: {report_dict['macro avg']['recall']:.4f}",
        f"macro_f1: {report_dict['macro avg']['f1-score']:.4f}",
        f"weighted_f1: {report_dict['weighted avg']['f1-score']:.4f}",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(report_text)
    print(f"[DONE] Saved classification report to: {report_txt_path}")
    print(f"[DONE] Saved test summary to: {summary_path}")

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig_path = FIGURES_DIR / "confusion_matrix_test.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"[DONE] Saved confusion matrix to: {fig_path}")

    # NEW: classification report table figure
    report_table_path = FIGURES_DIR / "classification_report_table_test.png"
    save_classification_report_table(report_dict, report_table_path)

    # NEW: per-class metrics bar chart
    per_class_metrics_path = FIGURES_DIR / "per_class_metrics_test.png"
    save_per_class_metrics(report_dict, per_class_metrics_path)


if __name__ == "__main__":
    main()