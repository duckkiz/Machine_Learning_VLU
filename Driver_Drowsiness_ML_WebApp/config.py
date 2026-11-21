from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

CLASSES = ["awake", "drowsy", "yawning", "looking_away"]
RANDOM_SEED = 42

BASE_MODEL_PATH = MODELS_DIR / "yolo11n-cls.pt"
BEST_MODEL_PATH = MODELS_DIR / "best.pt"

DEFAULT_EPOCHS = 60
DEFAULT_IMGSZ = 224
DEFAULT_BATCH = 16
DEFAULT_DEVICE = "cpu"
