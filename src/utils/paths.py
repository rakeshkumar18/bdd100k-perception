"""Centralized, immutable project path registry."""

from pathlib import Path
from src.utils.config import ConfigManager

cfg = ConfigManager()

# ---------------------------
# ROOTS
# ---------------------------
PROJECT_ROOT = cfg.project_root
DATASET_ROOT = cfg.dataset_root
CONFIG = cfg.config

# ---------------------------
# DATASET PATHS
# ---------------------------
DS = CONFIG["dataset"]

TRAIN_IMAGES = DATASET_ROOT / DS["train_images"]
VAL_IMAGES   = DATASET_ROOT / DS["val_images"]

TRAIN_LABELS = DATASET_ROOT / DS["train_labels"]
VAL_LABELS   = DATASET_ROOT / DS["val_labels"]

# ---------------------------
# OUTPUT ROOTS
# ---------------------------
ANALYSIS = CONFIG["analysis"]
PROCESSED = CONFIG["processed"]

OUTPUT_ROOT = PROJECT_ROOT / "outputs"

REPORTS_DIR    = OUTPUT_ROOT / ANALYSIS["reports_dir"]
FIGURES_DIR    = OUTPUT_ROOT / ANALYSIS["figures_dir"]
PROCESSED_DIR = OUTPUT_ROOT / PROCESSED["processed_dir"]
TRAINING_RUNS_DIR = OUTPUT_ROOT / ANALYSIS["training_runs_dir"]
MLFLOW_DIR = OUTPUT_ROOT / ANALYSIS["mlflow_dir"]
PREDICTIONS_DIR = OUTPUT_ROOT / ANALYSIS["predictions_dir"]

MLFLOW_DB = (
    MLFLOW_DIR / "mlflow.db"
)

MLFLOW_TRACKING_URI = (
    f"sqlite:///{MLFLOW_DB.resolve()}"
)

# ---------------------------
# ENSURE DIRECTORIES EXIST
# ---------------------------
def init_dirs():
    for p in (REPORTS_DIR, FIGURES_DIR, PROCESSED_DIR):
        p.mkdir(parents=True, exist_ok=True)

init_dirs()