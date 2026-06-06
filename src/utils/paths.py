"""
Centralized Path Management
"""

from pathlib import Path
from src.utils.config import ConfigManager

cfg = ConfigManager()
CONFIG = cfg.get()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (PROJECT_ROOT / CONFIG["dataset_root"]).resolve()

TRAIN_IMAGES = DATASET_ROOT / CONFIG["dataset"]["train_images"]
VAL_IMAGES = DATASET_ROOT / CONFIG["dataset"]["val_images"]

TRAIN_LABELS = DATASET_ROOT / CONFIG["dataset"]["train_labels"]
VAL_LABELS = DATASET_ROOT / CONFIG["dataset"]["val_labels"]

REPORT_DIR = PROJECT_ROOT / CONFIG["analysis"]["reports_dir"]
FIGURE_DIR = PROJECT_ROOT / CONFIG["analysis"]["figures_dir"]
PROCESSED_DIR = PROJECT_ROOT / CONFIG["processed"]["processed_dir"]

for directory in (REPORT_DIR, FIGURE_DIR, PROCESSED_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def validate_required_paths() -> None:
    """Validate all required dataset paths before analysis starts."""
    required_paths = [
        DATASET_ROOT,
        TRAIN_IMAGES,
        VAL_IMAGES,
        TRAIN_LABELS,
        VAL_LABELS,
    ]

    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_str = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required paths:\n{missing_str}")
