"""
Centralized Path Management
"""

from pathlib import Path

from src.utils.config import ConfigManager


cfg = ConfigManager()

CONFIG = cfg.get()

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

DATASET_ROOT = (
    PROJECT_ROOT /
    cfg.dataset_root
).resolve()

#
# Image Directories
#

TRAIN_IMAGES = (
    DATASET_ROOT /
    CONFIG["dataset"]["train_images"]
)

VAL_IMAGES = (
    DATASET_ROOT /
    CONFIG["dataset"]["val_images"]
)

TEST_IMAGES = (
    DATASET_ROOT /
    CONFIG["dataset"]["test_images"]
)

#
# Label Directories
#

TRAIN_LABELS = (
    DATASET_ROOT /
    CONFIG["dataset"]["train_labels"]
)

VAL_LABELS = (
    DATASET_ROOT /
    CONFIG["dataset"]["val_labels"]
)

TEST_LABELS = (
    DATASET_ROOT /
    CONFIG["dataset"]["test_labels"]
)

#
# Output Directories
#

REPORT_DIR = (
    PROJECT_ROOT /
    CONFIG["analysis"]["reports_dir"]
)

FIGURE_DIR = (
    PROJECT_ROOT /
    CONFIG["analysis"]["figures_dir"]
)

PROCESSED_DIR = (
    PROJECT_ROOT /
    CONFIG["processed"]["processed_dir"]
)

#
# Create output directories
#

for directory in [
    REPORT_DIR,
    FIGURE_DIR,
    PROCESSED_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

#
# Validation
#

required_paths = [
    DATASET_ROOT,
    TRAIN_IMAGES,
    VAL_IMAGES,
    TRAIN_LABELS,
    VAL_LABELS,
]

for path in required_paths:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing path: {path}"
        )