from src.utils.paths import (
    DATASET_ROOT,
    TRAIN_IMAGES,
    TRAIN_LABELS,
)


def test_dataset_root_exists():

    assert DATASET_ROOT.exists()


def test_train_images_exist():

    assert TRAIN_IMAGES.exists()


def test_train_labels_exist():

    assert TRAIN_LABELS.exists()