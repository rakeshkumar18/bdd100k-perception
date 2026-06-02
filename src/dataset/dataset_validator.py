from pathlib import Path
from PIL import Image

from src.utils.paths import (
    TRAIN_IMAGES,
    VAL_IMAGES,
    TEST_IMAGES,
    TRAIN_LABELS,
    VAL_LABELS,
    TEST_LABELS,
)


class DatasetValidator:

    SPLITS = {
        "train": {
            "images": TRAIN_IMAGES,
            "labels": TRAIN_LABELS,
        },
        "val": {
            "images": VAL_IMAGES,
            "labels": VAL_LABELS,
        },
        "test": {
            "images": TEST_IMAGES,
            "labels": TEST_LABELS,
        },
    }

    def validate_structure(self):

        missing = []

        for split, paths in self.SPLITS.items():

            if not paths["images"].exists():
                missing.append(paths["images"])

            if not paths["labels"].exists():
                missing.append(paths["labels"])

        return missing

    def validate_pairs(self, split):

        image_dir = self.SPLITS[split]["images"]
        label_dir = self.SPLITS[split]["labels"]

        image_names = {
            p.stem
            for p in image_dir.glob("*.jpg")
        }

        label_names = {
            p.stem
            for p in label_dir.glob("*.txt")
        }

        return {
            "missing_labels":
                image_names - label_names,
            "missing_images":
                label_names - image_names,
        }

    def validate_images(self, split):

        image_dir = self.SPLITS[split]["images"]

        corrupted = []

        for img_path in image_dir.glob("*.jpg"):

            try:
                with Image.open(img_path) as img:
                    img.verify()

            except Exception:
                corrupted.append(img_path)

        return corrupted

    def count_empty_labels(self, split):

        label_dir = self.SPLITS[split]["labels"]

        empty_files = []

        for file in label_dir.glob("*.txt"):

            if file.stat().st_size == 0:
                empty_files.append(file)

        return empty_files

    def validate_label_format(self, split):

        label_dir = self.SPLITS[split]["labels"]

        invalid_lines = []

        for label_file in label_dir.glob("*.txt"):

            with open(label_file, "r") as f:

                for line_num, line in enumerate(f, start=1):

                    values = line.strip().split()

                    if len(values) != 5:

                        invalid_lines.append(
                            (
                                label_file.name,
                                line_num,
                                line.strip(),
                            )
                        )

        return invalid_lines

    def validate_bbox_ranges(self, split):

        label_dir = self.SPLITS[split]["labels"]

        invalid_boxes = []

        for label_file in label_dir.glob("*.txt"):

            with open(label_file, "r") as f:

                for line_num, line in enumerate(f, start=1):

                    values = line.strip().split()

                    if len(values) != 5:
                        continue

                    try:

                        cls_id = int(values[0])

                        x = float(values[1])
                        y = float(values[2])
                        w = float(values[3])
                        h = float(values[4])

                        if not (
                            0 <= x <= 1 and
                            0 <= y <= 1 and
                            0 <= w <= 1 and
                            0 <= h <= 1
                        ):
                            invalid_boxes.append(
                                (
                                    label_file.name,
                                    line_num,
                                )
                            )

                    except ValueError:

                        invalid_boxes.append(
                            (
                                label_file.name,
                                line_num,
                            )
                        )

        return invalid_boxes