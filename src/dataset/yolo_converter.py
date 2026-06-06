"""Convert BDD100K object annotations into YOLO label files."""

from pathlib import Path

import pandas as pd


class YOLOConverter:
    """Convert tabular bounding-box annotations to YOLO text format."""

    def __init__(self, class_map: dict[str, int]) -> None:
        """Initialize the converter with a category-to-class-id mapping.

        Args:
            class_map: Mapping from dataset class names to YOLO class IDs.
        """
        self.class_map = class_map

    def convert_bbox(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        img_w: int = 1280,
        img_h: int = 720,
    ) -> tuple[float, float, float, float]:
        """Convert corner-format bounding boxes to normalized YOLO format.

        Args:
            x1: Left coordinate of the bounding box.
            y1: Top coordinate of the bounding box.
            x2: Right coordinate of the bounding box.
            y2: Bottom coordinate of the bounding box.
            img_w: Image width in pixels.
            img_h: Image height in pixels.

        Returns:
            A tuple of normalized `(x_center, y_center, width, height)`.
        """
        x_center = ((x1 + x2) / 2.0) / img_w
        y_center = ((y1 + y2) / 2.0) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h

        return x_center, y_center, width, height

    def convert_df(self, df: pd.DataFrame, output_dir: str | Path) -> None:
        """Convert a DataFrame of annotations into YOLO label text files.

        Args:
            df: DataFrame containing object annotations grouped by image.
            output_dir: Directory where YOLO label files will be written.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for image_name, group in df.groupby("image_name"):
            lines: list[str] = []

            for row in group.itertuples(index=False):
                category = row.category
                if category not in self.class_map:
                    continue

                class_id = self.class_map[category]
                x_center, y_center, width, height = self.convert_bbox(
                    row.x1,
                    row.y1,
                    row.x2,
                    row.y2,
                )

                lines.append(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}"
                )

            label_file = output_path / f"{Path(image_name).stem}.txt"
            with open(label_file, "w", encoding="utf-8") as file:
                file.write("\n".join(lines))

    def convert_csv(self, csv_path: str | Path, output_dir: str | Path) -> None:
        """Load annotations from CSV and convert them into YOLO label files.

        Args:
            csv_path: Path to the input CSV file.
            output_dir: Directory where YOLO label files will be written.
        """
        df = pd.read_csv(csv_path)
        self.convert_df(df=df, output_dir=output_dir)
