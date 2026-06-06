"""Build pandas DataFrames from scene-level BDD100K annotations."""

from pathlib import Path
from typing import List

import pandas as pd

from src.ingestion.schema import SceneAnnotation


class DataFrameBuilder:
    """Convert scene annotations into an object-level pandas DataFrame."""

    def build(self, scenes: List[SceneAnnotation]) -> pd.DataFrame:
        """Build a DataFrame containing one row per annotated object.

        Args:
            scenes: List of parsed scene annotations.

        Returns:
            A DataFrame containing valid object annotations and derived fields.
        """
        rows = []
        invalid_rows = []

        for scene in scenes:
            for obj in scene.objects:
                width = obj.bbox.width
                height = obj.bbox.height

                is_valid = width > 0 and height > 0

                row = {
                    "image_name": scene.image_name,
                    "weather": scene.weather,
                    "scene": scene.scene,
                    "timeofday": scene.timeofday,
                    "category": obj.category,
                    "x1": obj.bbox.x1,
                    "y1": obj.bbox.y1,
                    "x2": obj.bbox.x2,
                    "y2": obj.bbox.y2,
                    "width": width,
                    "height": height,
                    "occluded": obj.occluded,
                    "truncated": obj.truncated,
                    "is_valid_box": is_valid,
                }

                if not is_valid:
                    invalid_rows.append(row)
                    continue

                rows.append(row)

        df = pd.DataFrame(rows)

        if not df.empty:
            df["area"] = df["width"] * df["height"]
            df["aspect_ratio"] = df["width"] / df["height"]

        if invalid_rows:
            invalid_df = pd.DataFrame(invalid_rows)
            print(f"[WARN] Invalid boxes found: {len(invalid_rows)}")
            invalid_df.to_csv("outputs/reports/invalid_boxes.csv", index=True)

        return df

    def save_csv(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save a DataFrame to a CSV file.

        Args:
            df: DataFrame to save.
            output_path: Output CSV file path.
        """
        df.to_csv(output_path, index=False)
