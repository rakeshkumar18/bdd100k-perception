"""
Build DataFrame from SceneAnnotations
"""

from typing import List

import pandas as pd

from src.ingestion.schema import SceneAnnotation


class DataFrameBuilder:

    def build(
        self,
        scenes: List[SceneAnnotation]
    ) -> pd.DataFrame:

        rows = []

        for scene in scenes:

            for obj in scene.objects:

                rows.append(
                    {
                        "image_name": scene.image_name,
                        "weather": scene.weather,
                        "scene": scene.scene,
                        "timeofday": scene.timeofday,

                        "category": obj.category,

                        "x1": obj.bbox.x1,
                        "y1": obj.bbox.y1,
                        "x2": obj.bbox.x2,
                        "y2": obj.bbox.y2,

                        "width": obj.bbox.width,
                        "height": obj.bbox.height,
                        "area": obj.bbox.area,
                        "aspect_ratio": obj.bbox.aspect_ratio,

                        "occluded": obj.occluded,
                        "truncated": obj.truncated,
                    }
                )

        return pd.DataFrame(rows)

    def save_csv(
        self,
        df,
        output_path
    ):

        df.to_csv(
            output_path,
            index=False
        )