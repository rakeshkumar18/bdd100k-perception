import json
from pathlib import Path

from src.ingestion.schema import (
    BoundingBox,
    ObjectAnnotation,
    SceneAnnotation,
)


class BDDParser:

    def load_directory(
        self,
        label_dir: Path,
        max_files=None
    ):

        scenes = []

        json_files = sorted(
            label_dir.glob("*.json")
        )

        if max_files:
            json_files = json_files[:max_files]

        for file in json_files:

            try:

                scene = self.parse_file(file)

                scenes.append(scene)

            except Exception as e:

                print(
                    f"Skipping {file}: {e}"
                )

        return scenes

    def parse_file(self, file):

        with open(file, "r") as f:

            data = json.load(f)

        return self.parse_scene(data)

    def parse_scene(self, data):

        attrs = data.get(
            "attributes",
            {}
        )

        frame = data["frames"][0]

        objects = []

        for obj in frame["objects"]:

            if "box2d" not in obj:
                continue

            objects.append(
                self.parse_object(obj)
            )

        return SceneAnnotation(
            image_name=data["name"],
            weather=attrs.get(
                "weather",
                "unknown"
            ),
            scene=attrs.get(
                "scene",
                "unknown"
            ),
            timeofday=attrs.get(
                "timeofday",
                "unknown"
            ),
            objects=objects,
        )

    def parse_object(self, obj):

        bbox = obj["box2d"]

        attributes = obj.get(
            "attributes",
            {}
        )

        return ObjectAnnotation(
            category=obj["category"],
            bbox=BoundingBox(
                x1=bbox["x1"],
                y1=bbox["y1"],
                x2=bbox["x2"],
                y2=bbox["y2"],
            ),
            occluded=attributes.get(
                "occluded",
                False
            ),
            truncated=attributes.get(
                "truncated",
                False
            ),
        )