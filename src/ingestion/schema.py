"""
Data models for BDD100K Object Detection Dataset.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class BoundingBox:

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height

    @property
    def aspect_ratio(self):
        if self.height == 0:
            return 0
        return self.width / self.height


@dataclass
class ObjectAnnotation:

    category: str

    bbox: BoundingBox

    occluded: bool

    truncated: bool


@dataclass
class SceneAnnotation:

    image_name: str

    weather: str

    scene: str

    timeofday: str

    objects: List[ObjectAnnotation]
