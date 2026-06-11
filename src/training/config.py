"""
Training configuration dataclasses.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Configuration used for YOLO training.
    """

    # Model
    model_name: str
    data_yaml: str

    # Training
    epochs: int
    batch: int
    imgsz: int
    fraction: float
    # Hardware
    device: str
    workers: int
    cache: bool

    # Reproducibility
    seed: int

    # Experiment
    project: str
    run_name: str
    experiment_name: str

    # Augmentation
    hsv_h: float
    hsv_s: float
    hsv_v: float

    degrees: float
    translate: float
    scale: float
    fliplr: float