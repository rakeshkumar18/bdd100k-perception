"""
Utilities for loading training configuration.
"""

from pathlib import Path

import yaml

from src.training.config import TrainingConfig


def load_training_config() -> TrainingConfig:
    """
    Load training configuration from YAML files.

    Returns:
        Parsed TrainingConfig instance.
    """

    with open(
        Path("configs/model.yaml"),
        encoding="utf-8",
    ) as file:
        model_cfg = yaml.safe_load(file)

    with open(
        Path("configs/training.yaml"),
        encoding="utf-8",
    ) as file:
        train_cfg = yaml.safe_load(file)

    return TrainingConfig(
        # Model
        model_name=model_cfg["model"]["name"],
        data_yaml="configs/yolo_dataset.yaml",

        # Training
        epochs=train_cfg["training"]["epochs"],
        batch=train_cfg["training"]["batch"],
        imgsz=train_cfg["training"]["imgsz"],
        fraction=train_cfg["training"]["fraction"],
        # Hardware
        device=train_cfg["hardware"]["device"],
        workers=train_cfg["hardware"]["workers"],
        cache=train_cfg["hardware"]["cache"],

        # Reproducibility
        seed=train_cfg["reproducibility"]["seed"],

        # Experiment
        project=train_cfg["experiment"]["project"],
        run_name=train_cfg["experiment"]["run_name"],
        experiment_name=train_cfg["experiment"]["experiment_name"],

        # Augmentation
        hsv_h=train_cfg["augmentation"]["hsv_h"],
        hsv_s=train_cfg["augmentation"]["hsv_s"],
        hsv_v=train_cfg["augmentation"]["hsv_v"],

        degrees=train_cfg["augmentation"]["degrees"],
        translate=train_cfg["augmentation"]["translate"],
        scale=train_cfg["augmentation"]["scale"],
        fliplr=train_cfg["augmentation"]["fliplr"],
    )