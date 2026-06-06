"""Manage dataset configuration and environment-based project paths."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


class ConfigManager:
    """Load project configuration and expose dataset-related paths."""

    def __init__(self) -> None:
        """Initialize the configuration manager from project files and environment."""
        self.project_root = Path(__file__).resolve().parents[2]

        env_path = self.project_root / ".env"
        load_dotenv(env_path)

        config_path = self.project_root / "configs" / "dataset.yaml"
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    def get(self) -> dict:
        """Return the loaded YAML configuration."""
        return self.config

    @property
    def dataset_root(self) -> str:
        """Return the BDD100K dataset root path from the environment.

        Raises:
            ValueError: If the ``BDD100K_ROOT`` variable is not defined.
        """
        root = os.getenv("BDD100K_ROOT")

        if root:
            return root

        raise ValueError("BDD100K_ROOT not found in .env")
