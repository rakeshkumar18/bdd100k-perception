"""
Configuration Manager
"""

from pathlib import Path
import os
import yaml

from dotenv import load_dotenv


class ConfigManager:

    def __init__(self):

        self.project_root = (
            Path(__file__).resolve().parents[2]
        )

        env_path = self.project_root / ".env"

        load_dotenv(env_path)

        config_path = (
            self.project_root
            / "configs"
            / "dataset.yaml"
        )

        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def get(self):

        return self.config

    @property
    def dataset_root(self):

        root = os.getenv("BDD100K_ROOT")

        if root:
            return root

        raise ValueError(
            "BDD100K_ROOT not found in .env"
        )