from pathlib import Path
import yaml


class ConfigManager:

    _instance = None

    def __new__(cls):

        if cls._instance is None:

            cls._instance = super().__new__(cls)

            config_path = (
                Path(__file__)
                .parents[2]
                / "configs"
                / "dataset.yaml"
            )

            with open(config_path) as f:
                cls._instance.config = yaml.safe_load(f)

        return cls._instance

    def get(self):
        return self.config