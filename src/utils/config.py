"""Manage dataset configuration and environment-based project paths."""

from pathlib import Path
import yaml
from dotenv import load_dotenv


class ConfigManager:
    """Load project configuration and expose dataset-related paths."""

    def __init__(self) -> None:
        # Project root (stable anchor)
        self.project_root = Path(__file__).resolve().parents[2]

        # Load env ONLY for non-path configs (API keys, etc.)
        load_dotenv(self.project_root / ".env")

        # Load YAML config
        config_path = self.project_root / "configs" / "dataset.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Resolve dataset root ONCE here (canonical resolution)
        self._dataset_root = (self.project_root / self.config["dataset_root"]).resolve()

    @property
    def dataset_root(self) -> Path:
        """Absolute resolved dataset root."""
        return self._dataset_root
