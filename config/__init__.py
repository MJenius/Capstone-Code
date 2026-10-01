"""
Configuration loader and validator for the Watermarking Pipeline.
"""
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default.yaml"


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load pipeline configuration from a YAML file.
    Falls back to config/default.yaml if none is provided.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_base_dir() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent
