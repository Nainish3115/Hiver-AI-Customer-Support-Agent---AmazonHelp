"""
Configuration manager for Hiver Support Agent.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    raw_path: str
    chunk_size: int = 250000
    date_format: str = "%a %b %d %H:%M:%S %z %Y"


class Banking77Config(BaseModel):
    train_path: str
    test_path: str


class DatasetsConfig(BaseModel):
    twitter: DatasetConfig
    banking77: Banking77Config


class ProjectConfig(BaseModel):
    name: str = "hiver-support-agent"
    version: str = "0.1.0"
    target_brand: str = "AmazonHelp"


class PreprocessingConfig(BaseModel):
    sample_size: int = 10000
    min_thread_depth: int = 2
    max_thread_depth: int = 10


class AppConfig(BaseModel):
    project: ProjectConfig
    datasets: DatasetsConfig
    preprocessing: PreprocessingConfig


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load and parse configuration YAML into Pydantic models.
    """
    if config_path is None:
        # Default to configs/config.yaml relative to project root
        root_dir = Path(__file__).resolve().parent.parent.parent
        config_path = str(root_dir / "configs" / "config.yaml")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return AppConfig(**data)
