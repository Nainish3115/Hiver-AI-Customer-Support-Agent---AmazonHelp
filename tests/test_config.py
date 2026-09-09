"""
Unit tests for configuration loading and validation.
"""
import pytest
from pathlib import Path
from hiver_agent.config import load_config, AppConfig, ProjectConfig


def test_load_config_default():
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.project.name == "hiver-support-agent"
    assert config.project.target_brand == "AmazonHelp"
    assert Path(config.datasets.twitter.raw_path).name == "customer_support_on_twitter.csv"


def test_load_config_invalid_path():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")
