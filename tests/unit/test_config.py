"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from idfs.config import ProjectConfig, config_hash, load_config


class TestLoadConfig:
    def test_default_config(self) -> None:
        cfg = load_config()
        assert isinstance(cfg, ProjectConfig)
        assert cfg.seed == 42
        assert cfg.data.source == "synthetic"
        assert cfg.forecast.horizon == 14
        assert cfg.split.n_folds == 3

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = {
            "seed": 99,
            "data": {"source": "favorita", "synthetic_n_series": 100},
            "forecast": {"horizon": 7},
        }
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(yaml_content, f)

        cfg = load_config(config_file)
        assert cfg.seed == 99
        assert cfg.data.source == "favorita"
        assert cfg.data.synthetic_n_series == 100
        assert cfg.forecast.horizon == 7
        # Unchanged defaults
        assert cfg.split.n_folds == 3

    def test_missing_file_returns_defaults(self) -> None:
        cfg = load_config("/nonexistent/path.yaml")
        assert cfg.seed == 42

    def test_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        cfg = load_config(config_file)
        assert cfg.seed == 42


class TestConfigHash:
    def test_deterministic(self) -> None:
        cfg = load_config()
        h1 = config_hash(cfg)
        h2 = config_hash(cfg)
        assert h1 == h2
        assert len(h1) == 8

    def test_different_configs_different_hash(self) -> None:
        cfg1 = ProjectConfig(seed=42)
        cfg2 = ProjectConfig(seed=99)
        assert config_hash(cfg1) != config_hash(cfg2)
