"""Tests for src.config — Config loading, defaults, partial YAML."""

from pathlib import Path

import yaml

from src.config import (
    AffiliationConfig,
    CitationConfig,
    ClusteringConfig,
    Config,
    EmbeddingConfig,
    PipelineConfig,
    RelevanceConfig,
    VLLMConfig,
    _dict_to_dataclass,
    load_config,
    load_seed_papers_config,
)


class TestDictToDataclass:
    def test_valid_data(self):
        result = _dict_to_dataclass(PipelineConfig, {"papers_dir": "/my/dir", "log_level": "DEBUG"})
        assert result.papers_dir == "/my/dir"
        assert result.log_level == "DEBUG"

    def test_none_input_returns_defaults(self):
        result = _dict_to_dataclass(PipelineConfig, None)
        assert result.papers_dir == "papers"
        assert result.log_level == "INFO"

    def test_unknown_keys_ignored(self):
        result = _dict_to_dataclass(PipelineConfig, {"papers_dir": "/x", "unknown_key": 42})
        assert result.papers_dir == "/x"
        assert not hasattr(result, "unknown_key")

    def test_empty_dict_returns_defaults(self):
        result = _dict_to_dataclass(RelevanceConfig, {})
        assert result.enabled is True
        assert result.threshold == 0.3


class TestLoadConfig:
    def test_valid_yaml(self, tmp_path):
        config_data = {
            "pipeline": {"papers_dir": "/test/papers", "log_level": "DEBUG"},
            "relevance": {"threshold": 0.5},
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config_data))

        config = load_config(str(cfg_path))
        assert config.pipeline.papers_dir == "/test/papers"
        assert config.pipeline.log_level == "DEBUG"
        assert config.relevance.threshold == 0.5
        # Other sections should have defaults
        assert config.embedding.model_name == "Qwen/Qwen3-Embedding-0.6B"

    def test_missing_file_returns_defaults(self, tmp_path):
        config = load_config(str(tmp_path / "nonexistent.yaml"))
        assert isinstance(config, Config)
        assert config.pipeline.papers_dir == "papers"
        assert config.vllm.model_name == "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

    def test_partial_yaml_missing_sections(self, tmp_path):
        cfg_path = tmp_path / "partial.yaml"
        cfg_path.write_text(yaml.dump({"pipeline": {"papers_dir": "/x"}}))

        config = load_config(str(cfg_path))
        assert config.pipeline.papers_dir == "/x"
        assert config.relevance.enabled is True  # default
        assert config.citation.threshold == 0.2  # default

    def test_empty_yaml(self, tmp_path):
        cfg_path = tmp_path / "empty.yaml"
        cfg_path.write_text("")

        config = load_config(str(cfg_path))
        assert isinstance(config, Config)
        assert config.pipeline.papers_dir == "papers"

    def test_unknown_top_level_keys(self, tmp_path):
        cfg_path = tmp_path / "extra.yaml"
        cfg_path.write_text(yaml.dump({"pipeline": {"papers_dir": "/x"}, "unknown_section": {"a": 1}}))

        config = load_config(str(cfg_path))
        assert config.pipeline.papers_dir == "/x"

    def test_all_sections_populated(self, tmp_path):
        config_data = {
            "pipeline": {"papers_dir": "/p"},
            "vllm": {"model_name": "m"},
            "embedding": {"device": "cpu"},
            "clustering": {"n_clusters": 5},
            "relevance": {"threshold": 0.4},
            "affiliation": {"threshold": 0.5},
            "citation": {"threshold": 0.1},
        }
        cfg_path = tmp_path / "full.yaml"
        cfg_path.write_text(yaml.dump(config_data))

        config = load_config(str(cfg_path))
        assert config.pipeline.papers_dir == "/p"
        assert config.vllm.model_name == "m"
        assert config.embedding.device == "cpu"
        assert config.clustering.n_clusters == 5
        assert config.relevance.threshold == 0.4
        assert config.affiliation.threshold == 0.5
        assert config.citation.threshold == 0.1


class TestLoadSeedPapersConfig:
    def test_valid_seed_config(self, tmp_path):
        data = {"seed_papers": {"paper1.pdf": "Topic A", "paper2.pdf": "Topic B"}}
        path = tmp_path / "seeds.yaml"
        path.write_text(yaml.dump(data))

        result = load_seed_papers_config(str(path))
        assert result == {"paper1.pdf": "Topic A", "paper2.pdf": "Topic B"}

    def test_missing_file_returns_empty(self, tmp_path):
        result = load_seed_papers_config(str(tmp_path / "missing.yaml"))
        assert result == {}

    def test_empty_file_returns_empty(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        result = load_seed_papers_config(str(path))
        assert result == {}

    def test_no_seed_papers_key(self, tmp_path):
        path = tmp_path / "other.yaml"
        path.write_text(yaml.dump({"other_key": "value"}))
        result = load_seed_papers_config(str(path))
        assert result == {}

    def test_null_seed_papers_key(self, tmp_path):
        path = tmp_path / "null.yaml"
        path.write_text(yaml.dump({"seed_papers": None}))
        result = load_seed_papers_config(str(path))
        assert result == {}
