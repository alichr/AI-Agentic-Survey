"""Tests for src.config — Config loading, defaults, partial YAML."""

from pathlib import Path

import yaml

from src.config import (
    AffiliationConfig,
    CitationConfig,
    Config,
    EmbeddingConfig,
    HIndexConfig,
    KMeansClusteringConfig,
    PipelineConfig,
    SectionExtractionConfig,
    VLLMConfig,
    _dict_to_dataclass,
    load_config,
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
        result = _dict_to_dataclass(KMeansClusteringConfig, {})
        assert result.n_clusters == 10
        assert result.pca_components == 50


class TestLoadConfig:
    def test_valid_yaml(self, tmp_path):
        config_data = {
            "pipeline": {"papers_dir": "/test/papers", "log_level": "DEBUG"},
            "embedding": {"model_name": "Qwen/Qwen3-Embedding-4B"},
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(config_data))

        config = load_config(str(cfg_path))
        assert config.pipeline.papers_dir == "/test/papers"
        assert config.pipeline.log_level == "DEBUG"
        assert config.embedding.model_name == "Qwen/Qwen3-Embedding-4B"

    def test_citation_config_defaults(self, tmp_path):
        config = load_config(str(tmp_path / "nonexistent.yaml"))
        assert config.citation.source == "openalex"
        assert config.citation.openalex_api_key is None
        assert config.citation.openalex_email is None
        assert config.citation.rate_limit_rps == 10.0

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
            "kmeans": {"n_clusters": 5},
            "affiliation": {"threshold": 0.5},
            "citation": {"threshold": 0.1},
        }
        cfg_path = tmp_path / "full.yaml"
        cfg_path.write_text(yaml.dump(config_data))

        config = load_config(str(cfg_path))
        assert config.pipeline.papers_dir == "/p"
        assert config.vllm.model_name == "m"
        assert config.embedding.device == "cpu"
        assert config.kmeans.n_clusters == 5
        assert config.affiliation.threshold == 0.5
        assert config.citation.threshold == 0.1
