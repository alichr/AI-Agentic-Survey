"""YAML configuration loading and validation."""

import logging
from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    papers_dir: str = "papers"
    seed_papers_dir: str = "seed_papers"
    seed_papers_config: str = "config/seed_papers.yaml"
    output_dir: str = "output"
    cache_dir: str = "cache"
    log_level: str = "INFO"


@dataclass
class VLLMConfig:
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    base_url: str = "http://localhost:8000/v1"
    tensor_parallel_size: int = 2
    max_model_len: int = 64000
    gpu_memory_utilization: float = 0.90


@dataclass
class EmbeddingConfig:
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str = "cuda:0"
    batch_size: int = 32


@dataclass
class ClusteringConfig:
    n_clusters: int = 10
    method: str = "gmm"
    dimensionality_reduction: bool = False
    reduced_dim: int = 128


@dataclass
class RelevanceConfig:
    enabled: bool = True
    threshold: float = 0.3


@dataclass
class AffiliationConfig:
    enabled: bool = True
    threshold: float = 0.3
    university_rankings_path: str = "config/university_rankings.csv"
    company_tiers_path: str = "config/company_tiers.yaml"
    first_author_weight: float = 0.5
    last_author_weight: float = 0.5


@dataclass
class CitationConfig:
    enabled: bool = True
    threshold: float = 0.2
    semantic_scholar_api_key: Optional[str] = None
    rate_limit_rps: float = 5.0
    max_retries: int = 3
    cache_max_age_days: int = 30
    expected_citations: dict[int, int] = field(default_factory=lambda: {
        0: 5, 1: 15, 2: 40, 3: 80, 4: 120, 5: 160
    })
    expected_citations_slope: int = 30


@dataclass
class Config:
    """Top-level configuration container."""
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    relevance: RelevanceConfig = field(default_factory=RelevanceConfig)
    affiliation: AffiliationConfig = field(default_factory=AffiliationConfig)
    citation: CitationConfig = field(default_factory=CitationConfig)


def _dict_to_dataclass(cls, data: dict):
    """Convert a dict to a dataclass, ignoring unknown keys."""
    if data is None:
        return cls()
    valid_fields = {f.name for f in dataclass_fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)


def load_config(config_path: str) -> Config:
    """Load and validate configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated Config object with all sections populated.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file %s not found, using defaults", config_path)
        return Config()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    config = Config(
        pipeline=_dict_to_dataclass(PipelineConfig, raw.get("pipeline")),
        vllm=_dict_to_dataclass(VLLMConfig, raw.get("vllm")),
        embedding=_dict_to_dataclass(EmbeddingConfig, raw.get("embedding")),
        clustering=_dict_to_dataclass(ClusteringConfig, raw.get("clustering")),
        relevance=_dict_to_dataclass(RelevanceConfig, raw.get("relevance")),
        affiliation=_dict_to_dataclass(AffiliationConfig, raw.get("affiliation")),
        citation=_dict_to_dataclass(CitationConfig, raw.get("citation")),
    )

    logger.info("Loaded config from %s", config_path)
    return config


def load_seed_papers_config(config_path: str) -> dict[str, str]:
    """Load seed paper -> cluster label mapping from YAML.

    Args:
        config_path: Path to seed_papers.yaml.

    Returns:
        Dict mapping filename to cluster label.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("Seed papers config %s not found", config_path)
        return {}

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return raw.get("seed_papers", {}) or {}
