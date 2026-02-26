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
    model_name: str = "Qwen/Qwen3-Embedding-4B"
    device: str = "cuda:0"
    batch_size: int = 16


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
    source: str = "openalex"  # "openalex" or "semantic_scholar"
    threshold: float = 0.2
    semantic_scholar_api_key: Optional[str] = None
    openalex_api_key: Optional[str] = None
    openalex_email: Optional[str] = None
    rate_limit_rps: float = 10.0
    max_retries: int = 3
    cache_max_age_days: int = 30
    expected_citations: dict[int, int] = field(default_factory=lambda: {
        0: 5, 1: 15, 2: 40, 3: 80, 4: 120, 5: 160
    })
    expected_citations_slope: int = 30


@dataclass
class SectionExtractionConfig:
    max_concurrent: int = 20
    max_text_chars: int = 100000
    temperature: float = 0.3
    max_tokens: int = 1024


@dataclass
class KMeansClusteringConfig:
    n_clusters: int = 10
    n_init: int = 10
    random_state: int = 42
    enabled_views: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    weak_member_percentile: float = 90.0
    pca_components: int = 50  # PCA dim reduction before K-Means (0 = disabled)


@dataclass
class HIndexConfig:
    enabled: bool = True
    max_hindex_baseline: int = 50
    default_score: float = 0.3
    cache_max_age_days: int = 30
    rate_limit_rps: float = 1.0  # Semantic Scholar free tier: ~1 req/sec


@dataclass
class Config:
    """Top-level configuration container."""
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    section_extraction: SectionExtractionConfig = field(default_factory=SectionExtractionConfig)
    kmeans: KMeansClusteringConfig = field(default_factory=KMeansClusteringConfig)
    hindex: HIndexConfig = field(default_factory=HIndexConfig)
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
        section_extraction=_dict_to_dataclass(SectionExtractionConfig, raw.get("section_extraction")),
        kmeans=_dict_to_dataclass(KMeansClusteringConfig, raw.get("kmeans")),
        hindex=_dict_to_dataclass(HIndexConfig, raw.get("hindex")),
        affiliation=_dict_to_dataclass(AffiliationConfig, raw.get("affiliation")),
        citation=_dict_to_dataclass(CitationConfig, raw.get("citation")),
    )

    logger.info("Loaded config from %s", config_path)
    return config
