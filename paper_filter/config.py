from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class PipelineConfig:
    output_dir: str = "output"
    cache_dir: str = "cache"
    relevance_threshold: float = 5.0
    max_classify_concurrent: int = 20
    log_level: str = "INFO"


@dataclass
class VLLMConfig:
    base_url: str = "http://localhost:8001/v1"
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    temperature: float = 0.3
    max_tokens: int = 512


@dataclass
class DownloadConfig:
    max_concurrent: int = 5
    max_retries: int = 3
    retry_backoff: float = 2.0
    timeout: int = 60


@dataclass
class Config:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)


def _update_dataclass(instance, overrides: dict) -> None:
    valid_fields = {f.name for f in fields(instance)}
    for key, value in overrides.items():
        if key in valid_fields:
            setattr(instance, key, value)


def load_config(path: Optional[str] = None) -> Config:
    config = Config()
    if path is None:
        default_path = Path(__file__).parent.parent / "config" / "default_config.yaml"
        if default_path.exists():
            path = str(default_path)
    if path and Path(path).exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if "pipeline" in raw:
            _update_dataclass(config.pipeline, raw["pipeline"])
        if "vllm" in raw:
            _update_dataclass(config.vllm, raw["vllm"])
        if "download" in raw:
            _update_dataclass(config.download, raw["download"])
    return config
