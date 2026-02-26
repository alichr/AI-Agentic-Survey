"""Shared fixtures for all test modules."""

import json
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import yaml

from src.cache.cache_manager import CacheManager
from src.config import Config, load_config
from src.models.paper import Author, Paper, PaperMetadata, PaperScores

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# University rankings CSV
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_rankings_csv(tmp_path):
    """Write a small university rankings CSV and return its path."""
    csv_path = tmp_path / "rankings.csv"
    shutil.copy(FIXTURES_DIR / "sample_university_rankings.csv", csv_path)
    return str(csv_path)


# ---------------------------------------------------------------------------
# Company tiers YAML
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_tiers_yaml(tmp_path):
    """Write a small company tiers YAML and return its path."""
    yaml_path = tmp_path / "tiers.yaml"
    shutil.copy(FIXTURES_DIR / "sample_company_tiers.yaml", yaml_path)
    return str(yaml_path)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config(tmp_path, sample_rankings_csv, sample_tiers_yaml):
    """Create a test-friendly Config object pointing to temp dirs."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "cache"

    config_data = {
        "pipeline": {
            "papers_dir": str(papers_dir),
            "output_dir": str(output_dir),
            "cache_dir": str(cache_dir),
            "log_level": "WARNING",
        },
        "vllm": {
            "model_name": "test-model",
            "base_url": "http://localhost:9999/v1",
        },
        "embedding": {
            "model_name": "test-embedding",
            "device": "cpu",
            "batch_size": 8,
        },
        "kmeans": {"n_clusters": 3},
        "fusion": {"strategy": "consensus_weighted", "threshold": 0.3},
        "affiliation": {
            "enabled": True,
            "threshold": 0.3,
            "university_rankings_path": sample_rankings_csv,
            "company_tiers_path": sample_tiers_yaml,
        },
        "citation": {"enabled": True, "threshold": 0.2},
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))
    return load_config(str(config_path))


# ---------------------------------------------------------------------------
# Paper factory
# ---------------------------------------------------------------------------

@pytest.fixture
def make_paper(tmp_path):
    """Factory fixture for creating Paper objects with configurable fields."""

    def _make(
        filename="paper.pdf",
        venue="ICML",
        pdf_hash=None,
        first_page_text=None,
        title=None,
        abstract=None,
        authors=None,
        year=None,
        citation_count=None,
    ):
        pdf_path = tmp_path / filename
        if not pdf_path.exists():
            pdf_path.write_bytes(b"%PDF-1.4 fake content " + filename.encode())

        if pdf_hash is None:
            import hashlib
            pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

        metadata = None
        if title is not None:
            if authors is None:
                authors = []
            metadata = PaperMetadata(
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
            )

        scores = PaperScores(
            citation_count=citation_count,
        )

        return Paper(
            pdf_path=pdf_path,
            venue=venue,
            pdf_hash=pdf_hash,
            first_page_text=first_page_text,
            metadata=metadata,
            scores=scores,
        )

    return _make


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path):
    """Fresh CacheManager in a temp directory."""
    cm = CacheManager(str(tmp_path / "cache"))
    yield cm
    cm.close()


# ---------------------------------------------------------------------------
# Synthetic embeddings for clustering tests
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_embeddings():
    """30-point, 3-cluster numpy array with known separation (seed=42)."""
    rng = np.random.RandomState(42)
    centers = np.array([[5, 0], [0, 5], [-5, -5]], dtype=np.float32)
    points = []
    for center in centers:
        cluster_pts = rng.randn(10, 2).astype(np.float32) * 0.3 + center
        points.append(cluster_pts)
    return np.vstack(points)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_fitz():
    """Patch fitz (PyMuPDF) with a configurable mock."""

    def _make(pages_text=None):
        if pages_text is None:
            pages_text = ["First page text content here."]

        mock_doc = MagicMock()
        mock_pages = []
        for text in pages_text:
            page = MagicMock()
            page.get_text.return_value = text
            mock_pages.append(page)

        mock_doc.__len__ = MagicMock(return_value=len(mock_pages))
        mock_doc.__getitem__ = MagicMock(side_effect=lambda i: mock_pages[i])
        mock_doc.close = MagicMock()

        patcher = patch("src.extraction.pdf_extractor.fitz")
        mock_fitz_module = patcher.start()
        mock_fitz_module.open.return_value = mock_doc
        return patcher, mock_fitz_module

    return _make


@pytest.fixture
def mock_async_openai():
    """Patch AsyncOpenAI with a configurable JSON response."""

    def _make(response_json=None):
        if response_json is None:
            response_json = {
                "title": "Test Paper",
                "abstract": "Test abstract.",
                "authors": [{"name": "Test Author", "affiliation": "Test Uni"}],
                "year": 2024,
            }

        mock_message = MagicMock()
        mock_message.content = json.dumps(response_json)

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        patcher = patch("src.extraction.metadata_extractor.AsyncOpenAI", return_value=mock_client)
        mock_cls = patcher.start()
        return patcher, mock_client

    return _make


@pytest.fixture
def sample_llm_response():
    """Load the sample LLM response JSON."""
    with open(FIXTURES_DIR / "sample_llm_response.json") as f:
        return json.load(f)


@pytest.fixture
def sample_first_page_text():
    """Load the sample first page text."""
    return (FIXTURES_DIR / "sample_first_page.txt").read_text()
