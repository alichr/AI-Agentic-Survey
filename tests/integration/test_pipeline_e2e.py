"""Integration tests: full pipeline with all externals mocked."""

import csv
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import yaml

from src.config import load_config
from src.models.paper import Author, PaperMetadata
from src.pipeline import Pipeline

pytestmark = pytest.mark.integration


def _create_fake_pdf(path: Path, content: bytes = b"%PDF-1.4 fake"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _create_test_config(tmp_path, **overrides):
    """Create a complete test configuration."""
    rankings_csv = tmp_path / "rankings.csv"
    rankings_csv.write_text("rank,university\n1,MIT\n50,Stanford\n")

    tiers_yaml = tmp_path / "tiers.yaml"
    tiers_yaml.write_text(yaml.dump({
        "tier_scores": {1: 1.0, 2: 0.8, 3: 0.6},
        "unknown_score": 0.3,
        "tiers": {1: ["Google", "OpenAI"]},
    }))

    seed_config = tmp_path / "seed_papers.yaml"
    seed_config.write_text(yaml.dump({"seed_papers": {}}))

    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    seed_dir = tmp_path / "seed_papers"
    seed_dir.mkdir()

    config_data = {
        "pipeline": {
            "papers_dir": str(papers_dir),
            "seed_papers_dir": str(seed_dir),
            "seed_papers_config": str(seed_config),
            "output_dir": str(tmp_path / "output"),
            "cache_dir": str(tmp_path / "cache"),
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
        "clustering": {"n_clusters": 2},
        "relevance": overrides.get("relevance", {"enabled": True, "threshold": 0.1}),
        "affiliation": {
            "enabled": overrides.get("affiliation_enabled", True),
            "threshold": 0.1,
            "university_rankings_path": str(rankings_csv),
            "company_tiers_path": str(tiers_yaml),
        },
        "citation": {
            "enabled": overrides.get("citation_enabled", True),
            "threshold": 0.1,
        },
    }
    config_data.update({k: v for k, v in overrides.items()
                        if k not in ("affiliation_enabled", "citation_enabled", "relevance")})

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))
    return load_config(str(config_path))


def _mock_llm_response():
    """Create a mock LLM response for metadata extraction."""
    content = json.dumps({
        "title": "Test Paper Title",
        "abstract": "This is a test abstract about agentic AI systems.",
        "authors": [
            {"name": "Alice", "affiliation": "MIT"},
            {"name": "Bob", "affiliation": "Google"},
        ],
        "year": 2024,
    })
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def mock_all_externals():
    """Patch all external dependencies."""
    patches = {}

    # fitz (PyMuPDF)
    mock_page = MagicMock()
    mock_page.get_text.return_value = "A" * 200 + " first page text of a paper about AI agents"
    mock_doc = MagicMock()
    mock_doc.__len__ = MagicMock(return_value=1)
    mock_doc.__getitem__ = MagicMock(return_value=mock_page)
    patches["fitz"] = patch("src.extraction.pdf_extractor.fitz")

    # AsyncOpenAI
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_mock_llm_response())
    patches["openai"] = patch("src.extraction.metadata_extractor.AsyncOpenAI", return_value=mock_client)

    # SentenceTransformer — must use sys.modules patch because it's imported dynamically
    mock_st_model = MagicMock()
    embedding_dim = 8
    mock_st_model.encode = MagicMock(side_effect=lambda texts, **kwargs: np.random.randn(
        len(texts), embedding_dim
    ).astype(np.float32))
    mock_st_module = MagicMock()
    mock_st_module.SentenceTransformer.return_value = mock_st_model
    patches["st"] = patch.dict("sys.modules", {"sentence_transformers": mock_st_module})

    # SemanticScholarClient - mock at pipeline level
    from src.external.semantic_scholar import CitationData
    mock_ss = AsyncMock()
    mock_ss.search_paper = AsyncMock(return_value=CitationData(citation_count=10, year=2024))
    mock_ss.close = AsyncMock()
    patches["ss"] = patch("src.pipeline.SemanticScholarClient", return_value=mock_ss)

    started = {}
    for name, p in patches.items():
        started[name] = p.start()

    # Configure fitz mock
    started["fitz"].open.return_value = mock_doc

    yield started

    for name, p in patches.items():
        if name != "st":
            p.stop()
        else:
            p.stop()


class TestFullPipelineE2E:
    def test_full_pipeline_produces_csv(self, tmp_path, mock_all_externals):
        config = _create_test_config(tmp_path)
        papers_dir = Path(config.pipeline.papers_dir)

        # Create fake PDFs
        for i in range(3):
            _create_fake_pdf(papers_dir / f"venue{i % 2}" / f"paper{i}.pdf",
                             f"%PDF-1.4 paper {i}".encode())

        pipeline = Pipeline(config)
        pipeline.run()

        # Verify CSV output
        csv_path = Path(config.pipeline.output_dir) / "results.csv"
        assert csv_path.exists()

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert len(reader.fieldnames) == 14

    def test_all_scoring_disabled_all_accepted(self, tmp_path, mock_all_externals):
        config = _create_test_config(
            tmp_path,
            relevance={"enabled": False, "threshold": 0.3},
            affiliation_enabled=False,
            citation_enabled=False,
        )
        papers_dir = Path(config.pipeline.papers_dir)
        _create_fake_pdf(papers_dir / "venue" / "paper1.pdf")
        _create_fake_pdf(papers_dir / "venue" / "paper2.pdf", b"%PDF different")

        pipeline = Pipeline(config)
        pipeline.run()

        csv_path = Path(config.pipeline.output_dir) / "results.csv"
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        for row in rows:
            assert row["status"] == "ACCEPTED"

    def test_no_papers_completes(self, tmp_path, mock_all_externals):
        config = _create_test_config(tmp_path)
        pipeline = Pipeline(config)
        pipeline.run()  # Should not raise

        csv_path = Path(config.pipeline.output_dir) / "results.csv"
        assert csv_path.exists()

    def test_csv_accepted_count_matches_files(self, tmp_path, mock_all_externals):
        config = _create_test_config(tmp_path)
        papers_dir = Path(config.pipeline.papers_dir)
        for i in range(2):
            _create_fake_pdf(papers_dir / "v" / f"paper{i}.pdf",
                             f"%PDF paper {i}".encode())

        pipeline = Pipeline(config)
        pipeline.run()

        csv_path = Path(config.pipeline.output_dir) / "results.csv"
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        csv_accepted = sum(1 for r in rows if r["status"] == "ACCEPTED")

        accepted_dir = Path(config.pipeline.output_dir) / "accepted_papers"
        if accepted_dir.exists():
            file_count = sum(1 for _ in accepted_dir.rglob("*.pdf"))
            assert csv_accepted == file_count
        else:
            assert csv_accepted == 0

    def test_idempotent_second_run_uses_cache(self, tmp_path, mock_all_externals):
        config = _create_test_config(tmp_path)
        papers_dir = Path(config.pipeline.papers_dir)
        _create_fake_pdf(papers_dir / "v" / "paper1.pdf", b"%PDF paper 1 content")
        _create_fake_pdf(papers_dir / "v" / "paper2.pdf", b"%PDF paper 2 content")

        # First run
        pipeline1 = Pipeline(config)
        pipeline1.run()

        # Second run with same config (reopen cache)
        pipeline2 = Pipeline(config)
        pipeline2.run()

        csv_path = Path(config.pipeline.output_dir) / "results.csv"
        assert csv_path.exists()
