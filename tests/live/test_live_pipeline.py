"""Live tests: full pipeline with real vLLM server + real embedding model.

Run with: pytest tests/live/ -m live -v
Requires:
  - GPU with the embedding model accessible
  - vLLM server running at http://localhost:8000/v1
"""

import csv
from pathlib import Path

import pytest
import yaml

from src.config import load_config
from src.pipeline import Pipeline

pytestmark = [pytest.mark.live, pytest.mark.slow]


@pytest.fixture
def live_config(tmp_path):
    """Create a config for live testing with real models."""
    rankings_csv = tmp_path / "rankings.csv"
    rankings_csv.write_text("rank,university\n1,MIT\n50,Stanford\n100,ETH Zurich\n")

    tiers_yaml = tmp_path / "tiers.yaml"
    tiers_yaml.write_text(yaml.dump({
        "tier_scores": {1: 1.0, 2: 0.8, 3: 0.6},
        "unknown_score": 0.3,
        "tiers": {1: ["Google", "OpenAI", "Meta"]},
    }))

    seed_config = tmp_path / "seed_papers.yaml"
    seed_config.write_text(yaml.dump({"seed_papers": {}}))

    config_data = {
        "pipeline": {
            "papers_dir": str(tmp_path / "papers"),
            "seed_papers_dir": str(tmp_path / "seed_papers"),
            "seed_papers_config": str(seed_config),
            "output_dir": str(tmp_path / "output"),
            "cache_dir": str(tmp_path / "cache"),
            "log_level": "INFO",
        },
        "vllm": {
            "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
            "base_url": "http://localhost:8000/v1",
        },
        "embedding": {
            "model_name": "Qwen/Qwen3-Embedding-0.6B",
            "device": "cuda:0",
            "batch_size": 8,
        },
        "clustering": {"n_clusters": 2},
        "relevance": {"enabled": True, "threshold": 0.1},
        "affiliation": {
            "enabled": True,
            "threshold": 0.1,
            "university_rankings_path": str(rankings_csv),
            "company_tiers_path": str(tiers_yaml),
        },
        "citation": {
            "enabled": False,  # Don't hit Semantic Scholar in tests
            "threshold": 0.2,
        },
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))

    # Create papers directory
    (tmp_path / "papers").mkdir()
    (tmp_path / "seed_papers").mkdir()

    return load_config(str(config_path))


class TestLivePipeline:
    def test_full_pipeline_with_real_models(self, live_config, tmp_path):
        """Full pipeline with real vLLM + embedding model."""
        papers_dir = Path(live_config.pipeline.papers_dir)

        # Create a simple test PDF-like file (in practice you'd use real PDFs)
        # For live tests, you could place real PDFs here
        test_pdf = papers_dir / "test_venue" / "test_paper.pdf"
        test_pdf.parent.mkdir(parents=True, exist_ok=True)
        test_pdf.write_bytes(b"%PDF-1.4 " + b"Agentic AI Survey Paper Content " * 50)

        pipeline = Pipeline(live_config)
        pipeline.run()

        csv_path = Path(live_config.pipeline.output_dir) / "results.csv"
        assert csv_path.exists()

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["status"] in ("ACCEPTED", "REJECTED")
        # Scores should be populated (not empty)
        assert rows[0]["relevance_score"] != ""
