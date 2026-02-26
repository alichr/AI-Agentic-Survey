# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Multi-view clustering pipeline that automatically selects relevant papers for an Agentic AI survey. Takes a directory of academic PDFs organized by venue, extracts text/metadata via PyMuPDF + vLLM, splits into 5 sections, embeds each section independently, runs K-Means clustering per section view, and scores papers using h-index/affiliation/citation signals. All raw signals are output to CSV for downstream analysis.

## Commands

```bash
# Install
pip install -r requirements.txt
pip install -r requirements-test.txt

# Run pipeline (requires vLLM server running)
python -m src.main
python -m src.main --n-clusters 5 --no-citation --log-level DEBUG

# Start vLLM server (needs GPU)
bash scripts/start_vllm_server.sh

# Run all tests (excludes live/GPU tests by default via pyproject.toml addopts)
pytest tests/ -v --tb=short

# Run a single test
pytest tests/unit/test_config.py::TestLoadConfig::test_valid_yaml -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run live tests (requires running vLLM server + GPU)
pytest tests/ -m live -v
```

## Architecture

**8-stage pipeline** orchestrated by `src/pipeline.py`:

1. **Discovery** — Scan `papers/<Venue>/*.pdf`, create Paper objects with SHA256 hash
2. **Full-Text Extraction** — PyMuPDF (`src/extraction/pdf_extractor.py`)
3. **Metadata Extraction** — Async vLLM extracts title/abstract/authors/year (`src/extraction/metadata_extractor.py`)
4a. **Section Splitting** — Heuristic regex into 5 sections; LLM fallback for incomplete splits (`src/extraction/section_splitter.py`, `section_extractor.py`)
4b. **Section Summarization** — LLM ~400-word summaries per section for embedding (`src/extraction/section_summarizer.py`)
5. **Multi-View Embedding** — sentence-transformers (Qwen3-Embedding-4B), one embedding per section view (`src/embedding/embedding_model.py`)
6. **K-Means Clustering** — PCA + K-Means independently per view, weak-member detection at 90th percentile centroid distance (`src/embedding/kmeans_clustering.py`)
7. **Scoring** — Three independent scorers, all in `src/scoring/`:
   - `affiliation_scorer.py` — fuzzy-matches authors to QS university rankings + company tiers, per-author scores
   - `citation_scorer.py` — dual-source citation count (OpenAlex primary, Semantic Scholar fallback)
   - `hindex_scorer.py` — max author h-index via OpenAlex
8. **Output** — CSV report with all raw signals (`src/output/csv_writer.py`)

**Key design patterns:**
- **SQLite cache for resumability** (`src/cache/cache_manager.py`): All intermediate results keyed by PDF SHA256 hash. Pipeline can be interrupted and resumed without recomputation.
- **Async API clients** (`src/external/`): Rate-limited aiohttp clients for OpenAlex (10 RPS) and Semantic Scholar (1 RPS) with session reuse.
- **Incomplete paper filtering**: Papers missing section summaries for any enabled view are moved to `pipeline.incomplete_papers` and excluded from clustering but still appear in CSV output.

## Data Model

The central type is `Paper` (`src/models/paper.py`) with nested dataclasses:
- `PaperMetadata` — title, abstract, authors (list of `Author`), year
- `PaperSections` — 5 named sections (title_abstract_conclusion, introduction, related_work, method, experiments)
- `PaperScores` — per-view dicts for cluster_ids, centroid_distances, weak_member; plus quality signals (max_hindex, first/last_author_affiliation_score, citation_count)

`SectionType` enum maps view indices 0-4 to section names.

## Configuration

All parameters in `config/default_config.yaml`, loaded into nested dataclasses by `src/config.py`. CLI flags override config values (see `src/main.py` argparse). Key sections: `pipeline`, `vllm`, `embedding`, `kmeans`, `affiliation`, `citation`, `hindex`.

## Testing

- **pytest** with `asyncio_mode = "auto"` (pyproject.toml)
- Markers: `integration`, `live` (excluded by default), `slow`
- Key fixtures in `tests/conftest.py`: `sample_config`, `make_paper` (factory), `cache` (temp CacheManager), `mock_fitz`, `mock_async_openai`, `synthetic_embeddings`
- Tests mirror `src/` structure in `tests/unit/` and `tests/integration/`; `tests/live/` requires GPU + vLLM server
