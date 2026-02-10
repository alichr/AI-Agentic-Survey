# AI Survey Paper Selection Pipeline

Automatically select relevant papers for a survey on **Agentic AI**. Given a pool of academic PDFs organized by venue, the system scores each paper on three configurable metrics (relevance, affiliation prestige, citation+recency), applies thresholds, and outputs accepted papers with a detailed CSV report.

## Setup

### 1. Create virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

vLLM must be installed separately (requires specific CUDA version):

```bash
pip install vllm
```

### 2. Prepare input data

Place your candidate PDFs in `papers/`, organized by venue subfolders:

```
papers/
├── NeurIPS/
│   ├── paper1.pdf
│   └── paper2.pdf
├── ICML/
│   └── paper3.pdf
└── AAAI/
    └── paper4.pdf
```

Place seed/reference PDFs in `seed_papers/` and configure their topic labels in `config/seed_papers.yaml`:

```yaml
seed_papers:
  "toolformer.pdf": "Tool Use"
  "react_synergizing.pdf": "Reasoning and Acting"
  "autogpt.pdf": "Autonomous Agents"
```

### 3. Start vLLM server

```bash
bash scripts/start_vllm_server.sh
```

Or with custom settings:

```bash
VLLM_MODEL=Qwen/Qwen3-30B-A3B VLLM_TP_SIZE=3 bash scripts/start_vllm_server.sh
```

### 4. Run the pipeline

```bash
source venv/bin/activate
python -m src.main
```

With options:

```bash
python -m src.main --config config/default_config.yaml --log-level DEBUG
python -m src.main --no-citation          # Skip citation scoring
python -m src.main --papers-dir /path/to/pdfs --output-dir /path/to/output
```

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| 1. Discovery | Scan `papers/` and `seed_papers/` for PDFs |
| 2. PDF Extraction | Extract first-page text using PyMuPDF |
| 3. Metadata Extraction | LLM extracts title, abstract, authors, year via vLLM |
| 4. Embedding & Clustering | Embed with Qwen3-Embedding-0.6B, cluster with GMM, propagate seed labels |
| 5. Affiliation Scoring | Score authors by university rank (QS) and company tier |
| 6. Citation Scoring | Fetch citations from Semantic Scholar, score by citations/expected_by_age |
| 7. Aggregation | Accept papers passing ALL active thresholds (AND logic) |
| 8. Output | Write CSV report, copy accepted PDFs organized by cluster |

## Output

- `output/results.csv` — Full results with all scores and accept/reject status
- `output/accepted_papers/<cluster_label>/` — Accepted PDFs organized by topic

### CSV Columns

`pdf_path, venue, title, authors, first_author_affiliation, last_author_affiliation, year, relevance_score, cluster_id, cluster_label, affiliation_score, citation_score, citation_count, status`

## Configuration

All settings are in `config/default_config.yaml`. Key parameters:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `relevance` | `threshold` | 0.3 | Minimum GMM cluster probability |
| `affiliation` | `threshold` | 0.3 | Minimum affiliation score |
| `citation` | `threshold` | 0.2 | Minimum citation score |
| `clustering` | `n_clusters` | 10 | Number of GMM components |
| `embedding` | `model_name` | Qwen/Qwen3-Embedding-0.6B | Embedding model |
| `vllm` | `model_name` | Qwen/Qwen3-30B-A3B | LLM for metadata extraction |

Each scoring dimension can be disabled with `enabled: false`.

## Caching & Resumability

The pipeline uses SQLite (`cache/pipeline_cache.db`) to cache:
- Extracted PDF text (keyed by SHA256 of PDF content)
- LLM-extracted metadata
- Embedding vectors
- Semantic Scholar citation data (expires after 30 days)

If the pipeline is interrupted, re-running it skips already-processed papers.

## GPU Requirements

Tested with 3x NVIDIA A6000 (48GB each):
- **vLLM (Qwen3-30B-A3B)**: ~20GB/GPU with TP=3
- **Qwen3-Embedding-0.6B**: ~1.2GB on cuda:0

## Testing

Install test dependencies:

```bash
pip install -r requirements-test.txt
```

Run all unit and integration tests (excludes live/GPU tests by default):

```bash
pytest tests/ -v --tb=short
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

Run only live tests (requires GPU machine with vLLM running):

```bash
pytest tests/ -m live -v
```

**Test suite**: 266 tests across 20 test files (17 unit, 3 integration, 2 live).

## Project Structure

```
src/
├── models/paper.py              # Data models (Paper, Author, PaperMetadata, PaperScores)
├── config.py                    # YAML config loading and dataclass definitions
├── pipeline.py                  # Pipeline orchestrator (8 stages)
├── main.py                      # CLI entry point
├── cache/cache_manager.py       # SQLite cache for pipeline resumability
├── extraction/
│   ├── pdf_extractor.py         # PDF text extraction (PyMuPDF)
│   └── metadata_extractor.py    # Async LLM metadata extraction (vLLM)
├── embedding/
│   ├── embedding_model.py       # Sentence-transformers wrapper
│   └── clustering.py            # GMM clustering with seed label propagation
├── scoring/
│   ├── relevance_scorer.py      # GMM probability-based relevance
│   ├── affiliation_scorer.py    # University ranking + company tier scoring
│   ├── citation_scorer.py       # Citation count vs expected-by-age scoring
│   └── aggregator.py            # AND-logic threshold acceptance
├── external/
│   ├── semantic_scholar.py      # Async Semantic Scholar API client
│   ├── university_rankings.py   # QS rankings CSV loader with fuzzy matching
│   └── company_tiers.py         # Company tier YAML loader with fuzzy matching
└── output/
    ├── csv_writer.py            # CSV report generation
    └── file_organizer.py        # Accepted paper file organization
```

## Custom Rankings

- **Universities**: Edit `config/university_rankings.csv` or generate from QS Excel:
  ```bash
  python scripts/download_rankings.py --input qs_rankings.xlsx --output config/university_rankings.csv
  ```
- **Companies**: Edit `config/company_tiers.yaml` to add/remove companies and adjust tier assignments.
