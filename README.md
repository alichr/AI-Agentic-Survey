# AI Survey Paper Selection Pipeline

Automatically select relevant papers for a survey on **Agentic AI**. Given a pool of academic PDFs organized by venue, the system scores each paper on three configurable metrics (relevance, affiliation prestige, citation+recency), applies thresholds, and outputs accepted papers with a detailed CSV report.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install vllm

# 2. Start the LLM server (needs GPU)
bash scripts/start_vllm_server.sh

# 3. Place PDFs in papers/<Venue>/ and seed papers in seed_papers/
#    Configure seed labels in config/seed_papers.yaml

# 4. Run the pipeline
python -m src.main
```

## Prerequisites

- **Python** 3.10+
- **CUDA-capable GPU(s)** with drivers installed
- **vLLM** (installed separately — requires matching CUDA version)
- Internet access for Semantic Scholar citation lookups and HuggingFace model downloads

## Setup

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install vllm
```

### 2. Prepare input data

**Candidate papers** — place PDFs in `papers/`, organized by venue subfolders. The folder name becomes the venue label:

```
papers/
├── NeurIPS/
│   ├── attention_is_all_you_need.pdf
│   └── gpt3.pdf
├── ICLR/
│   ├── lora.pdf
│   └── autogen.pdf
├── NAACL/
│   └── bert.pdf
└── arXiv/
    └── llama.pdf
```

**Seed papers** — place reference PDFs in `seed_papers/` and map them to topic labels in `config/seed_papers.yaml`:

```yaml
seed_papers:
  "survey_of_llm_agents.pdf": "LLM-based Agents"
  "react_reasoning_acting.pdf": "Reasoning and Acting"
  "toolformer.pdf": "Tool Use"
  "generative_agents.pdf": "Multi-Agent Simulation"
  "voyager.pdf": "Embodied Agents"
  "metagpt.pdf": "Software Engineering Agents"
  "reflexion.pdf": "Self-Reflection"
```

Seed papers anchor the GMM clustering — each seed paper's cluster inherits its assigned label. Clusters without seeds get labeled "Miscellaneous-N".

### 3. Start vLLM server

```bash
bash scripts/start_vllm_server.sh
```

This serves `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` (MoE, 30.5B params / 3.3B active, FP8 quantized) with tensor parallelism across 2 GPUs.

Override defaults with environment variables:

```bash
VLLM_TP_SIZE=1 VLLM_MAX_MODEL_LEN=32768 bash scripts/start_vllm_server.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MODEL` | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | Model to serve |
| `VLLM_TP_SIZE` | `2` | Tensor parallel size (must divide 32 attention heads) |
| `VLLM_MAX_MODEL_LEN` | `64000` | Max context length in tokens |
| `VLLM_GPU_UTIL` | `0.90` | GPU memory utilization fraction |
| `VLLM_PORT` | `8000` | Server port |

Wait until you see the server log `Uvicorn running on ...` before running the pipeline.

### 4. Run the pipeline

```bash
python -m src.main
```

**CLI options:**

```bash
python -m src.main --config config/default_config.yaml   # Custom config file
python -m src.main --log-level DEBUG                      # Verbose logging
python -m src.main --no-citation                          # Skip citation scoring
python -m src.main --no-affiliation                       # Skip affiliation scoring
python -m src.main --no-relevance                         # Skip relevance scoring
python -m src.main --papers-dir /path/to/pdfs             # Override papers directory
python -m src.main --output-dir /path/to/output           # Override output directory
python -m src.main --n-clusters 5                          # Override number of GMM clusters
```

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| 1. Discovery | Scan `papers/` and `seed_papers/` for PDFs; detect venue from folder name |
| 2. PDF Extraction | Extract first-page text using PyMuPDF |
| 3. Metadata Extraction | LLM extracts title, abstract, authors, year via vLLM |
| 4. Embedding & Clustering | Embed with Qwen3-Embedding-0.6B, cluster with GMM, propagate seed labels |
| 5. Affiliation Scoring | Score authors by university rank (QS top 200) and company tier |
| 6. Citation Scoring | Fetch citations from Semantic Scholar, score by citations / expected-by-age |
| 7. Aggregation | Accept papers passing ALL active thresholds (AND logic) |
| 8. Output | Write CSV report, copy accepted PDFs organized by cluster |

## Output

After running, the pipeline produces:

- **`output/results.csv`** — Full results with all scores and accept/reject status
- **`output/accepted_papers/<cluster_label>/`** — Accepted PDFs organized by topic cluster

### CSV Columns

| Column | Description |
|--------|-------------|
| `pdf_path` | Path to the PDF file |
| `venue` | Venue detected from folder name (e.g., NeurIPS, ICLR) |
| `title` | Paper title extracted by LLM |
| `authors` | Authors separated by semicolons |
| `first_author_affiliation` | First author's institution |
| `last_author_affiliation` | Last (senior) author's institution |
| `year` | Publication year |
| `relevance_score` | GMM cluster probability (0-1) |
| `cluster_id` | Assigned cluster number |
| `cluster_label` | Cluster topic label (from seed papers) |
| `affiliation_score` | Weighted university/company score (0.3-1.0) |
| `citation_score` | Citation count vs expected-by-age (0-1, capped) |
| `citation_count` | Raw citation count from Semantic Scholar |
| `status` | ACCEPTED or REJECTED |

### Example Output

```
title,venue,affiliation_score,citation_score,cluster_label,status
Attention Is All You Need,NeurIPS,0.6500,1.0000,Miscellaneous-8,ACCEPTED
Chain-of-Thought Prompting...,NeurIPS,1.0000,1.0000,Reasoning and Acting,ACCEPTED
AutoGen: Enabling Next-Gen...,ICLR,0.9000,1.0000,Tool Use,ACCEPTED
```

## Configuration

All settings are in `config/default_config.yaml`:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `vllm` | `model_name` | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | LLM for metadata extraction |
| `vllm` | `tensor_parallel_size` | `2` | Number of GPUs for LLM |
| `vllm` | `max_model_len` | `64000` | Context window (tokens) |
| `embedding` | `model_name` | `Qwen/Qwen3-Embedding-0.6B` | Embedding model |
| `embedding` | `device` | `cuda:2` | GPU for embeddings (use a free GPU) |
| `clustering` | `n_clusters` | `10` | Number of GMM components |
| `relevance` | `threshold` | `0.3` | Minimum cluster probability to accept |
| `affiliation` | `threshold` | `0.3` | Minimum affiliation score to accept |
| `citation` | `threshold` | `0.2` | Minimum citation score to accept |

Each scoring dimension can be independently disabled with `enabled: false` in the config.

## Scoring Details

### Affiliation Scoring

Each author is scored by taking the max of their university rank score and company tier score. The paper's affiliation score is a weighted combination of the first and last author scores.

**University rank tiers** (QS World Rankings 2025, top 200):

| Rank | Score |
|------|-------|
| 1-10 | 1.00 |
| 11-25 | 0.95 |
| 26-50 | 0.90 |
| 51-75 | 0.85 |
| 76-100 | 0.80 |
| 101-150 | 0.70 |
| 151-200 | 0.60 |
| Unranked | 0.40 |

**Company tiers**: Tier 1 (Google, OpenAI, Meta AI, etc.) = 1.0, Tier 2 (Nvidia, Samsung, etc.) = 0.8, Tier 3 = 0.6, Unknown = 0.3.

University and company matching uses fuzzy string matching (via `thefuzz`) so "Dept. of CS, MIT" correctly matches "Massachusetts Institute of Technology".

### Citation Scoring

Papers are scored by comparing actual citations against expected citations for their age:

```
score = min(actual_citations / expected_citations_for_age, 1.0)
```

Expected citations per year are configurable. Citation data comes from the Semantic Scholar API (rate-limited, cached for 30 days).

## Caching & Resumability

The pipeline uses SQLite (`cache/pipeline_cache.db`) to cache:

- Extracted PDF text (keyed by SHA256 of PDF content)
- LLM-extracted metadata
- Embedding vectors (keyed by PDF hash + model name)
- Semantic Scholar citation data (expires after 30 days)

If the pipeline is interrupted, re-running it skips already-processed papers — making it safe to restart after failures.

## GPU Requirements

Tested on **3x NVIDIA RTX PRO 6000 Blackwell (96GB each)**:

| Component | GPU Memory | GPU(s) |
|-----------|-----------|--------|
| vLLM (Qwen3-30B-A3B-Instruct-FP8, TP=2) | ~16 GB/GPU | cuda:0, cuda:1 |
| Qwen3-Embedding-0.6B | ~1.2 GB | cuda:2 |

**Minimum requirements**: 1 GPU with 24GB+ VRAM. Use TP=1 and reduce `max_model_len` to 32768:

```bash
VLLM_TP_SIZE=1 VLLM_MAX_MODEL_LEN=32768 bash scripts/start_vllm_server.sh
```

Set `embedding.device: "cuda:0"` in the config when using a single GPU (embeddings run after vLLM calls, so they share the GPU sequentially).

## Testing

```bash
pip install -r requirements-test.txt

# Run all unit + integration tests (excludes live/GPU tests by default)
pytest tests/ -v --tb=short

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run only live tests (requires running vLLM server + GPU)
pytest tests/ -m live -v
```

**Test suite**: 277 tests across 20 test files (17 unit, 3 integration, 2 live).

## Project Structure

```
AI-Agentic-Survey/
├── config/
│   ├── default_config.yaml          # Main configuration
│   ├── seed_papers.yaml             # Seed paper -> topic label mapping
│   ├── university_rankings.csv      # QS 2025 top 200 universities
│   └── company_tiers.yaml           # 71 companies across 3 tiers
├── scripts/
│   ├── start_vllm_server.sh         # Launch vLLM server
│   └── download_rankings.py         # Generate rankings CSV from QS Excel
├── src/
│   ├── main.py                      # CLI entry point
│   ├── config.py                    # YAML config loading
│   ├── pipeline.py                  # Pipeline orchestrator (8 stages)
│   ├── models/paper.py              # Data models (Paper, Author, PaperMetadata)
│   ├── cache/cache_manager.py       # SQLite cache for resumability
│   ├── extraction/
│   │   ├── pdf_extractor.py         # PDF text extraction (PyMuPDF)
│   │   └── metadata_extractor.py    # Async LLM metadata extraction
│   ├── embedding/
│   │   ├── embedding_model.py       # Sentence-transformers wrapper
│   │   └── clustering.py            # GMM clustering + seed label propagation
│   ├── scoring/
│   │   ├── relevance_scorer.py      # Cluster probability scoring
│   │   ├── affiliation_scorer.py    # University + company scoring
│   │   ├── citation_scorer.py       # Citation vs expected-by-age scoring
│   │   └── aggregator.py            # AND-logic threshold acceptance
│   ├── external/
│   │   ├── semantic_scholar.py      # Async Semantic Scholar API client
│   │   ├── university_rankings.py   # QS rankings loader + fuzzy matching
│   │   └── company_tiers.py         # Company tier loader + fuzzy matching
│   └── output/
│       ├── csv_writer.py            # CSV report generation
│       └── file_organizer.py        # Organize accepted papers by cluster
└── tests/
    ├── unit/                        # 17 test files, ~220 tests
    ├── integration/                 # 3 test files, ~20 tests
    ├── live/                        # 2 test files (GPU required)
    ├── fixtures/                    # Test data files
    └── conftest.py                  # Shared pytest fixtures
```

## Custom Rankings

- **Universities**: Edit `config/university_rankings.csv` (columns: `rank,university`) or generate from a QS Excel file:
  ```bash
  python scripts/download_rankings.py --input qs_rankings.xlsx --output config/university_rankings.csv
  ```
- **Companies**: Edit `config/company_tiers.yaml` to add/remove companies and adjust tier assignments.
