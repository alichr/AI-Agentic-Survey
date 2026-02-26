<div align="center">

# AI Survey Paper Selection Pipeline

**Automated multi-view clustering pipeline for academic paper selection**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Given a pool of academic PDFs organized by venue, this system extracts text and metadata, splits papers into 5 sections, embeds each section independently, runs K-Means clustering per section view, and scores papers using h-index, affiliation, and citation signals. All raw signals are output to CSV for downstream analysis.

[Quick Start](#-quick-start) &bull; [Pipeline Overview](#-pipeline-stages) &bull; [Configuration](#%EF%B8%8F-configuration) &bull; [GPU Requirements](#-gpu-requirements)

</div>

---

## Pipeline Diagram

<div align="center">
<a href="docs/pipeline_diagram.pdf">
<img src="docs/pipeline_diagram.pdf" alt="Pipeline Diagram" width="800"/>
</a>
<p><i>Full pipeline visualization &mdash; <a href="docs/pipeline_diagram.pdf">view PDF</a></i></p>
</div>

---

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install vllm

# 3. Start the LLM server (needs GPU)
bash scripts/start_vllm_server.sh

# 4. Place PDFs in papers/<Venue>/

# 5. Run the pipeline
python -m src.main
```

---

## Prerequisites

| Requirement | Details |
|:------------|:--------|
| **Python** | 3.10+ |
| **GPU** | CUDA-capable with drivers installed |
| **vLLM** | Installed separately (requires matching CUDA version) |
| **Internet** | For OpenAlex / Semantic Scholar APIs and HuggingFace model downloads |

---

## Setup

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install vllm
```

### 2. Prepare input data

Place PDFs in `papers/`, organized by venue subfolders. The folder name becomes the venue label:

```
papers/
├── NeurIPS/
│   ├── paper1.pdf
│   └── paper2.pdf
├── ICLR/
│   └── paper3.pdf
└── arXiv/
    └── paper4.pdf
```

### 3. Start vLLM server

```bash
bash scripts/start_vllm_server.sh
```

This serves **Qwen/Qwen3-30B-A3B-Instruct-2507-FP8** (MoE, 30.5B params / 3.3B active, FP8 quantized) with tensor parallelism across 2 GPUs.

Override defaults with environment variables:

```bash
VLLM_TP_SIZE=1 VLLM_MAX_MODEL_LEN=32768 bash scripts/start_vllm_server.sh
```

<details>
<summary><b>vLLM environment variables</b></summary>

| Variable | Default | Description |
|:---------|:--------|:------------|
| `VLLM_MODEL` | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | Model to serve |
| `VLLM_TP_SIZE` | `2` | Tensor parallel size |
| `VLLM_MAX_MODEL_LEN` | `64000` | Max context length in tokens |
| `VLLM_GPU_UTIL` | `0.90` | GPU memory utilization fraction |
| `VLLM_PORT` | `8000` | Server port |

</details>

> **Note:** Wait until the server logs `Uvicorn running on ...` before running the pipeline.

### 4. Run the pipeline

```bash
python -m src.main
```

<details>
<summary><b>CLI options</b></summary>

```bash
python -m src.main --config config/default_config.yaml   # Custom config file
python -m src.main --log-level DEBUG                      # Verbose logging
python -m src.main --no-citation                          # Skip citation scoring
python -m src.main --no-affiliation                       # Skip affiliation scoring
python -m src.main --no-hindex                            # Skip h-index scoring
python -m src.main --papers-dir /path/to/pdfs             # Override papers directory
python -m src.main --output-dir /path/to/output           # Override output directory
python -m src.main --n-clusters 5                         # Override number of K-Means clusters
```

</details>

---

## Pipeline Stages

```
  PDFs ──> [1] Discovery ──> [2] Text Extraction ──> [3] Metadata Extraction
                                                            │
                    ┌───────────────────────────────────────┘
                    v
            [4a] Section Splitting ──> [4b] Section Summarization
                                               │
           ┌────────────┬──────────┬───────────┼───────────┐
           v            v          v           v           v
        View 0       View 1    View 2      View 3      View 4
      (Title+Abs+  (Intro)   (Related    (Method)   (Experiments)
       Conclusion)            Work)
           │            │          │           │           │
           └────────────┴──────────┴───────────┴───────────┘
                                   │
                    [5] Multi-View Embedding
                    [6] K-Means Clustering (per view)
                                   │
              ┌────────────────────┼────────────────────┐
              v                    v                    v
        [7a] Affiliation    [7b] Citation        [7c] H-Index
            Scoring            Scoring              Scoring
              └────────────────────┼────────────────────┘
                                   v
                          [8] CSV Output
```

| Stage | Description |
|:------|:------------|
| **1. Discovery** | Scan `papers/` for PDFs; detect venue from folder name |
| **2. Full-Text Extraction** | Extract all pages using PyMuPDF |
| **3. Metadata Extraction** | LLM extracts title, abstract, authors, year via vLLM |
| **4a. Section Splitting** | Heuristic regex into 5 sections (with LLM fallback) |
| **4b. Section Summarization** | Section-specific LLM summaries (~400 words each) |
| **5. Multi-View Embedding** | Embed section summaries independently (Qwen3-Embedding-4B) |
| **6. K-Means Clustering** | PCA + K-Means independently per section view |
| **7. Scoring** | H-index, affiliation, and citation scoring (OpenAlex + Semantic Scholar) |
| **8. Output** | CSV report with all raw signals |

---

## Output

The pipeline produces two files in the output directory:

| File | Description |
|:-----|:------------|
| `output/results.csv` | Full results with per-view cluster assignments and quality signals |
| `output/section_diagnostics.csv` | Per-paper section segmentation and summary diagnostics |

<details>
<summary><b>CSV column reference</b></summary>

| Column | Description |
|:-------|:------------|
| `title` | Paper title extracted by LLM |
| `venue` | Venue detected from folder name |
| `year` | Publication year |
| `first_author` | First author name |
| `first_author_affiliation` | First author's institution or company |
| `first_author_affiliation_score` | Affiliation quality score (0&ndash;1) |
| `last_author` | Last (senior) author name |
| `last_author_affiliation` | Last author's institution or company |
| `last_author_affiliation_score` | Affiliation quality score (0&ndash;1) |
| `cluster_<view>` | Cluster ID per section view (5 columns) |
| `centroid_dist_<view>` | Distance to cluster centroid per view (5 columns) |
| `max_hindex` | Maximum h-index among paper authors |
| `citation_count` | Raw citation count |
| `pdf_path` | Path to the PDF file |

The 5 section views are: `title_abstract_conclusion`, `introduction`, `related_work`, `method`, `experiments`.

</details>

---

## Configuration

All settings are in [`config/default_config.yaml`](config/default_config.yaml):

<details>
<summary><b>Full configuration reference</b></summary>

| Section | Parameter | Default | Description |
|:--------|:----------|:--------|:------------|
| `vllm` | `model_name` | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | LLM for metadata/section extraction |
| `vllm` | `base_url` | `http://localhost:8000/v1` | vLLM server URL |
| `embedding` | `model_name` | `Qwen/Qwen3-Embedding-4B` | Embedding model for section summaries |
| `embedding` | `device` | `cuda:0` | GPU for embeddings |
| `embedding` | `batch_size` | `16` | Embedding batch size |
| `kmeans` | `n_clusters` | `10` | Number of K-Means clusters |
| `kmeans` | `pca_components` | `50` | PCA dimensions before K-Means (0 = disabled) |
| `kmeans` | `enabled_views` | `[0,1,2,3,4]` | Section views to use |
| `hindex` | `enabled` | `true` | Enable h-index scoring |
| `affiliation` | `enabled` | `true` | Enable affiliation scoring |
| `citation` | `enabled` | `true` | Enable citation scoring |
| `citation` | `source` | `openalex` | Citation API (`openalex` or `semantic_scholar`) |

</details>

---

## Caching & Resumability

The pipeline uses **SQLite** (`cache/pipeline_cache.db`) to cache all intermediate results keyed by SHA256 of PDF content:

- Extracted PDF text, LLM-extracted metadata, section splits and summaries
- Embedding vectors (keyed by PDF hash + model name + view)
- Citation data and author h-index data (expires after 30 days)

> If the pipeline is interrupted, re-running it skips already-processed papers.

---

## GPU Requirements

Tested on **3x NVIDIA RTX PRO 6000 Blackwell (96GB each)**:

| Component | GPU Memory | GPU(s) |
|:----------|:-----------|:-------|
| vLLM (Qwen3-30B-A3B-Instruct-FP8, TP=2) | ~16 GB/GPU | cuda:0, cuda:1 |
| Qwen3-Embedding-4B | ~8 GB | cuda:2 |

<details>
<summary><b>Single-GPU setup (24GB+ VRAM)</b></summary>

Use TP=1 and reduce context length:

```bash
VLLM_TP_SIZE=1 VLLM_MAX_MODEL_LEN=32768 bash scripts/start_vllm_server.sh
```

Set `embedding.device: "cuda:0"` and `embedding.model_name: "Qwen/Qwen3-Embedding-0.6B"` in the config. Embeddings run after vLLM calls, so they share the GPU sequentially.

</details>

---

## Testing

```bash
pip install -r requirements-test.txt

# Unit + integration tests (excludes live/GPU tests by default)
pytest tests/ -v --tb=short

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Live tests (requires running vLLM server + GPU)
pytest tests/ -m live -v
```

---

## Project Structure

```
AI-Agentic-Survey/
├── config/
│   ├── default_config.yaml          # Main configuration
│   ├── university_rankings.csv      # QS 2025 top 200 universities
│   └── company_tiers.yaml           # 71 companies across 3 tiers
├── docs/
│   └── pipeline_diagram.pdf         # Pipeline visualization
├── scripts/
│   ├── start_vllm_server.sh         # Launch vLLM server
│   ├── download_rankings.py         # Generate rankings CSV from QS Excel
│   └── generate_pipeline_diagram.py # Generate pipeline visualization PDF
├── src/
│   ├── main.py                      # CLI entry point
│   ├── config.py                    # YAML config loading
│   ├── pipeline.py                  # Pipeline orchestrator (8 stages)
│   ├── models/paper.py              # Data models (Paper, Author, PaperScores)
│   ├── cache/cache_manager.py       # SQLite cache for resumability
│   ├── extraction/
│   │   ├── pdf_extractor.py         # PDF text extraction (PyMuPDF)
│   │   ├── metadata_extractor.py    # Async LLM metadata extraction
│   │   ├── section_extractor.py     # LLM section-start detection (fallback)
│   │   ├── section_splitter.py      # Heuristic regex section splitting
│   │   └── section_summarizer.py    # Section-specific LLM summarization
│   ├── embedding/
│   │   ├── embedding_model.py       # Sentence-transformers wrapper
│   │   └── kmeans_clustering.py     # PCA + multi-view K-Means clustering
│   ├── scoring/
│   │   ├── affiliation_scorer.py    # University + company scoring
│   │   ├── citation_scorer.py       # Dual-source citation scoring
│   │   └── hindex_scorer.py         # Author h-index scoring
│   ├── external/
│   │   ├── openalex.py              # Async OpenAlex API client
│   │   ├── semantic_scholar.py      # Async Semantic Scholar API client
│   │   ├── university_rankings.py   # QS rankings loader + fuzzy matching
│   │   └── company_tiers.py         # Company tier loader + fuzzy matching
│   └── output/
│       ├── csv_writer.py            # CSV report generation
│       └── section_diagnostics_csv.py # Section diagnostics CSV
└── tests/
    ├── unit/                        # Unit tests
    ├── integration/                 # Integration tests
    ├── live/                        # Live tests (GPU required)
    └── conftest.py                  # Shared pytest fixtures
```

---

<div align="center">
<sub>Built for the <b>Agentic AI Survey</b> paper selection process</sub>
</div>
