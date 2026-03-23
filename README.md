# Paper Filter

A CLI tool that filters academic conference papers by topic relevance using a local LLM. Two fully independent stages:

1. **`download`** — Fetch paper lists and download all PDFs into a folder
2. **`process`** — Take any folder of PDFs, extract metadata, classify relevance, output CSV

The stages share nothing. If automated download fails, you can manually download PDFs and point `process` at the folder.

## Supported Conferences

| Conference | Source | Years | PDF Download |
|---|---|---|---|
| CVPR | CVF Open Access | 2021–2025 | Yes (Wayback Machine fallback) |
| ICCV | CVF Open Access | 2021, 2023, 2025 | Yes (Wayback Machine fallback) |
| ECCV | ECVA | 2022, 2024 | Yes (Wayback Machine fallback) |
| ICML | PMLR (2021–2024), icml.cc (2025) | 2021–2025 | Yes (2021–2024), blocked (2025) |
| NeurIPS | proceedings.neurips.cc (2021–2024), neurips.cc (2025) | 2021–2025 | Yes (2021–2024), blocked (2025) |
| ICLR | iclr.cc | 2021–2026 | Blocked (OpenReview) |
| EMNLP | ACL Anthology XML | 2021–2025 | Yes |
| AAAI | OJS Platform (main track) | 2021–2026 | Yes |

**Blocked** = PDFs are on OpenReview which currently blocks automated downloads. Download them manually and use `process --pdf-dir`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start vLLM server (needed for 'process' only)
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --port 8001
```

## Usage

### Step 1: Download (no vLLM needed)

```bash
# Single conference
python -m paper_filter download -c eccv -y 2024

# All conferences
./scripts/run_all.sh download
```

Output: `output/pdfs/{CONF}_{YEAR}/` with all PDFs.

### Step 2: Process (needs vLLM)

```bash
# Point at the PDF folder
python -m paper_filter process \
  --pdf-dir output/pdfs/ECCV_2024 \
  -c eccv -y 2024 \
  -t "test-time learning" \
  --relevance-threshold 8

# With detailed scope for better precision
python -m paper_filter process \
  --pdf-dir output/pdfs/CVPR_2025 \
  -c cvpr -y 2025 \
  -t "test-time learning" \
  -d "Methods that adapt model parameters during inference, including TTA, TTT, and online adaptation. Excludes training-free inference methods." \
  --relevance-threshold 8

# Process manually downloaded PDFs
python -m paper_filter process \
  --pdf-dir /path/to/my/iclr-pdfs \
  -c iclr -y 2024 \
  -t "test-time learning" \
  --relevance-threshold 8

# All conferences
./scripts/run_all.sh process "test-time learning" 8

# Download + process in one go
./scripts/run_all.sh all "test-time learning" 8
```

### How It Works

1. **Scan** — Finds all PDFs in the input folder
2. **Extract** — LLM reads first 4 pages of each PDF, extracts title, abstract, and introduction
3. **Classify** — LLM scores relevance 0–10 using title + abstract + introduction
4. **Cleanup** — Copies relevant PDFs to `processed/` (input folder is never modified)
5. **Output** — Writes CSV with all papers and scores

Each step is cached per conference-year. Safe to interrupt and restart. The input PDF folder is never modified — you can re-run `process` with different topics or thresholds without re-downloading.

### Relevance Threshold

| Threshold | Use case |
|---|---|
| 6 | Broad — topic is one of several themes |
| 7 | Balanced — clearly related papers |
| **8 (recommended)** | High precision — topic is the paper's primary contribution |
| 9-10 | Very strict — only landmark papers |

### Topic Description (`-d`)

Optional but improves precision. Tells the LLM exactly what to include/exclude:

```bash
-t "vision-language models" \
-d "VLMs, image captioning, visual QA, visual grounding, multimodal pretraining. Excludes pure vision or pure NLP."
```

## Output

```
output/
├── pdfs/ECCV_2024/             # All downloaded PDFs (never modified by process)
├── processed/ECCV_2024/        # Copies of relevant PDFs that passed threshold
└── ECCV_2024_results.csv       # All papers with scores
```

CSV columns: title, abstract, introduction, relevance_score, relevance_reasoning, pdf_path

## Configuration

`config/default_config.yaml`:

```yaml
pipeline:
  output_dir: "output"
  cache_dir: "cache"
  relevance_threshold: 8
  max_classify_concurrent: 20

vllm:
  base_url: "http://localhost:8001/v1"
  model_name: "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
  temperature: 0.3
  max_tokens: 512

download:
  max_concurrent: 30
  max_retries: 3
  timeout: 60
```
