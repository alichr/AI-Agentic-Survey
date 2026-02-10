# Project Design Document — AI Survey Paper Selection System

## Overview

A pipeline to automatically select relevant papers for a survey on Agentic AI. Scores candidate PDFs on relevance (embedding+clustering), affiliation prestige (QS rankings + company tiers), and citation impact (Semantic Scholar), then applies configurable thresholds to accept/reject papers.

## Architecture

```
PDF Files (by venue)
        │
        ▼
  ┌─────────────┐
  │  Discovery   │  Scan folders, compute SHA256 hashes
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  PDF Extract │  PyMuPDF → first-page text
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  LLM Extract│  vLLM (Qwen3-30B-A3B) → title, abstract, authors, year
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Embed+GMM  │  Qwen3-Embedding-0.6B → GMM clustering → relevance score
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Affiliation│  QS rankings + company tiers → affiliation score
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Citations  │  Semantic Scholar API → citation score
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │  Aggregate  │  AND logic: all active scores >= thresholds
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │   Output    │  CSV report + accepted PDFs by cluster
  └─────────────┘
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Clustering method | GMM | Outputs soft membership probabilities; K-Means only gives hard assignments |
| Label propagation | Seed papers anchor clusters | User provides known papers with topic labels; GMM cluster inheriting that label |
| Affiliation scoring | max(uni_score, company_score) | An author at Google Research shouldn't be penalized for low university rank |
| Citation scoring | actual/expected_by_age | Normalizes for paper age; rewards recent papers with even modest citations |
| Acceptance logic | AND across all active scores | Conservative: paper must be relevant AND from good institution AND well-cited |
| Cache keys | SHA256 of PDF content | Survives file renames/moves |
| Cluster assignments | Not cached | They depend on the full paper set which may change between runs |

## Data Flow

1. **Paper** object created at discovery with `pdf_path`, `venue`, `pdf_hash`
2. **first_page_text** added at Stage 2
3. **PaperMetadata** (title, abstract, authors, year) added at Stage 3
4. **embedding** vector added at Stage 4
5. **PaperScores** (relevance, cluster, affiliation, citation) populated at Stages 4-6
6. **accepted** flag set at Stage 7

## Scoring Details

### Relevance Score
- Embedding text: `"{title} [SEP] {abstract}"`
- GMM with K components fitted on all papers (seed + candidates)
- Score = max probability across all clusters from `predict_proba()`

### Affiliation Score
- Per-author: `max(university_score, company_score)`
- University: QS rank → bucket score (Top 10=1.0, Top 50=0.9, ..., Unranked=0.2)
- Company: Tier 1=1.0, Tier 2=0.8, Tier 3=0.6, Unknown=0.3
- Combined: `first_weight * first_author + last_weight * last_author`

### Citation Score
- `score = min(actual_citations / expected_citations_by_age, 1.0)`
- Expected defaults: 0yr=5, 1yr=15, 2yr=40, 3yr=80, 4yr=120, 5yr=160, then +30/yr

## Module Inventory

| Module | Purpose |
|--------|---------|
| `src/models/paper.py` | Paper, PaperMetadata, PaperScores, Author dataclasses |
| `src/config.py` | YAML config loading with typed dataclass sections |
| `src/cache/cache_manager.py` | SQLite cache (pdf_text, metadata, embeddings, citations) |
| `src/extraction/pdf_extractor.py` | PyMuPDF text extraction |
| `src/extraction/metadata_extractor.py` | Async vLLM structured metadata extraction |
| `src/embedding/embedding_model.py` | sentence-transformers wrapper |
| `src/embedding/clustering.py` | GMM clustering with seed label propagation |
| `src/scoring/relevance_scorer.py` | Max cluster probability scoring |
| `src/scoring/affiliation_scorer.py` | University + company combined scoring |
| `src/scoring/citation_scorer.py` | Citations/expected with Semantic Scholar |
| `src/scoring/aggregator.py` | AND-logic threshold gating |
| `src/external/semantic_scholar.py` | Async API client with rate limiting |
| `src/external/university_rankings.py` | QS ranking fuzzy lookup |
| `src/external/company_tiers.py` | Company tier fuzzy lookup |
| `src/output/csv_writer.py` | 14-column CSV report |
| `src/output/file_organizer.py` | Copy accepted PDFs by cluster |
| `src/pipeline.py` | 8-stage orchestrator |
| `src/main.py` | CLI entry point |

## Configuration Files

| File | Purpose |
|------|---------|
| `config/default_config.yaml` | All pipeline parameters |
| `config/seed_papers.yaml` | Filename → topic label mapping |
| `config/company_tiers.yaml` | 3-tier company classification |
| `config/university_rankings.csv` | QS Top 100 universities |

## Status

- [x] Project structure
- [x] Configuration loading
- [x] SQLite cache with resume support
- [x] PDF text extraction
- [x] LLM metadata extraction (async)
- [x] Embedding model wrapper
- [x] GMM clustering + label propagation
- [x] University ranking lookup (fuzzy)
- [x] Company tier lookup (fuzzy)
- [x] Affiliation scoring
- [x] Citation scoring (Semantic Scholar)
- [x] Score aggregation (AND logic)
- [x] CSV output
- [x] File organization by cluster
- [x] Pipeline orchestrator
- [x] CLI entry point
