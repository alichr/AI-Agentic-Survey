"""Import paper bundles from export directories."""

import json
import logging
from pathlib import Path

import numpy as np

from src.models.paper import Author, Paper, PaperMetadata, PaperScores

logger = logging.getLogger(__name__)


def _paper_completeness_score(record: dict, has_embeddings: bool) -> int:
    """Score how much data a paper record has, for deduplication tie-breaking."""
    score = 0
    scores_dict = record.get("scores") or {}
    for v in scores_dict.values():
        if v is not None:
            score += 1
    if record.get("section_summaries"):
        score += len(record["section_summaries"])
    if has_embeddings:
        score += 10
    return score


def import_all_papers(input_dir: str | Path) -> tuple[list[Paper], list[Paper]]:
    """Import all paper bundles from a directory (recursive).

    Scans input_dir for papers.json files, loads the corresponding
    embeddings.npz, reconstructs Paper objects, and deduplicates by pdf_hash.

    Args:
        input_dir: Root directory to scan for export bundles.

    Returns:
        Tuple of (complete_papers, incomplete_papers).

    Raises:
        FileNotFoundError: If input_dir doesn't exist or contains no bundles.
        ValueError: If bundles have incompatible embedding dimensions.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Find all papers.json files
    json_files = sorted(input_dir.rglob("papers.json"))
    if not json_files:
        raise FileNotFoundError(f"No papers.json files found in {input_dir}")

    logger.info("Found %d export bundle(s) in %s", len(json_files), input_dir)

    # Collect all records with their embeddings, keyed by pdf_hash for dedup
    best_records: dict[str, tuple[dict, dict[str, np.ndarray], Path]] = {}

    for json_path in json_files:
        bundle_dir = json_path.parent
        npz_path = bundle_dir / "embeddings.npz"

        # Load JSON
        with open(json_path) as f:
            records = json.load(f)

        # Load embeddings
        embeddings: dict[str, np.ndarray] = {}
        if npz_path.exists():
            with np.load(npz_path) as npz:
                for key in npz.files:
                    embeddings[key] = npz[key]

        logger.info(
            "Loading bundle %s: %d papers, %d embeddings",
            bundle_dir, len(records), len(embeddings),
        )

        for record in records:
            pdf_hash = record["pdf_hash"]

            # Check which embeddings belong to this paper
            paper_emb_keys = [
                k for k in embeddings if k.startswith(f"{pdf_hash}__view_")
            ]
            has_embeddings = len(paper_emb_keys) > 0

            # Deduplication: keep the record with more data
            if pdf_hash in best_records:
                existing_record, existing_embs, existing_src = best_records[pdf_hash]
                existing_score = _paper_completeness_score(
                    existing_record,
                    bool(existing_embs),
                )
                new_score = _paper_completeness_score(record, has_embeddings)
                if new_score <= existing_score:
                    logger.warning(
                        "Duplicate paper %s (%.20s...) found in %s, "
                        "keeping version from %s (score %d >= %d)",
                        pdf_hash[:12], record.get("metadata", {}).get("title", "?"),
                        bundle_dir, existing_src, existing_score, new_score,
                    )
                    continue
                else:
                    logger.warning(
                        "Duplicate paper %s (%.20s...) — replacing version from %s "
                        "with %s (score %d > %d)",
                        pdf_hash[:12], record.get("metadata", {}).get("title", "?"),
                        existing_src, bundle_dir, new_score, existing_score,
                    )

            paper_embs = {k: embeddings[k] for k in paper_emb_keys}
            best_records[pdf_hash] = (record, paper_embs, bundle_dir)

    # Check embedding dimension consistency
    _check_embedding_dimensions(best_records)

    # Reconstruct Paper objects
    complete: list[Paper] = []
    incomplete: list[Paper] = []

    for pdf_hash, (record, paper_embs, _src) in best_records.items():
        paper = _record_to_paper(record, paper_embs)

        # Validate: if marked complete but missing embeddings, downgrade
        if record.get("is_complete", False) and not paper.section_embeddings:
            logger.warning(
                "Paper %s marked complete but has no embeddings — "
                "downgrading to incomplete",
                pdf_hash[:12],
            )
            incomplete.append(paper)
        elif record.get("is_complete", False):
            complete.append(paper)
        else:
            incomplete.append(paper)

    logger.info(
        "Imported %d unique papers: %d complete, %d incomplete",
        len(complete) + len(incomplete), len(complete), len(incomplete),
    )

    return complete, incomplete


def _check_embedding_dimensions(
    records: dict[str, tuple[dict, dict[str, np.ndarray], Path]],
) -> None:
    """Verify all embeddings have the same dimension per view."""
    view_dims: dict[str, int] = {}  # view_N -> expected dim

    for pdf_hash, (_record, paper_embs, src) in records.items():
        for key, emb in paper_embs.items():
            # key format: {hash}__view_{N}
            view_part = key.split("__")[-1]  # "view_N"
            dim = emb.shape[0] if emb.ndim == 1 else emb.shape[-1]

            if view_part not in view_dims:
                view_dims[view_part] = dim
            elif view_dims[view_part] != dim:
                raise ValueError(
                    f"Embedding dimension mismatch for {view_part}: "
                    f"expected {view_dims[view_part]}, got {dim} "
                    f"(paper {pdf_hash[:12]} from {src}). "
                    f"Were bundles produced with different embedding models?"
                )


def _record_to_paper(
    record: dict,
    paper_embs: dict[str, np.ndarray],
) -> Paper:
    """Reconstruct a Paper object from a JSON record + embeddings."""
    # Metadata
    metadata = None
    meta_dict = record.get("metadata")
    if meta_dict:
        authors = [
            Author(name=a["name"], affiliation=a.get("affiliation"))
            for a in meta_dict.get("authors", [])
        ]
        metadata = PaperMetadata(
            title=meta_dict.get("title"),
            abstract=meta_dict.get("abstract"),
            authors=authors,
            year=meta_dict.get("year"),
        )

    # Section summaries: str keys -> int keys
    section_summaries: dict[int, str] = {}
    for k, v in (record.get("section_summaries") or {}).items():
        if v:
            section_summaries[int(k)] = v

    # Embeddings: parse view number from key
    section_embeddings: dict[int, list[float]] = {}
    for key, emb in paper_embs.items():
        # key format: {hash}__view_{N}
        view_str = key.split("__view_")[-1]
        view = int(view_str)
        section_embeddings[view] = emb.tolist()

    # Scores
    scores_dict = record.get("scores") or {}
    scores = PaperScores(
        first_author_affiliation_score=scores_dict.get("first_author_affiliation_score"),
        last_author_affiliation_score=scores_dict.get("last_author_affiliation_score"),
        citation_count=scores_dict.get("citation_count"),
        max_hindex=scores_dict.get("max_hindex"),
    )

    return Paper(
        pdf_path=Path(record["pdf_path"]),
        venue=record.get("venue", "Unknown"),
        pdf_hash=record["pdf_hash"],
        metadata=metadata,
        section_summaries=section_summaries,
        section_embeddings=section_embeddings,
        scores=scores,
    )
