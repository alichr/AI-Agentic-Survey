"""Export papers to a portable bundle (papers.json + embeddings.npz)."""

import json
import logging
from pathlib import Path

import numpy as np

from src.models.paper import Paper

logger = logging.getLogger(__name__)


def export_papers(
    papers: list[Paper],
    incomplete_papers: list[Paper],
    export_dir: str | Path,
) -> None:
    """Export papers to a directory as papers.json + embeddings.npz.

    Args:
        papers: Complete papers (with embeddings and scores).
        incomplete_papers: Papers that failed section completeness filter.
        export_dir: Directory to write the bundle into.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    all_papers = papers + incomplete_papers
    complete_hashes = {p.pdf_hash for p in papers}

    # Build JSON records and embedding arrays
    records = []
    embeddings_dict: dict[str, np.ndarray] = {}

    for paper in all_papers:
        is_complete = paper.pdf_hash in complete_hashes

        # Metadata
        metadata_dict = None
        if paper.metadata:
            metadata_dict = {
                "title": paper.metadata.title,
                "abstract": paper.metadata.abstract,
                "authors": [
                    {"name": a.name, "affiliation": a.affiliation}
                    for a in paper.metadata.authors
                ],
                "year": paper.metadata.year,
            }

        # Section summaries keyed by view number (str)
        section_summaries = {
            str(k): v for k, v in paper.section_summaries.items()
        }

        # Scores (only quality signals, not clustering)
        scores_dict = {
            "first_author_affiliation_score": paper.scores.first_author_affiliation_score,
            "last_author_affiliation_score": paper.scores.last_author_affiliation_score,
            "citation_count": paper.scores.citation_count,
            "max_hindex": paper.scores.max_hindex,
        }

        records.append({
            "pdf_hash": paper.pdf_hash,
            "pdf_path": str(paper.pdf_path),
            "venue": paper.venue,
            "is_complete": is_complete,
            "metadata": metadata_dict,
            "section_summaries": section_summaries,
            "scores": scores_dict,
        })

        # Collect embeddings
        for view, emb in paper.section_embeddings.items():
            key = f"{paper.pdf_hash}__view_{view}"
            embeddings_dict[key] = np.array(emb, dtype=np.float32)

    # Write papers.json
    json_path = export_dir / "papers.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    # Write embeddings.npz
    npz_path = export_dir / "embeddings.npz"
    np.savez(npz_path, **embeddings_dict)

    logger.info(
        "Exported %d papers (%d complete, %d incomplete) to %s",
        len(all_papers), len(papers), len(incomplete_papers), export_dir,
    )
