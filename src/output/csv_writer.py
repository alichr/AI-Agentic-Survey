"""Write pipeline results to CSV."""

import csv
from pathlib import Path

from src.models.paper import Paper


def write_results_csv(papers: list[Paper], csv_path: Path):
    """Write scored papers to a CSV file.

    Args:
        papers: List of scored papers.
        csv_path: Output CSV path.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "title", "venue", "year", "first_author", "last_author",
        "cluster_id", "cluster_label", "relevance_score",
        "affiliation_score", "citation_score", "citation_count",
        "accepted", "pdf_path",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for paper in papers:
            writer.writerow({
                "title": paper.title or "",
                "venue": paper.venue,
                "year": paper.metadata.year if paper.metadata else "",
                "first_author": paper.first_author.name if paper.first_author else "",
                "last_author": paper.last_author.name if paper.last_author else "",
                "cluster_id": paper.scores.cluster_id if paper.scores.cluster_id is not None else "",
                "cluster_label": paper.scores.cluster_label or "",
                "relevance_score": f"{paper.scores.relevance_score:.4f}" if paper.scores.relevance_score is not None else "",
                "affiliation_score": f"{paper.scores.affiliation_score:.4f}" if paper.scores.affiliation_score is not None else "",
                "citation_score": f"{paper.scores.citation_score:.4f}" if paper.scores.citation_score is not None else "",
                "citation_count": paper.scores.citation_count if paper.scores.citation_count is not None else "",
                "accepted": paper.accepted,
                "pdf_path": str(paper.pdf_path),
            })
