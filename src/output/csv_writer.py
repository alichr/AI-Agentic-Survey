"""Write pipeline results to CSV with local and global scores."""

import csv
from pathlib import Path

from src.models.paper import SectionType, Paper

VIEW_NAMES = {
    0: "title_abstract_conclusion",
    1: "introduction",
    2: "related_work",
    3: "method",
    4: "experiments",
}


def write_results_csv(papers: list[Paper], csv_path: Path):
    """Write scored pipeline papers to a CSV file.

    Columns include per-view cluster assignments, author affiliations,
    and raw quality signals (max h-index, citation count).

    Args:
        papers: List of scored papers.
        csv_path: Output CSV path.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    views = list(SectionType)

    # Build column names
    fieldnames = [
        "title", "venue", "year",
        "first_author", "first_author_affiliation", "first_author_affiliation_score",
        "last_author", "last_author_affiliation", "last_author_affiliation_score",
    ]

    # Per-view: cluster_id + centroid_distance paired together
    for v in views:
        vname = VIEW_NAMES.get(v.value, f"v{v.value}")
        fieldnames.append(f"cluster_{vname}")
        fieldnames.append(f"centroid_dist_{vname}")

    # Raw quality signals
    fieldnames.extend([
        "max_hindex",
        "citation_count",
        "pdf_path",
    ])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for paper in papers:
            fa = paper.first_author
            la = paper.last_author
            row = {
                "title": paper.title or "",
                "venue": paper.venue,
                "year": paper.metadata.year if paper.metadata else "",
                "first_author": fa.name if fa else "",
                "first_author_affiliation": fa.affiliation or "" if fa else "",
                "first_author_affiliation_score": (
                    f"{paper.scores.first_author_affiliation_score:.4f}"
                    if paper.scores.first_author_affiliation_score is not None else ""
                ),
                "last_author": la.name if la else "",
                "last_author_affiliation": la.affiliation or "" if la else "",
                "last_author_affiliation_score": (
                    f"{paper.scores.last_author_affiliation_score:.4f}"
                    if paper.scores.last_author_affiliation_score is not None else ""
                ),
            }

            for v in views:
                vid = v.value
                vname = VIEW_NAMES.get(vid, f"v{vid}")
                row[f"cluster_{vname}"] = (
                    paper.scores.cluster_ids.get(vid, "")
                )
                dist = paper.scores.centroid_distances.get(vid)
                row[f"centroid_dist_{vname}"] = (
                    f"{dist:.4f}" if dist is not None else ""
                )

            row["max_hindex"] = (
                paper.scores.max_hindex if paper.scores.max_hindex is not None else ""
            )
            row["citation_count"] = (
                paper.scores.citation_count
                if paper.scores.citation_count is not None else ""
            )
            row["pdf_path"] = str(paper.pdf_path)

            writer.writerow(row)
