"""Organize accepted papers into output directory."""

import logging
import shutil
from pathlib import Path

from src.models.paper import Paper

logger = logging.getLogger(__name__)


def _sanitize_folder_name(name: str) -> str:
    """Convert a cluster label to a safe folder name."""
    return name.replace(" ", "_").replace("/", "-")


def organize_accepted_papers(papers: list[Paper], output_dir: Path) -> int:
    """Copy accepted papers into topic subdirectories under accepted_papers/.

    Each paper is placed in a subfolder named after its cluster label
    (which corresponds to the seed paper topics). Papers without a
    cluster label go into an "Uncategorized" folder.

    Args:
        papers: List of scored papers.
        output_dir: Directory to create accepted_papers/ in.

    Returns:
        Number of papers copied.
    """
    accepted_dir = output_dir / "accepted_papers"
    accepted_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for paper in papers:
        if not paper.accepted:
            continue

        label = paper.scores.cluster_label or "Uncategorized"
        topic_dir = accepted_dir / _sanitize_folder_name(label)
        topic_dir.mkdir(parents=True, exist_ok=True)

        dest = topic_dir / paper.pdf_path.name
        try:
            shutil.copy2(paper.pdf_path, dest)
            copied += 1
        except Exception as e:
            logger.error("Failed to copy %s: %s", paper.pdf_path, e)

    logger.info("Copied %d accepted papers to %s", copied, accepted_dir)
    return copied
