"""Organize accepted papers into output directory."""

import logging
import shutil
from pathlib import Path

from src.models.paper import Paper

logger = logging.getLogger(__name__)


def organize_accepted_papers(papers: list[Paper], output_dir: Path) -> int:
    """Copy accepted papers to the output directory.

    Args:
        papers: List of scored papers.
        output_dir: Directory to copy accepted papers into.

    Returns:
        Number of papers copied.
    """
    accepted_dir = output_dir / "accepted_papers"
    accepted_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for paper in papers:
        if not paper.accepted:
            continue
        dest = accepted_dir / paper.pdf_path.name
        try:
            shutil.copy2(paper.pdf_path, dest)
            copied += 1
        except Exception as e:
            logger.error("Failed to copy %s: %s", paper.pdf_path, e)

    logger.info("Copied %d accepted papers to %s", copied, accepted_dir)
    return copied
