"""Score aggregation and acceptance decision logic."""

import logging
from typing import Optional

from src.config import Config
from src.models.paper import Paper

logger = logging.getLogger(__name__)


class ScoreAggregator:
    """Applies threshold-based AND logic across all active scoring dimensions.

    A paper is ACCEPTED only if every active score >= its respective threshold.
    Scores can be individually enabled/disabled in the config.
    """

    def __init__(self, config: Config):
        self.relevance_enabled = config.relevance.enabled
        self.relevance_threshold = config.relevance.threshold
        self.affiliation_enabled = config.affiliation.enabled
        self.affiliation_threshold = config.affiliation.threshold
        self.citation_enabled = config.citation.enabled
        self.citation_threshold = config.citation.threshold

    def decide(self, paper: Paper) -> bool:
        """Determine whether a paper should be accepted.

        Args:
            paper: Paper with all scores computed.

        Returns:
            True if paper passes all active thresholds.
        """
        scores = paper.scores

        if self.relevance_enabled:
            if scores.relevance_score is None or scores.relevance_score < self.relevance_threshold:
                return False

        if self.affiliation_enabled:
            if scores.affiliation_score is None or scores.affiliation_score < self.affiliation_threshold:
                return False

        if self.citation_enabled:
            if scores.citation_score is None or scores.citation_score < self.citation_threshold:
                return False

        return True

    def decide_all(self, papers: list[Paper]) -> list[Paper]:
        """Apply acceptance decision to all papers.

        Args:
            papers: List of papers with scores.

        Returns:
            Same list with `accepted` field set on each paper.
        """
        accepted_count = 0
        for paper in papers:
            paper.accepted = self.decide(paper)
            if paper.accepted:
                accepted_count += 1

        total = len(papers)
        logger.info(
            "Aggregation complete: %d/%d papers accepted (%.1f%%)",
            accepted_count, total, 100 * accepted_count / total if total > 0 else 0,
        )

        # Log threshold details
        active_criteria = []
        if self.relevance_enabled:
            active_criteria.append(f"relevance>={self.relevance_threshold}")
        if self.affiliation_enabled:
            active_criteria.append(f"affiliation>={self.affiliation_threshold}")
        if self.citation_enabled:
            active_criteria.append(f"citation>={self.citation_threshold}")
        logger.info("Active criteria (AND): %s", " & ".join(active_criteria))

        return papers
