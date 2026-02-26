"""Author h-index scoring using external API data (OpenAlex or Semantic Scholar)."""

import asyncio
import logging
from typing import Optional, Protocol

from src.cache.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class HIndexClient(Protocol):
    """Any client that can fetch paper authors with h-index data."""

    async def get_paper_authors(self, paper_id: str) -> list: ...


class HIndexScorer:
    """Scores papers based on the maximum h-index of their authors.

    Formula: score = min(max_hindex / baseline, 1.0)
    An h-index of 50+ gives a perfect score of 1.0 (default baseline).
    """

    def __init__(self, client: HIndexClient,
                 cache: CacheManager,
                 max_hindex_baseline: int = 50,
                 default_score: float = 0.3,
                 cache_max_age_days: int = 30):
        self.client = client
        self.cache = cache
        self.baseline = max_hindex_baseline
        self.default_score = default_score
        self.cache_max_age_days = cache_max_age_days

    def compute_score(self, max_hindex: Optional[int]) -> float:
        """Compute h-index score from the maximum h-index among authors.

        Args:
            max_hindex: Maximum h-index across all paper authors.

        Returns:
            Score in [0, 1].
        """
        if max_hindex is None:
            return self.default_score
        return min(max_hindex / self.baseline, 1.0)

    async def fetch_and_score(self, title: str,
                              paper_id: Optional[str]) -> tuple[float, Optional[int]]:
        """Fetch author h-indices and compute score for a paper.

        Checks cache first, then queries Semantic Scholar API.

        Args:
            title: Paper title (used as cache key).
            paper_id: Semantic Scholar paper ID (needed for API call).

        Returns:
            Tuple of (score, max_hindex).
        """
        if not title:
            return self.default_score, None

        # Check cache
        cached = self.cache.get_author_hindex(title, self.cache_max_age_days)
        if cached is not None:
            return self.compute_score(cached["max_hindex"]), cached["max_hindex"]

        # Need paper_id to fetch authors
        if not paper_id:
            return self.default_score, None

        # Fetch from API
        authors = await self.client.get_paper_authors(paper_id)
        if not authors:
            return self.default_score, None

        # Find max h-index
        hindices = [a.hindex for a in authors if a.hindex is not None]
        max_hindex = max(hindices) if hindices else None

        # Cache the result
        authors_dicts = [
            {"author_id": a.author_id, "name": a.name, "hindex": a.hindex}
            for a in authors
        ]
        self.cache.set_author_hindex(title, authors_dicts, max_hindex or 0)

        score = self.compute_score(max_hindex)
        return score, max_hindex

    async def score_batch(self, papers_data: list[tuple[str, Optional[str]]]) -> list[tuple[float, Optional[int]]]:
        """Fetch and score h-indices for multiple papers.

        Args:
            papers_data: List of (title, paper_id) tuples.

        Returns:
            List of (score, max_hindex) tuples parallel to input.
        """
        tasks = [self.fetch_and_score(title, pid) for title, pid in papers_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error("H-index scoring failed for paper %d: %s", i, r)
                output.append((self.default_score, None))
            else:
                output.append(r)
        return output
