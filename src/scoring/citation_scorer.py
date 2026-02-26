"""Citation + recency scoring using external citation APIs."""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Protocol

from src.cache.cache_manager import CacheManager
from src.models.paper import Paper

logger = logging.getLogger(__name__)


class CitationClient(Protocol):
    """Any client that can search for a paper and return citation data."""

    async def search_paper(self, title: str): ...


class CitationScorer:
    """Scores papers based on citation count relative to expected citations by age.

    Formula: score = min(actual_citations / expected_citations, 1.0)
    This naturally rewards recent papers with even modest citations.

    Supports a fallback client: queries both sources and uses the higher
    citation count. This handles cases where OpenAlex has split preprint/
    published records with undercounted citations.
    """

    def __init__(self, client: CitationClient,
                 cache: CacheManager,
                 expected_citations: dict[int, int],
                 expected_citations_slope: int = 30,
                 cache_max_age_days: int = 30,
                 fallback_client: Optional[CitationClient] = None):
        self.client = client
        self.fallback_client = fallback_client
        self.cache = cache
        self.expected_citations = expected_citations
        self.slope = expected_citations_slope
        self.cache_max_age_days = cache_max_age_days
        self.current_year = datetime.now().year

    def _get_expected_citations(self, paper_age: int) -> float:
        """Look up expected citation count for a given paper age.

        Args:
            paper_age: Years since publication.

        Returns:
            Expected number of citations.
        """
        if paper_age < 0:
            paper_age = 0

        if paper_age in self.expected_citations:
            return float(self.expected_citations[paper_age])

        # For ages beyond the table, extrapolate linearly
        max_defined_age = max(self.expected_citations.keys())
        if paper_age > max_defined_age:
            base = self.expected_citations[max_defined_age]
            return float(base + self.slope * (paper_age - max_defined_age))

        # Interpolate if somehow between defined ages
        lower_ages = [a for a in self.expected_citations if a <= paper_age]
        if lower_ages:
            return float(self.expected_citations[max(lower_ages)])

        return 5.0  # Fallback minimum

    def compute_score(self, citation_count: int, year: Optional[int]) -> float:
        """Compute citation score for a paper.

        Args:
            citation_count: Actual citation count.
            year: Publication year.

        Returns:
            Score in [0, 1].
        """
        if year is None:
            year = self.current_year  # Assume recent if unknown

        paper_age = self.current_year - year
        expected = self._get_expected_citations(paper_age)

        if expected <= 0:
            return 1.0 if citation_count > 0 else 0.0

        return min(citation_count / expected, 1.0)

    async def fetch_and_score(self, paper: Paper) -> tuple[Optional[float], Optional[int]]:
        """Fetch citation data and compute score for a paper.

        Checks cache first, then queries Semantic Scholar API.

        Args:
            paper: Paper with metadata (needs title and year).

        Returns:
            Tuple of (score, citation_count), or (None, None) on failure.
        """
        title = paper.title
        if not title:
            return None, None

        # Resolve year with consistent fallback: API year -> paper metadata year -> None
        paper_year = paper.metadata.year if paper.metadata else None

        # Check cache
        cached = self.cache.get_citation(title, self.cache_max_age_days)
        if cached is not None:
            year = cached["year"] or paper_year
            score = self.compute_score(cached["citation_count"], year)
            return score, cached["citation_count"]

        # Fetch from primary API
        result = await self.client.search_paper(title)

        # Fetch from fallback API and keep the higher citation count.
        # This handles OpenAlex split-record issues where preprint and
        # published versions have separate (undercounted) citation totals.
        if self.fallback_client is not None:
            try:
                fallback = await self.fallback_client.search_paper(title)
                if fallback is not None:
                    if result is None:
                        result = fallback
                    elif fallback.citation_count > result.citation_count:
                        logger.debug(
                            "Fallback has higher citations for '%s': %d > %d",
                            title[:50], fallback.citation_count, result.citation_count,
                        )
                        # Keep the higher count but preserve primary paper_id
                        result.citation_count = fallback.citation_count
                        result.year = result.year or fallback.year
            except Exception as e:
                logger.debug("Fallback citation lookup failed for '%s': %s", title[:50], e)

        if result is None:
            logger.debug("No citation result for: %s", title[:60])
            return None, None

        # Cache the result (including paper_id for reuse by h-index scorer)
        self.cache.set_citation(title, result.citation_count, result.year, result.paper_id)

        year = result.year or paper_year
        score = self.compute_score(result.citation_count, year)
        return score, result.citation_count

    async def score_batch(self, papers: list[Paper]) -> list[tuple[Optional[float], Optional[int]]]:
        """Fetch and score citations for multiple papers.

        Args:
            papers: List of papers to score.

        Returns:
            List of (score, citation_count) tuples parallel to input.
        """
        tasks = [self.fetch_and_score(p) for p in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error("Citation scoring failed for paper %d: %s", i, r)
                output.append((None, None))
            else:
                output.append(r)
        return output
