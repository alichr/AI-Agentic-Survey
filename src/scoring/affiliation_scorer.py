"""Affiliation scoring combining university rankings and company tiers."""

import logging
from typing import Optional

from src.external.company_tiers import CompanyTiers
from src.external.university_rankings import UniversityRankings
from src.models.paper import Author

logger = logging.getLogger(__name__)


class AffiliationScorer:
    """Scores papers based on author affiliations.

    Combines university rankings and company tier lookups. For each author,
    takes the max of university score and company score. Then computes a
    weighted combination of first and last author scores.
    """

    def __init__(self, university_rankings: UniversityRankings,
                 company_tiers: CompanyTiers,
                 first_author_weight: float = 0.5,
                 last_author_weight: float = 0.5):
        self.uni_rankings = university_rankings
        self.company_tiers = company_tiers
        self.first_author_weight = first_author_weight
        self.last_author_weight = last_author_weight

    def score_author(self, author: Optional[Author]) -> float:
        """Score a single author based on their affiliation.

        Takes the max of university score and company score.

        Args:
            author: Author with affiliation, or None.

        Returns:
            Score in [0.3, 1.0].
        """
        if author is None or not author.affiliation:
            return 0.3  # Minimum score for unknown affiliation

        uni_score = self.uni_rankings.score_affiliation(author.affiliation)
        company_score = self.company_tiers.score_affiliation(author.affiliation)

        return max(uni_score, company_score)

    def score(self, first_author: Optional[Author],
              last_author: Optional[Author]) -> float:
        """Compute combined affiliation score for a paper.

        Args:
            first_author: First author of the paper.
            last_author: Last author (senior author) of the paper.

        Returns:
            Weighted combination of first and last author scores.
        """
        first_score = self.score_author(first_author)
        last_score = self.score_author(last_author)

        combined = (self.first_author_weight * first_score +
                    self.last_author_weight * last_score)

        return combined
