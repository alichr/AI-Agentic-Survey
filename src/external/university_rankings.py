"""QS World University Rankings lookup with fuzzy matching."""

import csv
import logging
from pathlib import Path
from typing import Optional

from thefuzz import fuzz, process

logger = logging.getLogger(__name__)

# Rank -> score mapping (finer-grained tiers for top-200 coverage)
RANK_SCORE_MAP = [
    (10, 1.0),    # Top 10: world-leading
    (25, 0.95),   # Top 25: elite
    (50, 0.90),   # Top 50: excellent
    (75, 0.85),   # Top 75: very strong
    (100, 0.80),  # Top 100: strong
    (150, 0.70),  # Top 150: well-regarded
    (200, 0.60),  # Top 200: reputable
]
UNRANKED_SCORE = 0.4


class UniversityRankings:
    """Looks up university rankings from QS data with fuzzy name matching."""

    def __init__(self, rankings_path: str):
        self.rankings: dict[str, int] = {}  # normalized name -> rank
        self._names: list[str] = []
        self._load(rankings_path)

    def _load(self, path: str):
        """Load rankings from CSV file."""
        csv_path = Path(path)
        if not csv_path.exists():
            logger.warning("University rankings file not found: %s", path)
            return

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rank_str = row.get("rank", "").strip()
                name = row.get("university", "").strip()
                if not rank_str or not name:
                    continue
                try:
                    rank = int(rank_str.split("-")[0]) if "-" in rank_str else int(rank_str)
                except ValueError:
                    continue
                self.rankings[name.lower()] = rank
                self._names.append(name.lower())

        logger.info("Loaded %d university rankings", len(self.rankings))

    def get_rank(self, affiliation: str) -> Optional[int]:
        """Look up the rank for an affiliation string using fuzzy matching.

        Args:
            affiliation: University name or affiliation string.

        Returns:
            University rank if a good match found, None otherwise.
        """
        if not affiliation or not self._names:
            return None

        aff_lower = affiliation.lower().strip()

        # Try exact match first
        if aff_lower in self.rankings:
            return self.rankings[aff_lower]

        # Try substring match
        for name, rank in self.rankings.items():
            if name in aff_lower or aff_lower in name:
                return rank

        # Fuzzy match
        result = process.extractOne(
            aff_lower, self._names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=85,
        )
        if result:
            matched_name = result[0]
            return self.rankings.get(matched_name)

        return None

    def rank_to_score(self, rank: Optional[int]) -> float:
        """Convert a university rank to a normalized score.

        Args:
            rank: University rank (1-based), or None if unranked.

        Returns:
            Score in [0.4, 1.0].
        """
        if rank is None:
            return UNRANKED_SCORE

        for threshold, score in RANK_SCORE_MAP:
            if rank <= threshold:
                return score

        return UNRANKED_SCORE

    def score_affiliation(self, affiliation: str) -> float:
        """Get the score for an affiliation string.

        Args:
            affiliation: University name or affiliation text.

        Returns:
            Score in [0.4, 1.0].
        """
        rank = self.get_rank(affiliation)
        return self.rank_to_score(rank)
