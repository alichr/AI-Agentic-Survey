"""Company tier lookup from configurable YAML tier list."""

import logging
from pathlib import Path
from typing import Optional

import yaml
from thefuzz import fuzz, process

logger = logging.getLogger(__name__)


class CompanyTiers:
    """Looks up company tier scores with fuzzy name matching."""

    def __init__(self, tiers_path: str):
        self.company_to_tier: dict[str, int] = {}  # lowercase name -> tier
        self.tier_scores: dict[int, float] = {}
        self.unknown_score: float = 0.3
        self._names: list[str] = []
        self._load(tiers_path)

    def _load(self, path: str):
        """Load company tiers from YAML file."""
        yaml_path = Path(path)
        if not yaml_path.exists():
            logger.warning("Company tiers file not found: %s", path)
            return

        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}

        self.tier_scores = {int(k): v for k, v in data.get("tier_scores", {}).items()}
        self.unknown_score = data.get("unknown_score", 0.3)

        tiers = data.get("tiers", {})
        for tier_num, companies in tiers.items():
            tier_num = int(tier_num)
            for company in companies:
                name_lower = company.lower()
                self.company_to_tier[name_lower] = tier_num
                self._names.append(name_lower)

        logger.info("Loaded %d companies across %d tiers",
                     len(self.company_to_tier), len(tiers))

    def get_tier(self, affiliation: str) -> Optional[int]:
        """Look up the tier for an affiliation string.

        Args:
            affiliation: Company name or affiliation string.

        Returns:
            Tier number if matched, None otherwise.
        """
        if not affiliation or not self._names:
            return None

        aff_lower = affiliation.lower().strip()

        # Exact match
        if aff_lower in self.company_to_tier:
            return self.company_to_tier[aff_lower]

        # Substring match: check if any known company name appears in the affiliation
        for name, tier in self.company_to_tier.items():
            if name in aff_lower:
                return tier

        # Fuzzy match
        result = process.extractOne(
            aff_lower, self._names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=80,
        )
        if result:
            matched_name = result[0]
            return self.company_to_tier.get(matched_name)

        return None

    def score_affiliation(self, affiliation: str) -> float:
        """Get the score for an affiliation string.

        Args:
            affiliation: Company name or affiliation text.

        Returns:
            Tier score if matched, unknown_score otherwise.
        """
        tier = self.get_tier(affiliation)
        if tier is not None:
            return self.tier_scores.get(tier, self.unknown_score)
        return self.unknown_score
