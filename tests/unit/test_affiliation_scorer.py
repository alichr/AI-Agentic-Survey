"""Tests for src.scoring.affiliation_scorer — Weighted author scoring."""

from unittest.mock import MagicMock

from src.models.paper import Author
from src.scoring.affiliation_scorer import AffiliationScorer


def _make_scorer(uni_score=0.4, company_score=0.3):
    """Create an AffiliationScorer with mock rankings/tiers."""
    uni_rankings = MagicMock()
    uni_rankings.score_affiliation.return_value = uni_score

    company_tiers = MagicMock()
    company_tiers.score_affiliation.return_value = company_score

    return AffiliationScorer(
        university_rankings=uni_rankings,
        company_tiers=company_tiers,
        first_author_weight=0.5,
        last_author_weight=0.5,
    )


class TestScoreAuthor:
    def test_ranked_university(self):
        scorer = _make_scorer(uni_score=0.9, company_score=0.3)
        author = Author("Alice", "MIT")
        score = scorer.score_author(author)
        assert score == 0.9  # max(0.9, 0.3)

    def test_tier1_company(self):
        scorer = _make_scorer(uni_score=0.4, company_score=1.0)
        author = Author("Bob", "Google")
        score = scorer.score_author(author)
        assert score == 1.0  # max(0.4, 1.0)

    def test_max_of_uni_and_company(self):
        scorer = _make_scorer(uni_score=0.7, company_score=0.8)
        author = Author("Carol", "Some Org")
        score = scorer.score_author(author)
        assert score == 0.8

    def test_none_author_returns_default(self):
        scorer = _make_scorer()
        score = scorer.score_author(None)
        assert score == 0.3

    def test_author_with_none_affiliation(self):
        scorer = _make_scorer()
        author = Author("Dave", None)
        score = scorer.score_author(author)
        assert score == 0.3

    def test_author_with_empty_affiliation(self):
        scorer = _make_scorer()
        author = Author("Eve", "")
        score = scorer.score_author(author)
        assert score == 0.3


class TestScore:
    def test_equal_weights(self):
        scorer = _make_scorer(uni_score=0.8, company_score=0.3)
        first = Author("A", "MIT")
        last = Author("B", "MIT")
        score = scorer.score(first, last)
        # Both return 0.8 (max of uni=0.8 and company=0.3)
        assert abs(score - 0.8) < 1e-6

    def test_asymmetric_weights(self):
        uni_rankings = MagicMock()
        company_tiers = MagicMock()

        # First author uni=1.0, company=0.3 -> 1.0
        # Last author uni=0.4, company=0.3 -> 0.4
        call_count = [0]

        def _uni_score(aff):
            call_count[0] += 1
            return 1.0 if call_count[0] == 1 else 0.4

        uni_rankings.score_affiliation.side_effect = _uni_score
        company_tiers.score_affiliation.return_value = 0.3

        scorer = AffiliationScorer(
            university_rankings=uni_rankings,
            company_tiers=company_tiers,
            first_author_weight=0.7,
            last_author_weight=0.3,
        )
        first = Author("A", "MIT")
        last = Author("B", "Unknown Uni")
        score = scorer.score(first, last)
        # 0.7 * 1.0 + 0.3 * 0.4 = 0.82
        assert abs(score - 0.82) < 1e-6

    def test_both_none_returns_default(self):
        scorer = _make_scorer()
        score = scorer.score(None, None)
        assert abs(score - 0.3) < 1e-6

    def test_first_none_last_present(self):
        scorer = _make_scorer(uni_score=0.9, company_score=0.3)
        score = scorer.score(None, Author("B", "MIT"))
        # 0.5 * 0.3 (None default) + 0.5 * 0.9 = 0.60
        assert abs(score - 0.60) < 1e-6
