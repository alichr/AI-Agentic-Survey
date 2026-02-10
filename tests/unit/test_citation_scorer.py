"""Tests for src.scoring.citation_scorer — Citation formula + async fetch."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cache.cache_manager import CacheManager
from src.external.semantic_scholar import CitationData, SemanticScholarClient
from src.models.paper import Author, Paper, PaperMetadata, PaperScores
from src.scoring.citation_scorer import CitationScorer


@pytest.fixture
def scorer_deps(tmp_path):
    """Create a CitationScorer with mock client and real cache."""
    mock_client = AsyncMock(spec=SemanticScholarClient)
    cache = CacheManager(str(tmp_path / "cache"))
    expected_citations = {0: 5, 1: 15, 2: 40, 3: 80, 4: 120, 5: 160}
    scorer = CitationScorer(
        client=mock_client,
        cache=cache,
        expected_citations=expected_citations,
        expected_citations_slope=30,
        cache_max_age_days=30,
    )
    yield scorer, mock_client, cache
    cache.close()


def _make_paper_with_title(title, year=None, tmp_path=None):
    from pathlib import Path
    return Paper(
        pdf_path=Path("/tmp/test.pdf"),
        venue="Test",
        pdf_hash="testhash",
        metadata=PaperMetadata(title=title, year=year),
    )


class TestGetExpectedCitations:
    def test_defined_age(self, scorer_deps):
        scorer, _, _ = scorer_deps
        assert scorer._get_expected_citations(0) == 5.0
        assert scorer._get_expected_citations(3) == 80.0
        assert scorer._get_expected_citations(5) == 160.0

    def test_extrapolated_age(self, scorer_deps):
        scorer, _, _ = scorer_deps
        # Age 7: base=160 + 30*(7-5) = 220
        assert scorer._get_expected_citations(7) == 220.0

    def test_negative_age_treated_as_zero(self, scorer_deps):
        scorer, _, _ = scorer_deps
        assert scorer._get_expected_citations(-1) == 5.0

    def test_interpolation(self, scorer_deps):
        scorer, _, _ = scorer_deps
        # Remove age 2 to test interpolation — actually the dict has it.
        # Let's test a scorer with gaps
        scorer.expected_citations = {0: 5, 3: 80, 5: 160}
        # Age 1: lower_ages = [0] -> returns expected_citations[0] = 5
        assert scorer._get_expected_citations(1) == 5.0


class TestComputeScore:
    def test_basic_score(self, scorer_deps):
        scorer, _, _ = scorer_deps
        current_year = datetime.now().year
        # Paper age 0, expected 5 citations, actual 5 -> 1.0
        score = scorer.compute_score(5, current_year)
        assert abs(score - 1.0) < 1e-6

    def test_below_expected(self, scorer_deps):
        scorer, _, _ = scorer_deps
        current_year = datetime.now().year
        # Paper age 0, expected 5, actual 2 -> 0.4
        score = scorer.compute_score(2, current_year)
        assert abs(score - 0.4) < 1e-6

    def test_above_expected_capped_at_one(self, scorer_deps):
        scorer, _, _ = scorer_deps
        current_year = datetime.now().year
        score = scorer.compute_score(100, current_year)
        assert abs(score - 1.0) < 1e-6

    def test_year_none_assumes_current(self, scorer_deps):
        scorer, _, _ = scorer_deps
        # year=None => age=0 => expected=5
        score = scorer.compute_score(5, None)
        assert abs(score - 1.0) < 1e-6

    def test_zero_citations(self, scorer_deps):
        scorer, _, _ = scorer_deps
        score = scorer.compute_score(0, datetime.now().year)
        assert abs(score - 0.0) < 1e-6

    def test_old_paper_score(self, scorer_deps):
        scorer, _, _ = scorer_deps
        # Paper from 5 years ago, expected 160, actual 80 -> 0.5
        score = scorer.compute_score(80, datetime.now().year - 5)
        assert abs(score - 0.5) < 1e-6


class TestFetchAndScore:
    async def test_cache_hit_no_api_call(self, scorer_deps):
        scorer, mock_client, cache = scorer_deps
        cache.set_citation("Test Paper", 50, 2023)

        paper = _make_paper_with_title("Test Paper", 2023)
        score, count = await scorer.fetch_and_score(paper)
        assert count == 50
        assert score is not None
        mock_client.search_paper.assert_not_called()

    async def test_cache_miss_api_hit(self, scorer_deps):
        scorer, mock_client, cache = scorer_deps
        mock_client.search_paper.return_value = CitationData(
            citation_count=25, year=2024, paper_id="abc"
        )

        paper = _make_paper_with_title("New Paper", 2024)
        score, count = await scorer.fetch_and_score(paper)
        assert count == 25
        assert score is not None
        mock_client.search_paper.assert_called_once_with("New Paper")

        # Verify cache was populated
        cached = cache.get_citation("New Paper", max_age_days=30)
        assert cached is not None
        assert cached["citation_count"] == 25

    async def test_no_title_returns_none(self, scorer_deps):
        scorer, mock_client, _ = scorer_deps
        paper = Paper(
            pdf_path=__import__("pathlib").Path("/tmp/t.pdf"),
            venue="V",
            pdf_hash="h",
        )
        score, count = await scorer.fetch_and_score(paper)
        assert score is None
        assert count is None

    async def test_api_returns_none(self, scorer_deps):
        scorer, mock_client, _ = scorer_deps
        mock_client.search_paper.return_value = None

        paper = _make_paper_with_title("Unknown Paper")
        score, count = await scorer.fetch_and_score(paper)
        assert score is None
        assert count is None


class TestFetchAndScoreYearFallback:
    """Tests for consistent year fallback between cached and uncached paths."""

    async def test_cached_path_uses_paper_year_when_api_year_is_none(self, scorer_deps):
        scorer, mock_client, cache = scorer_deps
        # Cache entry with no year (year=None)
        cache.set_citation("Test Paper", 50, None)
        paper = _make_paper_with_title("Test Paper", 2023)
        score, count = await scorer.fetch_and_score(paper)
        # Should use paper's year (2023) as fallback, not None
        assert count == 50
        assert score is not None
        expected_score = scorer.compute_score(50, 2023)
        assert abs(score - expected_score) < 1e-6

    async def test_uncached_path_uses_paper_year_when_api_year_is_none(self, scorer_deps):
        scorer, mock_client, cache = scorer_deps
        mock_client.search_paper.return_value = CitationData(
            citation_count=50, year=None, paper_id="abc"
        )
        paper = _make_paper_with_title("Test Paper", 2023)
        score, count = await scorer.fetch_and_score(paper)
        assert count == 50
        assert score is not None
        expected_score = scorer.compute_score(50, 2023)
        assert abs(score - expected_score) < 1e-6


class TestScoreBatch:
    async def test_processes_multiple_papers(self, scorer_deps):
        scorer, mock_client, cache = scorer_deps
        mock_client.search_paper.return_value = CitationData(
            citation_count=10, year=2024
        )

        papers = [
            _make_paper_with_title("Paper 1", 2024),
            _make_paper_with_title("Paper 2", 2024),
        ]
        results = await scorer.score_batch(papers)
        assert len(results) == 2
        for score, count in results:
            assert score is not None
            assert count == 10

    async def test_batch_handles_exception_gracefully(self, scorer_deps):
        """One paper raising an exception should not kill the entire batch."""
        scorer, mock_client, cache = scorer_deps

        call_count = 0
        async def side_effect(title):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("API exploded")
            return CitationData(citation_count=10, year=2024)

        mock_client.search_paper.side_effect = side_effect

        papers = [
            _make_paper_with_title("Exploding Paper", 2024),
            _make_paper_with_title("Good Paper", 2024),
        ]
        results = await scorer.score_batch(papers)
        assert len(results) == 2
        # First paper should return (None, None) due to exception
        assert results[0] == (None, None)
        # Second paper should succeed
        assert results[1][0] is not None
        assert results[1][1] == 10
