"""Tests for src.scoring.aggregator — AND-logic threshold tests."""

from pathlib import Path

from src.config import AffiliationConfig, CitationConfig, Config, RelevanceConfig
from src.models.paper import Paper, PaperScores
from src.scoring.aggregator import ScoreAggregator


def _make_config(relevance_enabled=True, relevance_threshold=0.3,
                 affiliation_enabled=True, affiliation_threshold=0.3,
                 citation_enabled=True, citation_threshold=0.2):
    config = Config()
    config.relevance = RelevanceConfig(enabled=relevance_enabled, threshold=relevance_threshold)
    config.affiliation = AffiliationConfig(enabled=affiliation_enabled, threshold=affiliation_threshold)
    config.citation = CitationConfig(enabled=citation_enabled, threshold=citation_threshold)
    return config


def _make_paper(relevance=None, affiliation=None, citation=None):
    return Paper(
        pdf_path=Path("/tmp/t.pdf"),
        venue="V",
        pdf_hash="h",
        scores=PaperScores(
            relevance_score=relevance,
            affiliation_score=affiliation,
            citation_score=citation,
        ),
    )


class TestDecide:
    def test_all_pass(self):
        agg = ScoreAggregator(_make_config())
        paper = _make_paper(relevance=0.5, affiliation=0.5, citation=0.5)
        assert agg.decide(paper) is True

    def test_relevance_below_threshold_rejected(self):
        agg = ScoreAggregator(_make_config())
        paper = _make_paper(relevance=0.1, affiliation=0.5, citation=0.5)
        assert agg.decide(paper) is False

    def test_affiliation_below_threshold_rejected(self):
        agg = ScoreAggregator(_make_config())
        paper = _make_paper(relevance=0.5, affiliation=0.1, citation=0.5)
        assert agg.decide(paper) is False

    def test_citation_below_threshold_rejected(self):
        agg = ScoreAggregator(_make_config())
        paper = _make_paper(relevance=0.5, affiliation=0.5, citation=0.1)
        assert agg.decide(paper) is False

    def test_none_relevance_rejected(self):
        agg = ScoreAggregator(_make_config())
        paper = _make_paper(relevance=None, affiliation=0.5, citation=0.5)
        assert agg.decide(paper) is False

    def test_none_affiliation_rejected(self):
        agg = ScoreAggregator(_make_config())
        paper = _make_paper(relevance=0.5, affiliation=None, citation=0.5)
        assert agg.decide(paper) is False

    def test_none_citation_rejected(self):
        agg = ScoreAggregator(_make_config())
        paper = _make_paper(relevance=0.5, affiliation=0.5, citation=None)
        assert agg.decide(paper) is False

    def test_relevance_disabled_passes_regardless(self):
        agg = ScoreAggregator(_make_config(relevance_enabled=False))
        paper = _make_paper(relevance=None, affiliation=0.5, citation=0.5)
        assert agg.decide(paper) is True

    def test_affiliation_disabled_passes_regardless(self):
        agg = ScoreAggregator(_make_config(affiliation_enabled=False))
        paper = _make_paper(relevance=0.5, affiliation=None, citation=0.5)
        assert agg.decide(paper) is True

    def test_citation_disabled_passes_regardless(self):
        agg = ScoreAggregator(_make_config(citation_enabled=False))
        paper = _make_paper(relevance=0.5, affiliation=0.5, citation=None)
        assert agg.decide(paper) is True

    def test_all_disabled_always_accepted(self):
        agg = ScoreAggregator(_make_config(
            relevance_enabled=False,
            affiliation_enabled=False,
            citation_enabled=False,
        ))
        paper = _make_paper(relevance=None, affiliation=None, citation=None)
        assert agg.decide(paper) is True

    def test_at_threshold_accepted(self):
        agg = ScoreAggregator(_make_config(
            relevance_threshold=0.3,
            affiliation_threshold=0.3,
            citation_threshold=0.2,
        ))
        paper = _make_paper(relevance=0.3, affiliation=0.3, citation=0.2)
        assert agg.decide(paper) is True

    def test_just_below_threshold_rejected(self):
        agg = ScoreAggregator(_make_config(relevance_threshold=0.3))
        paper = _make_paper(relevance=0.299, affiliation=0.5, citation=0.5)
        assert agg.decide(paper) is False


class TestDecideAll:
    def test_sets_accepted_field(self):
        agg = ScoreAggregator(_make_config())
        papers = [
            _make_paper(relevance=0.5, affiliation=0.5, citation=0.5),
            _make_paper(relevance=0.1, affiliation=0.5, citation=0.5),
        ]
        result = agg.decide_all(papers)
        assert result[0].accepted is True
        assert result[1].accepted is False

    def test_returns_same_list(self):
        agg = ScoreAggregator(_make_config())
        papers = [_make_paper(relevance=0.5, affiliation=0.5, citation=0.5)]
        result = agg.decide_all(papers)
        assert result is papers

    def test_empty_list(self):
        agg = ScoreAggregator(_make_config())
        result = agg.decide_all([])
        assert result == []
