"""Integration tests: multi-scorer -> aggregator flow."""

from pathlib import Path

import numpy as np
import pytest

from src.config import AffiliationConfig, CitationConfig, Config, RelevanceConfig
from src.embedding.clustering import PaperClusterer
from src.external.company_tiers import CompanyTiers
from src.external.university_rankings import UniversityRankings
from src.models.paper import Author, Paper, PaperMetadata, PaperScores
from src.scoring.affiliation_scorer import AffiliationScorer
from src.scoring.aggregator import ScoreAggregator
from src.scoring.relevance_scorer import RelevanceScorer

pytestmark = pytest.mark.integration


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


class TestEmbeddingClusteringRelevanceAggregator:
    def test_full_flow(self, synthetic_embeddings):
        """Embeddings -> clustering -> relevance -> aggregator."""
        clusterer = PaperClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(synthetic_embeddings, [], [])

        # Compute relevance scores
        scores = RelevanceScorer.score(result.cluster_probabilities)
        assert len(scores) == 30

        # Create papers with relevance scores
        config = Config()
        config.relevance = RelevanceConfig(enabled=True, threshold=0.3)
        config.affiliation = AffiliationConfig(enabled=False)
        config.citation = CitationConfig(enabled=False)
        agg = ScoreAggregator(config)

        papers = []
        for i, score in enumerate(scores):
            p = _make_paper(relevance=score)
            papers.append(p)

        agg.decide_all(papers)
        accepted = [p for p in papers if p.accepted]
        rejected = [p for p in papers if not p.accepted]
        assert len(accepted) + len(rejected) == 30

    def test_seed_papers_get_high_relevance(self, synthetic_embeddings):
        """Seed papers should have high relevance (tight cluster membership)."""
        clusterer = PaperClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(
            synthetic_embeddings,
            seed_indices=[0, 10, 20],
            seed_labels=["A", "B", "C"],
        )
        scores = RelevanceScorer.score(result.cluster_probabilities)
        # Seed papers (indices 0, 10, 20) near cluster centers should have high scores
        for idx in [0, 10, 20]:
            assert scores[idx] > 0.5


class TestAffiliationScoringAggregator:
    def test_affiliation_to_aggregator(self, sample_rankings_csv, sample_tiers_yaml):
        uni_rankings = UniversityRankings(sample_rankings_csv)
        company_tiers = CompanyTiers(sample_tiers_yaml)

        scorer = AffiliationScorer(
            university_rankings=uni_rankings,
            company_tiers=company_tiers,
        )

        # Paper with MIT author -> high affiliation score
        paper = Paper(
            pdf_path=Path("/tmp/t.pdf"),
            venue="V",
            pdf_hash="h",
            metadata=PaperMetadata(
                title="T",
                authors=[
                    Author("A", "Massachusetts Institute of Technology"),
                    Author("B", "Google"),
                ],
            ),
        )
        paper.scores.affiliation_score = scorer.score(paper.first_author, paper.last_author)

        config = Config()
        config.relevance = RelevanceConfig(enabled=False)
        config.affiliation = AffiliationConfig(enabled=True, threshold=0.3)
        config.citation = CitationConfig(enabled=False)
        agg = ScoreAggregator(config)

        assert agg.decide(paper) is True
        assert paper.scores.affiliation_score >= 0.3


class TestAllScorersCombined:
    def test_all_three_through_aggregator(self):
        config = Config()
        config.relevance = RelevanceConfig(enabled=True, threshold=0.3)
        config.affiliation = AffiliationConfig(enabled=True, threshold=0.3)
        config.citation = CitationConfig(enabled=True, threshold=0.2)
        agg = ScoreAggregator(config)

        # Paper passing all thresholds
        p1 = _make_paper(relevance=0.5, affiliation=0.5, citation=0.5)
        assert agg.decide(p1) is True

        # Paper failing one
        p2 = _make_paper(relevance=0.5, affiliation=0.5, citation=0.1)
        assert agg.decide(p2) is False

    def test_borderline_at_threshold(self):
        config = Config()
        config.relevance = RelevanceConfig(enabled=True, threshold=0.3)
        config.affiliation = AffiliationConfig(enabled=True, threshold=0.3)
        config.citation = CitationConfig(enabled=True, threshold=0.2)
        agg = ScoreAggregator(config)

        # Exactly at threshold
        p1 = _make_paper(relevance=0.3, affiliation=0.3, citation=0.2)
        assert agg.decide(p1) is True

        # Just below one threshold
        p2 = _make_paper(relevance=0.3, affiliation=0.3, citation=0.199)
        assert agg.decide(p2) is False
