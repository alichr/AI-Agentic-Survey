"""Tests for src.scoring.relevance_scorer — Pure numpy scoring."""

import numpy as np

from src.scoring.relevance_scorer import RelevanceScorer


class TestScore:
    def test_basic_scoring(self):
        probs = np.array([[0.1, 0.9], [0.5, 0.5]])
        scores = RelevanceScorer.score(probs)
        assert len(scores) == 2
        assert abs(scores[0] - 0.9) < 1e-6
        assert abs(scores[1] - 0.5) < 1e-6

    def test_single_cluster_all_one(self):
        probs = np.array([[1.0], [1.0], [1.0]])
        scores = RelevanceScorer.score(probs)
        for s in scores:
            assert abs(s - 1.0) < 1e-6

    def test_uniform_probabilities(self):
        k = 4
        probs = np.ones((3, k)) / k
        scores = RelevanceScorer.score(probs)
        for s in scores:
            assert abs(s - 1.0 / k) < 1e-6

    def test_returns_list_of_floats(self):
        probs = np.array([[0.3, 0.7]])
        scores = RelevanceScorer.score(probs)
        assert isinstance(scores, list)
        assert all(isinstance(s, float) for s in scores)

    def test_all_in_range(self):
        probs = np.random.dirichlet(np.ones(5), size=10)
        scores = RelevanceScorer.score(probs)
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_dominant_cluster(self):
        probs = np.array([[0.01, 0.01, 0.98]])
        scores = RelevanceScorer.score(probs)
        assert abs(scores[0] - 0.98) < 1e-6

    def test_empty_input_returns_empty(self):
        probs = np.array([]).reshape(0, 3)
        scores = RelevanceScorer.score(probs)
        assert scores == []
