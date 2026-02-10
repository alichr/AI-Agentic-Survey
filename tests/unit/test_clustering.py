"""Tests for src.embedding.clustering — GMM clustering (real sklearn, synthetic data)."""

import numpy as np

from src.embedding.clustering import ClusterResult, PaperClusterer


class TestFitPredict:
    def test_returns_cluster_result(self, synthetic_embeddings):
        clusterer = PaperClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(synthetic_embeddings, [], [])
        assert isinstance(result, ClusterResult)

    def test_cluster_ids_shape(self, synthetic_embeddings):
        clusterer = PaperClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(synthetic_embeddings, [], [])
        assert result.cluster_ids.shape == (30,)

    def test_cluster_probabilities_shape(self, synthetic_embeddings):
        clusterer = PaperClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(synthetic_embeddings, [], [])
        assert result.cluster_probabilities.shape == (30, 3)

    def test_probabilities_sum_to_one(self, synthetic_embeddings):
        clusterer = PaperClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(synthetic_embeddings, [], [])
        row_sums = result.cluster_probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_seed_label_propagation(self, synthetic_embeddings):
        clusterer = PaperClusterer(n_clusters=3, random_state=42)
        # Papers 0-9 are cluster 0, 10-19 cluster 1, 20-29 cluster 2 (approximately)
        result = clusterer.fit_predict(
            synthetic_embeddings,
            seed_indices=[0, 10, 20],
            seed_labels=["Topic A", "Topic B", "Topic C"],
        )
        # Each cluster should get a label from seeds
        labels = set(result.cluster_labels.values())
        assert "Topic A" in labels or "Topic B" in labels or "Topic C" in labels

    def test_unlabeled_clusters_get_miscellaneous(self, synthetic_embeddings):
        clusterer = PaperClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(synthetic_embeddings, [], [])
        for label in result.cluster_labels.values():
            assert label.startswith("Miscellaneous-")

    def test_majority_label_wins(self, synthetic_embeddings):
        clusterer = PaperClusterer(n_clusters=3, random_state=42)
        # Assign multiple seeds from the same group with different labels
        result = clusterer.fit_predict(
            synthetic_embeddings,
            seed_indices=[0, 1, 2],
            seed_labels=["Topic A", "Topic A", "Topic B"],
        )
        # The cluster containing indices 0,1,2 should be labeled "Topic A" (majority)
        cid = int(result.cluster_ids[0])
        assert result.cluster_labels[cid] == "Topic A"

    def test_n_clusters_capped_at_n_papers(self):
        # 5 papers but request 20 clusters
        embeddings = np.random.randn(5, 2).astype(np.float32)
        clusterer = PaperClusterer(n_clusters=20, random_state=42)
        result = clusterer.fit_predict(embeddings, [], [])
        assert result.cluster_ids.shape == (5,)
        assert result.cluster_probabilities.shape[0] == 5
        assert result.cluster_probabilities.shape[1] == 5  # capped

    def test_deterministic_with_seed(self, synthetic_embeddings):
        clusterer = PaperClusterer(n_clusters=3, random_state=42)
        r1 = clusterer.fit_predict(synthetic_embeddings, [], [])
        r2 = clusterer.fit_predict(synthetic_embeddings, [], [])
        np.testing.assert_array_equal(r1.cluster_ids, r2.cluster_ids)

    def test_gmm_is_fitted(self, synthetic_embeddings):
        clusterer = PaperClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit_predict(synthetic_embeddings, [], [])
        # GMM should have means_ attribute after fitting
        assert hasattr(result.gmm, "means_")

    def test_single_paper_no_crash(self):
        """Single paper should not crash GMM — returns cluster 0 with prob 1.0."""
        embeddings = np.random.randn(1, 8).astype(np.float32)
        clusterer = PaperClusterer(n_clusters=5, random_state=42)
        result = clusterer.fit_predict(embeddings, [], [])
        assert result.cluster_ids.shape == (1,)
        assert result.cluster_ids[0] == 0
        assert result.cluster_probabilities.shape == (1, 1)
        assert abs(result.cluster_probabilities[0, 0] - 1.0) < 1e-6
        assert result.gmm is None  # No GMM fitted for single paper


class TestPropagateLabels:
    def test_basic_propagation(self):
        cluster_ids = np.array([0, 0, 1, 1, 2, 2])
        labels = PaperClusterer._propagate_labels(
            cluster_ids,
            seed_indices=[0, 2, 4],
            seed_labels=["A", "B", "C"],
            n_clusters=3,
        )
        assert labels[0] == "A"
        assert labels[1] == "B"
        assert labels[2] == "C"

    def test_no_seeds(self):
        cluster_ids = np.array([0, 1, 2])
        labels = PaperClusterer._propagate_labels(cluster_ids, [], [], 3)
        assert labels[0] == "Miscellaneous-0"
        assert labels[1] == "Miscellaneous-1"
        assert labels[2] == "Miscellaneous-2"

    def test_majority_voting(self):
        cluster_ids = np.array([0, 0, 0, 1, 1])
        labels = PaperClusterer._propagate_labels(
            cluster_ids,
            seed_indices=[0, 1, 2],
            seed_labels=["X", "X", "Y"],
            n_clusters=2,
        )
        assert labels[0] == "X"  # majority
