"""GMM clustering with seed paper label propagation."""

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Result of clustering a set of papers."""
    cluster_ids: np.ndarray          # Shape (n_papers,) - hard cluster assignments
    cluster_probabilities: np.ndarray  # Shape (n_papers, n_clusters) - soft assignments
    cluster_labels: dict[int, str]    # Cluster ID -> human-readable label
    gmm: Optional[GaussianMixture]    # Fitted GMM model (None if n_papers < 2)


class PaperClusterer:
    """Clusters paper embeddings using Gaussian Mixture Models.

    After fitting, propagates topic labels from seed papers to clusters.
    """

    def __init__(self, n_clusters: int = 10, n_init: int = 5,
                 random_state: int = 42):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state

    def fit_predict(self, embeddings: np.ndarray,
                    seed_indices: list[int],
                    seed_labels: list[str]) -> ClusterResult:
        """Fit GMM on all embeddings and propagate seed labels.

        Args:
            embeddings: Array of shape (n_papers, embedding_dim) for all papers
                        (seed + candidate).
            seed_indices: Indices into `embeddings` corresponding to seed papers.
            seed_labels: Topic labels for each seed paper (parallel to seed_indices).

        Returns:
            ClusterResult with assignments, probabilities, and labels.
        """
        n_papers = embeddings.shape[0]
        if n_papers < 2:
            logger.warning("Only %d paper(s) — skipping GMM, assigning all to cluster 0", n_papers)
            cluster_ids = np.zeros(n_papers, dtype=int)
            cluster_probs = np.ones((n_papers, 1), dtype=np.float64)
            cluster_labels = self._propagate_labels(cluster_ids, seed_indices, seed_labels, 1)
            return ClusterResult(
                cluster_ids=cluster_ids,
                cluster_probabilities=cluster_probs,
                cluster_labels=cluster_labels,
                gmm=None,
            )

        actual_k = min(self.n_clusters, n_papers)

        logger.info("Fitting GMM with %d components on %d papers", actual_k, n_papers)

        gmm = GaussianMixture(
            n_components=actual_k,
            covariance_type="full",
            n_init=self.n_init,
            random_state=self.random_state,
        )
        gmm.fit(embeddings)

        cluster_ids = gmm.predict(embeddings)
        cluster_probs = gmm.predict_proba(embeddings)

        # Propagate labels from seed papers
        cluster_labels = self._propagate_labels(
            cluster_ids, seed_indices, seed_labels, actual_k
        )

        logger.info("Clustering complete. Cluster label mapping: %s", cluster_labels)

        return ClusterResult(
            cluster_ids=cluster_ids,
            cluster_probabilities=cluster_probs,
            cluster_labels=cluster_labels,
            gmm=gmm,
        )

    @staticmethod
    def _propagate_labels(cluster_ids: np.ndarray,
                          seed_indices: list[int],
                          seed_labels: list[str],
                          n_clusters: int) -> dict[int, str]:
        """Map cluster IDs to topic labels using seed papers.

        If multiple seed papers with different labels map to the same cluster,
        the most common label wins. Clusters without seed papers get
        'Miscellaneous-{id}'.
        """
        cluster_to_labels: dict[int, list[str]] = {}

        for idx, label in zip(seed_indices, seed_labels):
            cid = int(cluster_ids[idx])
            cluster_to_labels.setdefault(cid, []).append(label)

        labels = {}
        for cid in range(n_clusters):
            if cid in cluster_to_labels:
                counter = Counter(cluster_to_labels[cid])
                labels[cid] = counter.most_common(1)[0][0]
            else:
                labels[cid] = f"Miscellaneous-{cid}"

        return labels
