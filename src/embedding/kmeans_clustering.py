"""K-Means clustering with optional PCA for multi-view pipeline."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclass
class KMeansResult:
    """Result of K-Means clustering for a single section view."""
    section_type: int
    cluster_ids: np.ndarray       # shape (n_papers,) — hard assignments
    centroids: np.ndarray         # shape (k, dim)
    centroid_distances: np.ndarray # shape (n_papers,) — distance to assigned centroid
    n_clusters: int
    weak_members: np.ndarray      # shape (n_papers,) — boolean mask


class MultiViewKMeans:
    """K-Means clustering with PCA dimensionality reduction and weak member detection.

    High-dimensional embeddings (e.g., 1024-d) suffer from the curse of
    dimensionality — distances become less meaningful.  Reducing to ~50
    dimensions via PCA before K-Means produces tighter, better-separated
    clusters while preserving >90 % of the variance.
    """

    def __init__(self, n_clusters: int = 10, n_init: int = 10,
                 random_state: int = 42, weak_member_percentile: float = 90.0,
                 pca_components: Optional[int] = 50):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state
        self.weak_member_percentile = weak_member_percentile
        self.pca_components = pca_components

    def fit_predict(self, embeddings: np.ndarray, section_type: int) -> KMeansResult:
        """Cluster embeddings using K-Means and detect weak members.

        If pca_components is set, applies PCA dimensionality reduction
        before clustering to improve distance-based separation.

        Args:
            embeddings: Array of shape (n_papers, embed_dim), L2-normalized.
            section_type: Integer identifying which section view this is.

        Returns:
            KMeansResult with cluster assignments, centroids, distances,
            and weak member flags.
        """
        n_papers, orig_dim = embeddings.shape
        actual_k = min(self.n_clusters, n_papers)

        if actual_k < self.n_clusters:
            logger.warning(
                "Only %d papers for view %d, reducing k from %d to %d",
                n_papers, section_type, self.n_clusters, actual_k,
            )

        # Optional PCA dimensionality reduction
        reduced = embeddings
        if (self.pca_components is not None
                and self.pca_components < orig_dim
                and n_papers > self.pca_components):
            pca = PCA(
                n_components=self.pca_components,
                random_state=self.random_state,
            )
            reduced = pca.fit_transform(embeddings)
            variance_kept = pca.explained_variance_ratio_.sum()
            logger.info(
                "View %d: PCA %d → %d dims (%.1f%% variance retained)",
                section_type, orig_dim, self.pca_components,
                variance_kept * 100,
            )

        kmeans = KMeans(
            n_clusters=actual_k,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        cluster_ids = kmeans.fit_predict(reduced)
        centroids = kmeans.cluster_centers_

        # Compute distance from each paper to its assigned centroid
        centroid_distances = np.zeros(n_papers, dtype=np.float32)
        for i in range(n_papers):
            centroid_distances[i] = np.linalg.norm(
                reduced[i] - centroids[cluster_ids[i]]
            )

        # Weak member detection: papers above the percentile threshold
        threshold = np.percentile(centroid_distances, self.weak_member_percentile)
        weak_members = centroid_distances > threshold

        logger.info(
            "View %d: K-Means with k=%d, %d weak members (>%.4f distance)",
            section_type, actual_k, weak_members.sum(), threshold,
        )

        return KMeansResult(
            section_type=section_type,
            cluster_ids=cluster_ids,
            centroids=centroids,
            centroid_distances=centroid_distances,
            n_clusters=actual_k,
            weak_members=weak_members,
        )
