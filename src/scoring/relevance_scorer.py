"""Relevance scoring based on GMM cluster membership probability."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class RelevanceScorer:
    """Computes relevance scores from GMM cluster probabilities.

    The relevance score for a paper is the maximum probability across all
    clusters from the GMM's predict_proba output. This reflects how
    confidently the paper belongs to at least one coherent topic cluster.
    """

    @staticmethod
    def score(cluster_probabilities: np.ndarray) -> list[float]:
        """Compute relevance scores for all papers.

        Args:
            cluster_probabilities: Array of shape (n_papers, n_clusters)
                from GMM.predict_proba().

        Returns:
            List of relevance scores (one per paper), each in [0, 1].
        """
        if cluster_probabilities.size == 0:
            return []
        scores = np.max(cluster_probabilities, axis=1).tolist()
        logger.info(
            "Relevance scores: min=%.3f, max=%.3f, mean=%.3f",
            min(scores), max(scores), sum(scores) / len(scores),
        )
        return scores
