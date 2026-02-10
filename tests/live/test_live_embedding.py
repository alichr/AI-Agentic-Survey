"""Live tests: real embedding model on GPU.

Run with: pytest tests/live/ -m live -v
Requires GPU and the Qwen3-Embedding model accessible.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.live, pytest.mark.slow]


@pytest.fixture(scope="module")
def embedding_model():
    """Load a real embedding model (module-scoped for efficiency)."""
    from src.embedding.embedding_model import EmbeddingModel

    model = EmbeddingModel(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        device="cuda:0",
        batch_size=8,
    )
    return model


class TestLiveEmbedding:
    def test_model_loads_without_error(self, embedding_model):
        """Real embedding model can be loaded."""
        embedding_model._load_model()
        assert embedding_model._model is not None

    def test_produces_expected_dimensionality(self, embedding_model):
        """Embeddings have consistent, non-trivial dimensionality."""
        result = embedding_model.embed(["Test sentence about AI agents"])
        assert result.shape[0] == 1
        assert result.shape[1] > 100  # Typical embedding dims are 384+

    def test_similar_texts_high_cosine(self, embedding_model):
        """Semantically similar texts should have cosine similarity > 0.7."""
        texts = [
            "Large language model agents for autonomous task completion",
            "LLM-based autonomous agents that can complete complex tasks",
        ]
        embeddings = embedding_model.embed(texts)
        cos_sim = np.dot(embeddings[0], embeddings[1])
        # Embeddings are normalized, so dot product = cosine similarity
        assert cos_sim > 0.7

    def test_dissimilar_texts_low_cosine(self, embedding_model):
        """Unrelated texts should have lower cosine similarity."""
        texts = [
            "Large language model agents for autonomous task completion",
            "The history of Roman architecture and its influence on modern buildings",
        ]
        embeddings = embedding_model.embed(texts)
        cos_sim = np.dot(embeddings[0], embeddings[1])
        assert cos_sim < 0.5
