"""Embedding model wrapper using sentence-transformers with Qwen3-Embedding."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wraps a sentence-transformers model for computing paper embeddings.

    Uses Qwen3-Embedding-0.6B by default, loaded with flash_attention_2
    for efficient inference on GPU.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                 device: str = "cuda:0", batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is not None:
            return

        logger.info("Loading embedding model: %s on %s", self.model_name, self.device)
        from sentence_transformers import SentenceTransformer

        try:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs={"attn_implementation": "flash_attention_2"},
                trust_remote_code=True,
            )
        except Exception:
            logger.warning("flash_attention_2 not available, falling back to default attention")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
        logger.info("Embedding model loaded successfully")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            Numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])
        self._load_model()

        logger.info("Computing embeddings for %d texts (batch_size=%d)", len(texts), self.batch_size)
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_single(self, text: str) -> Optional[np.ndarray]:
        """Compute embedding for a single text."""
        if not text:
            return None
        result = self.embed([text])
        return result[0] if len(result) > 0 else None
