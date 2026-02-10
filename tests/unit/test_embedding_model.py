"""Tests for src.embedding.embedding_model — Embedding wrapper (SentenceTransformer mocked)."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embedding.embedding_model import EmbeddingModel


@pytest.fixture
def mock_st():
    """Patch SentenceTransformer via the sentence_transformers module."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
    )

    mock_st_module = MagicMock()
    mock_st_module.SentenceTransformer.return_value = mock_model

    with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
        yield mock_st_module.SentenceTransformer, mock_model


class TestEmbed:
    def test_calls_encode_with_correct_params(self, mock_st):
        mock_cls, mock_model = mock_st
        model = EmbeddingModel(model_name="test-model", device="cpu", batch_size=16)
        model.embed(["text1", "text2"])

        mock_model.encode.assert_called_once()
        call_kwargs = mock_model.encode.call_args
        assert call_kwargs.kwargs["batch_size"] == 16
        assert call_kwargs.kwargs["normalize_embeddings"] is True

    def test_returns_float32_numpy(self, mock_st):
        mock_cls, mock_model = mock_st
        model = EmbeddingModel(model_name="test-model", device="cpu")
        result = model.embed(["text1", "text2"])
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_empty_list_returns_empty_array(self, mock_st):
        mock_cls, mock_model = mock_st
        model = EmbeddingModel(model_name="test-model", device="cpu")
        result = model.embed([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_correct_shape(self, mock_st):
        mock_cls, mock_model = mock_st
        model = EmbeddingModel(model_name="test-model", device="cpu")
        result = model.embed(["text1", "text2"])
        assert result.shape == (2, 3)


class TestEmbedSingle:
    def test_returns_1d_array(self, mock_st):
        mock_cls, mock_model = mock_st
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        model = EmbeddingModel(model_name="test-model", device="cpu")
        result = model.embed_single("some text")
        assert result is not None
        assert result.shape == (3,)

    def test_empty_string_returns_none(self, mock_st):
        mock_cls, mock_model = mock_st
        model = EmbeddingModel(model_name="test-model", device="cpu")
        result = model.embed_single("")
        assert result is None

    def test_none_returns_none(self, mock_st):
        mock_cls, mock_model = mock_st
        model = EmbeddingModel(model_name="test-model", device="cpu")
        result = model.embed_single(None)
        assert result is None


class TestLazyLoading:
    def test_model_none_after_init(self):
        model = EmbeddingModel(model_name="test", device="cpu")
        assert model._model is None

    def test_model_loaded_on_first_embed(self, mock_st):
        mock_cls, mock_model = mock_st
        model = EmbeddingModel(model_name="test", device="cpu")
        model.embed(["text"])
        mock_cls.assert_called_once()

    def test_model_not_reloaded_on_second_embed(self, mock_st):
        mock_cls, mock_model = mock_st
        model = EmbeddingModel(model_name="test", device="cpu")
        model.embed(["text1"])
        model.embed(["text2"])
        # SentenceTransformer constructor only called once
        mock_cls.assert_called_once()


class TestFlashAttentionFallback:
    def test_fallback_on_flash_attention_error(self):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1]], dtype=np.float32)

        call_count = 0

        def _constructor(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("flash_attention_2 not available")
            return mock_model

        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.side_effect = _constructor

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            model = EmbeddingModel(model_name="test", device="cpu")
            result = model.embed(["text"])
            assert call_count == 2
            assert result.shape == (1, 1)
