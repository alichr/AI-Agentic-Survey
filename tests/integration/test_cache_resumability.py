"""Integration tests: cache persistence across close/reopen cycles."""

import numpy as np
import pytest

from src.cache.cache_manager import CacheManager

pytestmark = pytest.mark.integration


class TestCachePersistAcrossCloseReopen:
    def test_pdf_text_survives_close(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        cm1 = CacheManager(cache_dir)
        cm1.set_pdf_text("hash1", "first page", "full text")
        cm1.close()

        cm2 = CacheManager(cache_dir)
        assert cm2.get_pdf_text("hash1") == "first page"
        cm2.close()

    def test_metadata_survives_close(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        cm1 = CacheManager(cache_dir)
        authors = [{"name": "Alice", "affiliation": "MIT"}]
        cm1.set_metadata("hash1", "Title", "Abstract", authors, 2024)
        cm1.close()

        cm2 = CacheManager(cache_dir)
        result = cm2.get_metadata("hash1")
        assert result is not None
        assert result["title"] == "Title"
        assert result["authors"] == authors
        cm2.close()

    def test_embedding_survives_close(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        cm1 = CacheManager(cache_dir)
        cm1.set_embedding("hash1", emb, "model-a")
        cm1.close()

        cm2 = CacheManager(cache_dir)
        result = cm2.get_embedding("hash1", "model-a")
        assert result is not None
        np.testing.assert_allclose(result, emb, atol=1e-6)
        cm2.close()

    def test_citation_survives_close(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        cm1 = CacheManager(cache_dir)
        cm1.set_citation("Paper Title", 42, 2023)
        cm1.close()

        cm2 = CacheManager(cache_dir)
        result = cm2.get_citation("Paper Title", max_age_days=30)
        assert result is not None
        assert result["citation_count"] == 42
        cm2.close()

    def test_citation_expiry_on_reopen(self, tmp_path):
        import hashlib
        import time

        cache_dir = str(tmp_path / "cache")
        cm1 = CacheManager(cache_dir)
        cm1.set_citation("Old Paper", 10, 2020)

        # Inject old timestamp
        title_hash = hashlib.sha256("old paper".encode()).hexdigest()
        cm1._get_conn().execute(
            "UPDATE citations SET fetched_at = ? WHERE title_hash = ?",
            (time.time() - 86400 * 60, title_hash)
        )
        cm1._get_conn().commit()
        cm1.close()

        cm2 = CacheManager(cache_dir)
        result = cm2.get_citation("Old Paper", max_age_days=30)
        assert result is None
        cm2.close()

    def test_full_cycle_all_types(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        cm1 = CacheManager(cache_dir)
        cm1.set_pdf_text("h1", "text1")
        cm1.set_metadata("h1", "Title1", "Abs1", [{"name": "A"}], 2024)
        cm1.set_embedding("h1", emb, "model")
        cm1.set_citation("Title1", 99, 2024)
        cm1.close()

        cm2 = CacheManager(cache_dir)
        assert cm2.get_pdf_text("h1") == "text1"
        assert cm2.get_metadata("h1")["title"] == "Title1"
        np.testing.assert_allclose(cm2.get_embedding("h1", "model"), emb, atol=1e-6)
        assert cm2.get_citation("Title1", 30)["citation_count"] == 99
        cm2.close()
