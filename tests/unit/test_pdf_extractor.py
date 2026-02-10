"""Tests for src.extraction.pdf_extractor — PDF text extraction (fitz mocked)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.extraction.pdf_extractor import PDFExtractor


class TestExtractFirstPage:
    def test_returns_text_from_first_page(self, mock_fitz):
        patcher, mock_mod = mock_fitz(pages_text=["Page 1 content"])
        try:
            result = PDFExtractor.extract_first_page(Path("/fake.pdf"))
            assert result == "Page 1 content"
        finally:
            patcher.stop()

    def test_empty_pdf_returns_empty(self, mock_fitz):
        patcher, mock_mod = mock_fitz(pages_text=[])
        try:
            result = PDFExtractor.extract_first_page(Path("/fake.pdf"))
            assert result == ""
        finally:
            patcher.stop()

    def test_exception_returns_empty(self):
        with patch("src.extraction.pdf_extractor.fitz") as mock_mod:
            mock_mod.open.side_effect = RuntimeError("corrupt PDF")
            result = PDFExtractor.extract_first_page(Path("/fake.pdf"))
            assert result == ""

    def test_strips_whitespace(self, mock_fitz):
        patcher, mock_mod = mock_fitz(pages_text=["  text with spaces  "])
        try:
            result = PDFExtractor.extract_first_page(Path("/fake.pdf"))
            assert result == "text with spaces"
        finally:
            patcher.stop()


class TestExtractFullText:
    def test_concatenates_multiple_pages(self, mock_fitz):
        patcher, mock_mod = mock_fitz(pages_text=["Page 1", "Page 2", "Page 3"])
        try:
            result = PDFExtractor.extract_full_text(Path("/fake.pdf"))
            assert "Page 1" in result
            assert "Page 2" in result
            assert "Page 3" in result
        finally:
            patcher.stop()

    def test_max_pages_limit(self, mock_fitz):
        patcher, mock_mod = mock_fitz(pages_text=["P1", "P2", "P3", "P4"])
        try:
            result = PDFExtractor.extract_full_text(Path("/fake.pdf"), max_pages=2)
            assert "P1" in result
            assert "P2" in result
            assert "P3" not in result
        finally:
            patcher.stop()

    def test_exception_returns_empty(self):
        with patch("src.extraction.pdf_extractor.fitz") as mock_mod:
            mock_mod.open.side_effect = RuntimeError("error")
            result = PDFExtractor.extract_full_text(Path("/fake.pdf"))
            assert result == ""
