"""PDF text extraction using PyMuPDF (fitz)."""

import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extracts text from PDF files using PyMuPDF."""

    @staticmethod
    def extract_first_page(pdf_path: Path) -> str:
        """Extract text from the first page of a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Extracted text from the first page, or empty string on failure.
        """
        try:
            doc = fitz.open(str(pdf_path))
            if len(doc) == 0:
                logger.warning("Empty PDF: %s", pdf_path)
                doc.close()
                return ""
            text = doc[0].get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error("Failed to extract text from %s: %s", pdf_path, e)
            return ""

    @staticmethod
    def extract_full_text(pdf_path: Path, max_pages: int = 0) -> str:
        """Extract text from all pages of a PDF.

        Args:
            pdf_path: Path to the PDF file.
            max_pages: Maximum pages to extract (0 = all).

        Returns:
            Concatenated text from all pages.
        """
        try:
            doc = fitz.open(str(pdf_path))
            pages = min(len(doc), max_pages) if max_pages > 0 else len(doc)
            texts = []
            for i in range(pages):
                texts.append(doc[i].get_text())
            doc.close()
            return "\n".join(texts).strip()
        except Exception as e:
            logger.error("Failed to extract full text from %s: %s", pdf_path, e)
            return ""
