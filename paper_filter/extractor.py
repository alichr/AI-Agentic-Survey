"""Extract title, abstract, and introduction from a PDF file using LLM."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

import fitz  # PyMuPDF

# Suppress noisy MuPDF warnings (Screen annotations, color space, etc.)
fitz.TOOLS.mupdf_display_errors(False)

from openai import AsyncOpenAI

from paper_filter.config import VLLMConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Extract the title, abstract, and introduction section from this academic paper text.

Respond with valid JSON only:
{"title": "<full paper title>", "abstract": "<full abstract text>", "introduction": "<full introduction section text>"}

Rules:
- The title is the largest text at the top of the page.
- The abstract is usually labeled "Abstract" and appears before the introduction.
- The introduction starts after the abstract (usually labeled "1. Introduction" or "1 Introduction") and ends where section 2 begins.
- Include the full introduction text, not a summary.
- If you cannot find a section, return an empty string for it."""


@dataclass
class ExtractedMetadata:
    title: str
    abstract: str
    introduction: str


# Read up to 4 pages to capture introduction (usually ends within pages 1-3)
MAX_PAGES = 4


class PDFExtractor:
    def __init__(self, config: VLLMConfig, max_concurrent: int = 20) -> None:
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key="not-needed",
        )
        self._model = config.model_name
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def extract(self, pdf_path: str) -> ExtractedMetadata:
        """Extract title, abstract, and introduction from a PDF."""
        text = _read_pdf_text(pdf_path)
        if not text.strip():
            return ExtractedMetadata("", "", "")

        async with self._semaphore:
            return await self._extract_with_llm(text)

    async def _extract_with_llm(self, text: str) -> ExtractedMetadata:
        # Truncate to ~8000 chars — enough for title + abstract + introduction
        text = text[:8000]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=4096,
            )
            content = response.choices[0].message.content.strip()
            return _parse_response(content)
        except Exception as e:
            logger.warning("LLM extraction failed: %s", e)
            return ExtractedMetadata("", "", "")

    async def close(self) -> None:
        await self._client.close()


def _read_pdf_text(pdf_path: str) -> str:
    """Read text from the first pages of a PDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(min(MAX_PAGES, len(doc))):
            text += doc[page_num].get_text()
        doc.close()
        return text
    except Exception as e:
        logger.warning("Failed to read PDF %s: %s", pdf_path, e)
        return ""


def _parse_response(content: str) -> ExtractedMetadata:
    """Parse LLM JSON response."""
    if "```" in content:
        for block in content.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("{"):
                content = block
                break

    try:
        data = json.loads(content)
        return ExtractedMetadata(
            title=str(data.get("title", "")).strip(),
            abstract=str(data.get("abstract", "")).strip(),
            introduction=str(data.get("introduction", "")).strip(),
        )
    except json.JSONDecodeError:
        logger.warning("Failed to parse extraction response")
        return ExtractedMetadata("", "", "")
