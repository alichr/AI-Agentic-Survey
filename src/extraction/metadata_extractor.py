"""Async LLM-based metadata extraction via vLLM OpenAI-compatible API."""

import asyncio
import json
import logging
from typing import Optional

from openai import AsyncOpenAI

from src.models.paper import Author, PaperMetadata

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract the following from this academic paper's first page. Return valid JSON only:
{
  "title": "...",
  "abstract": "...",
  "authors": [{"name": "...", "affiliation": "..."}],
  "year": 2024
}
If a field is not found, use null. Do not include any text outside the JSON object.

First page text:
"""


class MetadataExtractor:
    """Extracts structured metadata from paper text using an LLM served by vLLM.

    Uses the OpenAI-compatible async API with concurrency control.
    """

    def __init__(self, base_url: str, model_name: str, max_concurrent: int = 50):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="not-needed",  # vLLM doesn't require a real key
        )
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def extract(self, first_page_text: str, retries: int = 1) -> Optional[PaperMetadata]:
        """Extract metadata from a paper's first-page text.

        Args:
            first_page_text: Raw text from the paper's first page.
            retries: Number of retries on JSON parse failure.

        Returns:
            PaperMetadata if extraction succeeds, None otherwise.
        """
        if not first_page_text or len(first_page_text.strip()) < 50:
            logger.warning("First page text too short for extraction")
            return None

        for attempt in range(1 + retries):
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": EXTRACTION_PROMPT + first_page_text[:3000]}
                        ],
                        temperature=0.7,
                        top_p=0.8,
                        max_tokens=2048,
                    )

                raw_content = response.choices[0].message.content
                if not raw_content:
                    logger.warning("LLM returned empty content")
                    if attempt < retries:
                        continue
                    return None
                content = raw_content.strip()
                return self._parse_response(content)

            except json.JSONDecodeError:
                if attempt < retries:
                    logger.warning("JSON parse failed, retrying (attempt %d)", attempt + 1)
                    continue
                logger.error("Failed to parse LLM response after %d attempts", 1 + retries)
                return None
            except Exception as e:
                logger.error("LLM extraction error: %s", e)
                return None

        return None

    @staticmethod
    def _parse_response(content: str) -> Optional[PaperMetadata]:
        """Parse LLM JSON response into PaperMetadata."""
        # Try to find JSON object in the response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
            raise json.JSONDecodeError("No JSON object found", content, 0)

        data = json.loads(content[start:end])

        authors = []
        if data.get("authors"):
            for a in data["authors"]:
                if isinstance(a, dict):
                    authors.append(Author(
                        name=a.get("name", "Unknown"),
                        affiliation=a.get("affiliation"),
                    ))
                elif isinstance(a, str):
                    authors.append(Author(name=a))

        return PaperMetadata(
            title=data.get("title"),
            abstract=data.get("abstract"),
            authors=authors,
            year=data.get("year"),
        )

    async def extract_batch(self, texts: list[tuple[str, str]]) -> dict[str, Optional[PaperMetadata]]:
        """Extract metadata for multiple papers concurrently.

        Args:
            texts: List of (pdf_hash, first_page_text) tuples.

        Returns:
            Dict mapping pdf_hash to PaperMetadata (or None on failure).
        """
        async def _extract_one(pdf_hash: str, text: str):
            result = await self.extract(text)
            return pdf_hash, result

        tasks = [_extract_one(h, t) for h, t in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for r in results:
            if isinstance(r, Exception):
                logger.error("Batch extraction error: %s", r)
                continue
            pdf_hash, metadata = r
            output[pdf_hash] = metadata

        return output
