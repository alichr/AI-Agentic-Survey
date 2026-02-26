"""Async LLM-based section segmentation via vLLM OpenAI-compatible API.

Uses a section-start approach: the LLM identifies the line number where each
section begins. We then split the text into contiguous chunks at those
boundaries, guaranteeing ~100% coverage with ~100 output tokens.
"""

import asyncio
import json
import logging
from typing import Optional

from openai import AsyncOpenAI

from src.models.paper import PaperSections

logger = logging.getLogger(__name__)

SECTION_PROMPT = """You are an expert at analyzing academic papers. The paper text below has line numbers.

Identify the line number where each of these sections STARTS in the paper:
1. title_abstract_conclusion — title and abstract (usually at the very beginning), AND conclusion (usually near the end)
2. introduction — the introduction section
3. related_work — related work, background, literature review
4. method — methodology, approach, framework, architecture
5. experiments — experiments, results, evaluation, ablation

Rules:
- A section may start at MULTIPLE line numbers (e.g. title+abstract at the top AND conclusion near the end).
- Return a JSON object mapping each section name to a list of starting line numbers.
- Every line in the paper will belong to whichever section was most recently started.
- If a section does not exist, use an empty list [].
- Include ALL major parts of the paper. Do not skip any section.

Example:
{"title_abstract_conclusion": [1, 580], "introduction": [22], "related_work": [95], "method": [160], "experiments": [310]}

This means: lines 1-21 are title/abstract, lines 22-94 are introduction, lines 95-159 are related work, lines 160-309 are method, lines 310-579 are experiments, lines 580+ are conclusion.

Paper text:
"""


class SectionExtractor:
    """Extracts structured sections from paper full text using an LLM served by vLLM.

    Uses a section-start approach: the LLM returns start-line numbers (~100 output
    tokens). We split the text into contiguous chunks at those boundaries, giving
    ~100% coverage regardless of paper length.
    """

    def __init__(self, base_url: str, model_name: str,
                 max_concurrent: int = 20, temperature: float = 0.3,
                 max_tokens: int = 1024, max_text_chars: int = 100000):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="not-needed",
        )
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_text_chars = max_text_chars

    async def extract(self, full_text: str, retries: int = 1) -> Optional[PaperSections]:
        """Segment a paper's full text into 5 sections.

        Args:
            full_text: Full paper text from PDF extraction.
            retries: Number of retries on JSON parse failure.

        Returns:
            PaperSections if extraction succeeds, None otherwise.
        """
        if not full_text or len(full_text.strip()) < 100:
            logger.warning("Full text too short for section extraction")
            return None

        truncated = full_text[:self.max_text_chars]
        lines = truncated.split("\n")
        # Build numbered text for the LLM
        numbered = "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))

        for attempt in range(1 + retries):
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": SECTION_PROMPT + numbered}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )

                raw_content = response.choices[0].message.content
                if not raw_content:
                    logger.warning("LLM returned empty content for section extraction")
                    if attempt < retries:
                        continue
                    return None
                content = raw_content.strip()
                return self._parse_response(content, lines)

            except json.JSONDecodeError:
                if attempt < retries:
                    logger.warning("JSON parse failed for sections, retrying (attempt %d)", attempt + 1)
                    continue
                logger.error("Failed to parse section extraction after %d attempts", 1 + retries)
                return None
            except Exception as e:
                logger.error("Section extraction error: %s", e)
                return None

        return None

    @staticmethod
    def _parse_response(content: str, lines: list[str]) -> Optional[PaperSections]:
        """Parse LLM response with section start lines and split text into contiguous chunks."""
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
            raise json.JSONDecodeError("No JSON object found", content, 0)

        data = json.loads(content[start:end])
        n_lines = len(lines)

        section_names = [
            "title_abstract_conclusion",
            "introduction",
            "related_work",
            "method",
            "experiments",
        ]

        # Build a sorted list of (line_number, section_name) boundaries
        boundaries = []
        for name in section_names:
            starts = data.get(name, [])
            if isinstance(starts, (int, float)):
                starts = [starts]
            for line_num in starts:
                line_num = int(line_num)
                if 1 <= line_num <= n_lines:
                    boundaries.append((line_num, name))

        if not boundaries:
            # LLM returned empty — treat entire text as title_abstract_conclusion
            return PaperSections(
                title_abstract_conclusion="\n".join(lines),
                introduction="",
                related_work="",
                method="",
                experiments="",
            )

        # Sort by line number
        boundaries.sort(key=lambda x: x[0])

        # Assign each line to the most recently started section
        section_lines: dict[str, list[str]] = {name: [] for name in section_names}

        boundary_idx = 0
        current_section = boundaries[0][1]

        for i in range(n_lines):
            line_num = i + 1  # 1-indexed

            # Advance to the latest boundary that starts at or before this line
            while (boundary_idx + 1 < len(boundaries)
                   and boundaries[boundary_idx + 1][0] <= line_num):
                boundary_idx += 1
                current_section = boundaries[boundary_idx][1]

            # Check if this line is exactly a boundary start
            if (boundary_idx < len(boundaries)
                    and boundaries[boundary_idx][0] == line_num):
                current_section = boundaries[boundary_idx][1]

            section_lines[current_section].append(lines[i])

        # Lines before the first boundary go to title_abstract_conclusion
        # (already handled since boundaries[0] is the first section start)

        return PaperSections(
            title_abstract_conclusion="\n".join(section_lines["title_abstract_conclusion"]),
            introduction="\n".join(section_lines["introduction"]),
            related_work="\n".join(section_lines["related_work"]),
            method="\n".join(section_lines["method"]),
            experiments="\n".join(section_lines["experiments"]),
        )

    @staticmethod
    def fallback_sections(full_text: str) -> PaperSections:
        """Create fallback sections when extraction fails — put all text in section 0."""
        return PaperSections(
            title_abstract_conclusion=full_text,
            introduction="",
            related_work="",
            method="",
            experiments="",
        )

    async def extract_batch(self, texts: list[tuple[str, str]]) -> dict[str, Optional[PaperSections]]:
        """Extract sections for multiple papers concurrently.

        Args:
            texts: List of (pdf_hash, full_text) tuples.

        Returns:
            Dict mapping pdf_hash to PaperSections (or None on failure).
        """
        async def _extract_one(pdf_hash: str, text: str):
            result = await self.extract(text)
            return pdf_hash, result

        tasks = [_extract_one(h, t) for h, t in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for r in results:
            if isinstance(r, Exception):
                logger.error("Batch section extraction error: %s", r)
                continue
            pdf_hash, sections = r
            output[pdf_hash] = sections

        await self.close()
        return output

    async def close(self):
        """Close the underlying HTTP client to avoid event-loop-closed errors."""
        await self.client.close()
