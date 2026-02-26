"""Async LLM-based section summarization via vLLM OpenAI-compatible API.

Takes raw section text (from heuristic splitting) and produces concise
~200-word summaries suitable for embedding-based clustering.
"""

import asyncio
import logging
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SECTION_PROMPTS = {
    "title_abstract_conclusion": """You are an expert at analyzing academic papers. Below is text from the Title, Abstract, and Conclusion of a research paper.

Write a detailed summary (~400 words) extracting:
- The exact paper title and the core research problem it addresses
- The main contribution or novelty claim (what is new compared to prior work?)
- The type of system or approach (e.g., multi-agent framework, LLM-based planner, RL policy, tool-use agent, etc.)
- The key domains or applications targeted (e.g., web navigation, code generation, robotics, game playing, scientific discovery)
- The main quantitative results or performance claims from the conclusion
- Any limitations or future work acknowledged by the authors

Be factual and specific. Use the exact technical terms from the paper. Do NOT include filler phrases like "This section discusses..." — jump straight into the content.

Text:
{section_text}

Summary:""",

    "introduction": """You are an expert at analyzing academic papers. Below is the Introduction section of a research paper.

Write a detailed summary (~400 words) extracting:
- The problem statement: what specific gap, challenge, or limitation does this paper address?
- The motivation: why is this problem important? What fails in current approaches?
- The proposed solution at a high level: what is the paper's main idea or approach?
- The key claims or hypotheses the paper intends to validate
- Any specific use cases, scenarios, or real-world applications mentioned
- How this work positions itself relative to the broader field (e.g., agentic AI, multi-agent systems, LLM reasoning, embodied agents)

Be factual and specific. Preserve the exact technical terms and problem framing from the paper. Do NOT include filler phrases — jump straight into the content.

Text:
{section_text}

Summary:""",

    "related_work": """You are an expert at analyzing academic papers. Below is the Related Work / Background section of a research paper.

Write a detailed summary (~400 words) extracting:
- The main research areas and subfields this paper connects to (list them explicitly)
- Key prior methods, systems, or frameworks cited and what they do (name them by name)
- What limitations of prior work does this paper identify? What gaps remain?
- How does the paper differentiate its approach from the closest related work?
- Any taxonomies, categorizations, or groupings of related work the authors present
- Foundational techniques or building blocks the paper relies on (e.g., specific LLM architectures, RL algorithms, planning frameworks, tool-use protocols)

Be factual and specific. Name the actual papers, methods, and systems referenced. Do NOT include filler phrases — jump straight into the content.

Text:
{section_text}

Summary:""",

    "method": """You are an expert at analyzing academic papers. Below is the Methodology / Approach section of a research paper.

Write a detailed summary (~400 words) extracting:
- The overall architecture or system design (components, modules, how they interact)
- The specific algorithms, models, or techniques used (name them: e.g., PPO, chain-of-thought, ReAct, tree search, etc.)
- The input/output specification: what does the system take in and produce?
- Key design decisions and their justifications (why this architecture over alternatives?)
- Any training procedures, optimization objectives, or loss functions
- How agents interact, communicate, or coordinate (if multi-agent)
- Any tool use, API calls, or external resources the system leverages
- Mathematical formulations or formal problem definitions if present

Be factual and specific. Use the exact technical terms, model names, and algorithm names from the paper. Do NOT include filler phrases — jump straight into the content.

Text:
{section_text}

Summary:""",

    "experiments": """You are an expert at analyzing academic papers. Below is the Experiments / Results section of a research paper.

Write a detailed summary (~400 words) extracting:
- The benchmarks, datasets, or environments used for evaluation (name them explicitly)
- The baseline methods compared against (name them explicitly)
- The main quantitative results: metrics, scores, and performance numbers
- Key ablation studies and what they reveal about which components matter
- Any qualitative analyses, case studies, or failure mode discussions
- Computational requirements mentioned (GPU hours, model sizes, inference costs)
- Statistical significance or variance information if reported
- The main takeaway: does the method achieve state-of-the-art? By how much?

Be factual and specific. Include actual numbers, percentages, and metric names. Do NOT include filler phrases — jump straight into the content.

Text:
{section_text}

Summary:""",
}

# Fallback for any section not in the map above
GENERIC_SUMMARY_PROMPT = """You are an expert at summarizing academic papers. Below is text from the "{section_name}" section of a research paper.

Write a detailed summary (~400 words) that captures the key ideas, methods, findings, or contributions of this section.

Be factual and specific. Do NOT include filler phrases like "This section discusses..." — jump straight into the content.

Section text:
{section_text}

Summary:"""

SECTION_DISPLAY_NAMES = {
    "title_abstract_conclusion": "Title, Abstract, and Conclusion",
    "introduction": "Introduction",
    "related_work": "Related Work",
    "method": "Methodology",
    "experiments": "Experiments and Results",
}

# Truncate input section text to this many chars (~25K tokens)
MAX_SECTION_INPUT_CHARS = 50000


class SectionSummarizer:
    """Summarizes raw section text into concise ~200-word summaries using an LLM."""

    def __init__(self, base_url: str, model_name: str,
                 max_concurrent: int = 20, temperature: float = 0.3,
                 max_tokens: int = 1024):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="not-needed",
        )
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def summarize(self, section_text: str, section_name: str,
                        retries: int = 1) -> Optional[str]:
        """Summarize a single section's text.

        Args:
            section_text: Raw text of the section.
            section_name: Section key (e.g., "introduction", "method").
            retries: Number of retries on failure.

        Returns:
            Summary string, or None on failure.
        """
        if not section_text or len(section_text.strip()) < 50:
            return None

        truncated = section_text[:MAX_SECTION_INPUT_CHARS]

        if section_name in SECTION_PROMPTS:
            prompt = SECTION_PROMPTS[section_name].format(section_text=truncated)
        else:
            display_name = SECTION_DISPLAY_NAMES.get(section_name, section_name)
            prompt = GENERIC_SUMMARY_PROMPT.format(
                section_name=display_name,
                section_text=truncated,
            )

        for attempt in range(1 + retries):
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )

                content = response.choices[0].message.content
                if not content or len(content.strip()) < 20:
                    logger.warning("LLM returned too-short summary for %s", section_name)
                    if attempt < retries:
                        continue
                    return None
                return content.strip()

            except Exception as e:
                if attempt < retries:
                    logger.warning("Summary failed for %s, retrying: %s", section_name, e)
                    continue
                logger.error("Section summarization error for %s: %s", section_name, e)
                return None

        return None

    async def summarize_all_sections(self, sections_dict: dict[str, str]) -> dict[str, str]:
        """Summarize all 5 sections of a paper concurrently.

        Args:
            sections_dict: Mapping of section_name -> raw text.

        Returns:
            Mapping of section_name -> summary text.
        """
        async def _summarize_one(name: str, text: str):
            summary = await self.summarize(text, name)
            return name, summary

        tasks = [
            _summarize_one(name, text)
            for name, text in sections_dict.items()
            if text and len(text.strip()) >= 50
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        summaries = {}
        for r in results:
            if isinstance(r, Exception):
                logger.error("Section summary error: %s", r)
                continue
            name, summary = r
            if summary:
                summaries[name] = summary

        return summaries

    async def summarize_batch(self, papers_sections: list[tuple[str, dict[str, str]]]
                              ) -> dict[str, dict[str, str]]:
        """Summarize sections for multiple papers concurrently.

        Args:
            papers_sections: List of (pdf_hash, {section_name: raw_text}) tuples.

        Returns:
            Dict mapping pdf_hash -> {section_name: summary}.
        """
        async def _process_paper(pdf_hash: str, sections: dict[str, str]):
            summaries = await self.summarize_all_sections(sections)
            return pdf_hash, summaries

        tasks = [_process_paper(h, s) for h, s in papers_sections]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for r in results:
            if isinstance(r, Exception):
                logger.error("Batch summary error: %s", r)
                continue
            pdf_hash, summaries = r
            output[pdf_hash] = summaries

        await self.close()
        return output

    async def close(self):
        """Close the underlying HTTP client to avoid event-loop-closed errors."""
        await self.client.close()
