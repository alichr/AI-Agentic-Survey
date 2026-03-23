from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from openai import AsyncOpenAI

from paper_filter.config import VLLMConfig
from paper_filter.models import Paper

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a strict academic paper relevance judge for a research survey. Determine whether a paper should be INCLUDED in a survey about a specific research topic.

Be strict and precise. A paper must make a DIRECT, SUBSTANTIVE contribution to the EXACT topic to score high.

Scoring criteria:
- 0-2: Not relevant. No meaningful connection to the topic.
- 3-4: Weak connection. Related general area but different focus.
- 5: Borderline. The paper uses or builds upon the topic but the topic is NOT the paper's main contribution.
- 6-7: Relevant. The topic is one of the paper's main themes. A survey author would cite this.
- 8-9: Highly relevant. The topic IS the paper's primary contribution.
- 10: Landmark contribution that defines or advances the topic.

CRITICAL rules — apply these BEFORE scoring:
1. Ask yourself: "Is the paper's PRIMARY contribution about [topic]?" If the answer is no, the score must be ≤ 5.
2. A paper that merely operates at inference/test time is NOT automatically about "test-time learning". A paper that is "training-free" is NOT automatically about test-time learning. Only papers that propose or study ADAPTATION/LEARNING during test time qualify.
3. A paper that uses a VLM/LLM as a tool for another task is NOT about VLMs/LLMs — it's about that task.
4. Adjacent concepts are NOT the same topic. Zero-shot learning ≠ test-time adaptation. Prompt engineering ≠ test-time learning. Model compression ≠ efficient inference.
5. When no abstract is available, be CONSERVATIVE. Without seeing the actual contribution, do not give more than 6 based on title alone, even if the title contains topic keywords.

Respond with valid JSON only:
{"score": <integer 0-10>, "reasoning": "<1-2 sentence explanation>"}"""

USER_PROMPT_TEMPLATE = """Research survey topic: {topic}
{topic_description_block}
Paper title: {title}
Paper abstract: {abstract}
{introduction_block}
First ask: "Is this paper's PRIMARY contribution about {topic}?" Then score accordingly."""


class RelevanceClassifier:
    def __init__(
        self,
        config: VLLMConfig,
        topic: str,
        topic_description: str | None = None,
        max_concurrent: int = 20,
    ) -> None:
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key="not-needed",
        )
        self._model = config.model_name
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens
        self._topic = topic
        self._topic_description = topic_description
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def classify(self, paper: Paper) -> tuple[float, str]:
        """Classify a paper's relevance to the topic.

        Returns (score, reasoning).
        """
        async with self._semaphore:
            return await self._classify_with_retry(paper)

    async def _classify_with_retry(
        self, paper: Paper, max_retries: int = 2
    ) -> tuple[float, str]:
        if self._topic_description:
            desc_block = f"Survey scope: {self._topic_description}\n"
        else:
            desc_block = ""

        if paper.introduction:
            intro_block = f"Paper introduction (first paragraphs): {paper.introduction[:2000]}\n"
        else:
            intro_block = ""

        user_prompt = USER_PROMPT_TEMPLATE.format(
            topic=self._topic,
            topic_description_block=desc_block,
            title=paper.title,
            abstract=paper.abstract or "(no abstract available)",
            introduction_block=intro_block,
        )

        for attempt in range(max_retries):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )

                content = response.choices[0].message.content.strip()
                return self._parse_response(content)

            except json.JSONDecodeError:
                if attempt < max_retries - 1:
                    logger.warning(
                        "JSON parse error for '%s', retrying...", paper.title[:50]
                    )
                    continue
                logger.error("Failed to parse LLM response for '%s'", paper.title[:50])
                return 0.0, "Classification failed: invalid JSON response"
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "LLM error for '%s': %s, retrying...", paper.title[:50], e
                    )
                    await asyncio.sleep(2 ** attempt)
                    continue
                logger.error("LLM classification failed for '%s': %s", paper.title[:50], e)
                return 0.0, f"Classification failed: {e}"

        return 0.0, "Classification failed after retries"

    def _parse_response(self, content: str) -> tuple[float, str]:
        # Try to extract JSON from the response
        # Handle cases where the model wraps JSON in markdown code blocks
        if "```" in content:
            lines = content.split("```")
            for block in lines:
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                if block.startswith("{"):
                    content = block
                    break

        data = json.loads(content)
        score = float(data.get("score", 0))
        reasoning = str(data.get("reasoning", ""))
        score = max(0.0, min(10.0, score))
        return score, reasoning

    async def close(self) -> None:
        await self._client.close()
