"""Async Semantic Scholar API client with rate limiting and retries."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"


@dataclass
class CitationData:
    """Citation information from Semantic Scholar."""
    citation_count: int
    year: Optional[int]
    paper_id: Optional[str] = None


@dataclass
class AuthorData:
    """Author information including h-index from Semantic Scholar."""
    author_id: Optional[str]
    name: str
    hindex: Optional[int] = None


class SemanticScholarClient:
    """Async client for the Semantic Scholar API.

    Implements rate limiting via a token-bucket approach and exponential
    backoff on 429 responses.
    """

    def __init__(self, api_key: Optional[str] = None,
                 rate_limit_rps: float = 5.0, max_retries: int = 3):
        self.api_key = api_key
        self.rate_limit_rps = rate_limit_rps
        self.max_retries = max_retries
        self._interval = 1.0 / rate_limit_rps
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                headers = {}
                if self.api_key:
                    headers["x-api-key"] = self.api_key
                self._session = aiohttp.ClientSession(headers=headers)
            return self._session

    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_request_time
            if elapsed < self._interval:
                await asyncio.sleep(self._interval - elapsed)
            self._last_request_time = asyncio.get_event_loop().time()

    async def search_paper(self, title: str) -> Optional[CitationData]:
        """Search for a paper by title and retrieve citation count.

        Uses the /paper/search/match endpoint for best title matching.

        Args:
            title: Paper title to search for.

        Returns:
            CitationData if found, None otherwise.
        """
        if not title:
            return None

        session = await self._get_session()
        url = f"{BASE_URL}/paper/search/match"
        params = {
            "query": title,
            "fields": "citationCount,year,paperId",
        }

        for attempt in range(self.max_retries):
            try:
                await self._rate_limit()
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data and "data" in data and data["data"]:
                            match = data["data"][0]
                            return CitationData(
                                citation_count=match.get("citationCount", 0),
                                year=match.get("year"),
                                paper_id=match.get("paperId"),
                            )
                        return None

                    elif resp.status == 429:
                        wait = 2 ** attempt * 2
                        logger.warning(
                            "Rate limited by Semantic Scholar, waiting %ds (attempt %d/%d)",
                            wait, attempt + 1, self.max_retries,
                        )
                        await asyncio.sleep(wait)
                        continue

                    elif resp.status == 404:
                        return None

                    else:
                        text = await resp.text()
                        logger.warning(
                            "Semantic Scholar API error %d: %s", resp.status, text[:200]
                        )
                        return None

            except aiohttp.ClientError as e:
                logger.warning("HTTP error searching for '%s': %s", title[:50], e)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
            except Exception as e:
                logger.error("Unexpected error searching for '%s': %s", title[:50], e)
                return None

        return None

    async def get_paper_authors(self, paper_id: str) -> list["AuthorData"]:
        """Fetch author data including h-index for a paper.

        Args:
            paper_id: Semantic Scholar paper ID.

        Returns:
            List of AuthorData with h-index information.
        """
        if not paper_id:
            return []

        session = await self._get_session()
        url = f"{BASE_URL}/paper/{paper_id}"
        params = {"fields": "authors.hIndex,authors.authorId,authors.name"}

        for attempt in range(self.max_retries):
            try:
                await self._rate_limit()
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        authors = []
                        for a in data.get("authors", []):
                            authors.append(AuthorData(
                                author_id=a.get("authorId"),
                                name=a.get("name", "Unknown"),
                                hindex=a.get("hIndex"),
                            ))
                        return authors

                    elif resp.status == 429:
                        wait = 2 ** attempt * 2
                        logger.warning(
                            "Rate limited fetching authors, waiting %ds (attempt %d/%d)",
                            wait, attempt + 1, self.max_retries,
                        )
                        await asyncio.sleep(wait)
                        continue

                    elif resp.status == 404:
                        return []

                    else:
                        text = await resp.text()
                        logger.warning(
                            "Semantic Scholar API error %d fetching authors: %s",
                            resp.status, text[:200],
                        )
                        return []

            except aiohttp.ClientError as e:
                logger.warning("HTTP error fetching authors for '%s': %s", paper_id, e)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
            except Exception as e:
                logger.error("Unexpected error fetching authors for '%s': %s", paper_id, e)
                return []

        return []

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
