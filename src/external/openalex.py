"""Async OpenAlex API client for citation data."""

import asyncio
import logging
import re
from typing import Optional
from urllib.parse import quote

import aiohttp

from src.external.semantic_scholar import AuthorData, CitationData

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"


class OpenAlexClient:
    """Async client for the OpenAlex API.

    Uses title.search filter to find papers and retrieve citation counts.
    Rate limiting is enforced via a simple interval-based approach.
    """

    def __init__(self, api_key: Optional[str] = None,
                 email: Optional[str] = None,
                 rate_limit_rps: float = 10.0,
                 max_retries: int = 3):
        self.api_key = api_key
        self.email = email
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
                # Explicitly request gzip/deflate only — aiohttp cannot
                # decode Brotli (br) without the optional brotli package.
                headers = {"Accept-Encoding": "gzip, deflate"}
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

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Lowercase, strip punctuation and extra whitespace for comparison."""
        t = title.lower()
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    @classmethod
    def _title_matches(cls, query_title: str, result_title: str,
                       threshold: float = 0.75) -> bool:
        """Check if a result title is close enough to the query title.

        Uses token overlap (Jaccard similarity) to catch fuzzy matches
        while rejecting completely different papers.
        """
        q_tokens = set(cls._normalize_title(query_title).split())
        r_tokens = set(cls._normalize_title(result_title).split())
        if not q_tokens or not r_tokens:
            return False
        overlap = len(q_tokens & r_tokens)
        union = len(q_tokens | r_tokens)
        similarity = overlap / union
        return similarity >= threshold

    @staticmethod
    def _sanitize_title(title: str) -> str:
        """Remove characters that break OpenAlex filter syntax.

        Colons are interpreted as filter key:value separators,
        commas as filter-list separators, and pipes as OR operators.
        """
        return title.replace(":", " ").replace(",", " ").replace("|", " ")

    def _build_params(self, title: str) -> dict:
        """Build query parameters for a title search."""
        safe_title = self._sanitize_title(title)
        params = {
            "filter": f"title.search:{safe_title}",
            "select": "id,display_name,title,cited_by_count,publication_year",
            "per_page": "1",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["mailto"] = self.email
        return params

    async def search_paper(self, title: str) -> Optional[CitationData]:
        """Search for a paper by title and retrieve citation count.

        Args:
            title: Paper title to search for.

        Returns:
            CitationData if found, None otherwise.
        """
        if not title:
            return None

        session = await self._get_session()
        url = f"{BASE_URL}/works"
        params = self._build_params(title)

        for attempt in range(self.max_retries):
            try:
                await self._rate_limit()
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results", [])
                        if not results:
                            return None

                        match = results[0]
                        result_title = match.get("title", "")

                        # Verify the returned title actually matches our query.
                        # OpenAlex title.search is fuzzy and can return wrong papers.
                        if not self._title_matches(title, result_title):
                            logger.debug(
                                "OpenAlex title mismatch: query='%s' got='%s'",
                                title[:60], result_title[:60],
                            )
                            return None

                        return CitationData(
                            citation_count=match.get("cited_by_count", 0),
                            year=match.get("publication_year"),
                            paper_id=match.get("id"),
                        )

                    elif resp.status == 429:
                        wait = 2 ** attempt * 2
                        logger.warning(
                            "Rate limited by OpenAlex, waiting %ds (attempt %d/%d)",
                            wait, attempt + 1, self.max_retries,
                        )
                        await asyncio.sleep(wait)
                        continue

                    elif resp.status == 403:
                        text = await resp.text()
                        logger.warning(
                            "OpenAlex 403 Forbidden (check API key): %s",
                            text[:200],
                        )
                        return None

                    else:
                        text = await resp.text()
                        logger.warning(
                            "OpenAlex API error %d: %s", resp.status, text[:200]
                        )
                        return None

            except aiohttp.ClientError as e:
                logger.warning("HTTP error searching OpenAlex for '%s': %s",
                               title[:50], e)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
            except Exception as e:
                logger.error("Unexpected error searching OpenAlex for '%s': %s",
                             title[:50], e)
                return None

        return None

    async def get_paper_authors(self, paper_id: str) -> list[AuthorData]:
        """Fetch author data including h-index for a paper.

        Uses two API calls:
        1. GET /works/{id} → extract author OpenAlex IDs
        2. GET /authors?filter=openalex:A1|A2|... → batch fetch h-indices

        Args:
            paper_id: OpenAlex work ID (e.g., "https://openalex.org/W...").

        Returns:
            List of AuthorData with h-index information.
        """
        if not paper_id:
            return []

        # Step 1: Get author IDs from the work
        session = await self._get_session()

        # Extract short ID if full URL given
        work_id = paper_id.split("/")[-1] if "/" in paper_id else paper_id
        url = f"{BASE_URL}/works/{work_id}"
        params = {"select": "authorships"}
        if self.email:
            params["mailto"] = self.email

        author_ids = []
        for attempt in range(self.max_retries):
            try:
                await self._rate_limit()
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for authorship in data.get("authorships", []):
                            author = authorship.get("author", {})
                            aid = author.get("id", "")
                            name = author.get("display_name", "Unknown")
                            if aid:
                                author_ids.append((aid, name))
                        break
                    elif resp.status == 429:
                        wait = 2 ** attempt * 2
                        logger.warning("Rate limited fetching work authors, waiting %ds", wait)
                        await asyncio.sleep(wait)
                        continue
                    else:
                        logger.warning("OpenAlex error %d fetching work %s", resp.status, work_id)
                        return []
            except Exception as e:
                logger.warning("Error fetching work authors for %s: %s", work_id, e)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue

        if not author_ids:
            return []

        # Step 2: Batch-fetch author h-indices
        # OpenAlex supports filtering by multiple IDs with pipe separator
        # Process in chunks of 50 (API limit per filter)
        all_authors: list[AuthorData] = []
        chunk_size = 50

        for i in range(0, len(author_ids), chunk_size):
            chunk = author_ids[i:i + chunk_size]
            # Build filter: short IDs joined with |
            short_ids = [aid.split("/")[-1] if "/" in aid else aid for aid, _ in chunk]
            id_filter = "|".join(short_ids)

            authors_url = f"{BASE_URL}/authors"
            authors_params = {
                "filter": f"openalex:{id_filter}",
                "select": "id,display_name,summary_stats",
                "per_page": str(chunk_size),
            }
            if self.email:
                authors_params["mailto"] = self.email

            for attempt in range(self.max_retries):
                try:
                    await self._rate_limit()
                    async with session.get(authors_url, params=authors_params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            for author in data.get("results", []):
                                stats = author.get("summary_stats", {})
                                all_authors.append(AuthorData(
                                    author_id=author.get("id"),
                                    name=author.get("display_name", "Unknown"),
                                    hindex=stats.get("h_index"),
                                ))
                            break
                        elif resp.status == 429:
                            wait = 2 ** attempt * 2
                            logger.warning("Rate limited fetching authors, waiting %ds", wait)
                            await asyncio.sleep(wait)
                            continue
                        else:
                            text = await resp.text()
                            logger.warning("OpenAlex error %d fetching authors: %s",
                                           resp.status, text[:200])
                            break
                except Exception as e:
                    logger.warning("Error fetching author batch: %s", e)
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    continue

        # Fill in names for authors that weren't returned by the batch
        # (use the names from step 1)
        returned_ids = {a.author_id for a in all_authors}
        for aid, name in author_ids:
            if aid not in returned_ids:
                all_authors.append(AuthorData(author_id=aid, name=name, hindex=None))

        return all_authors

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
