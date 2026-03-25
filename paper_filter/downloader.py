from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import aiohttp

from paper_filter.config import DownloadConfig
from paper_filter.models import Paper

logger = logging.getLogger(__name__)


class PDFDownloader:
    def __init__(self, config: DownloadConfig, output_dir: str, auth_token: str = "") -> None:
        self._config = config
        self._output_dir = output_dir
        self._auth_token = auth_token
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=self._config.max_concurrent,
                limit_per_host=self._config.max_concurrent,
            )
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            }
            if self._auth_token:
                headers["Authorization"] = f"Bearer {self._auth_token}"
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self._config.timeout),
                headers=headers,
            )
        return self._session

    async def download(self, paper: Paper) -> str | None:
        """Download a paper's PDF. Returns the local file path, or None on failure."""
        if not paper.pdf_url:
            return None

        # Create directory: output/pdfs/{conference}_{year}/
        dest_dir = Path(self._output_dir) / "pdfs" / f"{paper.conference}_{paper.year}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename from title
        safe_title = re.sub(r"[^\w\s-]", "", paper.title)[:80].strip()
        safe_title = re.sub(r"\s+", "_", safe_title)
        dest_path = dest_dir / f"{safe_title}.pdf"

        if dest_path.exists():
            return str(dest_path)

        async with self._semaphore:
            return await self._download_with_retry(paper.pdf_url, dest_path)

    async def _download_with_retry(self, url: str, dest_path: Path) -> str | None:
        session = await self._get_session()
        max_attempts = self._config.max_retries

        for attempt in range(max_attempts):
            try:
                async with session.get(url) as resp:
                    if resp.status == 429:
                        retry_after = float(resp.headers.get("Retry-After", 5))
                        logger.warning(
                            "Rate limited (429) %s, waiting %.0fs (attempt %d)",
                            url, retry_after, attempt + 1,
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status != 200:
                        logger.warning(
                            "HTTP %d downloading %s (attempt %d)",
                            resp.status, url, attempt + 1,
                        )
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(self._config.retry_backoff ** attempt)
                        continue

                    content = await resp.read()
                    dest_path.write_bytes(content)
                    return str(dest_path)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(
                    "Download error for %s (attempt %d): %s",
                    url, attempt + 1, e,
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(self._config.retry_backoff ** attempt)

        logger.error("Failed after %d attempts: %s", max_attempts, url)
        return None

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
