"""Fetcher for ECCV via ECVA (ecva.net/papers.php), with Wayback Machine fallback."""
from __future__ import annotations

import asyncio
import hashlib
import logging

import requests
from lxml import html

from paper_filter.fetchers.base import BaseFetcher
from paper_filter.models import Paper

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
TIMEOUT = 90
ECVA_URL = "https://www.ecva.net/papers.php"
WAYBACK_PREFIX = "https://web.archive.org/web/2025"


class ECVAFetcher(BaseFetcher):
    async def fetch(self, conference: str, year: int) -> list[Paper]:
        return await asyncio.to_thread(self._fetch_sync, conference.upper(), year)

    def _fetch_sync(self, conference: str, year: int) -> list[Paper]:
        content, use_wayback = self._fetch_page()
        if not content:
            return []

        tree = html.fromstring(content)

        xpath = (
            f"//button[contains(text(), 'ECCV {year} Papers')]"
            f"/following-sibling::div[@class='accordion-content'][1]//dl"
        )
        accordion = tree.xpath(xpath)
        if not accordion:
            return []

        papers = []
        for dl in accordion:
            titles = dl.xpath(".//dt/a/text()")
            pdf_links = dl.xpath(".//dd//a[contains(text(), 'pdf')]/@href")

            for title, pdf_link in zip(titles, pdf_links):
                title = title.strip()
                if use_wayback:
                    pdf_url = f"https://web.archive.org/web/2025id_/https://www.ecva.net/{pdf_link}"
                else:
                    pdf_url = f"https://www.ecva.net/{pdf_link}"
                source_id = hashlib.sha256(
                    f"ECCV_{year}_{title}".encode()
                ).hexdigest()[:16]
                papers.append(
                    Paper(
                        source_id=source_id,
                        title=title,
                        authors=[],
                        abstract="",
                        conference="ECCV",
                        year=year,
                        pdf_url=pdf_url,
                    )
                )
        return papers

    def _fetch_page(self) -> tuple[bytes, bool]:
        """Try ECVA direct, fall back to Wayback Machine. Returns (content, is_wayback)."""
        try:
            resp = requests.get(ECVA_URL, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            logger.info("Fetched from ECVA directly")
            return resp.content, False
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            logger.warning("ECVA unreachable, falling back to Wayback Machine: %s", e)

        try:
            resp = requests.get(
                f"{WAYBACK_PREFIX}/{ECVA_URL}", headers=HEADERS, timeout=TIMEOUT
            )
            resp.raise_for_status()
            logger.info("Fetched from Wayback Machine")
            return resp.content, True
        except Exception as e:
            logger.error("Both ECVA and Wayback Machine failed: %s", e)
            return b"", False
