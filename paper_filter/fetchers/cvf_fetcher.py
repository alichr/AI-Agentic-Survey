"""Fetcher for CVPR and ICCV via CVF Open Access, with Wayback Machine fallback."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time

import requests

from paper_filter.fetchers.base import BaseFetcher
from paper_filter.models import Paper

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
TIMEOUT = 90
MAX_RETRIES = 2

CVF_BASE = "https://openaccess.thecvf.com"
WAYBACK_PREFIX = "https://web.archive.org/web/2025"


class CVFFetcher(BaseFetcher):
    async def fetch(self, conference: str, year: int) -> list[Paper]:
        return await asyncio.to_thread(self._fetch_sync, conference.upper(), year)

    def _fetch_sync(self, conference: str, year: int) -> list[Paper]:
        cvf_url = f"{CVF_BASE}/{conference}{year}?day=all"

        # Try direct CVF first
        html_text, use_wayback = self._fetch_page(cvf_url)

        if not html_text:
            return []

        return self._parse_papers(html_text, conference, year, use_wayback)

    def _fetch_page(self, cvf_url: str) -> tuple[str, bool]:
        """Try CVF direct, fall back to Wayback Machine. Returns (html, is_wayback)."""
        # Try direct
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(cvf_url, headers=HEADERS, timeout=TIMEOUT)
                resp.raise_for_status()
                logger.info("Fetched from CVF directly")
                return resp.text, False
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning("CVF timeout (attempt %d/%d), retrying...", attempt + 1, MAX_RETRIES)
                    time.sleep(5)
                else:
                    logger.warning("CVF unreachable, falling back to Wayback Machine: %s", e)
            except requests.HTTPError as e:
                logger.warning("CVF HTTP error, falling back to Wayback Machine: %s", e)
                break

        # Fallback: Wayback Machine
        wayback_url = f"{WAYBACK_PREFIX}/{cvf_url}"
        try:
            resp = requests.get(wayback_url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            logger.info("Fetched from Wayback Machine")
            return resp.text, True
        except Exception as e:
            logger.error("Both CVF and Wayback Machine failed: %s", e)
            return "", False

    def _parse_papers(
        self, html_text: str, conference: str, year: int, use_wayback: bool
    ) -> list[Paper]:
        # Wayback rewrites URLs with /web/TIMESTAMP/... prefix,
        # so we use a flexible pattern that captures the original path
        pattern = re.compile(
            r'<dt class="ptitle">.*?'
            r'<a href="[^"]*(/content/{conf}\d{{4}}/(?:papers|html)/[^"]+\.html)"[^>]*>(.+?)</a>.*?'
            r'<a href="[^"]*(/content/{conf}\d{{4}}/papers/[^"]+_paper\.pdf)">pdf</a>'.format(
                conf=conference
            ),
            re.DOTALL,
        )
        matches = pattern.findall(html_text)

        papers = []
        for _, title, pdf_path in matches:
            title = title.strip()

            if use_wayback:
                # Use Wayback raw URL (id_) to get actual PDF without Wayback toolbar
                pdf_url = f"https://web.archive.org/web/2025id_/{CVF_BASE}{pdf_path}"
            else:
                pdf_url = f"{CVF_BASE}{pdf_path}"

            source_id = hashlib.sha256(
                f"{conference}_{year}_{title}".encode()
            ).hexdigest()[:16]
            papers.append(
                Paper(
                    source_id=source_id,
                    title=title,
                    authors=[],
                    abstract="",
                    conference=conference,
                    year=year,
                    pdf_url=pdf_url,
                )
            )
        return papers
