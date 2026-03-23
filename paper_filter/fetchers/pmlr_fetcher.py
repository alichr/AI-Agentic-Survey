"""Fetcher for ICML (2021-2024) via PMLR (proceedings.mlr.press)."""
from __future__ import annotations

import asyncio
import hashlib

import requests
from lxml import html

from paper_filter.fetchers.base import BaseFetcher
from paper_filter.models import Paper

HEADERS = {"User-Agent": "Mozilla/5.0 (Research Paper Downloader)"}
TIMEOUT = 60

# ICML year -> PMLR volume mapping
ICML_PMLR_VOLUMES = {
    2021: 139,
    2022: 162,
    2023: 202,
    2024: 235,
}


class PMLRFetcher(BaseFetcher):
    async def fetch(self, conference: str, year: int) -> list[Paper]:
        return await asyncio.to_thread(self._fetch_sync, year)

    def _fetch_sync(self, year: int) -> list[Paper]:
        vol = ICML_PMLR_VOLUMES.get(year)
        if vol is None:
            raise ValueError(
                f"No PMLR volume mapping for ICML {year}. "
                f"Available: {sorted(ICML_PMLR_VOLUMES.keys())}"
            )

        url = f"https://proceedings.mlr.press/v{vol}/"
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        tree = html.fromstring(resp.content)

        titles = tree.xpath('//p[@class="title"]')
        pdf_links = tree.xpath('//a[contains(@href, ".pdf")]/@href')

        papers = []
        for i, title_el in enumerate(titles):
            title = title_el.text_content().strip()
            pdf_url = pdf_links[i] if i < len(pdf_links) else ""
            if not title or not pdf_url:
                continue

            source_id = hashlib.sha256(
                f"ICML_{year}_{title}".encode()
            ).hexdigest()[:16]
            papers.append(
                Paper(
                    source_id=source_id,
                    title=title,
                    authors=[],
                    abstract="",
                    conference="ICML",
                    year=year,
                    pdf_url=pdf_url,
                )
            )
        return papers
