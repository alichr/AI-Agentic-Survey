"""Fetcher for NeurIPS (2021-2024) via proceedings.neurips.cc."""
from __future__ import annotations

import asyncio
import hashlib
import re

import requests
from lxml import html

from paper_filter.fetchers.base import BaseFetcher
from paper_filter.models import Paper

HEADERS = {"User-Agent": "Mozilla/5.0 (Research Paper Downloader)"}
TIMEOUT = 60


class NeurIPSProceedingsFetcher(BaseFetcher):
    async def fetch(self, conference: str, year: int) -> list[Paper]:
        return await asyncio.to_thread(self._fetch_sync, year)

    def _fetch_sync(self, year: int) -> list[Paper]:
        url = f"https://proceedings.neurips.cc/paper_files/paper/{year}"
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        tree = html.fromstring(resp.content)

        links = tree.xpath('//a[contains(@href, "-Abstract")]')
        papers = []
        for a in links:
            title = a.text_content().strip()
            href = a.get("href", "")
            if not title or not href:
                continue

            # Convert abstract URL to PDF URL
            # /paper_files/paper/2024/hash/XXX-Abstract-Conference.html
            # -> /paper_files/paper/2024/file/XXX-Paper-Conference.pdf
            # Handles all categories: Conference, Datasets_and_Benchmarks, etc.
            pdf_href = href.replace("/hash/", "/file/")
            pdf_href = re.sub(r"-Abstract(-\w+)?\.html$",
                              lambda m: f"-Paper{m.group(1) or ''}.pdf",
                              pdf_href)
            pdf_url = f"https://proceedings.neurips.cc{pdf_href}"

            source_id = hashlib.sha256(
                f"NEURIPS_{year}_{title}".encode()
            ).hexdigest()[:16]
            papers.append(
                Paper(
                    source_id=source_id,
                    title=title,
                    authors=[],
                    abstract="",
                    conference="NEURIPS",
                    year=year,
                    pdf_url=pdf_url,
                )
            )
        return papers
