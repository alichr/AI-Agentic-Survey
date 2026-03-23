"""Fetcher for ICLR (all years), ICML 2025+, NeurIPS 2025+ via conference virtual site JSON APIs."""
from __future__ import annotations

import asyncio
import hashlib
import re

import requests

from paper_filter.fetchers.base import BaseFetcher
from paper_filter.models import Paper

HEADERS = {"User-Agent": "Mozilla/5.0 (Research Paper Downloader)"}
TIMEOUT = 120

# Virtual site JSON URL patterns
VIRTUAL_SITE_URLS = {
    "ICLR": "https://iclr.cc/static/virtual/data/iclr-{year}-orals-posters.json",
    "ICML": "https://icml.cc/static/virtual/data/icml-{year}-orals-posters.json",
    "NEURIPS": "https://neurips.cc/static/virtual/data/neurips-{year}-orals-posters.json",
}


class VirtualSiteFetcher(BaseFetcher):
    """Fetches papers from conference virtual site JSON APIs (iclr.cc, icml.cc, neurips.cc)."""

    async def fetch(self, conference: str, year: int) -> list[Paper]:
        return await asyncio.to_thread(self._fetch_sync, conference.upper(), year)

    def _fetch_sync(self, conference: str, year: int) -> list[Paper]:
        url_template = VIRTUAL_SITE_URLS.get(conference)
        if not url_template:
            raise ValueError(f"No virtual site URL for {conference}")

        url = url_template.format(year=year)
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])

        papers = []
        for entry in results:
            if entry.get("eventtype") != "Poster":
                continue

            title = entry.get("name", "").strip()
            if not title:
                continue

            # Extract authors
            authors = []
            for author in entry.get("authors", []):
                if isinstance(author, dict):
                    name = author.get("fullname", "")
                else:
                    name = str(author)
                if name:
                    authors.append(name)

            abstract = entry.get("abstract", "") or ""

            # Get PDF URL: paper_pdf_url or paper_url (forum link -> pdf link)
            pdf_url = entry.get("paper_pdf_url") or ""
            if not pdf_url:
                paper_url = entry.get("paper_url") or ""
                if "openreview.net/forum" in paper_url:
                    pdf_url = paper_url.replace("/forum?id=", "/pdf?id=")
                else:
                    pdf_url = paper_url

            if "openreview.net/forum" in pdf_url:
                pdf_url = pdf_url.replace("/forum?id=", "/pdf?id=")

            # Build source_id from OpenReview forum ID or hash
            forum_match = re.search(r"id=([a-zA-Z0-9_-]+)", pdf_url)
            if forum_match:
                source_id = f"{conference}_{year}_{forum_match.group(1)}"
            else:
                source_id = hashlib.sha256(
                    f"{conference}_{year}_{title}".encode()
                ).hexdigest()[:16]

            papers.append(
                Paper(
                    source_id=source_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    conference=conference,
                    year=year,
                    pdf_url=pdf_url,
                )
            )
        return papers
