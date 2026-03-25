"""Fetcher for ICLR (and other OpenReview-hosted conferences) using the authenticated OpenReview API."""
from __future__ import annotations

import asyncio
import logging
import os

import openreview

from paper_filter.fetchers.base import BaseFetcher
from paper_filter.models import Paper

logger = logging.getLogger(__name__)

# Venue ID patterns per conference
VENUE_IDS = {
    "ICLR": "ICLR.cc/{year}/Conference",
    "ICML": "ICML.cc/{year}/Conference",
}


def get_openreview_token() -> str:
    """Authenticate with OpenReview and return a Bearer token."""
    username = os.environ.get("OPENREVIEW_USERNAME", "")
    password = os.environ.get("OPENREVIEW_PASSWORD", "")
    if not username or not password:
        raise RuntimeError(
            "Set OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD env vars"
        )
    client = openreview.api.OpenReviewClient(
        baseurl="https://api2.openreview.net",
        username=username,
        password=password,
    )
    return client.token


class OpenReviewAPIFetcher(BaseFetcher):
    """Fetches papers and PDFs via the authenticated OpenReview API."""

    def __init__(self) -> None:
        self.token: str = ""

    async def fetch(self, conference: str, year: int) -> list[Paper]:
        return await asyncio.to_thread(self._fetch_sync, conference.upper(), year)

    def _fetch_sync(self, conference: str, year: int) -> list[Paper]:
        username = os.environ.get("OPENREVIEW_USERNAME", "")
        password = os.environ.get("OPENREVIEW_PASSWORD", "")
        if not username or not password:
            raise RuntimeError(
                "Set OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD env vars"
            )

        venue_template = VENUE_IDS.get(conference)
        if not venue_template:
            raise ValueError(f"No venue ID for {conference}")

        venue_id = venue_template.format(year=year)
        logger.info("[FETCH] Connecting to OpenReview API for %s...", venue_id)

        client = openreview.api.OpenReviewClient(
            baseurl="https://api2.openreview.net",
            username=username,
            password=password,
        )
        self.token = client.token

        notes = client.get_all_notes(content={"venueid": venue_id})
        logger.info("[FETCH] Got %d papers from OpenReview API", len(notes))

        papers = []
        for note in notes:
            content = note.content
            title = content.get("title", {}).get("value", "").strip()
            if not title:
                continue

            authors = []
            for author in content.get("authors", {}).get("value", []):
                if isinstance(author, str):
                    authors.append(author)

            abstract = content.get("abstract", {}).get("value", "") or ""

            pdf_path = content.get("pdf", {}).get("value", "")
            if pdf_path:
                pdf_url = f"https://openreview.net{pdf_path}"
            else:
                pdf_url = ""

            source_id = f"{conference}_{year}_{note.id}"

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
