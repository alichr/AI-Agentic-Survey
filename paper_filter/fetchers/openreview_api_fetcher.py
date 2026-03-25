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
    "NEURIPS": "NeurIPS.cc/{year}/Conference",
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


# Accepted venue prefixes for v1 API filtering
_ACCEPTED_KEYWORDS = ("Poster", "Spotlight", "Oral")

# V1 invitation patterns
V1_INVITATIONS = {
    "ICLR": "ICLR.cc/{year}/Conference/-/Blind_Submission",
}


class OpenReviewV1Fetcher(BaseFetcher):
    """Fetches papers from the older OpenReview API v1 (ICLR 2021-2023)."""

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

        invitation_template = V1_INVITATIONS.get(conference)
        if not invitation_template:
            raise ValueError(f"No v1 invitation for {conference}")

        invitation = invitation_template.format(year=year)
        logger.info("[FETCH] Connecting to OpenReview API v1 for %s...", invitation)

        client = openreview.Client(
            baseurl="https://api.openreview.net",
            username=username,
            password=password,
        )
        self.token = client.token

        notes = client.get_all_notes(invitation=invitation)
        # Filter to accepted papers only
        accepted = [
            n for n in notes
            if any(kw in n.content.get("venue", "") for kw in _ACCEPTED_KEYWORDS)
        ]
        logger.info("[FETCH] Got %d accepted papers from OpenReview v1 API", len(accepted))

        papers = []
        for note in accepted:
            content = note.content
            title = content.get("title", "").strip()
            if not title:
                continue

            authors = content.get("authors", [])
            if not isinstance(authors, list):
                authors = []

            abstract = content.get("abstract", "") or ""

            pdf_path = content.get("pdf", "")
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
