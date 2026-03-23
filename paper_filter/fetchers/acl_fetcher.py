"""Fetcher for EMNLP via ACL Anthology XML (GitHub)."""
from __future__ import annotations

import asyncio
import hashlib
import urllib.request
import xml.etree.ElementTree as ET

from paper_filter.fetchers.base import BaseFetcher
from paper_filter.models import Paper

HEADERS = {"User-Agent": "Mozilla/5.0 (Research Paper Downloader)"}
TIMEOUT = 60

# Volumes to include (main conference tracks)
EMNLP_VOLUMES = ("main", "demo", "industry")


class ACLFetcher(BaseFetcher):
    async def fetch(self, conference: str, year: int) -> list[Paper]:
        return await asyncio.to_thread(self._fetch_sync, conference.upper(), year)

    def _fetch_sync(self, conference: str, year: int) -> list[Paper]:
        xml_url = (
            f"https://raw.githubusercontent.com/acl-org/acl-anthology"
            f"/master/data/xml/{year}.emnlp.xml"
        )
        req = urllib.request.Request(xml_url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
                xml_data = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []
            raise

        root = ET.fromstring(xml_data)
        papers = []

        for volume in root.findall(".//volume"):
            vol_id = volume.get("id")
            if vol_id not in EMNLP_VOLUMES:
                continue

            for paper_el in volume.findall(".//paper"):
                paper_id = paper_el.get("id")
                if not paper_id:
                    continue

                title_elem = paper_el.find("title")
                title = (
                    "".join(title_elem.itertext()).strip()
                    if title_elem is not None
                    else ""
                )
                if not title:
                    continue

                # Extract authors
                authors = []
                for author_el in paper_el.findall("author"):
                    first = author_el.findtext("first", "")
                    last = author_el.findtext("last", "")
                    name = f"{first} {last}".strip()
                    if name:
                        authors.append(name)

                # Extract abstract
                abstract_elem = paper_el.find("abstract")
                abstract = (
                    "".join(abstract_elem.itertext()).strip()
                    if abstract_elem is not None
                    else ""
                )

                anthology_id = f"{year}.emnlp-{vol_id}.{paper_id}"
                pdf_url = f"https://aclanthology.org/{anthology_id}.pdf"

                papers.append(
                    Paper(
                        source_id=anthology_id,
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        conference="EMNLP",
                        year=year,
                        pdf_url=pdf_url,
                    )
                )
        return papers
