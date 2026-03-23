"""Fetcher for AAAI via OJS platform (ojs.aaai.org). Main track (Technical Tracks) only."""
from __future__ import annotations

import asyncio
import hashlib
import time

import requests
from lxml import html

from paper_filter.fetchers.base import BaseFetcher
from paper_filter.models import Paper

HEADERS = {"User-Agent": "Mozilla/5.0 (Research Paper Downloader)"}
TIMEOUT = 60


class AAAIFetcher(BaseFetcher):
    async def fetch(self, conference: str, year: int) -> list[Paper]:
        return await asyncio.to_thread(self._fetch_sync, year)

    def _fetch_sync(self, year: int) -> list[Paper]:
        issue_urls = self._find_issue_urls(year)
        if not issue_urls:
            return []

        papers = []
        for issue_url in issue_urls:
            papers.extend(self._scrape_issue(issue_url, year))
            time.sleep(0.3)
        return papers

    def _find_issue_urls(self, year: int) -> list[str]:
        """Scrape AAAI archive pages to find Technical Tracks issue URLs."""
        yy = str(year)[2:]
        tag = f"AAAI-{yy}"

        issue_urls = []
        for page in range(1, 10):
            archive_url = (
                f"https://ojs.aaai.org/index.php/AAAI/issue/archive/{page}"
            )
            try:
                resp = requests.get(archive_url, headers=HEADERS, timeout=TIMEOUT)
                if resp.status_code != 200:
                    break
            except Exception:
                break

            tree = html.fromstring(resp.content)
            links = tree.xpath('//a[contains(@class, "title")]')
            if not links:
                links = tree.xpath("//h2/a")
            if not links:
                break

            found_year = False
            for a in links:
                text = a.text_content().strip()
                href = a.get("href", "")
                if text.startswith(tag) and "Technical Tracks" in text:
                    issue_urls.append(href)
                    found_year = True
                elif found_year and not text.startswith(tag):
                    break

            if issue_urls and not any(
                a.text_content().strip().startswith(tag) for a in links[-3:]
            ):
                break

            time.sleep(0.3)

        return issue_urls

    def _scrape_issue(self, issue_url: str, year: int) -> list[Paper]:
        """Scrape papers from a single AAAI OJS issue page."""
        try:
            resp = requests.get(issue_url, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code != 200:
                return []
        except Exception:
            return []

        tree = html.fromstring(resp.content)
        articles = tree.xpath("//div[contains(@class, 'obj_article_summary')]")

        papers = []
        for art in articles:
            title_el = art.xpath(".//h3[contains(@class, 'title')]/a")
            if not title_el:
                continue
            title = title_el[0].text_content().strip()

            # Get direct PDF link from galleys
            pdf_el = art.xpath(
                ".//a[contains(@class, 'obj_galley_link') and contains(@class, 'pdf')]/@href"
            )
            if pdf_el:
                pdf_url = pdf_el[0]
            else:
                pdf_url = title_el[0].get("href", "")

            if not title or not pdf_url:
                continue

            source_id = hashlib.sha256(
                f"AAAI_{year}_{title}".encode()
            ).hexdigest()[:16]
            papers.append(
                Paper(
                    source_id=source_id,
                    title=title,
                    authors=[],
                    abstract="",
                    conference="AAAI",
                    year=year,
                    pdf_url=pdf_url,
                )
            )
        return papers
