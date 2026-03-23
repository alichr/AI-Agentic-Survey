from __future__ import annotations

from abc import ABC, abstractmethod

from paper_filter.models import Paper


class BaseFetcher(ABC):
    @abstractmethod
    async def fetch(self, conference: str, year: int) -> list[Paper]:
        """Fetch all accepted papers from the given conference and year.

        Returns a list of Paper objects with source_id, title, authors,
        abstract, conference, year, and pdf_url populated.
        """
        ...
