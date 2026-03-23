from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Paper:
    source_id: str
    title: str
    authors: list[str]
    abstract: str
    conference: str
    year: int
    introduction: str = ""
    pdf_url: Optional[str] = None
    relevance_score: Optional[float] = None
    relevance_reasoning: Optional[str] = None
    pdf_path: Optional[str] = None

    def authors_json(self) -> str:
        return json.dumps(self.authors)

    @staticmethod
    def authors_from_json(s: str) -> list[str]:
        return json.loads(s)
