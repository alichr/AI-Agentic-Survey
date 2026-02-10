"""Data models for the paper selection pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Author:
    """Represents a paper author with affiliation."""
    name: str
    affiliation: Optional[str] = None


@dataclass
class PaperMetadata:
    """Metadata extracted from a paper's first page via LLM."""
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: list[Author] = field(default_factory=list)
    year: Optional[int] = None


@dataclass
class PaperScores:
    """Scores computed for a paper across all active metrics."""
    relevance_score: Optional[float] = None
    cluster_id: Optional[int] = None
    cluster_label: Optional[str] = None
    affiliation_score: Optional[float] = None
    citation_score: Optional[float] = None
    citation_count: Optional[int] = None


@dataclass
class Paper:
    """Represents a candidate paper flowing through the pipeline."""
    pdf_path: Path
    venue: str
    pdf_hash: str
    is_seed: bool = False
    seed_label: Optional[str] = None
    first_page_text: Optional[str] = None
    metadata: Optional[PaperMetadata] = None
    embedding: Optional[list[float]] = None
    scores: PaperScores = field(default_factory=PaperScores)
    accepted: bool = False

    @property
    def title(self) -> Optional[str]:
        return self.metadata.title if self.metadata else None

    @property
    def abstract(self) -> Optional[str]:
        return self.metadata.abstract if self.metadata else None

    @property
    def first_author(self) -> Optional[Author]:
        if self.metadata and self.metadata.authors:
            return self.metadata.authors[0]
        return None

    @property
    def last_author(self) -> Optional[Author]:
        if self.metadata and len(self.metadata.authors) > 1:
            return self.metadata.authors[-1]
        elif self.metadata and len(self.metadata.authors) == 1:
            return self.metadata.authors[0]
        return None

    @property
    def embedding_text(self) -> Optional[str]:
        """Text used for computing embeddings: title [SEP] abstract."""
        if self.metadata and self.metadata.title and self.metadata.abstract:
            return f"{self.metadata.title} [SEP] {self.metadata.abstract}"
        elif self.metadata and self.metadata.title:
            return self.metadata.title
        return None
