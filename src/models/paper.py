"""Data models for the paper selection pipeline."""

from dataclasses import dataclass, field
from enum import IntEnum
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


class SectionType(IntEnum):
    """The 5 section views used in multi-view clustering."""
    TITLE_ABSTRACT_CONCLUSION = 0
    INTRODUCTION = 1
    RELATED_WORK = 2
    METHOD = 3
    EXPERIMENTS = 4


@dataclass
class PaperSections:
    """Segmented paper text across 5 logical sections."""
    title_abstract_conclusion: Optional[str] = None
    introduction: Optional[str] = None
    related_work: Optional[str] = None
    method: Optional[str] = None
    experiments: Optional[str] = None

    def get_section(self, section_type: SectionType) -> Optional[str]:
        """Retrieve section text by SectionType enum."""
        mapping = {
            SectionType.TITLE_ABSTRACT_CONCLUSION: self.title_abstract_conclusion,
            SectionType.INTRODUCTION: self.introduction,
            SectionType.RELATED_WORK: self.related_work,
            SectionType.METHOD: self.method,
            SectionType.EXPERIMENTS: self.experiments,
        }
        return mapping.get(section_type)

    def as_dict(self) -> dict[int, Optional[str]]:
        """Return all sections keyed by SectionType int value."""
        return {
            SectionType.TITLE_ABSTRACT_CONCLUSION: self.title_abstract_conclusion,
            SectionType.INTRODUCTION: self.introduction,
            SectionType.RELATED_WORK: self.related_work,
            SectionType.METHOD: self.method,
            SectionType.EXPERIMENTS: self.experiments,
        }


@dataclass
class PaperScores:
    """Scores computed for a paper in the pipeline."""
    # Per-view clustering results
    cluster_ids: dict[int, int] = field(default_factory=dict)
    centroid_distances: dict[int, float] = field(default_factory=dict)
    weak_member: dict[int, bool] = field(default_factory=dict)
    # Quality signals
    max_hindex: Optional[int] = None
    first_author_affiliation_score: Optional[float] = None
    last_author_affiliation_score: Optional[float] = None
    citation_count: Optional[int] = None


@dataclass
class Paper:
    """Represents a candidate paper flowing through the pipeline."""
    pdf_path: Path
    venue: str
    pdf_hash: str
    full_text: Optional[str] = None
    first_page_text: Optional[str] = None
    metadata: Optional[PaperMetadata] = None
    sections: Optional[PaperSections] = None
    section_summaries: dict[int, str] = field(default_factory=dict)
    section_embeddings: dict[int, list[float]] = field(default_factory=dict)
    scores: PaperScores = field(default_factory=PaperScores)

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
