"""Tests for src.models.paper — Paper, Author, PaperMetadata, PaperScores."""

from pathlib import Path

from src.models.paper import Author, Paper, PaperMetadata, PaperScores


class TestAuthor:
    def test_creation_with_affiliation(self):
        a = Author(name="Alice", affiliation="MIT")
        assert a.name == "Alice"
        assert a.affiliation == "MIT"

    def test_creation_without_affiliation(self):
        a = Author(name="Bob")
        assert a.name == "Bob"
        assert a.affiliation is None


class TestPaperMetadata:
    def test_defaults(self):
        m = PaperMetadata()
        assert m.title is None
        assert m.abstract is None
        assert m.authors == []
        assert m.year is None

    def test_full_metadata(self):
        authors = [Author("A", "MIT"), Author("B", "Google")]
        m = PaperMetadata(title="T", abstract="Ab", authors=authors, year=2024)
        assert m.title == "T"
        assert len(m.authors) == 2
        assert m.year == 2024


class TestPaperScores:
    def test_defaults_all_none(self):
        s = PaperScores()
        assert s.relevance_score is None
        assert s.cluster_id is None
        assert s.cluster_label is None
        assert s.affiliation_score is None
        assert s.citation_score is None
        assert s.citation_count is None

    def test_set_values(self):
        s = PaperScores(relevance_score=0.9, cluster_id=2, cluster_label="Planning")
        assert s.relevance_score == 0.9
        assert s.cluster_id == 2
        assert s.cluster_label == "Planning"


class TestPaper:
    def _make_paper(self, **kwargs):
        defaults = dict(
            pdf_path=Path("/tmp/test.pdf"),
            venue="ICML",
            pdf_hash="abc123",
        )
        defaults.update(kwargs)
        return Paper(**defaults)

    def test_required_fields(self):
        p = self._make_paper()
        assert p.pdf_path == Path("/tmp/test.pdf")
        assert p.venue == "ICML"
        assert p.pdf_hash == "abc123"
        assert p.is_seed is False
        assert p.accepted is False

    def test_title_property_with_metadata(self):
        meta = PaperMetadata(title="My Title")
        p = self._make_paper(metadata=meta)
        assert p.title == "My Title"

    def test_title_property_no_metadata(self):
        p = self._make_paper()
        assert p.title is None

    def test_abstract_property(self):
        meta = PaperMetadata(abstract="My abstract")
        p = self._make_paper(metadata=meta)
        assert p.abstract == "My abstract"

    def test_abstract_property_no_metadata(self):
        p = self._make_paper()
        assert p.abstract is None

    def test_first_author_multiple(self):
        authors = [Author("A"), Author("B"), Author("C")]
        meta = PaperMetadata(authors=authors)
        p = self._make_paper(metadata=meta)
        assert p.first_author.name == "A"

    def test_last_author_multiple(self):
        authors = [Author("A"), Author("B"), Author("C")]
        meta = PaperMetadata(authors=authors)
        p = self._make_paper(metadata=meta)
        assert p.last_author.name == "C"

    def test_single_author_is_both_first_and_last(self):
        authors = [Author("Solo")]
        meta = PaperMetadata(authors=authors)
        p = self._make_paper(metadata=meta)
        assert p.first_author.name == "Solo"
        assert p.last_author.name == "Solo"

    def test_empty_authors_returns_none(self):
        meta = PaperMetadata(authors=[])
        p = self._make_paper(metadata=meta)
        assert p.first_author is None
        assert p.last_author is None

    def test_no_metadata_returns_none_authors(self):
        p = self._make_paper()
        assert p.first_author is None
        assert p.last_author is None

    def test_embedding_text_title_and_abstract(self):
        meta = PaperMetadata(title="T", abstract="A")
        p = self._make_paper(metadata=meta)
        assert p.embedding_text == "T [SEP] A"

    def test_embedding_text_title_only(self):
        meta = PaperMetadata(title="T")
        p = self._make_paper(metadata=meta)
        assert p.embedding_text == "T"

    def test_embedding_text_no_title(self):
        meta = PaperMetadata(abstract="A")
        p = self._make_paper(metadata=meta)
        assert p.embedding_text is None

    def test_embedding_text_no_metadata(self):
        p = self._make_paper()
        assert p.embedding_text is None

    def test_scores_default_factory(self):
        p = self._make_paper()
        assert isinstance(p.scores, PaperScores)
        assert p.scores.relevance_score is None
