"""Tests for src.output.csv_writer — CSV output."""

import csv
from pathlib import Path

from src.models.paper import Author, Paper, PaperMetadata, PaperScores
from src.output.csv_writer import CSV_COLUMNS, _fmt, write_results_csv


def _make_paper(
    title="Test Paper",
    accepted=True,
    relevance=0.85,
    affiliation=0.7,
    citation=0.5,
    citation_count=42,
    cluster_id=0,
    cluster_label="Planning",
    year=2024,
    authors=None,
    venue="ICML",
    pdf_path=None,
):
    if authors is None:
        authors = [
            Author("Alice Smith", "MIT"),
            Author("Bob Jones", "Google"),
        ]
    if pdf_path is None:
        pdf_path = Path("/tmp/test.pdf")

    return Paper(
        pdf_path=pdf_path,
        venue=venue,
        pdf_hash="abc123",
        metadata=PaperMetadata(
            title=title,
            abstract="Test abstract",
            authors=authors,
            year=year,
        ),
        scores=PaperScores(
            relevance_score=relevance,
            cluster_id=cluster_id,
            cluster_label=cluster_label,
            affiliation_score=affiliation,
            citation_score=citation,
            citation_count=citation_count,
        ),
        accepted=accepted,
    )


class TestWriteResultsCsv:
    def test_creates_file(self, tmp_path):
        papers = [_make_paper()]
        path = tmp_path / "results.csv"
        write_results_csv(papers, path)
        assert path.exists()

    def test_correct_header(self, tmp_path):
        papers = [_make_paper()]
        path = tmp_path / "results.csv"
        write_results_csv(papers, path)

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == CSV_COLUMNS

    def test_correct_row_count(self, tmp_path):
        papers = [_make_paper(), _make_paper(title="Paper 2")]
        path = tmp_path / "results.csv"
        write_results_csv(papers, path)

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2

    def test_accepted_status(self, tmp_path):
        papers = [_make_paper(accepted=True)]
        path = tmp_path / "results.csv"
        write_results_csv(papers, path)

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["status"] == "ACCEPTED"

    def test_rejected_status(self, tmp_path):
        papers = [_make_paper(accepted=False)]
        path = tmp_path / "results.csv"
        write_results_csv(papers, path)

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["status"] == "REJECTED"

    def test_none_scores_empty_strings(self, tmp_path):
        papers = [_make_paper(relevance=None, affiliation=None, citation=None, citation_count=None, cluster_id=None)]
        path = tmp_path / "results.csv"
        write_results_csv(papers, path)

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["relevance_score"] == ""
        assert row["affiliation_score"] == ""
        assert row["citation_score"] == ""
        assert row["citation_count"] == ""

    def test_score_formatting_4_decimals(self, tmp_path):
        papers = [_make_paper(relevance=0.123456789)]
        path = tmp_path / "results.csv"
        write_results_csv(papers, path)

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["relevance_score"] == "0.1235"

    def test_authors_semicolon_separated(self, tmp_path):
        authors = [Author("Alice", "MIT"), Author("Bob", "Google"), Author("Carol", "Meta")]
        papers = [_make_paper(authors=authors)]
        path = tmp_path / "results.csv"
        write_results_csv(papers, path)

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["authors"] == "Alice; Bob; Carol"

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "results.csv"
        write_results_csv([_make_paper()], path)
        assert path.exists()

    def test_empty_paper_list_header_only(self, tmp_path):
        path = tmp_path / "results.csv"
        write_results_csv([], path)

        with open(path, newline="") as f:
            lines = f.readlines()
        # Should have exactly one line: the header
        assert len(lines) == 1
        assert "pdf_path" in lines[0]

    def test_paper_without_metadata(self, tmp_path):
        paper = Paper(
            pdf_path=Path("/tmp/t.pdf"),
            venue="V",
            pdf_hash="h",
        )
        path = tmp_path / "results.csv"
        write_results_csv([paper], path)

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["title"] == ""
        assert row["authors"] == ""


class TestFmt:
    def test_none(self):
        assert _fmt(None) == ""

    def test_float(self):
        assert _fmt(0.5) == "0.5000"

    def test_zero(self):
        assert _fmt(0.0) == "0.0000"

    def test_one(self):
        assert _fmt(1.0) == "1.0000"
