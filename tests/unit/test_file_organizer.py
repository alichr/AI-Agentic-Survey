"""Tests for src.output.file_organizer — File copying by cluster."""

from pathlib import Path

from src.models.paper import Paper, PaperScores
from src.output.file_organizer import organize_accepted_papers


def _make_paper(tmp_path, filename="paper.pdf", accepted=True,
                cluster_label="Planning", create_pdf=True):
    pdf_path = tmp_path / "source" / filename
    if create_pdf:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

    return Paper(
        pdf_path=pdf_path,
        venue="ICML",
        pdf_hash="abc",
        scores=PaperScores(cluster_label=cluster_label),
        accepted=accepted,
    )


class TestOrganizeAcceptedPapers:
    def test_copies_accepted_paper(self, tmp_path):
        paper = _make_paper(tmp_path, "paper1.pdf", accepted=True, cluster_label="Planning")
        output_dir = tmp_path / "output"
        count = organize_accepted_papers([paper], output_dir)
        assert count == 1
        assert (output_dir / "accepted_papers" / "Planning" / "paper1.pdf").exists()

    def test_skips_rejected_papers(self, tmp_path):
        paper = _make_paper(tmp_path, "paper1.pdf", accepted=False, cluster_label="Planning")
        output_dir = tmp_path / "output"
        count = organize_accepted_papers([paper], output_dir)
        assert count == 0

    def test_creates_cluster_subdirectories(self, tmp_path):
        p1 = _make_paper(tmp_path, "p1.pdf", cluster_label="Planning")
        p2 = _make_paper(tmp_path, "p2.pdf", cluster_label="Tool Use")
        output_dir = tmp_path / "output"
        organize_accepted_papers([p1, p2], output_dir)
        assert (output_dir / "accepted_papers" / "Planning").is_dir()
        assert (output_dir / "accepted_papers" / "Tool Use").is_dir()

    def test_sanitizes_special_characters(self, tmp_path):
        paper = _make_paper(tmp_path, "paper.pdf", cluster_label="Topic/With:Special<Chars>")
        output_dir = tmp_path / "output"
        organize_accepted_papers([paper], output_dir)
        # Check that directory was created (special chars replaced)
        accepted_dir = output_dir / "accepted_papers"
        subdirs = list(accepted_dir.iterdir())
        assert len(subdirs) == 1
        # Name should not contain /:<>
        dir_name = subdirs[0].name
        assert "/" not in dir_name
        assert ":" not in dir_name
        assert "<" not in dir_name
        assert ">" not in dir_name

    def test_missing_source_pdf_skipped(self, tmp_path):
        paper = _make_paper(tmp_path, "missing.pdf", accepted=True, create_pdf=False)
        output_dir = tmp_path / "output"
        count = organize_accepted_papers([paper], output_dir)
        assert count == 0

    def test_returns_correct_count(self, tmp_path):
        papers = [
            _make_paper(tmp_path, "p1.pdf", accepted=True),
            _make_paper(tmp_path, "p2.pdf", accepted=True),
            _make_paper(tmp_path, "p3.pdf", accepted=False),
        ]
        output_dir = tmp_path / "output"
        count = organize_accepted_papers(papers, output_dir)
        assert count == 2

    def test_no_cluster_label_uncategorized(self, tmp_path):
        paper = _make_paper(tmp_path, "paper.pdf", cluster_label=None)
        output_dir = tmp_path / "output"
        organize_accepted_papers([paper], output_dir)
        assert (output_dir / "accepted_papers" / "Uncategorized" / "paper.pdf").exists()

    def test_empty_papers_list(self, tmp_path):
        output_dir = tmp_path / "output"
        count = organize_accepted_papers([], output_dir)
        assert count == 0
