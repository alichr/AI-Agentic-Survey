"""Tests for src.external.university_rankings — CSV loading + fuzzy matching."""

import csv
from pathlib import Path

from src.external.university_rankings import UNRANKED_SCORE, UniversityRankings


def _write_rankings_csv(tmp_path, rows):
    """Helper to write a test CSV."""
    path = tmp_path / "rankings.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "university"])
        writer.writeheader()
        for rank, name in rows:
            writer.writerow({"rank": rank, "university": name})
    return str(path)


class TestLoad:
    def test_load_valid_csv(self, tmp_path):
        path = _write_rankings_csv(tmp_path, [
            ("1", "MIT"),
            ("10", "ETH Zurich"),
        ])
        ur = UniversityRankings(path)
        assert len(ur.rankings) == 2

    def test_missing_file_empty_rankings(self, tmp_path):
        ur = UniversityRankings(str(tmp_path / "nonexistent.csv"))
        assert ur.rankings == {}
        assert ur._names == []

    def test_rank_range_format(self, tmp_path):
        path = _write_rankings_csv(tmp_path, [("101-150", "Some University")])
        ur = UniversityRankings(path)
        assert ur.rankings["some university"] == 101

    def test_invalid_rank_skipped(self, tmp_path):
        path = _write_rankings_csv(tmp_path, [
            ("abc", "Bad Rank Uni"),
            ("5", "Good Uni"),
        ])
        ur = UniversityRankings(path)
        assert len(ur.rankings) == 1


class TestGetRank:
    def test_exact_match(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        rank = ur.get_rank("Massachusetts Institute of Technology")
        assert rank == 1

    def test_case_insensitive(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        rank = ur.get_rank("MASSACHUSETTS INSTITUTE OF TECHNOLOGY")
        assert rank == 1

    def test_substring_match(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        # "University of Oxford" should match "Department of CS, University of Oxford"
        rank = ur.get_rank("Department of CS, University of Oxford")
        assert rank == 3

    def test_fuzzy_match_typo(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        # Slight misspelling
        rank = ur.get_rank("Masachusetts Institute of Technology")
        assert rank == 1

    def test_no_match_returns_none(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        rank = ur.get_rank("Completely Unknown School of Whatever")
        assert rank is None

    def test_empty_string_returns_none(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        assert ur.get_rank("") is None

    def test_none_returns_none(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        assert ur.get_rank(None) is None


class TestRankToScore:
    def test_rank_1(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        assert ur.rank_to_score(1) == 1.0

    def test_rank_10(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        assert ur.rank_to_score(10) == 1.0

    def test_rank_11(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        assert ur.rank_to_score(11) == 0.95

    def test_rank_25(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        assert ur.rank_to_score(25) == 0.95

    def test_rank_26(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        assert ur.rank_to_score(26) == 0.90

    def test_rank_50(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        assert ur.rank_to_score(50) == 0.90

    def test_rank_75(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        assert ur.rank_to_score(75) == 0.85

    def test_rank_100(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        assert ur.rank_to_score(100) == 0.80

    def test_rank_150(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        assert ur.rank_to_score(150) == 0.70

    def test_rank_200(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        assert ur.rank_to_score(200) == 0.60

    def test_rank_none(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        assert ur.rank_to_score(None) == UNRANKED_SCORE

    def test_rank_above_200(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        assert ur.rank_to_score(500) == UNRANKED_SCORE


class TestScoreAffiliation:
    def test_end_to_end(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        score = ur.score_affiliation("Massachusetts Institute of Technology")
        assert score == 1.0

    def test_unknown_affiliation(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        score = ur.score_affiliation("Random Unranked Place")
        assert score == UNRANKED_SCORE

    def test_mid_rank(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        # University of Toronto rank=25 => score 0.95
        score = ur.score_affiliation("University of Toronto")
        assert score == 0.95

    def test_rank_150_score(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        # Indian Institute of Technology Delhi rank=150 => score 0.70
        score = ur.score_affiliation("Indian Institute of Technology Delhi")
        assert score == 0.70

    def test_rank_200_score(self, sample_rankings_csv):
        ur = UniversityRankings(sample_rankings_csv)
        # Arizona State University rank=200 => score 0.60
        score = ur.score_affiliation("Arizona State University")
        assert score == 0.60
