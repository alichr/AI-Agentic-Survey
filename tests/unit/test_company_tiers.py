"""Tests for src.external.company_tiers — YAML loading + fuzzy matching."""

from src.external.company_tiers import CompanyTiers


class TestLoad:
    def test_load_valid_yaml(self, sample_tiers_yaml):
        ct = CompanyTiers(sample_tiers_yaml)
        assert len(ct.company_to_tier) > 0
        assert ct.tier_scores[1] == 1.0
        assert ct.tier_scores[2] == 0.8
        assert ct.tier_scores[3] == 0.6

    def test_missing_file_empty(self, tmp_path):
        ct = CompanyTiers(str(tmp_path / "nonexistent.yaml"))
        assert ct.company_to_tier == {}
        assert ct._names == []

    def test_unknown_score(self, sample_tiers_yaml):
        ct = CompanyTiers(sample_tiers_yaml)
        assert ct.unknown_score == 0.3


class TestGetTier:
    def test_exact_match(self, sample_tiers_yaml):
        ct = CompanyTiers(sample_tiers_yaml)
        assert ct.get_tier("Google") == 1

    def test_case_insensitive(self, sample_tiers_yaml):
        ct = CompanyTiers(sample_tiers_yaml)
        assert ct.get_tier("GOOGLE") == 1

    def test_substring_match(self, sample_tiers_yaml):
        ct = CompanyTiers(sample_tiers_yaml)
        # "google" is in "Google Research Lab"
        tier = ct.get_tier("Google Research Lab")
        assert tier == 1

    def test_fuzzy_match(self, sample_tiers_yaml):
        ct = CompanyTiers(sample_tiers_yaml)
        # Close enough to "Google" (substring match covers most cases;
        # test with a variation close enough for fuzzy match)
        tier = ct.get_tier("Gooogle")
        assert tier == 1

    def test_no_match_returns_none(self, sample_tiers_yaml):
        ct = CompanyTiers(sample_tiers_yaml)
        assert ct.get_tier("Random Startup XYZ") is None

    def test_empty_string_returns_none(self, sample_tiers_yaml):
        ct = CompanyTiers(sample_tiers_yaml)
        assert ct.get_tier("") is None

    def test_none_returns_none(self, sample_tiers_yaml):
        ct = CompanyTiers(sample_tiers_yaml)
        assert ct.get_tier(None) is None

    def test_tier2_match(self, sample_tiers_yaml):
        ct = CompanyTiers(sample_tiers_yaml)
        assert ct.get_tier("Samsung") == 2

    def test_tier3_match(self, sample_tiers_yaml):
        ct = CompanyTiers(sample_tiers_yaml)
        assert ct.get_tier("Uber") == 3


class TestScoreAffiliation:
    def test_tier1_score(self, sample_tiers_yaml):
        ct = CompanyTiers(sample_tiers_yaml)
        assert ct.score_affiliation("Google") == 1.0

    def test_tier2_score(self, sample_tiers_yaml):
        ct = CompanyTiers(sample_tiers_yaml)
        assert ct.score_affiliation("Samsung") == 0.8

    def test_tier3_score(self, sample_tiers_yaml):
        ct = CompanyTiers(sample_tiers_yaml)
        assert ct.score_affiliation("Uber") == 0.6

    def test_unknown_score(self, sample_tiers_yaml):
        ct = CompanyTiers(sample_tiers_yaml)
        assert ct.score_affiliation("Unknown Corp") == 0.3
