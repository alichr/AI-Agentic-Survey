"""Tests for src.main — CLI argparse."""

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.main import main, setup_logging


class TestSetupLogging:
    def test_sets_debug_level(self):
        # Reset root logger handlers to allow basicConfig to work
        root = logging.getLogger()
        root.handlers.clear()
        setup_logging("DEBUG")
        assert root.level == logging.DEBUG

    def test_sets_info_level(self):
        root = logging.getLogger()
        root.handlers.clear()
        setup_logging("INFO")
        assert root.level == logging.INFO

    def test_sets_warning_level(self):
        root = logging.getLogger()
        root.handlers.clear()
        setup_logging("WARNING")
        assert root.level == logging.WARNING

    def test_sets_error_level(self):
        root = logging.getLogger()
        root.handlers.clear()
        setup_logging("ERROR")
        assert root.level == logging.ERROR

    def test_case_insensitive(self):
        root = logging.getLogger()
        root.handlers.clear()
        setup_logging("debug")
        assert root.level == logging.DEBUG


def _make_mock_config():
    """Create a mock Config with proper nested attribute access."""
    config = MagicMock()
    config.pipeline = MagicMock()
    config.pipeline.papers_dir = "papers"
    config.pipeline.output_dir = "output"
    config.pipeline.log_level = "INFO"
    config.relevance = MagicMock()
    config.relevance.enabled = True
    config.affiliation = MagicMock()
    config.affiliation.enabled = True
    config.citation = MagicMock()
    config.citation.enabled = True
    return config


class TestMainArgparse:
    def test_default_args(self):
        config = _make_mock_config()
        with patch("sys.argv", ["main"]):
            with patch("src.main.load_config", return_value=config) as mock_load:
                with patch("src.main.Pipeline") as mock_pipeline:
                    with patch("src.main.setup_logging"):
                        main()
                        mock_load.assert_called_once_with("config/default_config.yaml")

    def test_custom_config_path(self):
        config = _make_mock_config()
        with patch("sys.argv", ["main", "--config", "/custom/config.yaml"]):
            with patch("src.main.load_config", return_value=config) as mock_load:
                with patch("src.main.Pipeline"):
                    with patch("src.main.setup_logging"):
                        main()
                        mock_load.assert_called_once_with("/custom/config.yaml")

    def test_short_config_flag(self):
        config = _make_mock_config()
        with patch("sys.argv", ["main", "-c", "/custom/config.yaml"]):
            with patch("src.main.load_config", return_value=config) as mock_load:
                with patch("src.main.Pipeline"):
                    with patch("src.main.setup_logging"):
                        main()
                        mock_load.assert_called_once_with("/custom/config.yaml")

    def test_papers_dir_override(self):
        config = _make_mock_config()
        with patch("sys.argv", ["main", "--papers-dir", "/my/papers"]):
            with patch("src.main.load_config", return_value=config):
                with patch("src.main.Pipeline"):
                    with patch("src.main.setup_logging"):
                        main()
                        assert config.pipeline.papers_dir == "/my/papers"

    def test_output_dir_override(self):
        config = _make_mock_config()
        with patch("sys.argv", ["main", "--output-dir", "/my/output"]):
            with patch("src.main.load_config", return_value=config):
                with patch("src.main.Pipeline"):
                    with patch("src.main.setup_logging"):
                        main()
                        assert config.pipeline.output_dir == "/my/output"

    def test_log_level_override(self):
        config = _make_mock_config()
        with patch("sys.argv", ["main", "--log-level", "DEBUG"]):
            with patch("src.main.load_config", return_value=config):
                with patch("src.main.Pipeline"):
                    with patch("src.main.setup_logging"):
                        main()
                        assert config.pipeline.log_level == "DEBUG"

    def test_no_relevance_flag(self):
        config = _make_mock_config()
        with patch("sys.argv", ["main", "--no-relevance"]):
            with patch("src.main.load_config", return_value=config):
                with patch("src.main.Pipeline"):
                    with patch("src.main.setup_logging"):
                        main()
                        assert config.relevance.enabled is False

    def test_no_affiliation_flag(self):
        config = _make_mock_config()
        with patch("sys.argv", ["main", "--no-affiliation"]):
            with patch("src.main.load_config", return_value=config):
                with patch("src.main.Pipeline"):
                    with patch("src.main.setup_logging"):
                        main()
                        assert config.affiliation.enabled is False

    def test_no_citation_flag(self):
        config = _make_mock_config()
        with patch("sys.argv", ["main", "--no-citation"]):
            with patch("src.main.load_config", return_value=config):
                with patch("src.main.Pipeline"):
                    with patch("src.main.setup_logging"):
                        main()
                        assert config.citation.enabled is False

    def test_multiple_flags_combined(self):
        config = _make_mock_config()
        with patch("sys.argv", ["main", "--no-relevance", "--no-affiliation", "--no-citation",
                                 "--log-level", "ERROR"]):
            with patch("src.main.load_config", return_value=config):
                with patch("src.main.Pipeline"):
                    with patch("src.main.setup_logging"):
                        main()
                        assert config.relevance.enabled is False
                        assert config.affiliation.enabled is False
                        assert config.citation.enabled is False
                        assert config.pipeline.log_level == "ERROR"

    def test_invalid_log_level_raises(self):
        with patch("sys.argv", ["main", "--log-level", "INVALID"]):
            with pytest.raises(SystemExit):
                main()
