"""Tests for src.extraction.metadata_extractor — Async LLM extraction."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.extraction.metadata_extractor import MetadataExtractor
from src.models.paper import Author, PaperMetadata


def _make_mock_response(content_str):
    """Helper to create a mock OpenAI response."""
    mock_message = MagicMock()
    mock_message.content = content_str

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


class TestExtract:
    @pytest.fixture(autouse=True)
    def setup_extractor(self):
        with patch("src.extraction.metadata_extractor.AsyncOpenAI") as mock_cls:
            self.mock_client = AsyncMock()
            mock_cls.return_value = self.mock_client
            self.extractor = MetadataExtractor("http://localhost:8000/v1", "test-model")
            yield

    async def test_valid_json_response(self):
        response_json = {
            "title": "Test Paper",
            "abstract": "Test abstract.",
            "authors": [{"name": "Alice", "affiliation": "MIT"}],
            "year": 2024,
        }
        self.mock_client.chat.completions.create = AsyncMock(
            return_value=_make_mock_response(json.dumps(response_json))
        )

        result = await self.extractor.extract("A" * 100)
        assert result is not None
        assert result.title == "Test Paper"
        assert result.abstract == "Test abstract."
        assert len(result.authors) == 1
        assert result.authors[0].name == "Alice"
        assert result.year == 2024

    async def test_invalid_json_retry_then_success(self):
        valid_json = json.dumps({
            "title": "Paper",
            "abstract": "Ab",
            "authors": [],
            "year": 2024,
        })
        self.mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                _make_mock_response("not valid json"),
                _make_mock_response(valid_json),
            ]
        )

        result = await self.extractor.extract("A" * 100, retries=1)
        assert result is not None
        assert result.title == "Paper"

    async def test_all_retries_fail_returns_none(self):
        self.mock_client.chat.completions.create = AsyncMock(
            return_value=_make_mock_response("garbage output no json here")
        )

        result = await self.extractor.extract("A" * 100, retries=1)
        assert result is None

    async def test_short_text_returns_none(self):
        result = await self.extractor.extract("short")
        assert result is None

    async def test_empty_text_returns_none(self):
        result = await self.extractor.extract("")
        assert result is None

    async def test_exception_returns_none(self):
        self.mock_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("connection error")
        )
        result = await self.extractor.extract("A" * 100)
        assert result is None

    async def test_none_content_returns_none(self):
        """LLM response with None content should not crash."""
        self.mock_client.chat.completions.create = AsyncMock(
            return_value=_make_mock_response(None)
        )
        result = await self.extractor.extract("A" * 100, retries=0)
        assert result is None

    async def test_none_content_with_retry_succeeds(self):
        """None content on first attempt, valid JSON on retry."""
        valid_json = json.dumps({
            "title": "Paper",
            "abstract": "Ab",
            "authors": [],
            "year": 2024,
        })
        self.mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                _make_mock_response(None),
                _make_mock_response(valid_json),
            ]
        )
        result = await self.extractor.extract("A" * 100, retries=1)
        assert result is not None
        assert result.title == "Paper"


class TestParseResponse:
    def test_valid_json(self):
        content = json.dumps({
            "title": "T",
            "abstract": "A",
            "authors": [{"name": "X", "affiliation": "Y"}],
            "year": 2024,
        })
        result = MetadataExtractor._parse_response(content)
        assert result.title == "T"
        assert result.authors[0].name == "X"

    def test_json_with_surrounding_text(self):
        content = 'Here is the JSON:\n{"title": "T", "abstract": "A", "authors": [], "year": 2024}\nDone.'
        result = MetadataExtractor._parse_response(content)
        assert result.title == "T"

    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            MetadataExtractor._parse_response("no json here at all")

    def test_null_fields(self):
        content = json.dumps({
            "title": None,
            "abstract": None,
            "authors": None,
            "year": None,
        })
        result = MetadataExtractor._parse_response(content)
        assert result.title is None
        assert result.authors == []

    def test_string_authors(self):
        content = json.dumps({
            "title": "T",
            "abstract": "A",
            "authors": ["Alice", "Bob"],
            "year": 2024,
        })
        result = MetadataExtractor._parse_response(content)
        assert len(result.authors) == 2
        assert result.authors[0].name == "Alice"
        assert result.authors[0].affiliation is None


class TestExtractBatch:
    @pytest.fixture(autouse=True)
    def setup_extractor(self):
        with patch("src.extraction.metadata_extractor.AsyncOpenAI") as mock_cls:
            self.mock_client = AsyncMock()
            mock_cls.return_value = self.mock_client
            self.extractor = MetadataExtractor("http://localhost:8000/v1", "test-model")
            yield

    async def test_batch_processes_multiple(self):
        response = _make_mock_response(json.dumps({
            "title": "Paper",
            "abstract": "Ab",
            "authors": [],
            "year": 2024,
        }))
        self.mock_client.chat.completions.create = AsyncMock(return_value=response)

        texts = [("hash1", "A" * 100), ("hash2", "B" * 100)]
        result = await self.extractor.extract_batch(texts)
        assert "hash1" in result
        assert "hash2" in result
        assert result["hash1"] is not None

    async def test_batch_handles_exception_in_one(self):
        valid_response = _make_mock_response(json.dumps({
            "title": "Good",
            "abstract": "A",
            "authors": [],
            "year": 2024,
        }))

        call_count = 0

        async def _side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("fail")
            return valid_response

        self.mock_client.chat.completions.create = AsyncMock(side_effect=_side_effect)

        texts = [("hash1", "A" * 100), ("hash2", "B" * 100)]
        result = await self.extractor.extract_batch(texts)
        # At least one should succeed
        assert len(result) >= 1
