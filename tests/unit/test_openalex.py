"""Tests for src.external.openalex — OpenAlex API client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.external.openalex import OpenAlexClient
from src.external.semantic_scholar import CitationData


class TestOpenAlexClient:
    @pytest.fixture
    def client(self):
        return OpenAlexClient(api_key="test-key", email="test@example.com")

    def test_build_params_with_key_and_email(self, client):
        params = client._build_params("Attention Is All You Need")
        assert params["filter"] == "title.search:Attention Is All You Need"
        assert params["select"] == "id,title,cited_by_count,publication_year"
        assert params["per_page"] == "1"
        assert params["api_key"] == "test-key"
        assert params["mailto"] == "test@example.com"

    def test_build_params_no_key(self):
        client = OpenAlexClient()
        params = client._build_params("Test Paper")
        assert "api_key" not in params
        assert "mailto" not in params

    async def test_search_paper_success(self, client):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "results": [{
                "id": "https://openalex.org/W123",
                "title": "Attention Is All You Need",
                "cited_by_count": 100000,
                "publication_year": 2017,
            }]
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False
        client._session = mock_session

        result = await client.search_paper("Attention Is All You Need")
        assert isinstance(result, CitationData)
        assert result.citation_count == 100000
        assert result.year == 2017
        assert result.paper_id == "https://openalex.org/W123"

    async def test_search_paper_no_results(self, client):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"results": []})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False
        client._session = mock_session

        result = await client.search_paper("Nonexistent Paper XYZ")
        assert result is None

    async def test_search_paper_empty_title(self, client):
        result = await client.search_paper("")
        assert result is None

    async def test_search_paper_none_title(self, client):
        result = await client.search_paper(None)
        assert result is None

    async def test_close_session(self, client):
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        client._session = mock_session

        await client.close()
        mock_session.close.assert_called_once()
        assert client._session is None
