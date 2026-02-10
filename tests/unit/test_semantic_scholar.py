"""Tests for src.external.semantic_scholar — Async aiohttp client."""

import re

import aiohttp
import pytest
from aioresponses import aioresponses

from src.external.semantic_scholar import BASE_URL, CitationData, SemanticScholarClient

SEARCH_URL_PATTERN = re.compile(
    r"https://api\.semanticscholar\.org/graph/v1/paper/search/match.*"
)


class TestSearchPaper:
    async def test_200_valid_data(self):
        with aioresponses() as m:
            m.get(SEARCH_URL_PATTERN, payload={
                "data": [{"citationCount": 42, "year": 2024, "paperId": "abc123"}]
            })
            client = SemanticScholarClient(api_key=None, rate_limit_rps=1000.0, max_retries=2)
            result = await client.search_paper("Test Paper Title")
            assert isinstance(result, CitationData)
            assert result.citation_count == 42
            assert result.year == 2024
            assert result.paper_id == "abc123"
            await client.close()

    async def test_404_returns_none(self):
        with aioresponses() as m:
            m.get(SEARCH_URL_PATTERN, status=404)
            client = SemanticScholarClient(api_key=None, rate_limit_rps=1000.0, max_retries=1)
            result = await client.search_paper("Nonexistent Paper")
            assert result is None
            await client.close()

    async def test_200_empty_data(self):
        with aioresponses() as m:
            m.get(SEARCH_URL_PATTERN, payload={"data": []})
            client = SemanticScholarClient(api_key=None, rate_limit_rps=1000.0, max_retries=1)
            result = await client.search_paper("Empty Result")
            assert result is None
            await client.close()

    async def test_200_no_data_key(self):
        with aioresponses() as m:
            m.get(SEARCH_URL_PATTERN, payload={})
            client = SemanticScholarClient(api_key=None, rate_limit_rps=1000.0, max_retries=1)
            result = await client.search_paper("No Data")
            assert result is None
            await client.close()

    async def test_429_then_200_retry_succeeds(self):
        with aioresponses() as m:
            m.get(SEARCH_URL_PATTERN, status=429)
            m.get(SEARCH_URL_PATTERN, payload={
                "data": [{"citationCount": 10, "year": 2023, "paperId": "xyz"}]
            })
            client = SemanticScholarClient(api_key=None, rate_limit_rps=1000.0, max_retries=3)
            result = await client.search_paper("Retry Paper")
            assert result is not None
            assert result.citation_count == 10
            await client.close()

    async def test_all_retries_429_returns_none(self):
        with aioresponses() as m:
            for _ in range(5):
                m.get(SEARCH_URL_PATTERN, status=429)
            client = SemanticScholarClient(api_key=None, rate_limit_rps=1000.0, max_retries=2)
            result = await client.search_paper("Rate Limited")
            assert result is None
            await client.close()

    async def test_500_returns_none(self):
        with aioresponses() as m:
            m.get(SEARCH_URL_PATTERN, status=500, body="Internal Server Error")
            client = SemanticScholarClient(api_key=None, rate_limit_rps=1000.0, max_retries=1)
            result = await client.search_paper("Server Error")
            assert result is None
            await client.close()

    async def test_client_error_with_retry(self):
        with aioresponses() as m:
            m.get(SEARCH_URL_PATTERN, exception=aiohttp.ClientError("fail"))
            m.get(SEARCH_URL_PATTERN, exception=aiohttp.ClientError("fail"))
            m.get(SEARCH_URL_PATTERN, payload={
                "data": [{"citationCount": 5, "year": 2024, "paperId": "p1"}]
            })
            client = SemanticScholarClient(api_key=None, rate_limit_rps=1000.0, max_retries=3)
            result = await client.search_paper("Flaky Connection")
            assert result is not None
            assert result.citation_count == 5
            await client.close()

    async def test_empty_title_returns_none(self):
        client = SemanticScholarClient(api_key=None, rate_limit_rps=1000.0)
        result = await client.search_paper("")
        assert result is None
        await client.close()

    async def test_none_title_returns_none(self):
        client = SemanticScholarClient(api_key=None, rate_limit_rps=1000.0)
        result = await client.search_paper(None)
        assert result is None
        await client.close()

    async def test_api_key_in_headers(self):
        with aioresponses() as m:
            m.get(SEARCH_URL_PATTERN, payload={
                "data": [{"citationCount": 1, "year": 2024, "paperId": "p"}]
            })
            client = SemanticScholarClient(api_key="test-key-123", rate_limit_rps=1000.0, max_retries=1)
            result = await client.search_paper("Key Test")
            assert result is not None
            # Verify the session was created with the key
            session = await client._get_session()
            assert session._default_headers.get("x-api-key") == "test-key-123"
            await client.close()


class TestSession:
    async def test_lazy_session_creation(self):
        client = SemanticScholarClient(api_key=None, rate_limit_rps=1000.0)
        assert client._session is None
        session = await client._get_session()
        assert session is not None
        assert isinstance(session, aiohttp.ClientSession)
        await client.close()

    async def test_close_works(self):
        client = SemanticScholarClient(api_key=None, rate_limit_rps=1000.0)
        await client._get_session()
        await client.close()
        assert client._session is None or client._session.closed
