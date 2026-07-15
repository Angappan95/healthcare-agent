"""Pytest configuration and fixtures for distributed multi-agent tests."""

import pytest
import os
from typing import Generator


@pytest.fixture(scope="session")
def api_keys_configured() -> bool:
    """Check if required API keys are configured."""
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_tavily = bool(os.getenv("TAVILY_API_KEY"))
    
    return (has_gemini or has_openai) and has_tavily


@pytest.fixture(scope="session")
def supervisor_url() -> str:
    """Supervisor API base URL."""
    return os.getenv("SUPERVISOR_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def pubmed_service_url() -> str:
    """PubMed service URL."""
    return os.getenv("PUBMED_SERVICE_URL", "http://localhost:8001")


@pytest.fixture(scope="session")
def websearch_service_url() -> str:
    """WebSearch service URL."""
    return os.getenv("WEBSEARCH_SERVICE_URL", "http://localhost:8002")


@pytest.fixture(scope="session")
def synthesis_service_url() -> str:
    """Synthesis service URL."""
    return os.getenv("SYNTHESIS_SERVICE_URL", "http://localhost:8003")


@pytest.fixture
def sample_query() -> str:
    """Sample healthcare query for testing."""
    return "What are the symptoms of diabetes?"


@pytest.fixture
def sample_pubmed_state() -> dict:
    """Sample PubMed agent state."""
    return {
        "query": "diabetes treatment",
        "keywords": [],
        "pmids": [],
        "article_markdown": "",
        "error": None
    }


@pytest.fixture
def sample_websearch_state() -> dict:
    """Sample WebSearch agent state."""
    return {
        "query": "diabetes symptoms",
        "search_results": [],
        "credible_results": [],
        "web_search_markdown": "",
        "error": None
    }


@pytest.fixture
def sample_synthesis_state() -> dict:
    """Sample Synthesis agent state."""
    return {
        "query": "diabetes information",
        "context": {},
        "combined_content": "",
        "response": "",
        "sources_used": [],
        "error": None
    }
