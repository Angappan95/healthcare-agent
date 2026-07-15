"""
Unit tests for LangGraph agent graphs.

Tests individual graph execution and state transformations.
Does not require Docker services to be running.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.agents.agent_graphs import (
    create_pubmed_graph,
    create_websearch_graph,
    create_synthesis_graph
)
from src.models.state import PubMedState, WebSearchState, SynthesisState

class TestPubMedGraph:
    """Unit tests for PubMed agent graph."""

    @patch('src.utils.pubmed_utils.search_articles')
    @patch('src.utils.pubmed_utils.extract_pubmed_articles_as_markdown')
    @patch('src.tools.helper.get_llm')
    def test_pubmed_graph_success(
        self,
        mock_get_llm,
        mock_extract,
        mock_search,
        sample_pubmed_state: dict
    ):
        """Test successful PubMed graph execution."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="diabetes, treatment, insulin")
        mock_get_llm.return_value = mock_llm
        
        mock_search.return_value = ["12345678", "87654321"]
        mock_extract.return_value = "## Article 1\n**Title:** Diabetes Research"
        
        # Execute graph
        graph = create_pubmed_graph()
        result = graph.invoke(sample_pubmed_state)
        
        # Verify state updates
        assert "keywords" in result
        assert len(result["keywords"]) > 0
        assert "pmids" in result
        assert len(result["pmids"]) == 2
        assert "article_markdown" in result
        assert len(result["article_markdown"]) > 0

    @patch('src.utils.pubmed_utils.search_articles')
    @patch('src.tools.helper.get_llm')
    def test_pubmed_graph_no_results(
        self,
        mock_get_llm,
        mock_search,
        sample_pubmed_state: dict
    ):
        """Test PubMed graph when no articles found."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="rare, disease, xyz")
        mock_get_llm.return_value = mock_llm
        
        mock_search.return_value = []  # No results
        
        # Execute graph
        graph = create_pubmed_graph()
        result = graph.invoke(sample_pubmed_state)
        
        # Verify state
        assert "keywords" in result
        assert "pmids" in result
        assert len(result["pmids"]) == 0

    @patch('src.tools.helper.get_llm')
    def test_pubmed_graph_keyword_extraction_fallback(
        self,
        mock_get_llm,
        sample_pubmed_state: dict
    ):
        """Test fallback when keyword extraction fails."""
        # Setup mock to raise exception
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        mock_get_llm.return_value = mock_llm
        
        # Execute graph
        graph = create_pubmed_graph()
        result = graph.invoke(sample_pubmed_state)
        
        # Should fallback to query split
        assert "keywords" in result
        assert len(result["keywords"]) > 0

class TestWebSearchGraph:
    """Unit tests for WebSearch agent graph."""

    @patch('tavily.TavilyClient')
    def test_websearch_graph_success(
        self,
        mock_tavily_class,
        sample_websearch_state: dict
    ):
        """Test successful WebSearch graph execution."""
        # Setup mock
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {
            "results": [
                {"url": "https://cdc.gov/diabetes", "title": "CDC Diabetes", "content": "CDC info about diabetes"},
                {"url": "https://mayoclinic.org/diabetes", "title": "Mayo Clinic", "content": "Mayo Clinic info"}
            ]
        }
        mock_tavily_class.return_value = mock_tavily
        
        # Execute graph
        graph = create_websearch_graph()
        result = graph.invoke(sample_websearch_state)
        
        # Verify state
        assert "search_results" in result
        assert "credible_results" in result
        assert "web_search_markdown" in result

    @patch('tavily.TavilyClient')
    def test_websearch_graph_filter_credible(
        self,
        mock_tavily_class,
        sample_websearch_state: dict
    ):
        """Test filtering for credible sources."""
        # Setup mock with mixed sources
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {
            "results": [
                {"url": "https://cdc.gov/info", "title": "CDC", "content": "CDC content"},
                {"url": "https://random-blog.com/post", "title": "Blog", "content": "Blog content"},
                {"url": "https://nih.gov/research", "title": "NIH", "content": "NIH content"}
            ]
        }
        mock_tavily_class.return_value = mock_tavily
        
        # Execute graph
        graph = create_websearch_graph()
        result = graph.invoke(sample_websearch_state)
        
        # Should filter to credible sources only
        assert "credible_results" in result
        # CDC and NIH should be included, blog should be filtered

class TestSynthesisGraph:
    """Unit tests for Synthesis agent graph."""

    @patch('src.tools.helper.get_llm')
    def test_synthesis_graph_success(
        self,
        mock_get_llm,
        sample_synthesis_state: dict
    ):
        """Test successful Synthesis graph execution."""
        # Setup mock
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="## Comprehensive Response\n\nDiabetes information synthesized from multiple sources.\n\n**Sources:** PubMed, WebSearch"
        )
        mock_get_llm.return_value = mock_llm
        
        # Add context
        sample_synthesis_state["context"] = {
            "pubmed_markdown": "## PubMed Article\nContent",
            "websearch_markdown": "## Web Results\nContent"
        }
        
        # Execute graph
        graph = create_synthesis_graph()
        result = graph.invoke(sample_synthesis_state)
        
        # Verify state
        assert "combined_content" in result
        assert "response" in result
        assert len(result["response"]) > 0
        assert "sources_used" in result

    @patch('src.tools.helper.get_llm')
    def test_synthesis_graph_single_source(
        self,
        mock_get_llm,
        sample_synthesis_state: dict
    ):
        """Test synthesis with single source."""
        # Setup mock
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Response based on PubMed data only"
        )
        mock_get_llm.return_value = mock_llm
        
        # Add only PubMed context
        sample_synthesis_state["context"] = {
            "pubmed_markdown": "## PubMed Article\nContent"
        }
        
        # Execute graph
        graph = create_synthesis_graph()
        result = graph.invoke(sample_synthesis_state)
        
        # Should still work
        assert "response" in result
        assert len(result["response"]) > 0

class TestGraphStateManagement:
    """Test state management across graph nodes."""

    def test_pubmed_state_structure(self):
        """Test PubMedState structure."""
        from src.models.state import PubMedState
        
        # Verify required fields
        assert "query" in PubMedState.__annotations__
        assert "keywords" in PubMedState.__annotations__
        assert "pmids" in PubMedState.__annotations__
        assert "article_markdown" in PubMedState.__annotations__
        assert "error" in PubMedState.__annotations__

    def test_websearch_state_structure(self):
        """Test WebSearchState structure."""
        from src.models.state import WebSearchState
        
        assert "query" in WebSearchState.__annotations__
        assert "search_results" in WebSearchState.__annotations__
        assert "credible_results" in WebSearchState.__annotations__
        assert "web_search_markdown" in WebSearchState.__annotations__
        assert "error" in WebSearchState.__annotations__

    def test_synthesis_state_structure(self):
        """Test SynthesisState structure."""
        from src.models.state import SynthesisState
        
        assert "query" in SynthesisState.__annotations__
        assert "context" in SynthesisState.__annotations__
        assert "combined_content" in SynthesisState.__annotations__
        assert "response" in SynthesisState.__annotations__
        assert "sources_used" in SynthesisState.__annotations__
        assert "error" in SynthesisState.__annotations__
