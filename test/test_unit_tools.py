"""
Unit tests for RemoteGraph supervisor tools.

Tests supervisor tools with mocked RemoteGraph connections.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.agents.supervisor_tools import (
    search_pubmed_literature,
    search_web_sources,
    synthesize_information
)

class TestSupervisorTools:
    """Test supervisor RemoteGraph tools."""

    @patch('src.agents.supervisor_tools.pubmed_graph')
    def test_search_pubmed_tool_success(self, mock_graph):
        """Test successful PubMed tool invocation."""
        # Mock RemoteGraph response
        mock_graph.invoke.return_value = {
            "keywords": ["diabetes", "treatment"],
            "pmids": ["12345", "67890"],
            "article_markdown": "## Article\n**Title:** Research",
            "error": None
        }
        
        # Call tool
        result = search_pubmed_literature.invoke({"query": "diabetes treatment"})
        
        # Verify
        assert result["status"] == "success"
        assert "data" in result
        assert "keywords" in result["data"]
        assert len(result["data"]["pmids"]) == 2

    @patch('src.agents.supervisor_tools.pubmed_graph')
    def test_search_pubmed_tool_error(self, mock_graph):
        """Test PubMed tool error handling."""
        # Mock RemoteGraph to raise exception
        mock_graph.invoke.side_effect = Exception("Connection error")
        
        # Call tool
        result = search_pubmed_literature.invoke({"query": "test query"})
        
        # Should handle error gracefully
        assert result["status"] == "error"
        assert "error" in result

    @patch('src.agents.supervisor_tools.websearch_graph')
    def test_search_web_tool_success(self, mock_graph):
        """Test successful WebSearch tool invocation."""
        # Mock RemoteGraph response
        mock_graph.invoke.return_value = {
            "search_results": [{"url": "https://cdc.gov", "content": "Info"}],
            "credible_results": [{"url": "https://cdc.gov", "content": "Info"}],
            "web_search_markdown": "## Web Results\nContent",
            "error": None
        }
        
        # Call tool
        result = search_web_sources.invoke({"query": "diabetes symptoms"})
        
        # Verify
        assert result["status"] == "success"
        assert "data" in result
        assert "web_search_markdown" in result["data"]

    @patch('src.agents.supervisor_tools.websearch_graph')
    def test_search_web_tool_error(self, mock_graph):
        """Test WebSearch tool error handling."""
        # Mock RemoteGraph to raise exception
        mock_graph.invoke.side_effect = Exception("API error")
        
        # Call tool
        result = search_web_sources.invoke({"query": "test query"})
        
        # Should handle error gracefully
        assert result["status"] == "error"
        assert "error" in result

    @patch('src.agents.supervisor_tools.synthesis_graph')
    def test_synthesize_tool_success(self, mock_graph):
        """Test successful Synthesis tool invocation."""
        # Mock RemoteGraph response
        mock_graph.invoke.return_value = {
            "response": "Synthesized response about diabetes",
            "sources_used": ["pubmed", "websearch"],
            "error": None
        }
        
        # Call tool
        result = synthesize_information.invoke({
            "query": "What is diabetes?",
            "context": {"pubmed_markdown": "## Article", "websearch_markdown": "## Web"}
        })
        
        # Verify
        assert result["status"] == "success"
        assert "data" in result
        assert "response" in result["data"]
        assert len(result["data"]["sources_used"]) == 2

    @patch('src.agents.supervisor_tools.synthesis_graph')
    def test_synthesize_tool_error(self, mock_graph):
        """Test Synthesis tool error handling."""
        # Mock RemoteGraph to raise exception
        mock_graph.invoke.side_effect = Exception("LLM error")
        
        # Call tool
        result = synthesize_information.invoke({"query": "test", "context": {}})
        
        # Should handle error gracefully
        assert result["status"] == "error"
        assert "error" in result

class TestToolIntegration:
    """Test tool integration patterns."""

    @patch('src.agents.supervisor_tools.pubmed_graph')
    @patch('src.agents.supervisor_tools.synthesis_graph')
    def test_pubmed_to_synthesis_flow(self, mock_synthesis, mock_pubmed):
        """Test flow from PubMed to Synthesis."""
        # Mock PubMed response
        mock_pubmed.invoke.return_value = {
            "article_markdown": "## Research Article",
            "pmids": ["12345"],
            "error": None
        }
        
        # Mock Synthesis response
        mock_synthesis.invoke.return_value = {
            "response": "Final synthesized response",
            "sources_used": ["pubmed"],
            "error": None
        }
        
        # Execute flow
        pubmed_result = search_pubmed_literature.invoke({"query": "diabetes"})
        assert pubmed_result["status"] == "success"
        
        synthesis_result = synthesize_information.invoke({
            "query": "diabetes",
            "context": {"pubmed_markdown": pubmed_result["data"]["article_markdown"]}
        })
        assert synthesis_result["status"] == "success"
        assert "response" in synthesis_result["data"]

    @patch('src.agents.supervisor_tools.pubmed_graph')
    @patch('src.agents.supervisor_tools.websearch_graph')
    @patch('src.agents.supervisor_tools.synthesis_graph')
    def test_multi_source_to_synthesis_flow(
        self,
        mock_synthesis,
        mock_websearch,
        mock_pubmed
    ):
        """Test flow with multiple sources."""
        # Mock PubMed
        mock_pubmed.invoke.return_value = {
            "article_markdown": "## PubMed Article",
            "error": None
        }
        
        # Mock WebSearch
        mock_websearch.invoke.return_value = {
            "web_search_markdown": "## Web Results",
            "error": None
        }
        
        # Mock Synthesis
        mock_synthesis.invoke.return_value = {
            "response": "Multi-source synthesis",
            "sources_used": ["pubmed", "websearch"],
            "error": None
        }
        
        # Execute flow
        pubmed_result = search_pubmed_literature.invoke({"query": "diabetes"})
        web_result = search_web_sources.invoke({"query": "diabetes"})
        
        synthesis_result = synthesize_information.invoke({
            "query": "diabetes",
            "context": {
                "pubmed_markdown": pubmed_result["data"]["article_markdown"],
                "websearch_markdown": web_result["data"]["web_search_markdown"]
            }
        })
        
        assert synthesis_result["status"] == "success"
        assert len(synthesis_result["data"]["sources_used"]) == 2
