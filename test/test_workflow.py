"""
Tests for the LangGraph workflow.

Tests cover:
- Workflow initialization and compilation
- Individual node functionality
- Full workflow execution
"""

import pytest
import logging
from unittest.mock import patch, MagicMock
from src.workflow import (
    WorkflowState,
    receive_query,
    extract_keywords,
    search_pubmed_articles,
    extract_article_info,
    generate_response,
    build_workflow,
    run_workflow
)

logger = logging.getLogger(__name__)


class TestWorkflowState:
    """Test WorkflowState schema."""

    def test_workflow_state_initialization(self):
        """Test that WorkflowState can be initialized with correct fields."""
        state: WorkflowState = {
            "user_query": "Test query",
            "extracted_keywords": ["keyword1", "keyword2"],
            "pmids": ["12345", "67890"],
            "article_markdown": "# Article\nContent here",
            "response": "Generated response"
        }
        assert state["user_query"] == "Test query"
        assert len(state["extracted_keywords"]) == 2
        assert len(state["pmids"]) == 2


class TestReceiveQuery:
    """Test receive_query node."""

    def test_receive_query_stores_input(self):
        """Test that receive_query correctly stores user query in state."""
        query = "What are the treatments for malaria?"
        state: WorkflowState = {
            "user_query": query,
            "extracted_keywords": [],
            "pmids": [],
            "article_markdown": "",
            "response": ""
        }

        result = receive_query(state)
        assert result["user_query"] == query


class TestExtractKeywords:
    """Test extract_keywords node."""

    def test_extract_keywords_with_query(self):
        """Test keyword extraction from user query."""
        state: WorkflowState = {
            "user_query": "What are the latest treatments for COVID-19 pandemic?",
            "extracted_keywords": [],
            "pmids": [],
            "article_markdown": "",
            "response": ""
        }

        result = extract_keywords(state)
        assert len(result["extracted_keywords"]) > 0
        logger.info(f"Extracted keywords: {result['extracted_keywords']}")


class TestSearchPubmedArticles:
    """Test search_pubmed_articles node."""

    @patch('src.workflow.search_articles')
    def test_search_with_keywords(self, mock_search):
        """Test PubMed search with extracted keywords."""
        mock_search.return_value = ["12345", "67890", "11111"]

        state: WorkflowState = {
            "user_query": "COVID-19 treatment",
            "extracted_keywords": ["COVID-19", "treatment"],
            "pmids": [],
            "article_markdown": "",
            "response": ""
        }

        result = search_pubmed_articles(state)
        assert len(result["pmids"]) == 3
        assert result["pmids"] == ["12345", "67890", "11111"]
        mock_search.assert_called_once()

    def test_search_without_keywords(self):
        """Test search behavior when no keywords extracted."""
        state: WorkflowState = {
            "user_query": "Some query",
            "extracted_keywords": [],
            "pmids": [],
            "article_markdown": "",
            "response": ""
        }

        result = search_pubmed_articles(state)
        assert result["pmids"] == []
        assert "No keywords" in result["article_markdown"]


class TestExtractArticleInfo:
    """Test extract_article_info node."""

    @patch('src.workflow.extract_pubmed_articles_as_markdown')
    def test_extract_with_pmids(self, mock_extract):
        """Test article extraction with PMIDs."""
        mock_markdown = "# Article 1\nContent here\n---\n# Article 2\nMore content"
        mock_extract.return_value = mock_markdown

        state: WorkflowState = {
            "user_query": "test",
            "extracted_keywords": ["test"],
            "pmids": ["12345", "67890"],
            "article_markdown": "",
            "response": ""
        }

        result = extract_article_info(state)
        assert result["article_markdown"] == mock_markdown
        mock_extract.assert_called_once_with(["12345", "67890"])

    def test_extract_without_pmids(self):
        """Test extraction behavior when no PMIDs available."""
        state: WorkflowState = {
            "user_query": "test",
            "extracted_keywords": [],
            "pmids": [],
            "article_markdown": "",
            "response": ""
        }

        result = extract_article_info(state)
        assert "No articles found" in result["article_markdown"]


class TestGenerateResponse:
    """Test generate_response node."""

    @patch('src.workflow.ChatOpenAI')
    def test_generate_response_with_articles(self, mock_llm_class):
        """Test response generation with article content."""
        mock_llm = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Generated response based on articles"
        mock_llm.invoke.return_value = mock_message
        mock_llm_class.return_value = mock_llm

        state: WorkflowState = {
            "user_query": "What is COVID-19?",
            "extracted_keywords": ["COVID-19"],
            "pmids": ["12345"],
            "article_markdown": "# COVID-19 Article\nContent about COVID-19",
            "response": ""
        }

        result = generate_response(state)
        assert len(result["response"]) > 0
        mock_llm.invoke.assert_called_once()

    @patch('src.workflow.ChatOpenAI')
    def test_generate_response_without_articles(self, mock_llm_class):
        """Test response generation without article content."""
        mock_llm = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Response without articles"
        mock_llm.invoke.return_value = mock_message
        mock_llm_class.return_value = mock_llm

        state: WorkflowState = {
            "user_query": "What is COVID-19?",
            "extracted_keywords": ["COVID-19"],
            "pmids": [],
            "article_markdown": "No articles found.",
            "response": ""
        }

        result = generate_response(state)
        assert len(result["response"]) > 0


class TestWorkflowBuilder:
    """Test workflow graph building."""

    def test_build_workflow(self):
        """Test that workflow graph builds successfully."""
        graph = build_workflow()
        assert graph is not None
        logger.info("Workflow graph built successfully")


class TestFullWorkflow:
    """Test full workflow execution."""

    @patch('src.workflow.search_articles')
    @patch('src.workflow.extract_pubmed_articles_as_markdown')
    @patch('src.workflow.ChatOpenAI')
    def test_run_workflow_integration(self, mock_llm_class, mock_extract, mock_search):
        """Test full workflow execution with mocked components."""
        # Setup mocks
        mock_search.return_value = ["12345", "67890"]
        mock_extract.return_value = "# Article 1\nContent here"

        mock_llm = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Comprehensive response"
        mock_llm.invoke.return_value = mock_message
        mock_llm_class.return_value = mock_llm

        # Run workflow
        result = run_workflow("What is malaria treatment?")

        # Assertions
        assert result["user_query"] == "What is malaria treatment?"
        assert len(result["extracted_keywords"]) > 0
        assert len(result["pmids"]) > 0
        assert len(result["article_markdown"]) > 0
        assert result["response"] == "Comprehensive response"

        # Verify mocks were called
        mock_search.assert_called_once()
        mock_extract.assert_called_once()
        mock_llm.invoke.assert_called_once()

        logger.info("Full workflow integration test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
