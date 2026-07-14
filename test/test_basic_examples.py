"""
Tests converted from examples.py - Basic workflow usage patterns.

Tests cover:
- Basic workflow usage
- Intermediate results access
- Custom workflow building
"""

import pytest
import logging
from unittest.mock import patch, MagicMock
from src.workflow import run_workflow, build_workflow, WorkflowState

logger = logging.getLogger(__name__)


class TestBasicWorkflowExample:
    """Test basic workflow usage patterns."""

    @patch('src.workflow.search_articles')
    @patch('src.workflow.extract_pubmed_articles_as_markdown')
    @patch('src.workflow.ChatOpenAI')
    def test_basic_workflow_usage(self, mock_llm_class, mock_extract, mock_search):
        """Test basic workflow execution and result access."""
        # Setup mocks
        mock_search.return_value = ["12345", "67890"]
        mock_extract.return_value = "## Article\n**Title:** Test Article"

        mock_llm = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Sample response about diabetes symptoms"
        mock_llm.invoke.return_value = mock_message
        mock_llm_class.return_value = mock_llm

        # Test basic usage
        result = run_workflow("What are the symptoms of diabetes?")

        assert result['user_query'] == "What are the symptoms of diabetes?"
        assert len(result['extracted_keywords']) > 0
        assert len(result['pmids']) > 0
        assert result['response'] == "Sample response about diabetes symptoms"
        logger.info("Basic workflow example test passed")

    @patch('src.workflow.search_articles')
    @patch('src.workflow.extract_pubmed_articles_as_markdown')
    @patch('src.workflow.ChatOpenAI')
    def test_intermediate_results_access(self, mock_llm_class, mock_extract, mock_search):
        """Test accessing intermediate results at each step."""
        # Setup mocks
        mock_search.return_value = ["11111", "22222"]
        mock_extract.return_value = "# Heart Disease Article\nContent about heart disease"

        mock_llm = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Heart disease information response"
        mock_llm.invoke.return_value = mock_message
        mock_llm_class.return_value = mock_llm

        # Run workflow
        result = run_workflow("What is heart disease?")

        # Verify intermediate results
        assert result['user_query'] == "What is heart disease?"
        assert isinstance(result['extracted_keywords'], list)
        assert len(result['extracted_keywords']) > 0
        assert result['pmids'] == ["11111", "22222"]
        assert len(result['article_markdown']) > 0
        assert len(result['response']) > 0
        logger.info("Intermediate results test passed")


class TestCustomWorkflowBuilding:
    """Test custom workflow building patterns."""

    @patch('src.workflow.search_articles')
    @patch('src.workflow.extract_pubmed_articles_as_markdown')
    @patch('src.workflow.ChatOpenAI')
    def test_custom_workflow_building(self, mock_llm_class, mock_extract, mock_search):
        """Test building and running workflow manually."""
        # Setup mocks
        mock_search.return_value = ["33333", "44444"]
        mock_extract.return_value = "## Hypertension Article"

        mock_llm = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Information about hypertension"
        mock_llm.invoke.return_value = mock_message
        mock_llm_class.return_value = mock_llm

        # Build workflow manually
        graph = build_workflow()
        assert graph is not None

        # Initialize state
        state: WorkflowState = {
            "user_query": "What is hypertension?",
            "extracted_keywords": [],
            "pmids": [],
            "article_markdown": "",
            "response": ""
        }

        # Run workflow
        result = graph.invoke(state)

        # Verify results
        assert result['user_query'] == "What is hypertension?"
        assert len(result['extracted_keywords']) > 0
        assert len(result['pmids']) > 0
        assert len(result['article_markdown']) > 0
        assert len(result['response']) > 0
        logger.info("Custom workflow building test passed")

    def test_workflow_state_initialization(self):
        """Test WorkflowState initialization for custom workflows."""
        state: WorkflowState = {
            "user_query": "Test query",
            "extracted_keywords": [],
            "pmids": [],
            "article_markdown": "",
            "response": ""
        }

        assert isinstance(state, dict)
        assert "user_query" in state
        assert "extracted_keywords" in state
        assert "pmids" in state
        assert "article_markdown" in state
        assert "response" in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
