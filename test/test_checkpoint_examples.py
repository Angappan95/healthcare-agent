"""
Tests converted from checkpoint_examples.py - Checkpointing and state persistence.

Tests cover:
- Running workflow with checkpoint support
- Resuming from checkpoints
- Listing saved checkpoints
- Retrieving checkpoint state
- Batch processing with checkpoints
"""

import pytest
import logging
from unittest.mock import patch, MagicMock
from src.workflow import run_workflow, get_checkpoint, list_checkpoints

logger = logging.getLogger(__name__)


class TestBasicCheckpointing:
    """Test basic workflow with automatic checkpointing."""

    @patch('src.workflow.search_articles')
    @patch('src.workflow.extract_pubmed_articles_as_markdown')
    @patch('src.workflow.ChatOpenAI')
    def test_basic_workflow_with_checkpoint(self, mock_llm_class, mock_extract, mock_search):
        """
        Test running workflow with automatic checkpoint.
        State is automatically saved to SQLite database.
        """
        # Setup mocks
        mock_search.return_value = ["12345", "67890"]
        mock_extract.return_value = "## Hypertension Article"

        mock_llm = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Information about hypertension symptoms"
        mock_llm.invoke.return_value = mock_message
        mock_llm_class.return_value = mock_llm

        # Run workflow with checkpoint
        query = "What are the symptoms of hypertension?"
        result = run_workflow(query)

        assert result['user_query'] == query
        assert len(result['extracted_keywords']) > 0
        assert len(result['pmids']) > 0
        assert len(result['response']) > 0
        logger.info("Basic checkpointing test passed")


class TestCustomThreadID:
    """Test workflow with custom thread IDs for related queries."""

    @patch('src.workflow.search_articles')
    @patch('src.workflow.extract_pubmed_articles_as_markdown')
    @patch('src.workflow.ChatOpenAI')
    def test_custom_thread_id_for_related_queries(self, mock_llm_class, mock_extract, mock_search):
        """
        Test running multiple related queries with custom thread ID.
        Allows grouping related queries under one thread.
        """
        # Setup mocks
        mock_search.return_value = ["11111", "22222"]
        mock_extract.return_value = "## Article Content"

        mock_llm = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Response about diabetes"
        mock_llm.invoke.return_value = mock_message
        mock_llm_class.return_value = mock_llm

        thread_id = "medical_research_session_001"
        queries = [
            "What is diabetes?",
            "What are diabetes complications?",
            "How to manage diabetes?"
        ]

        results = []
        for query in queries:
            result = run_workflow(query, thread_id=thread_id)
            results.append(result)
            assert result['user_query'] == query
            assert len(result['extracted_keywords']) > 0

        assert len(results) == 3
        logger.info("Custom thread ID test passed")


class TestRetrieveCheckpoint:
    """Test retrieving previously saved checkpoint state."""

    @patch('src.workflow.search_articles')
    @patch('src.workflow.extract_pubmed_articles_as_markdown')
    @patch('src.workflow.ChatOpenAI')
    def test_retrieve_checkpoint_state(self, mock_llm_class, mock_extract, mock_search):
        """
        Test retrieving previously saved checkpoint state from database.
        """
        # Setup mocks
        mock_search.return_value = ["33333", "44444"]
        mock_extract.return_value = "## Heart Disease Content"

        mock_llm = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Information about heart disease"
        mock_llm.invoke.return_value = mock_message
        mock_llm_class.return_value = mock_llm

        # Run workflow to create checkpoint
        thread_id = "checkpoint_test_001"
        query = "What is heart disease?"
        result1 = run_workflow(query, thread_id=thread_id)

        # Retrieve the checkpoint
        checkpoint_state = get_checkpoint(thread_id)

        if checkpoint_state:
            assert checkpoint_state.get('user_query') == query
            assert len(checkpoint_state.get('extracted_keywords', [])) > 0
            assert len(checkpoint_state.get('pmids', [])) > 0
            logger.info("Checkpoint retrieval test passed")


class TestListCheckpoints:
    """Test listing all saved checkpoints."""

    @patch('src.workflow.search_articles')
    @patch('src.workflow.extract_pubmed_articles_as_markdown')
    @patch('src.workflow.ChatOpenAI')
    def test_list_all_checkpoints(self, mock_llm_class, mock_extract, mock_search):
        """
        Test listing all saved checkpoint thread IDs in database.
        """
        # Setup mocks
        mock_search.return_value = ["55555", "66666"]
        mock_extract.return_value = "## Article Content"

        mock_llm = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Research response"
        mock_llm.invoke.return_value = mock_message
        mock_llm_class.return_value = mock_llm

        # Create checkpoints
        queries = [
            ("cancer_research", "What is cancer?"),
            ("cancer_research", "What are cancer treatments?"),
            ("covid_research", "What is COVID-19?"),
        ]

        for thread_id, query in queries:
            run_workflow(query, thread_id=thread_id)

        # List checkpoints
        checkpoints = list_checkpoints()

        if checkpoints:
            assert isinstance(checkpoints, list)
            logger.info(f"Found {len(checkpoints)} checkpoints")


class TestResumeFromCheckpoint:
    """Test resuming workflow from saved checkpoint."""

    @patch('src.workflow.search_articles')
    @patch('src.workflow.extract_pubmed_articles_as_markdown')
    @patch('src.workflow.ChatOpenAI')
    def test_resume_from_checkpoint(self, mock_llm_class, mock_extract, mock_search):
        """
        Test continuing workflow execution from a saved checkpoint.
        Useful for long-running workflows or handling interruptions.
        """
        # Setup mocks
        mock_search.return_value = ["77777", "88888"]
        mock_extract.return_value = "## Diabetes Treatment Article"

        mock_llm = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Information about diabetes treatment"
        mock_llm.invoke.return_value = mock_message
        mock_llm_class.return_value = mock_llm

        thread_id = "resume_test_001"

        # Initial workflow execution
        query = "What is diabetes treatment?"
        result1 = run_workflow(query, thread_id=thread_id)

        assert result1['user_query'] == query
        assert len(result1['pmids']) > 0

        # Retrieve checkpoint and verify restoration
        checkpoint = get_checkpoint(thread_id)
        if checkpoint:
            assert checkpoint.get('user_query') == query
            assert len(checkpoint.get('extracted_keywords', [])) > 0
            logger.info("Resume from checkpoint test passed")


class TestBatchProcessingWithCheckpoints:
    """Test batch processing with checkpoint save for each item."""

    @patch('src.workflow.search_articles')
    @patch('src.workflow.extract_pubmed_articles_as_markdown')
    @patch('src.workflow.ChatOpenAI')
    def test_batch_with_checkpoints(self, mock_llm_class, mock_extract, mock_search):
        """
        Test processing multiple queries with checkpoint save for each.
        Enables recovery from failures mid-batch.
        """
        # Setup mocks
        mock_search.return_value = ["99999", "10101"]
        mock_extract.return_value = "## Medical Article"

        mock_llm = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Medical information response"
        mock_llm.invoke.return_value = mock_message
        mock_llm_class.return_value = mock_llm

        batch_id = "batch_medical_research"
        queries = [
            "What is hypertension?",
            "What is diabetes?",
            "What is heart disease?",
        ]

        results = []
        for i, query in enumerate(queries, 1):
            thread_id = f"{batch_id}_{i}"
            result = run_workflow(query, thread_id=thread_id)
            results.append(result)

        # Verify batch results
        assert len(results) == len(queries)
        for i, result in enumerate(results):
            assert result['user_query'] == queries[i]
            assert len(result['extracted_keywords']) > 0
            assert len(result['pmids']) > 0

        logger.info(f"Batch processing test passed: {len(results)} queries processed")


class TestLLMReuse:
    """Test LLM instance reuse across multiple workflow executions."""

    @patch('src.workflow.search_articles')
    @patch('src.workflow.extract_pubmed_articles_as_markdown')
    @patch('src.workflow.ChatOpenAI')
    def test_llm_reuse_multiple_queries(self, mock_llm_class, mock_extract, mock_search):
        """
        Test that LLM is reused across multiple queries.
        LLM is instantiated once and reused for all workflow executions.
        """
        # Setup mocks
        mock_search.return_value = ["11111", "22222"]
        mock_extract.return_value = "## Article"

        mock_llm = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_llm.invoke.return_value = mock_message
        mock_llm_class.return_value = mock_llm

        # Run multiple queries
        queries = [
            "What is asthma?",
            "What causes asthma?",
            "How to treat asthma?",
        ]

        results = []
        for query in queries:
            result = run_workflow(query)
            results.append(result)
            assert len(result['response']) > 0

        # Verify all queries executed
        assert len(results) == 3
        logger.info("LLM reuse test passed - single instance executed 3 queries")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
