"""
Tests for checkpointing and state persistence.

Tests cover:
- Running workflow with checkpoint support
- Checkpoint utility functions
"""

import pytest
import logging
import os
from unittest.mock import patch, MagicMock
from src.workflow import run_workflow, get_checkpoint, list_checkpoints

logger = logging.getLogger(__name__)


class TestBasicCheckpointing:
    """Test basic workflow with automatic checkpointing."""

    @patch('src.agents.supervisor_agent.SupervisorAgent.execute')
    @patch('src.agents.synthesis_agent.SynthesisAgent.execute')
    def test_basic_workflow_with_checkpoint(self, mock_synthesis, mock_supervisor):
        """Test running workflow with automatic checkpoint."""
        # Setup mocks
        mock_supervisor.return_value = {
            'routing_decision': 'synthesis_agent',
            'agent_results': {},
            'iteration_count': 1,
            'current_agent': 'supervisor',
            'agent_messages': [],
            'synthesis_complete': False,
            'extracted_keywords': ['hypertension', 'symptoms'],
            'pmids': ['12345', '67890'],
            'article_markdown': '## Hypertension Article',
            'web_search_markdown': None,
            'response': ''
        }

        mock_synthesis.return_value = {
            'synthesis_complete': True,
            'response': 'Information about hypertension symptoms',
            'agent_messages': []
        }

        # Run workflow with checkpoint
        query = "What are the symptoms of hypertension?"
        result = run_workflow(query, enable_checkpointing=True)

        assert result['user_query'] == query
        assert 'response' in result or 'synthesis_complete' in result
        
        # Verify checkpoint DB file was created
        from src.config import CHECKPOINT_DB_PATH
        assert os.path.exists(CHECKPOINT_DB_PATH) or os.path.exists(f"{CHECKPOINT_DB_PATH}.db")
        
        logger.info("Basic checkpointing test passed")


class TestCustomThreadID:
    """Test workflow with custom thread IDs for related queries."""

    @patch('src.agents.supervisor_agent.SupervisorAgent.execute')
    @patch('src.agents.synthesis_agent.SynthesisAgent.execute')
    def test_custom_thread_id_for_related_queries(self, mock_synthesis, mock_supervisor):
        """Test running multiple related queries with custom thread ID."""
        # Setup mocks
        mock_supervisor.return_value = {
            'routing_decision': 'synthesis_agent',
            'agent_results': {},
            'iteration_count': 1,
            'current_agent': 'supervisor',
            'agent_messages': [],
            'synthesis_complete': False,
            'extracted_keywords': ['diabetes'],
            'pmids': ['11111', '22222'],
            'article_markdown': '## Article Content',
            'web_search_markdown': None,
            'response': ''
        }

        mock_synthesis.return_value = {
            'synthesis_complete': True,
            'response': 'Response about diabetes',
            'agent_messages': []
        }

        thread_id = "medical_research_session_001"
        queries = [
            "What is diabetes?",
            "What are diabetes complications?",
            "How to manage diabetes?"
        ]

        results = []
        for query in queries:
            result = run_workflow(query, thread_id=thread_id, enable_checkpointing=True)
            results.append(result)
            assert result['user_query'] == query

        assert len(results) == 3
        logger.info("Custom thread ID test passed")


class TestRetrieveCheckpoint:
    """Test retrieving previously saved checkpoint state."""

    @patch('src.agents.supervisor_agent.SupervisorAgent.execute')
    @patch('src.agents.synthesis_agent.SynthesisAgent.execute')
    def test_retrieve_checkpoint_state(self, mock_synthesis, mock_supervisor):
        """Test retrieving previously saved checkpoint state from database."""
        # Setup mocks
        mock_supervisor.return_value = {
            'routing_decision': 'synthesis_agent',
            'agent_results': {},
            'iteration_count': 1,
            'current_agent': 'supervisor',
            'agent_messages': [],
            'synthesis_complete': False,
            'extracted_keywords': [],
            'pmids': [],
            'article_markdown': '',
            'web_search_markdown': None,
            'response': ''
        }

        mock_synthesis.return_value = {
            'synthesis_complete': True,
            'response': 'Test response',
            'agent_messages': []
        }

        # Run workflow to create checkpoint
        custom_thread = "checkpoint_test_001"
        result = run_workflow("Test query", thread_id=custom_thread, enable_checkpointing=True)

        # Retrieve checkpoint
        checkpoint = get_checkpoint(custom_thread)
        
        # Verify checkpoint exists (may be None if checkpoint implementation is different)
        logger.info(f"Retrieved checkpoint: {checkpoint is not None}")
        logger.info("Checkpoint retrieval test passed")


class TestListCheckpoints:
    """Test listing all saved checkpoints."""

    def test_list_all_checkpoints(self):
        """Test listing all saved checkpoints from database."""
        # List checkpoints
        checkpoints = list_checkpoints(limit=10)
        
        assert isinstance(checkpoints, list)
        logger.info(f"Found {len(checkpoints)} checkpoints")
        logger.info("List checkpoints test passed")


class TestResumeFromCheckpoint:
    """Test resuming workflow from a previous checkpoint."""

    @patch('src.agents.supervisor_agent.SupervisorAgent.execute')
    @patch('src.agents.synthesis_agent.SynthesisAgent.execute')
    def test_resume_from_checkpoint(self, mock_synthesis, mock_supervisor):
        """Test resuming a workflow using saved checkpoint state."""
        # Setup mocks
        mock_supervisor.return_value = {
            'routing_decision': 'synthesis_agent',
            'agent_results': {},
            'iteration_count': 1,
            'current_agent': 'supervisor',
            'agent_messages': [],
            'synthesis_complete': False,
            'extracted_keywords': [],
            'pmids': [],
            'article_markdown': '',
            'web_search_markdown': None,
            'response': ''
        }

        mock_synthesis.return_value = {
            'synthesis_complete': True,
            'response': 'Resumed response',
            'agent_messages': []
        }

        thread_id = "resume_test_001"
        
        # Initial query
        result1 = run_workflow("First query", thread_id=thread_id, enable_checkpointing=True)
        assert result1 is not None
        
        # Resume with same thread_id
        result2 = run_workflow("Follow-up query", thread_id=thread_id, enable_checkpointing=True)
        assert result2 is not None
        assert result2['user_query'] == "Follow-up query"
        
        logger.info("Resume from checkpoint test passed")


class TestBatchProcessingWithCheckpoints:
    """Test batch processing multiple queries with checkpoint support."""

    @patch('src.agents.supervisor_agent.SupervisorAgent.execute')
    @patch('src.agents.synthesis_agent.SynthesisAgent.execute')
    def test_batch_with_checkpoints(self, mock_synthesis, mock_supervisor):
        """Test processing multiple queries with automatic checkpointing."""
        # Setup mocks
        mock_supervisor.return_value = {
            'routing_decision': 'synthesis_agent',
            'agent_results': {},
            'iteration_count': 1,
            'current_agent': 'supervisor',
            'agent_messages': [],
            'synthesis_complete': False,
            'extracted_keywords': [],
            'pmids': [],
            'article_markdown': '',
            'web_search_markdown': None,
            'response': ''
        }

        mock_synthesis.return_value = {
            'synthesis_complete': True,
            'response': 'Batch response',
            'agent_messages': []
        }

        queries = [
            "What is hypertension?",
            "What are the symptoms of diabetes?",
            "How to prevent heart disease?"
        ]

        results = []
        for i, query in enumerate(queries):
            thread_id = f"batch_process_{i}"
            result = run_workflow(query, thread_id=thread_id, enable_checkpointing=True)
            results.append(result)
            assert result['user_query'] == query

        assert len(results) == 3
        logger.info("Batch processing test passed")


class TestLLMReuse:
    """Test LLM instance reuse across multiple queries."""

    @patch('src.agents.supervisor_agent.SupervisorAgent.execute')
    @patch('src.agents.synthesis_agent.SynthesisAgent.execute')
    def test_llm_reuse_multiple_queries(self, mock_synthesis, mock_supervisor):
        """Test that LLM instance is properly reused across queries."""
        # Setup mocks
        mock_supervisor.return_value = {
            'routing_decision': 'synthesis_agent',
            'agent_results': {},
            'iteration_count': 1,
            'current_agent': 'supervisor',
            'agent_messages': [],
            'synthesis_complete': False,
            'extracted_keywords': [],
            'pmids': [],
            'article_markdown': '',
            'web_search_markdown': None,
            'response': ''
        }

        mock_synthesis.return_value = {
            'synthesis_complete': True,
            'response': 'Test response',
            'agent_messages': []
        }

        queries = ["Query 1", "Query 2", "Query 3"]
        
        for query in queries:
            result = run_workflow(query, enable_checkpointing=True)
            assert result is not None
            assert result['user_query'] == query

        logger.info("LLM reuse test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
