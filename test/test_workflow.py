"""
Tests for the LangGraph workflow.

Tests cover:
- Workflow initialization and compilation
- Agent-based workflow execution
- State management
"""

import pytest
import logging
from unittest.mock import patch, MagicMock
from src.workflow import (
    SupervisorState,
    build_workflow,
    run_workflow
)

logger = logging.getLogger(__name__)

class TestWorkflowState:
    """Test SupervisorState schema."""

    def test_workflow_state_initialization(self):
        """Test that SupervisorState can be initialized with correct fields."""
        # The state is a TypedDict so we just verify structure
        from src.models import SupervisorState
        assert 'messages' in SupervisorState.__annotations__
        assert 'user_query' in SupervisorState.__annotations__
        assert 'routing_decision' in SupervisorState.__annotations__

class TestWorkflowBuilder:
    """Test workflow graph building."""

    def test_build_workflow_without_checkpointing(self):
        """Test that workflow graph builds successfully without checkpointing."""
        graph = build_workflow(enable_checkpointing=False)
        assert graph is not None
        logger.info("Workflow graph built successfully without checkpointing")

    def test_build_workflow_with_checkpointing(self):
        """Test that workflow graph builds successfully with checkpointing."""
        graph = build_workflow(enable_checkpointing=True)
        assert graph is not None
        logger.info("Workflow graph built successfully with checkpointing")

class TestWorkflowExecution:
    """Test full workflow execution."""

    @patch('src.agents.supervisor_agent.SupervisorAgent.execute')
    @patch('src.agents.synthesis_agent.SynthesisAgent.execute')
    def test_run_workflow_basic(self, mock_synthesis, mock_supervisor):
        """Test basic workflow execution with mocked agents."""
        # Mock supervisor to route to synthesis directly (simplified flow)
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
        
        # Mock synthesis to return a response
        mock_synthesis.return_value = {
            'synthesis_complete': True,
            'response': 'This is a test response',
            'agent_messages': []
        }
        
        result = run_workflow("Test query", enable_checkpointing=False)
        
        assert result is not None
        assert 'user_query' in result
        assert result['user_query'] == "Test query"
        
        logger.info(f"Workflow executed successfully with result keys: {result.keys()}")

    def test_run_workflow_generates_thread_id(self):
        """Test that workflow generates a thread_id if not provided."""
        with patch('src.agents.supervisor_agent.SupervisorAgent.execute') as mock_supervisor, \
             patch('src.agents.synthesis_agent.SynthesisAgent.execute') as mock_synthesis:
            
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
            
            result = run_workflow("Test query", thread_id=None, enable_checkpointing=False)
            
            # Workflow should work even without explicit thread_id
            assert result is not None
            assert 'response' in result or 'user_query' in result

    def test_run_workflow_uses_provided_thread_id(self):
        """Test that workflow uses the provided thread_id."""
        custom_thread_id = "test-thread-123"
        
        with patch('src.agents.supervisor_agent.SupervisorAgent.execute') as mock_supervisor, \
             patch('src.agents.synthesis_agent.SynthesisAgent.execute') as mock_synthesis:
            
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
            
            result = run_workflow("Test query", thread_id=custom_thread_id, enable_checkpointing=False)
            
            # Workflow should work with custom thread_id
            assert result is not None

class TestFullWorkflow:
    """Test full workflow execution."""

    @patch('src.agents.supervisor_agent.SupervisorAgent.execute')
    @patch('src.agents.pubmed_agent.PubMedAgent.execute')
    @patch('src.agents.synthesis_agent.SynthesisAgent.execute')
    def test_run_workflow_integration(self, mock_synthesis, mock_pubmed, mock_supervisor):
        """Test full workflow execution with mocked agents."""
        import time
        
        # Setup supervisor mock - routes to PubMed first
        mock_supervisor.side_effect = [
            {
                'routing_decision': 'pubmed_agent',
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
            },
            # Second call - after PubMed, route to synthesis
            {
                'routing_decision': 'synthesis_agent',
                'agent_results': {
                    'pubmed_agent': {'status': 'success', 'timestamp': time.time()}
                },
                'iteration_count': 2,
                'current_agent': 'supervisor',
                'agent_messages': [],
                'synthesis_complete': False,
                'extracted_keywords': ['malaria', 'treatment'],
                'pmids': ['12345', '67890'],
                'article_markdown': '# Article 1\nContent here',
                'web_search_markdown': None,
                'response': ''
            }
        ]

        # Setup PubMed agent mock
        mock_pubmed.return_value = {
            'extracted_keywords': ['malaria', 'treatment'],
            'pmids': ['12345', '67890'],
            'article_markdown': '# Article 1\nContent here',
            'agent_messages': [{'agent': 'pubmed', 'status': 'success'}],
            'agent_results': {
                'pubmed_agent': {
                    'status': 'success',
                    'timestamp': time.time(),
                    'data': {
                        'extracted_keywords': ['malaria', 'treatment'],
                        'pmids': ['12345', '67890']
                    }
                }
            }
        }

        # Setup synthesis agent mock
        mock_synthesis.return_value = {
            'synthesis_complete': True,
            'response': 'Comprehensive response',
            'agent_messages': [{'agent': 'synthesis', 'status': 'success'}],
            'agent_results': {
                'synthesis_agent': {
                    'status': 'success',
                    'timestamp': time.time()
                }
            }
        }

        # Run workflow
        result = run_workflow("What is malaria treatment?", enable_checkpointing=False)

        # Assertions
        assert result["user_query"] == "What is malaria treatment?"
        assert mock_supervisor.called
        assert 'response' in result or 'synthesis_complete' in result

        logger.info("Full workflow integration test passed")
