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
from src.workflow import run_workflow, build_workflow
from src.models import SupervisorState

logger = logging.getLogger(__name__)

class TestBasicWorkflowExample:
    """Test basic workflow usage patterns."""

    @patch('src.agents.supervisor_agent.SupervisorAgent.execute')
    @patch('src.agents.synthesis_agent.SynthesisAgent.execute')
    def test_basic_workflow_usage(self, mock_synthesis, mock_supervisor):
        """Test basic workflow execution and result access."""
        # Setup mocks
        mock_supervisor.return_value = {
            'routing_decision': 'synthesis_agent',
            'agent_results': {},
            'iteration_count': 1,
            'current_agent': 'supervisor',
            'agent_messages': [],
            'synthesis_complete': False,
            'extracted_keywords': ['diabetes', 'symptoms'],
            'pmids': ['12345', '67890'],
            'article_markdown': '## Article\n**Title:** Test Article',
            'web_search_markdown': None,
            'response': ''
        }

        mock_synthesis.return_value = {
            'synthesis_complete': True,
            'response': 'Sample response about diabetes symptoms',
            'agent_messages': []
        }

        # Test basic usage
        result = run_workflow("What are the symptoms of diabetes?", enable_checkpointing=False)

        assert result['user_query'] == "What are the symptoms of diabetes?"
        assert 'response' in result or 'synthesis_complete' in result
        logger.info("Basic workflow example test passed")

    @patch('src.agents.supervisor_agent.SupervisorAgent.execute')
    @patch('src.agents.pubmed_agent.PubMedAgent.execute')
    @patch('src.agents.synthesis_agent.SynthesisAgent.execute')
    def test_intermediate_results_access(self, mock_synthesis, mock_pubmed, mock_supervisor):
        """Test accessing intermediate results at each step."""
        # Setup mocks
        mock_supervisor.return_value = {
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
        }

        mock_pubmed.return_value = {
            'extracted_keywords': ['heart', 'disease'],
            'pmids': ['11111', '22222'],
            'article_markdown': '# Heart Disease Article\nContent about heart disease',
            'agent_messages': [{'agent': 'pubmed', 'status': 'success'}]
        }

        mock_synthesis.return_value = {
            'synthesis_complete': True,
            'response': 'Heart disease information response',
            'agent_messages': []
        }

        # Run workflow
        result = run_workflow("What is heart disease?", enable_checkpointing=False)

        # Verify result structure
        assert result['user_query'] == "What is heart disease?"
        assert 'extracted_keywords' in result or 'pmids' in result or 'response' in result
        logger.info("Intermediate results test passed")

class TestCustomWorkflowBuilding:
    """Test custom workflow building patterns."""

    @patch('src.agents.supervisor_agent.SupervisorAgent.execute')
    @patch('src.agents.synthesis_agent.SynthesisAgent.execute')
    def test_custom_workflow_building(self, mock_synthesis, mock_supervisor):
        """Test building and running workflow manually."""
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
            'response': 'Information about hypertension',
            'agent_messages': []
        }

        # Build workflow manually
        graph = build_workflow(enable_checkpointing=False)
        assert graph is not None
        logger.info("Custom workflow building test passed")

    def test_workflow_state_initialization(self):
        """Test SupervisorState schema for custom workflows."""
        # Test that SupervisorState has required fields
        from src.models import SupervisorState
        assert 'user_query' in SupervisorState.__annotations__
        assert 'routing_decision' in SupervisorState.__annotations__
        assert 'messages' in SupervisorState.__annotations__
        assert 'response' in SupervisorState.__annotations__
