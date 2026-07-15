"""
Test suite for multi-agent supervisor workflow.

Tests routing logic, agent execution, error handling, and checkpointing.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.workflow import (
    SupervisorState,
    supervisor_node,
    pubmed_agent_node,
    websearch_agent_node,
    synthesis_agent_node,
    route_from_supervisor,
    route_after_agents,
    run_workflow
)
from src.agents.supervisor_agent import SupervisorAgent
from src.agents.pubmed_agent import PubMedAgent
from src.agents.websearch_agent import WebSearchAgent
from src.agents.synthesis_agent import SynthesisAgent

class TestSupervisorRouting:
    """Test supervisor routing logic."""
    
    def test_route_to_pubmed_agent(self):
        """Test routing decision to PubMed agent."""
        state: SupervisorState = {
            'routing_decision': 'pubmed_agent',
            'user_query': 'Test query',
            'extracted_keywords': [],
            'pmids': [],
            'article_markdown': '',
            'response': '',
            'current_agent': 'supervisor',
            'agent_messages': [],
            'agent_results': {},
            'iteration_count': 1,
            'conversation_history': [],
            'pubmed_results': None,
            'websearch_results': None,
            'web_search_results': None,
            'web_search_markdown': None,
            'synthesis_complete': False
        }
        
        result = route_from_supervisor(state)
        assert result == 'pubmed_agent'
    
    def test_route_to_websearch_agent(self):
        """Test routing decision to WebSearch agent."""
        state: SupervisorState = {
            'routing_decision': 'websearch_agent',
            'user_query': 'What are symptoms of flu?',
            'extracted_keywords': [],
            'pmids': [],
            'article_markdown': '',
            'response': '',
            'current_agent': 'supervisor',
            'agent_messages': [],
            'agent_results': {},
            'iteration_count': 1,
            'conversation_history': [],
            'pubmed_results': None,
            'websearch_results': None,
            'web_search_results': None,
            'web_search_markdown': None,
            'synthesis_complete': False
        }
        
        result = route_from_supervisor(state)
        assert result == 'websearch_agent'
    
    def test_route_to_both_agents(self):
        """Test routing decision for both agents."""
        state: SupervisorState = {
            'routing_decision': 'both_agents',
            'user_query': 'Diabetes treatments and research',
            'extracted_keywords': [],
            'pmids': [],
            'article_markdown': '',
            'response': '',
            'current_agent': 'supervisor',
            'agent_messages': [],
            'agent_results': {},
            'iteration_count': 1,
            'conversation_history': [],
            'pubmed_results': None,
            'websearch_results': None,
            'web_search_results': None,
            'web_search_markdown': None,
            'synthesis_complete': False
        }
        
        result = route_from_supervisor(state)
        # When both_agents, route to PubMed first
        assert result == 'pubmed_agent'
    
    def test_route_after_pubmed_to_websearch(self):
        """Test routing after PubMed when both agents requested."""
        state: SupervisorState = {
            'routing_decision': 'both_agents',
            'user_query': 'Test',
            'extracted_keywords': ['diabetes'],
            'pmids': ['12345'],
            'article_markdown': 'Article content',
            'response': '',
            'current_agent': 'pubmed_agent',
            'agent_messages': [],
            'agent_results': {
                'pubmed_agent': {'status': 'success'}
            },
            'iteration_count': 1,
            'conversation_history': [],
            'pubmed_results': {'keywords': ['diabetes']},
            'websearch_results': None,
            'web_search_results': None,
            'web_search_markdown': None,
            'synthesis_complete': False
        }
        
        result = route_after_agents(state)
        # Should route to WebSearch next
        assert result == 'websearch_agent'
    
    def test_route_to_synthesis_after_agents(self):
        """Test routing to synthesis when both agents done."""
        state: SupervisorState = {
            'routing_decision': 'both_agents',
            'user_query': 'Test',
            'extracted_keywords': ['diabetes'],
            'pmids': ['12345'],
            'article_markdown': 'Article content',
            'response': '',
            'current_agent': 'websearch_agent',
            'agent_messages': [],
            'agent_results': {
                'pubmed_agent': {'status': 'success'},
                'websearch_agent': {'status': 'success'}
            },
            'iteration_count': 2,
            'conversation_history': [],
            'pubmed_results': {'keywords': ['diabetes']},
            'websearch_results': {'result_count': 3},
            'web_search_results': [],
            'web_search_markdown': 'Web content',
            'synthesis_complete': False
        }
        
        result = route_after_agents(state)
        # Should route to synthesis
        assert result == 'synthesis_agent'

class TestAgentNodes:
    """Test agent node execution with mocks."""
    
    @patch('src.workflow.get_supervisor_agent')
    def test_supervisor_node_execution(self, mock_get_agent):
        """Test supervisor node executes successfully."""
        mock_agent = Mock(spec=SupervisorAgent)
        mock_agent.execute_with_retry.return_value = {
            'routing_decision': 'pubmed_agent',
            'iteration_count': 1
        }
        mock_get_agent.return_value = mock_agent
        
        state: SupervisorState = {
            'user_query': 'Test query',
            'routing_decision': '',
            'iteration_count': 0,
            'extracted_keywords': [],
            'pmids': [],
            'article_markdown': '',
            'response': '',
            'current_agent': '',
            'agent_messages': [],
            'agent_results': {},
            'conversation_history': [],
            'pubmed_results': None,
            'websearch_results': None,
            'web_search_results': None,
            'web_search_markdown': None,
            'synthesis_complete': False
        }
        
        result = supervisor_node(state)
        
        assert mock_agent.execute_with_retry.called
        assert result['routing_decision'] == 'pubmed_agent'
    
    @patch('src.workflow.get_pubmed_agent')
    def test_pubmed_node_execution(self, mock_get_agent):
        """Test PubMed agent node executes successfully."""
        mock_agent = Mock(spec=PubMedAgent)
        mock_agent.execute_with_retry.return_value = {
            'extracted_keywords': ['diabetes'],
            'pmids': ['12345'],
            'article_markdown': 'Article content'
        }
        mock_get_agent.return_value = mock_agent
        
        state: SupervisorState = {
            'user_query': 'What is diabetes?',
            'routing_decision': 'pubmed_agent',
            'iteration_count': 1,
            'extracted_keywords': [],
            'pmids': [],
            'article_markdown': '',
            'response': '',
            'current_agent': 'supervisor',
            'agent_messages': [],
            'agent_results': {},
            'conversation_history': [],
            'pubmed_results': None,
            'websearch_results': None,
            'web_search_results': None,
            'web_search_markdown': None,
            'synthesis_complete': False
        }
        
        result = pubmed_agent_node(state)
        
        assert mock_agent.execute_with_retry.called
        assert result['extracted_keywords'] == ['diabetes']
        assert result['pmids'] == ['12345']
    
    @patch('src.workflow.get_websearch_agent')
    def test_websearch_node_execution(self, mock_get_agent):
        """Test WebSearch agent node executes successfully."""
        mock_agent = Mock(spec=WebSearchAgent)
        mock_agent.execute_with_retry.return_value = {
            'web_search_results': [{'title': 'Test', 'url': 'http://test.com'}],
            'web_search_markdown': 'Web content'
        }
        mock_get_agent.return_value = mock_agent
        
        state: SupervisorState = {
            'user_query': 'What are flu symptoms?',
            'routing_decision': 'websearch_agent',
            'iteration_count': 1,
            'extracted_keywords': [],
            'pmids': [],
            'article_markdown': '',
            'response': '',
            'current_agent': 'supervisor',
            'agent_messages': [],
            'agent_results': {},
            'conversation_history': [],
            'pubmed_results': None,
            'websearch_results': None,
            'web_search_results': None,
            'web_search_markdown': None,
            'synthesis_complete': False
        }
        
        result = websearch_agent_node(state)
        
        assert mock_agent.execute_with_retry.called
        assert 'web_search_results' in result
    
    @patch('src.workflow.get_synthesis_agent')
    def test_synthesis_node_execution(self, mock_get_agent):
        """Test Synthesis agent node executes successfully."""
        mock_agent = Mock(spec=SynthesisAgent)
        mock_agent.execute_with_retry.return_value = {
            'response': 'Synthesized response with citations',
            'synthesis_complete': True
        }
        mock_get_agent.return_value = mock_agent
        
        state: SupervisorState = {
            'user_query': 'Test query',
            'routing_decision': 'synthesis_agent',
            'iteration_count': 2,
            'extracted_keywords': ['diabetes'],
            'pmids': ['12345'],
            'article_markdown': 'Article content',
            'response': '',
            'current_agent': 'pubmed_agent',
            'agent_messages': [],
            'agent_results': {
                'pubmed_agent': {'status': 'success'}
            },
            'conversation_history': [],
            'pubmed_results': {'keywords': ['diabetes']},
            'websearch_results': None,
            'web_search_results': None,
            'web_search_markdown': None,
            'synthesis_complete': False
        }
        
        result = synthesis_agent_node(state)
        
        assert mock_agent.execute_with_retry.called
        assert result['response'] == 'Synthesized response with citations'
        assert result['synthesis_complete'] is True

class TestErrorHandling:
    """Test error handling and graceful degradation."""
    
    @patch('src.workflow.get_pubmed_agent')
    def test_pubmed_agent_failure_graceful(self, mock_get_agent):
        """Test PubMed agent failure doesn't crash workflow."""
        mock_agent = Mock(spec=PubMedAgent)
        mock_agent.execute_with_retry.side_effect = Exception("PubMed API error")
        mock_agent._record_error = Mock()
        mock_get_agent.return_value = mock_agent
        
        state: SupervisorState = {
            'user_query': 'Test query',
            'routing_decision': 'pubmed_agent',
            'iteration_count': 1,
            'extracted_keywords': [],
            'pmids': [],
            'article_markdown': '',
            'response': '',
            'current_agent': 'supervisor',
            'agent_messages': [],
            'agent_results': {},
            'conversation_history': [],
            'pubmed_results': None,
            'websearch_results': None,
            'web_search_results': None,
            'web_search_markdown': None,
            'synthesis_complete': False
        }
        
        # Should not raise exception
        result = pubmed_agent_node(state)
        
        # Error should be recorded
        assert mock_agent._record_error.called

class TestWorkflowIntegration:
    """Integration tests for full workflow (with mocks)."""
    
    @patch('src.workflow.build_workflow')
    def test_run_workflow_generates_thread_id(self, mock_build):
        """Test run_workflow generates thread_id when not provided."""
        mock_graph = Mock()
        mock_graph.invoke.return_value = {
            'response': 'Test response',
            'user_query': 'Test',
            'synthesis_complete': True
        }
        mock_build.return_value = mock_graph
        
        result = run_workflow("Test query", thread_id=None)
        
        # Should generate UUID and invoke graph
        assert mock_graph.invoke.called
        assert 'response' in result
    
    @patch('src.workflow.build_workflow')
    def test_run_workflow_uses_provided_thread_id(self, mock_build):
        """Test run_workflow uses provided thread_id."""

        mock_graph = Mock()
        mock_graph.invoke.return_value = {
            'response': 'Test response',
            'user_query': 'Test',
            'synthesis_complete': True
        }
        mock_build.return_value = mock_graph
        
        result = run_workflow("Test query", thread_id="abc-123")
        
        # Should use provided thread_id
        # Config is passed as second positional argument
        call_args = mock_graph.invoke.call_args
        config = call_args[0][1] if len(call_args[0]) > 1 else {}
        assert config.get('configurable', {}).get('thread_id') == "abc-123"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
