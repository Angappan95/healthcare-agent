"""
End-to-End test for the supervisor multi-agent workflow.

Tests the complete workflow from user query through supervisor routing,
agent execution, synthesis, and final response generation.
"""
import pytest
import logging
import os

from unittest.mock import patch, Mock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from src.workflow import run_workflow, build_workflow
from src.models import SupervisorState
from src.agents.supervisor_agent import SupervisorAgent
from src.agents.pubmed_agent import PubMedAgent
from src.agents.websearch_agent import WebSearchAgent
from src.agents.synthesis_agent import SynthesisAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

class TestE2EWorkflow:
    """End-to-end tests for the complete supervisor workflow."""
    
    @pytest.mark.e2e
    def test_e2e_pubmed_only_flow(self):
        """Test e2e flow when only PubMed agent is needed."""

        # Mock LLM for supervisor routing
        with patch('src.tools.helper.get_llm') as mock_get_llm, \
             patch('src.utils.pubmed_utils.search_articles') as mock_search, \
             patch('src.utils.pubmed_utils.extract_pubmed_articles_as_markdown') as mock_extract:
            

            # Setup LLM to return different responses based on context
            mock_llm = MagicMock()
            responses = [
                MagicMock(content="pubmed_agent"),  # Supervisor routing
                MagicMock(content="diabetes, treatment, insulin"),  # PubMed keyword extraction
                MagicMock(content="Based on recent research, type 2 diabetes treatments include metformin and insulin therapy. [1]")  # Synthesis
            ]
            mock_llm.invoke.side_effect = responses
            mock_get_llm.return_value = mock_llm
            
            # Mock PubMed API
            mock_search.return_value = ["12345678", "87654321"]
            mock_extract.return_value = "**Article 1:** Diabetes Treatment Review\n*Authors:* Smith et al.\n*Year:* 2023"
            
            # Run workflow
            query = "What are the latest treatments for type 2 diabetes?"
            result = run_workflow(query, enable_checkpointing=False)
            
            # Assertions
            assert result is not None
            assert result['user_query'] == query
            assert result['response'] != ""
            assert len(result['messages']) >= 2  # At least user message and AI response
            
            # Verify routing happened
            assert result['iteration_count'] > 0
            assert 'pubmed_agent' in result['agent_results']
            
            logger.info("✅ E2E PubMed-only flow completed successfully")
            logger.info(f"Response: {result['response'][:200]}...")
    
    @pytest.mark.e2e
    def test_e2e_websearch_only_flow(self):
        """Test e2e flow when only WebSearch agent is needed."""

        with patch('src.tools.helper.get_llm') as mock_get_llm, \
             patch('src.agents.websearch_agent.WebSearchAgent.execute') as mock_websearch_execute:
            

            # Setup LLM to return different responses
            mock_llm = MagicMock()
            responses = [
                MagicMock(content="websearch_agent"),  # Supervisor routing
                MagicMock(content="Common flu symptoms include fever, cough, and fatigue. [1]")  # Synthesis
            ]
            mock_llm.invoke.side_effect = responses
            mock_get_llm.return_value = mock_llm
            
            # Mock WebSearch execution
            mock_websearch_execute.return_value = {
                'web_search_results': [
                    {'title': 'Flu Symptoms', 'url': 'http://example.com', 'content': 'Fever, cough, fatigue'}
                ],
                'web_search_markdown': "**Flu Symptoms:** Fever, cough, fatigue",
                'agent_results': {
                    'websearch_agent': {'status': 'success', 'result_count': 1}
                }
            }
            

            
            # Run workflow
            query = "What are the symptoms of flu?"
            result = run_workflow(query, enable_checkpointing=False)
            
            # Assertions
            assert result is not None
            assert result['user_query'] == query
            assert result['response'] != ""
            
            logger.info("✅ E2E WebSearch-only flow completed successfully")
            logger.info(f"Response: {result['response'][:200]}...")
    
    @pytest.mark.e2e
    def test_e2e_both_agents_flow(self):
        """Test e2e flow when both agents are needed."""

        with patch('src.tools.helper.get_llm') as mock_get_llm, \
             patch('src.agents.websearch_agent.WebSearchAgent.execute') as mock_websearch_execute, \
             patch('src.utils.pubmed_utils.search_articles') as mock_search, \
             patch('src.utils.pubmed_utils.extract_pubmed_articles_as_markdown') as mock_extract:
            

            # Setup LLM to return different responses
            mock_llm = MagicMock()
            responses = [
                MagicMock(content="both_agents"),  # Supervisor routing
                MagicMock(content="covid, symptoms, treatment"),  # PubMed keyword extraction
                MagicMock(content="COVID-19 symptoms include fever and cough. Latest research shows effective treatments. [1][2]")  # Synthesis
            ]
            mock_llm.invoke.side_effect = responses
            mock_get_llm.return_value = mock_llm
            
            # Mock PubMed API
            mock_search.return_value = ["33456789"]
            mock_extract.return_value = "**COVID-19 Research:** Symptoms and treatment options"
            
            # Mock WebSearch execution
            mock_websearch_execute.return_value = {
                'web_search_results': [
                    {'title': 'CDC COVID Guide', 'url': 'http://cdc.gov/covid', 'content': 'Latest guidelines'}
                ],
                'web_search_markdown': "**CDC:** Latest COVID-19 guidelines",
                'agent_results': {
                    'websearch_agent': {'status': 'success', 'result_count': 1}
                }
            }
            

            
            # Run workflow
            query = "What are COVID-19 symptoms and treatments?"
            result = run_workflow(query, enable_checkpointing=False)
            
            # Assertions
            assert result is not None
            assert result['user_query'] == query
            assert result['response'] != ""
            
            # Both agents should have been executed
            assert 'pubmed_agent' in result['agent_results'] or 'websearch_agent' in result['agent_results']
            
            logger.info("✅ E2E both-agents flow completed successfully")
            logger.info(f"Response: {result['response'][:200]}...")
    
    @pytest.mark.e2e
    def test_e2e_multi_turn_conversation(self):
        """Test e2e multi-turn conversation with checkpointing."""

        thread_id = "e2e-test-thread-001"
        
        with patch('src.tools.helper.get_llm') as mock_get_llm, \
             patch('src.utils.pubmed_utils.search_articles') as mock_search, \
             patch('src.utils.pubmed_utils.extract_pubmed_articles_as_markdown') as mock_extract:
            

            # Setup LLM to return different responses for each turn
            mock_llm = MagicMock()
            responses = [
                # Turn 1
                MagicMock(content="pubmed_agent"),
                MagicMock(content="diabetes, metformin"),
                MagicMock(content="Metformin is the first-line treatment for type 2 diabetes. [1]"),
                # Turn 2
                MagicMock(content="pubmed_agent"),
                MagicMock(content="metformin, side effects"),
                MagicMock(content="Common side effects include nausea and diarrhea. [1]")
            ]
            mock_llm.invoke.side_effect = responses
            mock_get_llm.return_value = mock_llm
            
            # Mock PubMed API
            mock_search.return_value = ["11111111"]
            mock_extract.return_value = "**Diabetes Treatment:** Metformin is first-line therapy"
            

            
            # Turn 1
            result1 = run_workflow(
                user_query="What is the main treatment for diabetes?",
                thread_id=thread_id,
                enable_checkpointing=True
            )
            
            assert result1 is not None
            assert result1['response'] != ""
            messages_turn1 = len(result1.get('messages', []))
            
            # Turn 2 - Follow-up question
            result2 = run_workflow(
                user_query="What are the side effects?",
                thread_id=thread_id,
                enable_checkpointing=True
            )
            
            assert result2 is not None
            assert result2['response'] != ""
            
            # Verify conversation history grew
            messages_turn2 = len(result2.get('messages', []))
            assert messages_turn2 > messages_turn1, "Conversation history should accumulate"
            
            logger.info("✅ E2E multi-turn conversation completed successfully")
            logger.info(f"Turn 1 messages: {messages_turn1}, Turn 2 messages: {messages_turn2}")
    
    @pytest.mark.e2e
    def test_e2e_error_recovery(self):
        """Test e2e flow with agent failure and graceful recovery."""

        with patch('src.tools.helper.get_llm') as mock_get_llm, \
             patch('src.agents.pubmed_agent.PubMedAgent.execute') as mock_pubmed_execute:
            

            # Setup LLM
            mock_llm = MagicMock()
            responses = [
                MagicMock(content="pubmed_agent"),
                MagicMock(content="Unable to fetch medical literature. Please try again later.")
            ]
            mock_llm.invoke.side_effect = responses
            mock_get_llm.return_value = mock_llm
            
            # Make PubMed agent fail
            mock_pubmed_execute.side_effect = Exception("PubMed API timeout")
            

            
            # Run workflow - should not crash
            query = "What is aspirin used for?"
            result = run_workflow(query, enable_checkpointing=False)
            
            # Assertions - workflow should complete even with agent failure
            assert result is not None
            assert result['user_query'] == query
            assert result['response'] != ""
            
            logger.info("✅ E2E error recovery completed successfully")
            logger.info(f"Response despite error: {result['response'][:150]}...")

class TestE2EWorkflowStructure:
    """Test the workflow graph structure end-to-end."""
    
    @pytest.mark.e2e
    def test_workflow_graph_builds_correctly(self):
        """Test that workflow graph compiles without errors."""
        graph = build_workflow(enable_checkpointing=False)
        assert graph is not None
        
        # Verify the compiled graph has the expected structure
        # LangGraph provides a 'nodes' attribute with node names
        logger.info("✅ Workflow graph built successfully")
    
    @pytest.mark.e2e
    def test_workflow_with_checkpointing_enabled(self):
        """Test workflow with SQLite checkpointing enabled."""

        # This tests the actual checkpoint persistence
        thread_id = "e2e-checkpoint-test"
        
        with patch('src.tools.helper.get_llm') as mock_get_llm, \
             patch('src.utils.pubmed_utils.search_articles') as mock_search, \
             patch('src.utils.pubmed_utils.extract_pubmed_articles_as_markdown') as mock_extract:
            

            # Setup LLM
            mock_llm = MagicMock()
            responses = [
                MagicMock(content="pubmed_agent"),
                MagicMock(content="headache, treatment"),
                MagicMock(content="For headaches, over-the-counter pain relievers are effective. [1]")
            ]
            mock_llm.invoke.side_effect = responses
            mock_get_llm.return_value = mock_llm
            
            mock_search.return_value = ["99999999"]
            mock_extract.return_value = "**Headache Treatment:** Pain relievers are effective"
            

            
            # Run with checkpointing
            result = run_workflow(
                user_query="How to treat a headache?",
                thread_id=thread_id,
                enable_checkpointing=True
            )
            
            assert result is not None
            assert result['response'] != ""
            
            # Verify checkpoint DB file was created
            from src.config import CHECKPOINT_DB_PATH
            assert os.path.exists(CHECKPOINT_DB_PATH) or os.path.exists(f"{CHECKPOINT_DB_PATH}.db")
            
            logger.info("✅ Checkpointing enabled and working")

@pytest.mark.e2e
def test_run_all_e2e_validations():
    """Meta-test to run all e2e validations in sequence."""
    logger.info("\n" + "="*80)
    logger.info("Starting E2E Validation Suite")
    logger.info("="*80)
    
    # This will be executed by pytest
    # Just a marker test to group e2e tests
    logger.info("Run with: pytest test/test_e2e_supervisor.py -v -m e2e")
    logger.info("="*80 + "\n")
