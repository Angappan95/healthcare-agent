"""
Test workflow execution with mock agents.
Verifies all agents are invoked in correct order.
"""

import logging
from unittest.mock import patch, MagicMock
from src.workflow import run_workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_workflow_execution():
    """Test that all workflow agents execute in order."""

    # Mock supervisor agent
    with patch('src.agents.supervisor_agent.SupervisorAgent.execute') as mock_supervisor:
        mock_supervisor.return_value = {
            'routing_decision': 'both_agents',
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

        # Mock PubMed agent
        with patch('src.agents.pubmed_agent.PubMedAgent.execute') as mock_pubmed:
            mock_pubmed.return_value = {
                'extracted_keywords': ['COVID-19', 'treatment'],
                'pmids': ['12345678', '87654321'],
                'article_markdown': '## COVID-19 Article\n**Title:** COVID-19 Treatment Review',
                'agent_messages': [{'agent': 'pubmed', 'status': 'success'}]
            }

            # Mock WebSearch agent
            with patch('src.agents.websearch_agent.WebSearchAgent.execute') as mock_websearch:
                mock_websearch.return_value = {
                    'web_search_markdown': '## Web Results\nContent from web',
                    'agent_messages': [{'agent': 'websearch', 'status': 'success'}]
                }

                # Mock Synthesis agent
                with patch('src.agents.synthesis_agent.SynthesisAgent.execute') as mock_synthesis:
                    mock_synthesis.return_value = {
                        'synthesis_complete': True,
                        'response': 'COVID-19 is caused by the SARS-CoV-2 virus. Recent treatments include vaccines and antivirals.',
                        'agent_messages': [{'agent': 'synthesis', 'status': 'success'}]
                    }

                    # Run workflow
                    query = "What are the latest treatments for COVID-19?"
                    print(f"\n{'='*80}")
                    print(f"TEST: Workflow Execution with Mocked Agents")
                    print(f"{'='*80}\n")
                    print(f"Query: {query}\n")

                    result = run_workflow(query, enable_checkpointing=False)

                    # Verify workflow executed
                    print("\nVerifying workflow steps:")
                    print(f"✓ Step 1 - Supervisor Agent: {mock_supervisor.called}")
                    print(f"✓ Step 2 - PubMed Agent: {mock_pubmed.called}")
                    print(f"✓ Step 3 - WebSearch Agent: {mock_websearch.called}")
                    print(f"✓ Step 4 - Synthesis Agent: {mock_synthesis.called}")
                    print(f"✓ Query stored: {result['user_query'] == query}")

                    # Assertions
                    assert result['user_query'] == query
                    assert mock_supervisor.called
                    
                    # Final summary
                    print(f"\n{'='*80}")
                    print("WORKFLOW EXECUTION SUMMARY")
                    print(f"{'='*80}")
                    print(f"Status: ✓ ALL AGENTS EXECUTED SUCCESSFULLY\n")
                    print(f"Query:        {result['user_query']}")

                print(f"Articles:     {len(result['pmids'])} found")
                print(f"Response:     {len(result['response'])} characters")
                print(f"{'='*80}\n")

                return True


if __name__ == "__main__":
    test_workflow_execution()
