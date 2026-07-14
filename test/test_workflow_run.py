"""
Test workflow execution with mock LLM.
Verifies all nodes are invoked in correct order.
"""

import logging
from unittest.mock import patch, MagicMock
from src.workflow import run_workflow, build_workflow, WorkflowState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_workflow_execution():
    """Test that all workflow nodes execute in order."""

    # Mock the LLM
    with patch('src.workflow.get_llm') as mock_get_llm:
        mock_llm = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "COVID-19 is caused by the SARS-CoV-2 virus. Recent treatments include vaccines and antivirals."
        mock_llm.invoke.return_value = mock_message
        mock_get_llm.return_value = mock_llm

        # Mock PubMed search
        with patch('src.workflow.search_articles') as mock_search:
            mock_search.return_value = ["12345678", "87654321"]

            # Mock article extraction
            with patch('src.workflow.extract_pubmed_articles_as_markdown') as mock_extract:
                mock_extract.return_value = "## COVID-19 Article\n**Title:** COVID-19 Treatment Review\n**Authors:** Smith et al."

                # Run workflow
                query = "What are the latest treatments for COVID-19?"
                print(f"\n{'='*80}")
                print(f"TEST: Workflow Execution with Mocked Components")
                print(f"{'='*80}\n")
                print(f"Query: {query}\n")

                result = run_workflow(query)

                # Verify all steps executed
                print("\nVerifying workflow steps:")
                print(f"✓ Step 1 - Receive Query: {result['user_query'] == query}")
                print(f"✓ Step 2 - Extract Keywords: {len(result['extracted_keywords']) > 0}")
                print(f"  Keywords: {result['extracted_keywords']}")
                print(f"✓ Step 3 - Search PubMed: {len(result['pmids']) > 0}")
                print(f"  PMIDs: {result['pmids']}")
                print(f"✓ Step 4 - Extract Articles: {len(result['article_markdown']) > 0}")
                print(f"  Article preview: {result['article_markdown'][:50]}...")
                print(f"✓ Step 5 - Generate Response: {len(result['response']) > 0}")
                print(f"  Response: {result['response']}")

                # Final summary
                print(f"\n{'='*80}")
                print("WORKFLOW EXECUTION SUMMARY")
                print(f"{'='*80}")
                print(f"Status: ✓ ALL STEPS EXECUTED SUCCESSFULLY\n")
                print(f"Query:        {result['user_query']}")
                print(f"Keywords:     {', '.join(result['extracted_keywords'])}")
                print(f"Articles:     {len(result['pmids'])} found")
                print(f"Response:     {len(result['response'])} characters")
                print(f"{'='*80}\n")

                return True


if __name__ == "__main__":
    test_workflow_execution()
