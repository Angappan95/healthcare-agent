"""Examples using the LangGraph workflow."""

import logging
from src.workflow import run_workflow, build_workflow, WorkflowState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Example 1: Basic Usage
def example_basic():
    """Run workflow and display results."""
    print("\n" + "="*80)
    print("BASIC WORKFLOW EXAMPLE")
    print("="*80)
    result = run_workflow("What are the symptoms of diabetes?")
    print(f"Query: {result['user_query']}")
    print(f"Keywords: {result['extracted_keywords']}")
    print(f"Articles: {len(result['pmids'])}")
    print(f"Response: {result['response'][:300]}...")


# Example 2: Intermediate Results
def example_intermediate():
    """Access state at each step."""
    print("\n" + "="*80)
    print("INTERMEDIATE RESULTS EXAMPLE")
    print("="*80)
    result = run_workflow("What is heart disease?")
    print(f"Step 1 - Query: {result['user_query']}")
    print(f"Step 2 - Keywords: {result['extracted_keywords']}")
    print(f"Step 3 - PMIDs: {result['pmids']}")
    print(f"Step 4 - Articles Found: {len(result['pmids'])}")
    print(f"Step 5 - Response: {result['response'][:200]}...")


# Example 3: Custom Workflow Building
def example_custom():
    """Build and run workflow manually."""
    print("\n" + "="*80)
    print("CUSTOM WORKFLOW EXAMPLE")
    print("="*80)
    graph = build_workflow()
    state: WorkflowState = {
        "user_query": "What is hypertension?",
        "extracted_keywords": [],
        "pmids": [],
        "article_markdown": "",
        "response": ""
    }
    result = graph.invoke(state)
    print(f"Query: {result['user_query']}")
    print(f"Keywords: {result['extracted_keywords']}")
    print(f"Articles: {len(result['pmids'])}")
    print(f"Response: {result['response'][:200]}...")


if __name__ == "__main__":
    example_basic()
    # example_intermediate()
    # example_custom()
