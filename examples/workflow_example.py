"""Example: Running a multi-agent workflow with the healthcare agent."""

import logging
from src.workflow import run_workflow

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    
    # Run multi-agent workflow
    result = run_workflow("What are the latest treatments for type 2 diabetes?")
    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"Query: {result['user_query']}\nKeywords: {result['extracted_keywords']}")
    print(f"Articles: {len(result['pmids'])}\nResponse:\n{result['response']}")
