import logging
from src.workflow import run_workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    result = run_workflow("What are the latest treatments for COVID-19?")
    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"Query: {result['user_query']}")
    print(f"Keywords: {', '.join(result['extracted_keywords'])}")
    print(f"Articles: {len(result['pmids'])}")
    print(f"Response:\n{result['response']}")
    print("="*80)
