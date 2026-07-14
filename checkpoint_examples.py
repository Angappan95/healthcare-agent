"""
Examples demonstrating checkpointing and state persistence.

Shows:
1. Running workflow with checkpoint support
2. Resuming from checkpoints
3. Listing saved checkpoints
4. Retrieving checkpoint state
"""

import logging
from src.workflow import run_workflow, get_checkpoint, list_checkpoints

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")


# ============================================================================
# EXAMPLE 1: Basic Workflow with Auto-Generated Thread ID
# ============================================================================

def example_1_basic_with_checkpoint():
    """
    Run workflow with automatic thread ID.
    State is automatically saved to SQLite database.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Workflow with Checkpointing")
    print("="*80)

    query = "What are the symptoms of hypertension?"
    result = run_workflow(query)

    print(f"\nQuery: {result['user_query']}")
    print(f"Keywords: {result['extracted_keywords']}")
    print(f"Articles: {len(result['pmids'])}")
    print(f"Response: {result['response'][:200]}...")


# ============================================================================
# EXAMPLE 2: Workflow with Custom Thread ID
# ============================================================================

def example_2_custom_thread_id():
    """
    Run workflow with custom thread ID.
    Allows grouping related queries under one thread.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Thread ID for Related Queries")
    print("="*80)

    thread_id = "medical_research_session_001"
    queries = [
        "What is diabetes?",
        "What are diabetes complications?",
        "How to manage diabetes?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}] {query}")
        result = run_workflow(query, thread_id=thread_id)
        print(f"  Keywords: {result['extracted_keywords']}")
        print(f"  Articles: {len(result['pmids'])}")
        print(f"  Checkpoint saved to thread: {thread_id}")


# ============================================================================
# EXAMPLE 3: Retrieve Checkpoint State
# ============================================================================

def example_3_retrieve_checkpoint():
    """
    Retrieve previously saved checkpoint state from database.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Retrieve Checkpoint State")
    print("="*80)

    # First, run a workflow to create a checkpoint
    thread_id = "checkpoint_test_001"
    query = "What is heart disease?"
    print(f"\nRunning workflow: {query}")
    result1 = run_workflow(query, thread_id=thread_id)

    # Now retrieve the checkpoint
    print(f"\nRetrieving checkpoint for thread_id: {thread_id}")
    checkpoint_state = get_checkpoint(thread_id)

    if checkpoint_state:
        print("✓ Checkpoint found!")
        print(f"  Query: {checkpoint_state.get('user_query')}")
        print(f"  Keywords: {checkpoint_state.get('extracted_keywords')}")
        print(f"  Articles: {len(checkpoint_state.get('pmids', []))}")
    else:
        print("✗ Checkpoint not found")


# ============================================================================
# EXAMPLE 4: List All Saved Checkpoints
# ============================================================================

def example_4_list_checkpoints():
    """
    List all saved checkpoint thread IDs in database.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: List All Saved Checkpoints")
    print("="*80)

    # Create a few checkpoints first
    queries = [
        ("cancer_research", "What is cancer?"),
        ("cancer_research", "What are cancer treatments?"),
        ("covid_research", "What is COVID-19?"),
    ]

    print("\nCreating checkpoints...")
    for thread_id, query in queries:
        print(f"  [{thread_id}] {query}")
        run_workflow(query, thread_id=thread_id)

    # List all checkpoints
    print("\nListing all saved checkpoints...")
    checkpoints = list_checkpoints()

    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoints:")
        for i, thread_id in enumerate(checkpoints, 1):
            print(f"  {i}. {thread_id}")
    else:
        print("No checkpoints found")


# ============================================================================
# EXAMPLE 5: Resume Workflow from Checkpoint
# ============================================================================

def example_5_resume_from_checkpoint():
    """
    Continue workflow execution from a saved checkpoint.
    Useful for long-running workflows or handling interruptions.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Resume from Checkpoint")
    print("="*80)

    thread_id = "resume_test_001"

    # First run
    print("\n[Step 1] Initial workflow execution")
    query = "What is diabetes treatment?"
    result1 = run_workflow(query, thread_id=thread_id)
    print(f"  Status: Completed")
    print(f"  Articles found: {len(result1['pmids'])}")

    # Simulate interruption and resume
    print("\n[Step 2] Retrieved checkpoint from database")
    checkpoint = get_checkpoint(thread_id)
    if checkpoint:
        print(f"  ✓ State restored from checkpoint")
        print(f"  Query: {checkpoint.get('user_query')}")
        print(f"  Keywords: {checkpoint.get('extracted_keywords')}")
        print(f"  Articles: {len(checkpoint.get('pmids', []))}")
    else:
        print(f"  ✗ Could not retrieve checkpoint")


# ============================================================================
# EXAMPLE 6: Batch Processing with Checkpoints
# ============================================================================

def example_6_batch_with_checkpoints():
    """
    Process multiple queries with checkpoint save for each.
    Enables recovery from failures mid-batch.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Batch Processing with Checkpoints")
    print("="*80)

    batch_id = "batch_medical_research"
    queries = [
        "What is hypertension?",
        "What is diabetes?",
        "What is heart disease?",
    ]

    results = []
    for i, query in enumerate(queries, 1):
        thread_id = f"{batch_id}_{i}"
        print(f"\n[{i}/{len(queries)}] {query}")
        try:
            result = run_workflow(query, thread_id=thread_id)
            results.append(result)
            print(f"  ✓ Success - Checkpoint: {thread_id}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print(f"\n\nBatch Summary:")
    print(f"  Total: {len(queries)}")
    print(f"  Successful: {len(results)}")
    print(f"  Total articles: {sum(len(r['pmids']) for r in results)}")


# ============================================================================
# EXAMPLE 7: LLM Reuse (Implicit)
# ============================================================================

def example_7_llm_reuse():
    """
    Demonstrate LLM reuse - LLM is instantiated once at module level
    and reused for all workflow executions, improving performance.
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: LLM Reuse (Single Instance)")
    print("="*80)

    print("\nLLM Reuse Benefits:")
    print("  ✓ LLM initialized once at module level")
    print("  ✓ Connection pooling and caching")
    print("  ✓ Faster response times")
    print("  ✓ Lower resource usage")
    print("  ✓ Better rate limiting handling")

    # Run multiple queries - LLM instance is reused
    queries = [
        "What is asthma?",
        "What causes asthma?",
        "How to treat asthma?",
    ]

    print("\nExecuting queries with reused LLM instance...")
    for i, query in enumerate(queries, 1):
        print(f"\n  [{i}] {query}")
        result = run_workflow(query)
        print(f"      Response length: {len(result['response'])} chars")

    print("\n✓ All queries completed with single LLM instance")


# ============================================================================
# Main: Run All Examples
# ============================================================================

def main():
    """Run all checkpoint examples."""
    print("\n" + "="*80)
    print("Checkpoint & State Persistence Examples")
    print("="*80)

    examples = [
        ("Basic with Checkpoint", example_1_basic_with_checkpoint),
        ("Custom Thread ID", example_2_custom_thread_id),
        ("Retrieve Checkpoint", example_3_retrieve_checkpoint),
        ("List Checkpoints", example_4_list_checkpoints),
        ("Resume from Checkpoint", example_5_resume_from_checkpoint),
        ("Batch with Checkpoints", example_6_batch_with_checkpoints),
        ("LLM Reuse", example_7_llm_reuse),
    ]

    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n✗ Example '{name}' failed: {e}")

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    # Run individual examples
    example_1_basic_with_checkpoint()
    # example_2_custom_thread_id()
    # example_3_retrieve_checkpoint()
    # example_4_list_checkpoints()
    # example_5_resume_from_checkpoint()
    # example_6_batch_with_checkpoints()
    # example_7_llm_reuse()

    # Or run all:
    # main()
