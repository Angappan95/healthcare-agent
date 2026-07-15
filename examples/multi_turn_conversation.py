"""
Example: Multi-turn conversation using LangGraph's built-in message handling.

This demonstrates how to have follow-up conversations where the system
maintains context from previous turns using BaseMessage types and add_messages reducer.
"""

import logging
from langchain_core.messages import HumanMessage, AIMessage
from src.workflow import run_workflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

def print_response(result, turn_number):
    """Pretty print the response."""
    print(f"\n{'='*80}")
    print(f"TURN {turn_number}")
    print('='*80)
    print(f"Query: {result['user_query']}")
    print(f"\nResponse:\n{result['response']}")
    
    # Show message count
    messages = result.get('messages', [])
    print(f"\nMessages in conversation: {len(messages)}")
    print('='*80)

def main():
    """Run a multi-turn conversation."""
    
    # Important: Use the same thread_id for all turns to maintain conversation history
    thread_id = "multi-turn-demo-001"
    
    print("Multi-Turn Conversation Demo")
    print("=" * 80)
    print("This demonstrates conversation history tracking using LangGraph's")
    print("built-in message handling with add_messages reducer.")
    print("Each query builds on the previous context.\n")
    
    # Turn 1: Initial question
    print("\n🔵 TURN 1: Initial Question")
    result1 = run_workflow(
        user_query="What are the main treatments for type 2 diabetes?",
        thread_id=thread_id,
        enable_checkpointing=True
    )
    print_response(result1, 1)
    
    # Turn 2: Follow-up question (references previous context)
    print("\n🔵 TURN 2: Follow-up Question")
    result2 = run_workflow(
        user_query="What are the side effects of those medications?",
        thread_id=thread_id,
        enable_checkpointing=True
    )
    print_response(result2, 2)
    
    # Turn 3: Another follow-up
    print("\n🔵 TURN 3: Another Follow-up")
    result3 = run_workflow(
        user_query="Are there any lifestyle changes that can help reduce the need for medication?",
        thread_id=thread_id,
        enable_checkpointing=True
    )
    print_response(result3, 3)
    
    # Show full conversation history from messages
    print("\n\n📚 FINAL CONVERSATION HISTORY (from messages)")
    print('='*80)
    messages = result3.get('messages', [])
    
    turn = 0
    for i in range(0, len(messages), 2):
        turn += 1
        if i < len(messages):
            user_msg = messages[i]
            ai_msg = messages[i + 1] if i + 1 < len(messages) else None
            
            print(f"\nTurn {turn}:")
            if isinstance(user_msg, HumanMessage):
                print(f"  User: {user_msg.content[:200]}")
            
            if ai_msg and isinstance(ai_msg, AIMessage):
                print(f"  Assistant: {ai_msg.content[:150]}...")
                # Show metadata if available
                sources = ai_msg.additional_kwargs.get('sources_used', {})
                if sources:
                    print(f"  Sources: PubMed={sources.get('pubmed', False)}, "
                          f"WebSearch={sources.get('websearch', False)}")
    
    print("\n" + "="*80)
    print(f"✅ Conversation complete with {len(messages)} messages total")
    print("💡 Note: With >2000 tokens, SummarizationMiddleware automatically summarizes older messages")
    print("    (Keeps last 2 messages intact for immediate context)")

if __name__ == "__main__":
    main()
