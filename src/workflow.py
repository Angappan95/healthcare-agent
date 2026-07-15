"""Multi-agent healthcare workflow with supervisor pattern."""

import logging
import uuid
from typing import Optional, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.agents.supervisor_agent import SupervisorAgent
from src.agents.pubmed_agent import PubMedAgent
from src.agents.websearch_agent import WebSearchAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.tools.helper import get_llm
from src.config import CHECKPOINT_DB_PATH
from src.models import SupervisorState, WorkflowState

logger = logging.getLogger(__name__)

_supervisor_agent = None
_pubmed_agent = None
_websearch_agent = None
_synthesis_agent = None

def get_supervisor_agent() -> SupervisorAgent:
    global _supervisor_agent
    if _supervisor_agent is None:
        _supervisor_agent = SupervisorAgent(llm=get_llm())
    return _supervisor_agent

def get_pubmed_agent() -> PubMedAgent:
    global _pubmed_agent
    if _pubmed_agent is None:
        _pubmed_agent = PubMedAgent(llm=get_llm())
    return _pubmed_agent

def get_websearch_agent() -> WebSearchAgent:
    global _websearch_agent
    if _websearch_agent is None:
        _websearch_agent = WebSearchAgent()
    return _websearch_agent

def get_synthesis_agent() -> SynthesisAgent:
    global _synthesis_agent
    if _synthesis_agent is None:
        _synthesis_agent = SynthesisAgent(llm=get_llm())
    return _synthesis_agent

def supervisor_node(state: SupervisorState) -> SupervisorState:
    """Supervisor node - routes to appropriate agents."""
    agent = get_supervisor_agent()
    
    try:
        # Circuit breaker is handled automatically by execute_with_retry
        result = agent.execute_with_retry(state)
        return result
        
    except Exception as e:
        logger.error(f"Supervisor node failed: {e}")
        # Graceful fallback: go to synthesis with whatever we have
        state['routing_decision'] = 'synthesis_agent'
        return state

def pubmed_agent_node(state: SupervisorState) -> SupervisorState:
    """PubMed agent node - searches medical literature."""
    agent = get_pubmed_agent()
    
    try:
        # Circuit breaker is handled automatically by execute_with_retry
        result = agent.execute_with_retry(state)
        return result
        
    except Exception as e:
        logger.error(f"PubMed agent failed: {e}")
        # Record error but don't fail workflow
        agent._record_error(state, e)
        return state

def websearch_agent_node(state: SupervisorState) -> SupervisorState:
    """WebSearch agent node - searches general health information."""
    agent = get_websearch_agent()
    
    try:
        # Circuit breaker is handled automatically by execute_with_retry
        result = agent.execute_with_retry(state)
        return result
        
    except Exception as e:
        logger.error(f"WebSearch agent failed: {e}")
        # Record error but don't fail workflow
        agent._record_error(state, e)
        return state

def synthesis_agent_node(state: SupervisorState) -> SupervisorState:
    """Synthesis agent node - collates and synthesizes results."""
    agent = get_synthesis_agent()
    
    try:
        # Circuit breaker is handled automatically by execute_with_retry
        result = agent.execute_with_retry(state)
        return result
        
    except Exception as e:
        logger.error(f"Synthesis agent failed: {e}")
        # Must provide some response
        state['response'] = f"Error synthesizing results: {str(e)}. Please try again."
        return state

# ============================================================================
# Routing Logic
# ============================================================================

def route_from_supervisor(state: SupervisorState) -> str:
    """
    Route from supervisor to next agent based on routing_decision.
    
    Returns:
        Next node name or END
    """
    decision = state.get('routing_decision', 'end')
    
    logger.info(f"Routing from supervisor: {decision}")
    
    # Map routing decisions to nodes
    if decision == 'pubmed_agent':
        return 'pubmed_agent'
    elif decision == 'websearch_agent':
        return 'websearch_agent'
    elif decision == 'both_agents':
        # LangGraph doesn't support parallel edges easily in this pattern
        # So we'll route to PubMed first, then WebSearch
        # The supervisor will handle this in next iteration
        return 'pubmed_agent'
    elif decision == 'synthesis_agent':
        return 'synthesis_agent'
    elif decision == 'end':
        return END
    else:
        logger.warning(f"Unknown routing decision: {decision}, defaulting to synthesis")
        return 'synthesis_agent'

def route_after_agents(state: SupervisorState) -> str:
    """
    Route after agent execution - back to supervisor or to synthesis.
    
    Returns:
        Next node name
    """
    agent_results = state.get('agent_results', {})
    routing_decision = state.get('routing_decision', '')
    
    # If both_agents routing and only PubMed done, go to WebSearch
    if routing_decision == 'both_agents':
        has_pubmed = 'pubmed_agent' in agent_results
        has_websearch = 'websearch_agent' in agent_results
        
        if has_pubmed and not has_websearch:
            logger.info("Both agents requested: PubMed done, routing to WebSearch")
            return 'websearch_agent'
    
    # Check if we should synthesize
    if state.get('synthesis_complete'):
        return END
    
    # Check if any agent has results
    has_results = any(
        result.get('status') == 'success' 
        for result in agent_results.values()
    )
    
    if has_results:
        logger.info("Agent results available, routing to synthesis")
        return 'synthesis_agent'
    
    # No results yet, back to supervisor
    logger.info("No results yet, routing back to supervisor")
    return 'supervisor'

def build_workflow(enable_checkpointing: bool = True) -> StateGraph:
    """Build and compile the multi-agent supervisor workflow."""
    workflow = StateGraph(SupervisorState)
    
    # Add agent nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("pubmed_agent", pubmed_agent_node)
    workflow.add_node("websearch_agent", websearch_agent_node)
    workflow.add_node("synthesis_agent", synthesis_agent_node)
    
    # Entry point: START → supervisor
    workflow.add_edge(START, "supervisor")
    
    # Supervisor makes routing decisions via conditional edges
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "pubmed_agent": "pubmed_agent",
            "websearch_agent": "websearch_agent",
            "synthesis_agent": "synthesis_agent",
            END: END
        }
    )
    
    # After PubMed agent: check if need WebSearch or go to synthesis
    workflow.add_conditional_edges(
        "pubmed_agent",
        route_after_agents,
        {
            "websearch_agent": "websearch_agent",
            "synthesis_agent": "synthesis_agent",
            "supervisor": "supervisor",
            END: END
        }
    )
    
    # After WebSearch agent: go to synthesis
    workflow.add_conditional_edges(
        "websearch_agent",
        route_after_agents,
        {
            "synthesis_agent": "synthesis_agent",
            "supervisor": "supervisor",
            END: END
        }
    )
    
    # After synthesis: workflow complete
    workflow.add_edge("synthesis_agent", END)
    
    # Compile with checkpointing if enabled
    if enable_checkpointing:

        # Create SqliteSaver directly - from_conn_string returns a context manager
        import sqlite3
        conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        graph = workflow.compile(checkpointer=checkpointer)
        logger.info(f"Workflow compiled with checkpointing (DB: {CHECKPOINT_DB_PATH})")
    else:
        graph = workflow.compile()
        logger.info("Workflow compiled without checkpointing")
    
    return graph

def run_workflow(user_query: str, thread_id: Optional[str] = None, 
                enable_checkpointing: bool = True) -> SupervisorState:
    """Execute the multi-agent supervisor workflow."""
    # Generate thread_id if not provided
    if thread_id is None:
        thread_id = str(uuid.uuid4())
        logger.info(f"Generated new thread_id: {thread_id}")
    else:
        logger.info(f"Using existing thread_id: {thread_id}")
    
    # Build workflow
    graph = build_workflow(enable_checkpointing=enable_checkpointing)
    
    # Initialize state with HumanMessage for the user query
    initial_state: SupervisorState = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "routing_decision": "",
        "agent_results": {},
        "iteration_count": 0,
        "current_agent": "",
        "agent_messages": [],
        "synthesis_complete": False,
        "extracted_keywords": [],
        "pmids": [],
        "article_markdown": "",
        "web_search_markdown": None,
        "response": ""
    }
    
    # Configuration for checkpointing
    config = {"configurable": {"thread_id": thread_id}} if enable_checkpointing else {}
    
    logger.info(f"Starting multi-agent workflow: {user_query}")
    
    try:
        final_state = graph.invoke(initial_state, config)
        logger.info("Multi-agent workflow completed successfully")
        return final_state
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        # Return state with error
        initial_state['response'] = f"Workflow error: {str(e)}"
        return initial_state

def get_checkpoint(thread_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve conversation checkpoint for given thread."""

    try:
        import sqlite3
        with sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False) as conn:
            checkpointer = SqliteSaver(conn)
            config = {"configurable": {"thread_id": thread_id}}
            
            # Get latest checkpoint
            checkpoint = checkpointer.get(config)
            
            if checkpoint:
                logger.info(f"Retrieved checkpoint for thread {thread_id}")
                return checkpoint
            else:
                logger.info(f"No checkpoint found for thread {thread_id}")
                return None
            
    except Exception as e:
        logger.error(f"Failed to retrieve checkpoint: {e}")
        return None

def list_checkpoints(limit: int = 10) -> List[Dict[str, Any]]:
    """List all conversation checkpoints."""

    try:
        import sqlite3
        with sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False) as conn:
            checkpointer = SqliteSaver(conn)
            
            # List all checkpoints
            checkpoints = []
        for checkpoint in checkpointer.list():
            checkpoints.append({
                'thread_id': checkpoint['config']['configurable']['thread_id'],
                'checkpoint_id': checkpoint['checkpoint_id'],
                'timestamp': checkpoint.get('timestamp')
            })
            
            if len(checkpoints) >= limit:
                break
        
        logger.info(f"Listed {len(checkpoints)} checkpoints")
        return checkpoints
        
    except Exception as e:
        logger.error(f"Failed to list checkpoints: {e}")
        return []

# ============================================================================
# Helper Functions for Working with Messages
# ============================================================================

def get_conversation_turns(state: SupervisorState) -> List[Dict[str, Any]]:
    """
    Extract conversation turns from messages for easier viewing.
    
    Returns list of turns with format:
    [
        {
            'user_query': str,
            'response': str,
            'sources_used': dict
        },
        ...
    ]
    """
    messages = state.get('messages', [])
    turns = []
    
    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            user_msg = messages[i]
            ai_msg = messages[i + 1]
            
            if isinstance(user_msg, HumanMessage) and isinstance(ai_msg, AIMessage):
                turn = {
                    'user_query': user_msg.content,
                    'response': ai_msg.content,
                    'sources_used': ai_msg.additional_kwargs.get('sources_used', {})
                }
                turns.append(turn)
    
    return turns

def print_conversation_history(state: SupervisorState):
    """Pretty print the conversation history from messages."""
    turns = get_conversation_turns(state)
    
    if not turns:
        print("No conversation history yet.")
        return
    
    print("\n" + "="*80)
    print("CONVERSATION HISTORY")
    print("="*80)
    
    for i, turn in enumerate(turns, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {turn['user_query']}")
        print(f"\nAssistant: {turn['response'][:200]}...")
        sources = turn['sources_used']
        if sources:
            print(f"Sources: PubMed={sources.get('pubmed', False)}, "
                  f"WebSearch={sources.get('websearch', False)}")
    
    print("\n" + "="*80)
    print(f"Total turns: {len(turns)}")
