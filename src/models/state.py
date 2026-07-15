"""State models for the healthcare agent workflow."""

from typing import TypedDict, Optional, List, Dict, Any, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class SupervisorState(TypedDict):
    """State schema for multi-agent workflow."""
    # Built-in message handling with automatic reducer
    messages: Annotated[list[BaseMessage], add_messages]
    # Workflow control
    user_query: str
    routing_decision: str
    agent_results: Dict[str, Any]
    iteration_count: int
    current_agent: str
    agent_messages: List[Dict[str, Any]]
    synthesis_complete: Optional[bool]
    # Agent-specific data
    extracted_keywords: List[str]
    pmids: List[str]
    article_markdown: str
    web_search_markdown: Optional[str]
    # Final output
    response: str

WorkflowState = SupervisorState
