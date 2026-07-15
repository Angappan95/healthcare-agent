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

# Agent-specific state definitions
class PubMedState(TypedDict):
    """State for PubMed agent graph."""
    query: str
    keywords: List[str]
    pmids: List[str]
    article_markdown: str
    error: Optional[str]

class WebSearchState(TypedDict):
    """State for WebSearch agent graph."""
    query: str
    search_results: List[Dict]
    credible_results: List[Dict]
    web_search_markdown: str
    error: Optional[str]

class SynthesisState(TypedDict):
    """State for Synthesis agent graph."""
    query: str
    context: Dict[str, Any]
    combined_content: str
    response: str
    sources_used: List[str]
    error: Optional[str]
