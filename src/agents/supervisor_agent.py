"""Supervisor Agent - Orchestrates workflow and routes to specialized agents."""

import logging
from typing import Any, Dict, List, Literal
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from src.agents.base_agent import BaseAgent, AgentExecutionError
from src.config import MAX_AGENT_ITERATIONS, ENABLE_PARALLEL_AGENTS
from src.prompts.prompt import get_query_routing_prompt

logger = logging.getLogger(__name__)

# Type hints for routing decisions
RoutingDecision = Literal["pubmed_agent", "websearch_agent", "both_agents", "synthesis_agent", "end"]

class SupervisorAgent(BaseAgent):
    """Supervisor agent that coordinates multi-agent workflow."""
    
    def __init__(self, llm=None):
        """Initialize Supervisor Agent."""
        super().__init__(
            name="supervisor",
            description="Orchestrates multi-agent workflow and makes routing decisions"
        )
        self._llm = llm
        
    def _get_llm(self):
        """Get LLM instance (lazy initialization)."""
        if self._llm is None:
            from src.tools.helper import get_llm
            self._llm = get_llm()
        return self._llm
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute supervisor logic - analyze query and make routing decision."""
        try:
            # Check iteration count
            iteration = state.get('iteration_count', 0)
            if iteration >= MAX_AGENT_ITERATIONS:
                self.logger.warning(f"Max iterations ({MAX_AGENT_ITERATIONS}) reached")
                state['routing_decision'] = 'synthesis_agent'
                return self._update_state(state, {'routing_decision': 'synthesis_agent'})
            
            # Increment iteration
            state['iteration_count'] = iteration + 1
            
            # Determine routing based on workflow state
            routing = self._determine_routing(state)
            
            self.logger.info(f"Supervisor routing decision: {routing}")
            
            # Update state
            updates = {
                'routing_decision': routing,
                'current_agent': 'supervisor'
            }
            
            # Add routing message for transparency
            if 'agent_messages' not in state:
                state['agent_messages'] = []
            
            state['agent_messages'].append({
                'agent': 'supervisor',
                'type': 'routing',
                'decision': routing,
                'iteration': state['iteration_count']
            })
            
            return self._update_state(state, updates)
            
        except Exception as e:
            self.logger.error(f"Supervisor execution failed: {str(e)}")
            self._record_error(state, e)
            raise AgentExecutionError(f"Supervisor failed: {str(e)}")
    
    def _determine_routing(self, state: Dict[str, Any]) -> RoutingDecision:
        """Determine next agent to invoke based on state."""
        iteration = state.get('iteration_count', 0)
        agent_results = state.get('agent_results', {})
        
        # First iteration: route based on query analysis
        if iteration == 1:
            return self._analyze_query_and_route(state)
        
        # Check if agents have been invoked
        pubmed_done = 'pubmed_agent' in agent_results
        websearch_done = 'websearch_agent' in agent_results
        synthesis_done = 'synthesis_agent' in agent_results
        
        # If synthesis is done, end workflow
        if synthesis_done:
            return 'end'
        
        # If at least one agent has results, proceed to synthesis
        if pubmed_done or websearch_done:
            return 'synthesis_agent'
        
        # Fallback: analyze query again
        return self._analyze_query_and_route(state)
    
    def _format_conversation_history(self, state: Dict[str, Any]) -> str:
        """Format conversation history from messages for LLM context.
        
        Note: SummarizationMiddleware already filters messages, so we format
        all messages in state (they're already managed by the middleware).
        """
        messages = state.get('messages', [])
        if len(messages) <= 1:  # Only current message
            return "[No previous conversation]"
        
        formatted = []
        # Format all messages except the last one (current query)
        # The middleware already keeps only relevant messages (last 2 + summary)
        for msg in messages[:-1]:
            if isinstance(msg, HumanMessage):
                formatted.append(f"User: {msg.content[:200]}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"Assistant: {msg.content[:200]}...")
        
        return "\n".join(formatted) if formatted else "[No previous conversation]"
    
    def _analyze_query_and_route(self, state: Dict[str, Any]) -> RoutingDecision:
        """
        Use LLM to analyze query intent and decide routing.
        
        Returns:
            Routing decision
        """
        user_query = state.get('user_query', '')
        conversation_context = self._format_conversation_history(state)
        
        try:
            llm = self._get_llm()
            

            # Use centralized prompt
            prompt = get_query_routing_prompt(user_query, conversation_context)

            self.logger.debug("Analyzing query for routing...")
            message = llm.invoke(prompt)
            
            # Extract routing decision
            if isinstance(message.content, str):
                decision_text = message.content.strip().lower()
            elif isinstance(message.content, list):
                decision_text = message.content[0].get('text', '').strip().lower()
            else:
                decision_text = str(message.content).strip().lower()
            
            # Map to valid routing decision
            if 'both' in decision_text:
                routing = 'both_agents'
            elif 'websearch' in decision_text or 'web' in decision_text:
                routing = 'websearch_agent'
            elif 'pubmed' in decision_text:
                routing = 'pubmed_agent'
            else:
                # Default to both for safety
                self.logger.warning(f"Unclear routing decision: {decision_text}, defaulting to both_agents")
                routing = 'both_agents'
            
            self.logger.info(f"LLM routing decision: {routing}")
            return routing
            
        except Exception as e:
            self.logger.error(f"LLM routing failed: {e}, defaulting to both_agents")
            # Fallback: use both agents for comprehensive coverage
            return 'both_agents'
    
    def should_route_parallel(self, routing: RoutingDecision) -> bool:
        """Determine if agents should execute in parallel."""
        return ENABLE_PARALLEL_AGENTS and routing == 'both_agents'
