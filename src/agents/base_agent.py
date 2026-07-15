"""Base agent abstraction with retry logic and circuit breaker."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pybreaker import CircuitBreaker
from src.config import MAX_RETRIES, RETRY_BACKOFF_FACTOR, AGENT_TIMEOUT

logger = logging.getLogger(__name__)

class AgentExecutionError(Exception):
    """Raised when agent execution fails."""
    pass

class AgentTimeoutError(Exception):
    """Raised when agent execution exceeds timeout."""
    pass

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the healthcare system.
    
    Provides common functionality:
    - Retry logic with exponential backoff
    - Error handling and logging
    - Timeout management
    - Agent metadata (name, capabilities, description)
    """
    
    def __init__(self, name: str, description: str, fail_max: int = 5, reset_timeout: int = 60):
        """
        Initialize base agent.
        
        Args:
            name: Agent identifier (e.g., "pubmed_agent", "supervisor")
            description: Human-readable description of agent's purpose
            fail_max: Number of consecutive failures before circuit opens
            reset_timeout: Seconds before attempting to reset circuit
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"agent.{name}")
        self._execution_count = 0
        self._total_duration = 0.0
        self._failure_count = 0
        
        # Each agent has its own circuit breaker
        self.circuit_breaker = CircuitBreaker(
            fail_max=fail_max,
            reset_timeout=reset_timeout,
            name=name
        )
        
    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent's primary function."""
        pass
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent's capabilities and metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "execution_count": self._execution_count,
            "average_duration": self._total_duration / max(self._execution_count, 1),
            "failure_count": self._failure_count,
            "failure_rate": self._failure_count / max(self._execution_count, 1)
        }
    
    def execute_with_retry(self, state: Dict[str, Any], max_retries: int = MAX_RETRIES) -> Dict[str, Any]:
        """
        Execute agent with circuit breaker, retry logic, and timeout handling.
        
        The circuit breaker automatically tracks successes and failures.
        """
        # Check circuit breaker state
        if self.circuit_breaker.current_state == 'open':
            self.logger.error(f"{self.name} circuit breaker is OPEN - skipping execution")
            return self._record_error(state, AgentExecutionError("Circuit breaker open"))
        
        # Wrap execution in circuit breaker
        try:
            return self.circuit_breaker.call(self._execute_with_retry_internal, state, max_retries)
        except Exception as e:
            # Circuit breaker will automatically track this failure
            self.logger.error(f"{self.name} execution failed: {e}")
            if self.circuit_breaker.current_state == 'open':
                self.logger.warning(f"{self.name} circuit breaker opened after repeated failures")
            return self._record_error(state, e)
    
    def _execute_with_retry_internal(self, state: Dict[str, Any], max_retries: int) -> Dict[str, Any]:
        """Internal retry logic - called by circuit breaker."""
        start_time = time.time()
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"{self.name} execution attempt {attempt + 1}/{max_retries + 1}")
                
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > AGENT_TIMEOUT:
                    raise AgentTimeoutError(f"{self.name} exceeded timeout of {AGENT_TIMEOUT}s")
                
                # Execute agent
                result = self.execute(state)
                
                # Track metrics
                duration = time.time() - start_time
                self._execution_count += 1
                self._total_duration += duration
                
                self.logger.info(f"{self.name} completed successfully in {duration:.2f}s")
                return result
                
            except Exception as e:
                last_error = e
                self._failure_count += 1
                self.logger.warning(f"{self.name} attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries:
                    # Exponential backoff
                    backoff = RETRY_BACKOFF_FACTOR ** attempt
                    self.logger.info(f"Retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    self.logger.error(f"{self.name} failed after {max_retries + 1} attempts")
        
        # All retries exhausted
        raise AgentExecutionError(
            f"{self.name} failed after {max_retries + 1} attempts. Last error: {str(last_error)}"
        )
    
    def _update_state(self, state: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update state with agent results."""
        state.update(updates)
        
        # Update agent tracking
        state['current_agent'] = self.name
        
        # Add to agent results
        if 'agent_results' not in state:
            state['agent_results'] = {}
        
        state['agent_results'][self.name] = {
            'timestamp': time.time(),
            'status': 'success',
            'data': updates
        }
        
        return state
    
    def _record_error(self, state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        """Record agent error in state."""
        if 'agent_results' not in state:
            state['agent_results'] = {}
            
        state['agent_results'][self.name] = {
            'timestamp': time.time(),
            'status': 'error',
            'error': str(error),
            'error_type': type(error).__name__
        }
        
        # Add to agent messages for visibility
        if 'agent_messages' not in state:
            state['agent_messages'] = []
            
        state['agent_messages'].append({
            'agent': self.name,
            'type': 'error',
            'message': str(error),
            'timestamp': time.time()
        })
        
        return state


