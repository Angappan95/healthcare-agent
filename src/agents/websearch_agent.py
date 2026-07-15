"""WebSearch Agent - Searches web for health information using Tavily API."""

import logging
import os
from typing import Any, Dict, List
from src.agents.base_agent import BaseAgent, AgentExecutionError
from src.config import (
    TAVILY_API_KEY,
    TAVILY_MAX_RESULTS,
    TAVILY_SEARCH_DEPTH,
    CREDIBLE_HEALTH_DOMAINS
)

logger = logging.getLogger(__name__)

class WebSearchAgent(BaseAgent):
    """Specialized agent for web search using Tavily API."""
    
    def __init__(self, api_key: str = None):
        """Initialize WebSearch Agent."""
        super().__init__(
            name="websearch_agent",
            description="Searches web for general health information and clinical guidelines"
        )
        self.api_key = api_key or os.environ.get('TAVILY_API_KEY') or TAVILY_API_KEY
        self._tavily_client = None
        
    def _get_tavily_client(self):
        """Get or create Tavily client instance."""
        if self._tavily_client is None:
            if not self.api_key:
                raise AgentExecutionError(
                    "Tavily API key not configured. Set TAVILY_API_KEY environment variable."
                )
            
            try:
                from tavily import TavilyClient
                self._tavily_client = TavilyClient(api_key=self.api_key)
                self.logger.info("Tavily client initialized")
            except ImportError:
                raise AgentExecutionError(
                    "tavily-python not installed. Run: pip install tavily-python"
                )
            except Exception as e:
                raise AgentExecutionError(f"Failed to initialize Tavily client: {str(e)}")
        
        return self._tavily_client
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search workflow."""
        try:
            user_query = state.get('user_query', '')
            if not user_query:
                raise AgentExecutionError("No user_query provided to WebSearch Agent")
            
            self.logger.info(f"WebSearch Agent processing query: {user_query}")
            
            # Step 1: Refine query for health context
            refined_query = self._refine_query_for_health(user_query)
            self.logger.info(f"Refined query: {refined_query}")
            
            # Step 2: Search using Tavily
            search_results = self._search_tavily(refined_query)
            self.logger.info(f"Found {len(search_results)} web results")
            
            # Step 3: Filter for credible sources
            filtered_results = self._filter_credible_sources(search_results)
            self.logger.info(f"Filtered to {len(filtered_results)} credible sources")
            
            # Step 4: Format as markdown
            markdown = self._format_as_markdown(filtered_results)
            self.logger.info(f"Formatted results ({len(markdown)} chars)")
            
            # Update state with results
            updates = {
                'web_search_markdown': markdown
            }
            
            return self._update_state(state, updates)
            
        except Exception as e:
            self.logger.error(f"WebSearch Agent execution failed: {str(e)}")
            self._record_error(state, e)
            raise AgentExecutionError(f"WebSearch Agent failed: {str(e)}")
    
    def _refine_query_for_health(self, user_query: str) -> str:
        """
        Refine query to focus on health/medical context.
        
        Args:
            user_query: Original user query
            
        Returns:
            Refined query with health context
        """
        # Add health context if not already present
        health_keywords = ['health', 'medical', 'medicine', 'treatment', 'disease', 'symptom', 'diagnosis']
        
        query_lower = user_query.lower()
        has_health_context = any(keyword in query_lower for keyword in health_keywords)
        
        if not has_health_context:
            # Add health context
            refined = f"{user_query} health medical information"
        else:
            refined = user_query
        
        return refined
    
    def _search_tavily(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using Tavily API.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        try:
            client = self._get_tavily_client()
            
            # Perform search
            response = client.search(
                query=query,
                max_results=TAVILY_MAX_RESULTS,
                search_depth=TAVILY_SEARCH_DEPTH,
                include_domains=CREDIBLE_HEALTH_DOMAINS if CREDIBLE_HEALTH_DOMAINS else None
            )
            
            results = response.get('results', [])
            self.logger.debug(f"Tavily returned {len(results)} results")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Tavily search failed: {e}")
            # Return empty results rather than failing completely
            return []
    
    def _filter_credible_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter results to keep only credible health sources.
        
        Args:
            results: Raw search results
            
        Returns:
            Filtered results from credible domains
        """
        if not CREDIBLE_HEALTH_DOMAINS:
            return results  # No filtering if no domains specified
        
        filtered = []
        for result in results:
            url = result.get('url', '').lower()
            
            # Check if URL contains any credible domain
            is_credible = any(domain.lower() in url for domain in CREDIBLE_HEALTH_DOMAINS)
            
            if is_credible:
                result['credibility_score'] = 0.8  # Credible web source
                filtered.append(result)
            else:
                # Include non-credible sources but with lower score
                result['credibility_score'] = 0.5
                filtered.append(result)
        
        # Sort by credibility score (highest first)
        filtered.sort(key=lambda x: x.get('credibility_score', 0), reverse=True)
        
        return filtered
    
    def _format_as_markdown(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results as markdown (matching PubMed format).
        
        Args:
            results: Filtered search results
            
        Returns:
            Markdown-formatted results
        """
        if not results:
            return "No web search results found."
        
        markdown_parts = []
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No Title')
            url = result.get('url', '')
            content = result.get('content', '')
            score = result.get('credibility_score', 0.5)
            
            # Format similar to PubMed articles
            article_md = f"""## Web Result {i}: {title}

**Source:** {url}
**Credibility Score:** {score:.1f}/1.0

### Content
{content[:500]}...

---"""
            
            markdown_parts.append(article_md)
        
        return '\n'.join(markdown_parts)
