"""Synthesis Agent - Collates and synthesizes results from multiple sources."""

import logging
from typing import Any, Dict, List
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from src.agents.base_agent import BaseAgent, AgentExecutionError
from src.config import SOURCE_CREDIBILITY, ARTICLE_CHAR_LIMIT
from src.prompts.prompt import get_synthesis_prompt

logger = logging.getLogger(__name__)

class SynthesisAgent(BaseAgent):
    """Specialized agent for synthesizing multi-source information."""
    
    def __init__(self, llm=None):
        """Initialize Synthesis Agent."""
        super().__init__(
            name="synthesis_agent",
            description="Synthesizes information from multiple sources into coherent response"
        )
        self._llm = llm
        
    def _get_llm(self):
        """Get LLM instance (lazy initialization)."""
        if self._llm is None:
            from src.tools.helper import get_llm
            self._llm = get_llm()
        return self._llm
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute synthesis workflow."""
        try:
            self.logger.info("Synthesis Agent collating results...")
            
            # Step 1: Check what results are available
            agent_results = state.get('agent_results', {})
            pubmed_data = agent_results.get('pubmed_agent', {})
            websearch_data = agent_results.get('websearch_agent', {})
            
            has_pubmed = pubmed_data.get('status') == 'success'
            has_websearch = websearch_data.get('status') == 'success'
            
            self.logger.info(f"Available sources - PubMed: {has_pubmed}, WebSearch: {has_websearch}")
            
            if not has_pubmed and not has_websearch:
                raise AgentExecutionError("No successful agent results to synthesize")
            
            # Step 2: Collate content
            collated_content = self._collate_content(state, has_pubmed, has_websearch)
            
            # Step 3: Generate synthesized response
            response = self._generate_synthesis(state, collated_content, has_pubmed, has_websearch)
            
            # Step 4: Add disclaimers if needed
            final_response = self._add_disclaimers(response, has_pubmed, has_websearch, agent_results)
            
            # Add AI response to messages (add_messages reducer handles appending)
            # Include metadata about sources used
            ai_message = AIMessage(
                content=final_response,
                additional_kwargs={
                    'sources_used': {
                        'pubmed': has_pubmed,
                        'websearch': has_websearch
                    }
                }
            )
            
            # Update state
            updates = {
                'messages': [ai_message],  # add_messages reducer will append this
                'response': final_response,
                'synthesis_complete': True
            }
            
            return self._update_state(state, updates)
            
        except Exception as e:
            self.logger.error(f"Synthesis Agent execution failed: {str(e)}")
            self._record_error(state, e)
            raise AgentExecutionError(f"Synthesis Agent failed: {str(e)}")
    
    def _collate_content(self, state: Dict[str, Any], has_pubmed: bool, has_websearch: bool) -> Dict[str, Any]:
        """
        Collate content from available sources.
        
        Args:
            state: Current state
            has_pubmed: Whether PubMed results available
            has_websearch: Whether WebSearch results available
            
        Returns:
            Dictionary with collated content and metadata
        """
        collated = {
            'sources': [],
            'content': []
        }
        
        # Add PubMed content
        if has_pubmed:
            pubmed_markdown = state.get('article_markdown', '')
            if pubmed_markdown and pubmed_markdown != 'No PubMed articles found.':
                # Limit article content
                articles = pubmed_markdown.split('\n---\n')
                limited_articles = [article[:ARTICLE_CHAR_LIMIT] for article in articles]
                collated['content'].append({
                    'source': 'PubMed',
                    'credibility': SOURCE_CREDIBILITY['pubmed'],
                    'content': '\n---\n'.join(limited_articles)
                })
                collated['sources'].append('PubMed (peer-reviewed literature)')
        
        # Add WebSearch content
        if has_websearch:
            web_markdown = state.get('web_search_markdown', '')
            if web_markdown and web_markdown != 'No web search results found.':
                # Limit web content
                web_articles = web_markdown.split('\n---\n')
                limited_web = [article[:ARTICLE_CHAR_LIMIT] for article in web_articles]
                collated['content'].append({
                    'source': 'WebSearch',
                    'credibility': SOURCE_CREDIBILITY['credible_web'],
                    'content': '\n---\n'.join(limited_web)
                })
                collated['sources'].append('Web Search (health websites)')
        
        return collated
    
    def _format_conversation_history(self, state: Dict[str, Any]) -> str:
        """Format conversation history from messages for synthesis context.
        
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
                formatted.append(f"User asked: {msg.content[:150]}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"We answered: {msg.content[:150]}...")
        
        return "\n".join(formatted) if formatted else "[No previous conversation]"
    
    def _generate_synthesis(self, state: Dict[str, Any], collated: Dict[str, Any], 
                           has_pubmed: bool, has_websearch: bool) -> str:
        """
        Generate synthesized response using LLM.
        
        Args:
            state: Current state
            collated: Collated content from sources
            has_pubmed: PubMed available
            has_websearch: WebSearch available
            
        Returns:
            Synthesized response text
        """
        user_query = state.get('user_query', '')
        sources_str = ', '.join(collated['sources'])
        conversation_context = self._format_conversation_history(state)
        
        # Build content sections
        content_sections = []
        for item in collated['content']:
            content_sections.append(f"=== {item['source']} Results (Credibility: {item['credibility']}) ===\n{item['content']}")
        
        combined_content = '\n\n'.join(content_sections)
        

        # Use centralized prompt
        prompt = get_synthesis_prompt(
            user_query=user_query,
            conversation_context=conversation_context,
            sources_str=sources_str,
            combined_content=combined_content
        )

        try:
            self.logger.info("Generating synthesized response with LLM...")
            llm = self._get_llm()
            message = llm.invoke(prompt)
            
            # Extract response text
            if isinstance(message.content, str):
                response = message.content
            elif isinstance(message.content, list):
                response = message.content[0].get('text', str(message.content))
            else:
                response = str(message.content)
            
            self.logger.info(f"Synthesis generated ({len(response)} chars)")
            return response
            
        except Exception as e:
            self.logger.error(f"LLM synthesis failed: {e}")
            # Fallback: concatenate content without LLM synthesis
            fallback = f"Query: {user_query}\n\nInformation gathered from {sources_str}:\n\n{combined_content}"
            return fallback
    
    def _add_disclaimers(self, response: str, has_pubmed: bool, has_websearch: bool, 
                        agent_results: Dict[str, Any]) -> str:
        """
        Add disclaimers for partial results or errors.
        
        Args:
            response: Base response text
            has_pubmed: PubMed successful
            has_websearch: WebSearch successful
            agent_results: All agent results including errors
            
        Returns:
            Response with disclaimers
        """
        disclaimers = []
        
        # Check for partial results
        if not has_pubmed and has_websearch:
            pubmed_error = agent_results.get('pubmed_agent', {}).get('error', 'unknown error')
            disclaimers.append(
                f"⚠️ Note: PubMed search failed ({pubmed_error}). "
                "Response based only on web search results."
            )
        
        if has_pubmed and not has_websearch:
            websearch_error = agent_results.get('websearch_agent', {}).get('error', 'unknown error')
            disclaimers.append(
                f"⚠️ Note: Web search failed ({websearch_error}). "
                "Response based only on PubMed literature."
            )
        
        # Add medical disclaimer
        disclaimers.append(
            "\n---\n**Medical Disclaimer:** This information is for educational purposes only "
            "and should not be considered medical advice. Always consult with a qualified "
            "healthcare professional for medical concerns."
        )
        
        # Combine response with disclaimers
        if disclaimers:
            return response + '\n\n' + '\n\n'.join(disclaimers)
        
        return response
