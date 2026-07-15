"""PubMed Agent - Searches medical literature and extracts article information."""

import logging
from typing import Any, Dict, List
from src.agents.base_agent import BaseAgent, AgentExecutionError
from src.utils.pubmed_utils import search_articles, extract_pubmed_articles_as_markdown
from src.config import ARTICLE_CHAR_LIMIT

logger = logging.getLogger(__name__)

class PubMedAgent(BaseAgent):
    """Specialized agent for searching PubMed medical literature."""
    
    def __init__(self, llm=None):
        """Initialize PubMed Agent."""
        super().__init__(
            name="pubmed_agent",
            description="Searches PubMed for peer-reviewed medical literature"
        )
        self._llm = llm
        
    def _get_llm(self):
        """Get LLM instance (lazy initialization)."""
        if self._llm is None:
            # Import here to avoid circular dependency
            from src.tools.helper import get_llm
            self._llm = get_llm()
        return self._llm
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute PubMed search workflow."""
        try:
            user_query = state.get('user_query', '')
            if not user_query:
                raise AgentExecutionError("No user_query provided to PubMed Agent")
            
            self.logger.info(f"PubMed Agent processing query: {user_query}")
            
            # Step 1: Extract keywords
            keywords = self._extract_keywords(user_query)
            self.logger.info(f"Extracted keywords: {keywords}")
            
            # Step 2: Search PubMed
            pmids = self._search_pubmed(keywords)
            self.logger.info(f"Found {len(pmids)} PubMed articles")
            
            # Step 3: Extract article information
            article_markdown = self._extract_articles(pmids)
            self.logger.info(f"Extracted article information ({len(article_markdown)} chars)")
            
            # Update state with results
            updates = {
                'extracted_keywords': keywords,
                'pmids': pmids,
                'article_markdown': article_markdown
            }
            
            return self._update_state(state, updates)
            
        except Exception as e:
            self.logger.error(f"PubMed Agent execution failed: {str(e)}")
            self._record_error(state, e)
            raise AgentExecutionError(f"PubMed Agent failed: {str(e)}")
    
    def _extract_keywords(self, user_query: str) -> List[str]:
        """Extract 1-2 key medical keywords from query using LLM."""
        try:
            llm = self._get_llm()
            
            prompt = f"""Extract the 1-2 MOST relevant medical and scientific keywords from the following query for searching PubMed.

Focus on the core medical concept(s):
- Primary medical condition, disease, or symptom
- Key treatment or drug (if mentioned)

Query: {user_query}

Return ONLY 1 or 2 keywords maximum, comma-separated, without any explanation or additional text.
Example: "diabetes, metformin" or "COVID-19" """

            self.logger.debug("Extracting keywords using LLM...")
            message = llm.invoke(prompt)
            
            # Extract keywords from LLM response
            response_text = message.content if isinstance(message.content, str) else message.content[0]['text']
            keywords = [k.strip() for k in response_text.split(',') if k.strip()]
            
            return keywords[:2]  # Maximum 2 keywords
            
        except Exception as e:
            self.logger.warning(f"LLM keyword extraction failed: {e}, using fallback")
            # Fallback to simple extraction
            stopwords = {'what', 'is', 'are', 'a', 'an', 'the', 'and', 'or', 'of', 'in', 'for', 'to', 'from'}
            words = user_query.lower().split()
            keywords = [w.strip('?.,!') for w in words if w.lower() not in stopwords and len(w) > 2]
            return keywords[:2]  # Maximum 2 keywords
    
    def _search_pubmed(self, keywords: List[str]) -> List[str]:
        """Search PubMed with keywords."""
        if not keywords:
            self.logger.warning("No keywords provided for PubMed search")
            return []
        
        search_query = " OR ".join(keywords)
        self.logger.debug(f"PubMed search query: {search_query}")
        
        try:
            pmids = search_articles(search_query, max_results=5)
            return pmids
        except Exception as e:
            self.logger.error(f"PubMed search failed: {e}")
            return []
    
    def _extract_articles(self, pmids: List[str]) -> str:
        """Extract article metadata and format as markdown."""
        if not pmids:
            return "No PubMed articles found."
        
        try:
            article_markdown = extract_pubmed_articles_as_markdown(pmids)
            return article_markdown
        except Exception as e:
            self.logger.error(f"Article extraction failed: {e}")
            return f"Error extracting articles: {str(e)}"
