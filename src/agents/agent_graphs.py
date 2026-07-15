from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import logging

from src.models.state import PubMedState, WebSearchState, SynthesisState

logger = logging.getLogger(__name__)

def create_pubmed_graph():
    """Create PubMed agent as LangGraph."""
    from src.utils.pubmed_utils import search_articles, extract_pubmed_articles_as_markdown
    from src.tools.helper import get_llm
    from src.prompts.prompt import get_keyword_extraction_prompt
    
    llm = get_llm()
    
    def extract_keywords(state: PubMedState) -> PubMedState:
        try:
            response = llm.invoke(get_keyword_extraction_prompt(state["query"]))
            keywords = [k.strip() for k in response.content.strip().split(',')]
            return {**state, "keywords": keywords}
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return {**state, "keywords": state["query"].split()[:3]}
    
    def search_pubmed(state: PubMedState) -> PubMedState:
        try:
            pmids = search_articles(state["keywords"])
            return {**state, "pmids": pmids}
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return {**state, "pmids": [], "error": str(e)}
    
    def extract_articles(state: PubMedState) -> PubMedState:
        try:
            markdown = extract_pubmed_articles_as_markdown(state["pmids"]) if state["pmids"] else ""
            return {**state, "article_markdown": markdown}
        except Exception as e:
            logger.error(f"Article extraction failed: {e}")
            return {**state, "article_markdown": "", "error": str(e)}
    
    graph = StateGraph(PubMedState)
    graph.add_node("extract_keywords", extract_keywords)
    graph.add_node("search", search_pubmed)
    graph.add_node("extract", extract_articles)
    
    graph.set_entry_point("extract_keywords")
    graph.add_edge("extract_keywords", "search")
    graph.add_edge("search", "extract")
    graph.add_edge("extract", END)
    
    # Compile without checkpointer - services can add it if needed
    return graph.compile()

def create_websearch_graph():
    """Create WebSearch agent as LangGraph."""
    import os
    from src.config import CREDIBLE_HEALTH_DOMAINS, SOURCE_CREDIBILITY
    
    tavily_client = None
    try:
        from tavily import TavilyClient
        if api_key := os.getenv('TAVILY_API_KEY'):
            tavily_client = TavilyClient(api_key=api_key)
    except ImportError:
        logger.error("Tavily not installed")
    
    def search_web(state: WebSearchState) -> WebSearchState:
        if not tavily_client:
            return {**state, "search_results": [], "error": "Tavily not configured"}
        try:
            response = tavily_client.search(
                query=f"{state['query']} health information CDC Mayo Clinic NIH",
                max_results=5,
                search_depth="basic"
            )
            return {**state, "search_results": response.get('results', [])}
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {**state, "search_results": [], "error": str(e)}
    
    def filter_credible(state: WebSearchState) -> WebSearchState:
        results = state["search_results"]
        credible = [r for r in results if any(d in r.get('url', '') for d in CREDIBLE_HEALTH_DOMAINS)]
        return {**state, "credible_results": credible if credible else results[:3]}
    
    def format_markdown(state: WebSearchState) -> WebSearchState:
        results = state["credible_results"]
        if not results:
            return {**state, "web_search_markdown": "No results found."}
        
        markdown = "\n---\n".join(
            f"**{i}. {r.get('title', 'Untitled')}**\n"
            f"*URL:* {r.get('url', '')}\\n"
            f"*Content:* {r.get('content', '')[:300]}...\\n"
            for i, r in enumerate(results, 1)
        )
        return {**state, "web_search_markdown": markdown}
    
    graph = StateGraph(WebSearchState)
    graph.add_node("search", search_web)
    graph.add_node("filter", filter_credible)
    graph.add_node("format", format_markdown)
    
    graph.set_entry_point("search")
    graph.add_edge("search", "filter")
    graph.add_edge("filter", "format")
    graph.add_edge("format", END)
    
    # Compile without checkpointer - services can add it if needed
    return graph.compile()

def create_synthesis_graph():
    """Create Synthesis agent as LangGraph."""
    from src.tools.helper import get_llm
    from src.prompts.prompt import get_synthesis_prompt
    
    llm = get_llm()
    
    def collate_content(state: SynthesisState) -> SynthesisState:
        context = state.get("context", {})
        parts = []
        
        if pm := context.get('pubmed_results', {}).get('article_markdown'):
            parts.append(f"**PubMed Literature:**\n{pm[:2000]}")
        if ws := context.get('websearch_results', {}).get('web_search_markdown'):
            parts.append(f"**Web Sources:**\n{ws[:2000]}")
        
        return {**state, "combined_content": "\n\n---\n\n".join(parts)}
    
    def synthesize(state: SynthesisState) -> SynthesisState:
        if not state.get("combined_content"):
            return {**state, "response": "No information available.", "sources_used": []}
        
        context = state.get("context", {})
        has_pubmed = bool(context.get('pubmed_results', {}).get('article_markdown'))
        has_websearch = bool(context.get('websearch_results', {}).get('web_search_markdown'))
        
        sources = []
        if has_pubmed:
            sources.append("PubMed (peer-reviewed literature)")
        if has_websearch:
            sources.append("Web Search (credible health websites)")
        
        try:
            prompt = get_synthesis_prompt(
                user_query=state["query"],
                combined_content=state["combined_content"],
                sources_str=", ".join(sources),
                conversation_context="[No previous conversation]"
            )
            response = llm.invoke(prompt)
            result = response.content.strip() + (
                "\n\n⚠️ **Medical Disclaimer:** This information is for educational purposes only. "
                "Consult a qualified healthcare professional for medical guidance."
            )
            return {
                **state,
                "response": result,
                "sources_used": ["PubMed" if has_pubmed else None, "WebSearch" if has_websearch else None]
            }
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {**state, "response": f"Error: {str(e)}", "sources_used": [], "error": str(e)}
    
    graph = StateGraph(SynthesisState)
    graph.add_node("collate", collate_content)
    graph.add_node("synthesize", synthesize)
    
    graph.set_entry_point("collate")
    graph.add_edge("collate", "synthesize")
    graph.add_edge("synthesize", END)
    
    # Compile without checkpointer - services can add it if needed
    return graph.compile()
