import os
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from langgraph.pregel.remote import RemoteGraph
import logging

logger = logging.getLogger(__name__)

# Initialize RemoteGraphs for each agent
pubmed_graph = RemoteGraph("pubmed_agent", url=os.getenv("PUBMED_SERVICE_URL", "http://localhost:8001"))
websearch_graph = RemoteGraph("websearch_agent", url=os.getenv("WEBSEARCH_SERVICE_URL", "http://localhost:8002"))
synthesis_graph = RemoteGraph("synthesis_agent", url=os.getenv("SYNTHESIS_SERVICE_URL", "http://localhost:8003"))

@tool
def search_pubmed_literature(query: str) -> Dict[str, Any]:
    """Search peer-reviewed medical literature. Use for scientific evidence, clinical trials, research studies."""
    try:
        result = pubmed_graph.invoke({"query": query})
        return {
            "status": "success",
            "agent": "pubmed_agent",
            "data": {
                "keywords": result.get("keywords", []),
                "pmids": result.get("pmids", []),
                "article_markdown": result.get("article_markdown", ""),
                "article_count": len(result.get("pmids", []))
            }
        }
    except Exception as e:
        logger.error(f"PubMed agent failed: {e}")
        return {"status": "error", "agent": "pubmed_agent", "error": str(e), "data": {}}

@tool
def search_web_sources(query: str) -> Dict[str, Any]:
    """Search credible health websites (CDC, NIH, Mayo Clinic). Use for symptoms, treatments, patient guidelines."""
    try:
        result = websearch_graph.invoke({"query": query})
        return {
            "status": "success",
            "agent": "websearch_agent",
            "data": {
                "search_results": result.get("credible_results", []),
                "web_search_markdown": result.get("web_search_markdown", ""),
                "result_count": len(result.get("credible_results", []))
            }
        }
    except Exception as e:
        logger.error(f"WebSearch agent failed: {e}")
        return {"status": "error", "agent": "websearch_agent", "error": str(e), "data": {}}

@tool
def synthesize_information(query: str, pubmed_results: Optional[Dict] = None, 
                          websearch_results: Optional[Dict] = None) -> Dict[str, Any]:
    """Synthesize information from PubMed/WebSearch to generate comprehensive response."""
    try:
        context = {}
        if pubmed_results:
            context['pubmed_results'] = pubmed_results
        if websearch_results:
            context['websearch_results'] = websearch_results
        
        result = synthesis_graph.invoke({"query": query, "context": context})
        return {
            "status": "success",
            "agent": "synthesis_agent",
            "data": {
                "response": result.get("response", ""),
                "sources_used": result.get("sources_used", [])
            }
        }
    except Exception as e:
        logger.error(f"Synthesis agent failed: {e}")
        return {"status": "error", "agent": "synthesis_agent", "error": str(e), "data": {}}

# Tool collection for LangGraph
supervisor_tools = [
    search_pubmed_literature,
    search_web_sources,
    synthesize_information
]
