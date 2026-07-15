import os
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from langgraph.pregel.remote import RemoteGraph
import logging

logger = logging.getLogger(__name__)

PUBMED_URL = os.getenv("PUBMED_SERVICE_URL", "http://localhost:8001")
WEBSEARCH_URL = os.getenv("WEBSEARCH_SERVICE_URL", "http://localhost:8002")
SYNTHESIS_URL = os.getenv("SYNTHESIS_SERVICE_URL", "http://localhost:8003")

pubmed_graph = RemoteGraph("pubmed_agent", url=PUBMED_URL)
websearch_graph = RemoteGraph("websearch_agent", url=WEBSEARCH_URL)
synthesis_graph = RemoteGraph("synthesis_agent", url=SYNTHESIS_URL)

@tool
def search_pubmed_literature(query: str) -> Dict[str, Any]:
    """Search peer-reviewed medical literature for scientific evidence, clinical trials, and research studies."""
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
        logger.error(f"PubMed failed: {e}")
        return {"status": "error", "agent": "pubmed_agent", "error": str(e), "data": {}}

@tool
def search_web_sources(query: str) -> Dict[str, Any]:
    """Search credible health websites (CDC, NIH, Mayo Clinic) for symptoms, treatments, and patient guidelines."""
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
        logger.error(f"WebSearch failed: {e}")
        return {"status": "error", "agent": "websearch_agent", "error": str(e), "data": {}}

@tool
def synthesize_information(query: str, pubmed_results: Optional[Dict] = None, 
                          websearch_results: Optional[Dict] = None) -> Dict[str, Any]:
    """Synthesize information from PubMed and WebSearch to generate a comprehensive response."""
    try:
        context = {}
        if pubmed_results:
            context["pubmed_results"] = pubmed_results
        if websearch_results:
            context["websearch_results"] = websearch_results
        
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
        logger.error(f"Synthesis failed: {e}")
        return {"status": "error", "agent": "synthesis_agent", "error": str(e), "data": {}}

supervisor_tools = [
    search_pubmed_literature,
    search_web_sources,
    synthesize_information
]
