import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agents.agent_graphs import create_websearch_graph
from src.models.state import WebSearchState
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WebSearch Agent Service")
graph = create_websearch_graph()

class WebSearchRequest(BaseModel):
    query: str

class WebSearchResponse(BaseModel):
    search_results: list[dict]
    credible_results: list[dict]
    web_search_markdown: str
    error: str | None = None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "websearch-agent"}

@app.post("/invoke", response_model=WebSearchResponse)
async def invoke_graph(request: WebSearchRequest):
    try:
        initial_state: WebSearchState = {
            "query": request.query,
            "search_results": [],
            "credible_results": [],
            "web_search_markdown": "",
            "error": None
        }
        result = graph.invoke(initial_state)
        return WebSearchResponse(**result)
    except Exception as e:
        logger.error(f"WebSearch graph invocation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
    uvicorn.run(app, host="0.0.0.0", port=port)
