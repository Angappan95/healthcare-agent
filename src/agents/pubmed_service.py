import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agents.agent_graphs import create_pubmed_graph
from src.models.state import PubMedState
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PubMed Agent Service")
graph = create_pubmed_graph()

class PubMedRequest(BaseModel):
    query: str

class PubMedResponse(BaseModel):
    keywords: list[str]
    pmids: list[str]
    article_markdown: str
    error: str | None = None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "pubmed-agent"}

@app.post("/invoke", response_model=PubMedResponse)
async def invoke_graph(request: PubMedRequest):
    try:
        initial_state: PubMedState = {
            "query": request.query,
            "keywords": [],
            "pmids": [],
            "article_markdown": "",
            "error": None
        }
        result = graph.invoke(initial_state)
        return PubMedResponse(**result)
    except Exception as e:
        logger.error(f"PubMed graph invocation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    uvicorn.run(app, host="0.0.0.0", port=port)
