import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
from src.agents.agent_graphs import create_synthesis_graph
from src.models.state import SynthesisState
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Synthesis Agent Service")
graph = create_synthesis_graph()

class SynthesisRequest(BaseModel):
    query: str
    context: dict[str, Any]

class SynthesisResponse(BaseModel):
    response: str
    sources_used: list[str]
    combined_content: str
    error: str | None = None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "synthesis-agent"}

@app.post("/invoke", response_model=SynthesisResponse)
async def invoke_graph(request: SynthesisRequest):
    try:
        initial_state: SynthesisState = {
            "query": request.query,
            "context": request.context,
            "combined_content": "",
            "response": "",
            "sources_used": [],
            "error": None
        }
        result = graph.invoke(initial_state)
        return SynthesisResponse(**result)
    except Exception as e:
        logger.error(f"Synthesis graph invocation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8003
    uvicorn.run(app, host="0.0.0.0", port=port)
