import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional
from src.agents.agent_graphs import create_pubmed_graph
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PubMed Agent Service")
graph = create_pubmed_graph()

class InvokeRequest(BaseModel):
    input: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "pubmed-agent"}

@app.post("/runs/stream")
def runs_stream(request: InvokeRequest):
    """LangGraph RemoteGraph endpoint with SSE streaming."""
    import uuid
    def event_stream():
        try:
            run_id = str(uuid.uuid4())
            # Send metadata event
            yield f"event: metadata\n"
            yield f"data: {json.dumps({'run_id': run_id})}\n\n"
            
            # Execute graph
            result = graph.invoke(request.input, config=request.config)
            logger.info(f"PubMed completed: {list(result.keys())}")
            
            # Send values event (what RemoteGraph expects for final state)
            yield f"event: values\n"
            yield f"data: {json.dumps(result)}\n\n"
            
            # Send end event
            yield f"event: end\n"
            yield f"data: {json.dumps({'run_id': run_id})}\n\n"
        except Exception as e:
            logger.error(f"PubMed failed: {e}")
            yield f"event: error\n"
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    logger.info(f"Starting PubMed Agent on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
