import json
import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class InvokeRequest(BaseModel):
    input: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None


def create_agent_service(app_title: str, service_name: str, graph) -> FastAPI:
    """Create a FastAPI service that exposes a LangGraph SSE endpoint."""
    app = FastAPI(title=app_title)

    @app.get("/health")
    def health_check():
        return {"status": "healthy", "service": service_name}

    @app.post("/runs/stream")
    def runs_stream(request: InvokeRequest):
        """LangGraph RemoteGraph endpoint with SSE streaming."""

        def event_stream():
            try:
                run_id = str(uuid.uuid4())
                yield "event: metadata\n"
                yield f"data: {json.dumps({'run_id': run_id})}\n\n"

                result = graph.invoke(request.input, config=request.config)
                logger.info("%s completed: %s", app_title, list(result.keys()))

                yield "event: values\n"
                yield f"data: {json.dumps(result)}\n\n"

                yield "event: end\n"
                yield f"data: {json.dumps({'run_id': run_id})}\n\n"
            except Exception as exc:
                logger.error("%s failed: %s", app_title, exc)
                yield "event: error\n"
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
