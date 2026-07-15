from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging
import uuid
import time
from datetime import datetime

from langgraph.prebuilt import create_react_agent
from src.tools.helper import get_llm
from src.agents.supervisor_tools import supervisor_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Healthcare Agent Supervisor API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5)
    conversation_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    sources_used: list[str]
    conversation_id: str
    request_id: str
    timestamp: str
    execution_time: Optional[float] = None

from langchain_core.messages import SystemMessage

llm = get_llm()

# Create system message for the agent
system_message = SystemMessage(content="""You are a healthcare information supervisor.

Analyze queries and call:
- search_pubmed_literature: for scientific/research queries
- search_web_sources: for general health information
- synthesize_information: always call at the end to generate final response

For complex queries, use both PubMed and WebSearch before synthesizing.""")

supervisor_agent = create_react_agent(
    model=llm,
    tools=supervisor_tools
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "supervisor"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    start = time.time()
    request_id = str(uuid.uuid4())
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    try:

        result = None
        sources_used = []
        
        for event in supervisor_agent.stream(
            {"messages": [system_message, ("user", request.query)]},
            config={"configurable": {"thread_id": conversation_id}}
        ):
            if "agent" in event and (messages := event["agent"].get("messages")):
                last_message = messages[-1]
                if hasattr(last_message, 'content'):
                    content = last_message.content
                    # Handle both string and list content
                    if isinstance(content, list):
                        # Extract text from message list
                        result = " ".join(
                            item.get('text', '') if isinstance(item, dict) else str(item)
                            for item in content
                        )
                    else:
                        result = str(content)
            
            # Track which tools were called
            if "tools" in event:
                for tool_call in event.get("tools", {}).get("messages", []):
                    if hasattr(tool_call, 'name'):
                        if 'pubmed' in tool_call.name.lower():
                            sources_used.append("PubMed")
                        elif 'websearch' in tool_call.name.lower():
                            sources_used.append("WebSearch")
                        elif 'synthesis' in tool_call.name.lower():
                            sources_used.append("Synthesis")
        
        if not result:
            raise Exception("Agent did not produce a response")
        
        # Deduplicate sources
        sources_used = list(dict.fromkeys(sources_used)) or ["AI Assistant"]
        
        return QueryResponse(
            response=result,
            sources_used=sources_used,
            conversation_id=conversation_id,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            execution_time=time.time() - start
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Healthcare Agent Supervisor",
        "version": "2.0.0",
        "description": "Multi-agent healthcare information retrieval system",
        "endpoints": {
            "POST /query": "Submit a healthcare query",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    import sys
    
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    logger.info(f"Starting Supervisor API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
