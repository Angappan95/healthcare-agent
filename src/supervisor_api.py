from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import uuid
import time
import re
from datetime import datetime

from langgraph.prebuilt import create_react_agent
from src.tools.helper import get_llm, activate_fallback_llm, is_rate_limit_error
from src.agents.supervisor_tools import supervisor_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Healthcare Agent Supervisor API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5)
    conversation_id: Optional[str] = None

class SourceDetail(BaseModel):
    agent: str
    urls: List[str] = []
    description: str = ""

class QueryResponse(BaseModel):
    response: str
    sources_used: list[str]  # Keep for backward compatibility
    source_details: List[SourceDetail] = []  # Enhanced source information
    conversation_id: str
    request_id: str
    timestamp: str
    execution_time: Optional[float] = None

from langchain_core.messages import SystemMessage

# Create system message for the agent
system_message = SystemMessage(content="""You are a healthcare information supervisor.

Your job is to answer health questions by gathering information and providing a final answer.

For each query:
1. Call search_web_sources to get general health information
2. Optionally call search_pubmed_literature for research/scientific queries  
3. After gathering information, ALWAYS provide a final answer based on the information you found

IMPORTANT: You must ALWAYS generate a final text response to answer the user's question. 
Do not just call tools - use the information from tools to create a comprehensive answer.""")

def get_supervisor_agent():
    """Get or create supervisor agent with current LLM (supports fallback)."""
    return create_react_agent(
        model=get_llm(),  # Dynamically get LLM to support fallback
        tools=supervisor_tools
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "supervisor"}

def extract_urls(text: str) -> List[str]:
    """Extract URLs from text."""
    url_pattern = r'https?://[^\s<>"\)\]]+'
    return re.findall(url_pattern, text)

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    start = time.time()
    request_id = str(uuid.uuid4())
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    try:
        result = None
        sources_used = []
        source_details: Dict[str, SourceDetail] = {}
        
        # Add run name and metadata for better tracing
        run_config = {
            "configurable": {"thread_id": conversation_id},
            "run_name": f"Healthcare Query: {request.query[:50]}",
            "metadata": {
                "conversation_id": conversation_id,
                "request_id": request_id,
                "query": request.query
            }
        }
        
        try:
            # First attempt with primary LLM
            for event in get_supervisor_agent().stream(
                {"messages": [system_message, ("user", request.query)]},
                config=run_config
            ):
                logger.info(f"Agent event keys: {event.keys()}")
                
                if "agent" in event and (messages := event["agent"].get("messages")):
                    last_message = messages[-1]
                    logger.info(f"Agent message type: {type(last_message).__name__}")
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
                        logger.info(f"Agent produced response: {len(result)} chars")
            
                # Track which tools were called and collect their responses
                if "tools" in event:
                    tool_messages = event.get("tools", {}).get("messages", [])
                    logger.info(f"Tools event: {len(tool_messages)} tool messages")
                    for tool_msg in tool_messages:
                        if hasattr(tool_msg, 'name'):
                            tool_name = tool_msg.name
                            tool_content = str(tool_msg.content) if hasattr(tool_msg, 'content') else ""
                            
                            # Extract URLs from tool response
                            urls = extract_urls(tool_content)
                        
                        if 'pubmed' in tool_name.lower():
                            agent_name = "PubMed Agent"
                            sources_used.append("PubMed")
                            if agent_name not in source_details:
                                source_details[agent_name] = SourceDetail(
                                    agent=agent_name,
                                    urls=[],
                                    description="Scientific literature from NCBI PubMed"
                                )
                            source_details[agent_name].urls.extend(urls)
                            
                        elif 'websearch' in tool_name.lower() or 'web' in tool_name.lower():
                            agent_name = "WebSearch Agent"
                            sources_used.append("WebSearch")
                            if agent_name not in source_details:
                                source_details[agent_name] = SourceDetail(
                                    agent=agent_name,
                                    urls=[],
                                    description="Credible health sources (CDC, NIH, Mayo Clinic)"
                                )
                            source_details[agent_name].urls.extend(urls)
                            
                        elif 'synthesis' in tool_name.lower():
                            agent_name = "Synthesis Agent"
                            sources_used.append("Synthesis")
                            if agent_name not in source_details:
                                source_details[agent_name] = SourceDetail(
                                    agent=agent_name,
                                    urls=[],
                                    description="Multi-source information synthesis"
                                )
                            source_details[agent_name].urls.extend(urls)
        
        except Exception as agent_error:
            # Check if it's a rate limit error
            if is_rate_limit_error(agent_error):
                logger.warning(f"⚠️ Rate limit hit, switching to fallback LLM")
                activate_fallback_llm()
                
                # Retry with fallback LLM
                for event in get_supervisor_agent().stream(
                    {"messages": [system_message, ("user", request.query)]},
                    config=run_config
                ):
                    if "agent" in event and (messages := event["agent"].get("messages")):
                        last_message = messages[-1]
                        if hasattr(last_message, 'content'):
                            content = last_message.content
                            if isinstance(content, list):
                                result = " ".join(
                                    item.get('text', '') if isinstance(item, dict) else str(item)
                                    for item in content
                                )
                            else:
                                result = str(content)
                    
                    if "tools" in event:
                        for tool_msg in event.get("tools", {}).get("messages", []):
                            if hasattr(tool_msg, 'name'):
                                tool_name = tool_msg.name
                                tool_content = str(tool_msg.content) if hasattr(tool_msg, 'content') else ""
                                urls = extract_urls(tool_content)
                                
                                if 'pubmed' in tool_name.lower():
                                    agent_name = "PubMed Agent"
                                    sources_used.append("PubMed")
                                    if agent_name not in source_details:
                                        source_details[agent_name] = SourceDetail(
                                            agent=agent_name,
                                            urls=[],
                                            description="Scientific literature from NCBI PubMed"
                                        )
                                    source_details[agent_name].urls.extend(urls)
                                elif 'websearch' in tool_name.lower() or 'web' in tool_name.lower():
                                    agent_name = "WebSearch Agent"
                                    sources_used.append("WebSearch")
                                    if agent_name not in source_details:
                                        source_details[agent_name] = SourceDetail(
                                            agent=agent_name,
                                            urls=[],
                                            description="Credible health sources (CDC, NIH, Mayo Clinic)"
                                        )
                                    source_details[agent_name].urls.extend(urls)
                                elif 'synthesis' in tool_name.lower():
                                    agent_name = "Synthesis Agent"
                                    sources_used.append("Synthesis")
                                    if agent_name not in source_details:
                                        source_details[agent_name] = SourceDetail(
                                            agent=agent_name,
                                            urls=[],
                                            description="Multi-source information synthesis"
                                        )
                                    source_details[agent_name].urls.extend(urls)
                
                logger.info("✅ Fallback LLM succeeded")
            else:
                # Not a rate limit error, re-raise
                raise agent_error
        
        if not result:
            raise Exception("Agent did not produce a response")
        
        # Deduplicate sources and URLs
        sources_used = list(dict.fromkeys(sources_used)) or ["AI Assistant"]
        for detail in source_details.values():
            detail.urls = list(dict.fromkeys(detail.urls))  # Remove duplicate URLs
        
        return QueryResponse(
            response=result,
            sources_used=sources_used,
            source_details=list(source_details.values()),
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
