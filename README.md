# Healthcare Agent - Distributed Multi-Agent System

A production-ready healthcare information retrieval system using **LangGraph distributed multi-agent architecture** with RemoteGraph communication. Each agent runs as an independent microservice with Docker deployment.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User / Client                            │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP REST API
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         Supervisor Agent (FastAPI + LangGraph ReAct)        │
│                        Port 8000                             │
│  - Analyzes query intent                                     │
│  - Uses RemoteGraph to call sub-agents                       │
│  - Orchestrates multi-agent workflow                         │
└─────────┬──────────────┬──────────────┬─────────────────────┘
          │              │              │
          │ RemoteGraph  │ RemoteGraph  │ RemoteGraph
          ▼              ▼              ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   PubMed    │  │  WebSearch  │  │  Synthesis  │
│   Agent     │  │   Agent     │  │   Agent     │
│  Port 8001  │  │  Port 8002  │  │  Port 8003  │
│             │  │             │  │             │
│ LangGraph   │  │ LangGraph   │  │ LangGraph   │
│ StateGraph  │  │ StateGraph  │  │ StateGraph  │
│ + Server    │  │ + Server    │  │ + Server    │
└─────────────┘  └─────────────┘  └─────────────┘
```

### Key Components

1. **Supervisor Agent** (Port 8000)
   - FastAPI REST API for external clients
   - LangGraph ReAct agent for intelligent orchestration
   - RemoteGraph tools to communicate with sub-agents
   - Multi-turn conversation support

2. **PubMed Agent** (Port 8001)
   - LangGraph StateGraph: `extract_keywords → search_pubmed → extract_articles`
   - Searches peer-reviewed medical literature (NCBI PubMed)
   - Exposed via LangGraphServer

3. **WebSearch Agent** (Port 8002)
   - LangGraph StateGraph: `search_web → filter_credible → format_markdown`
   - Searches credible health sources (CDC, NIH, Mayo Clinic) via Tavily
   - Exposed via LangGraphServer

4. **Synthesis Agent** (Port 8003)
   - LangGraph StateGraph: `collate_content → synthesize`
   - Merges multi-source information with citations
   - LLM-based synthesis with conflict resolution
   - Exposed via LangGraphServer

### LangGraph Features

✅ **Native StateGraph**: Each agent is a proper LangGraph graph with type-safe state  
✅ **RemoteGraph**: Distributed graph communication (no manual HTTP)  
✅ **Checkpointing**: SqliteSaver for conversation history (upgradeable to Redis)  
✅ **Streaming**: Built-in support for streaming responses  
✅ **Type Safety**: TypedDict states ensure correct data flow  

## 🚀 Quick Start

### Prerequisites

```bash
# 1. Verify Docker installation
docker --version
docker-compose --version

# 2. Set up environment variables
cat > .env << EOF
# Required: At least one LLM API key
GEMINI_API_KEY=your_gemini_api_key_here
# OR
OPENAI_API_KEY=your_openai_api_key_here

# Required: For web search
TAVILY_API_KEY=your_tavily_api_key_here
EOF
```

### Start All Services

```bash
# Make scripts executable
chmod +x start-services.sh stop-services.sh

# Build and start all 4 services
./start-services.sh
```

**What happens:**
1. Builds Docker images for all services
2. Starts containers in dependency order
3. Runs health checks for each service
4. Displays service URLs

**Expected output:**
```
✅ All services are running!

Service URLs:
- Supervisor API: http://localhost:8000
- Supervisor Docs: http://localhost:8000/docs
- PubMed Agent: http://localhost:8001
- WebSearch Agent: http://localhost:8002
- Synthesis Agent: http://localhost:8003
```

### Test the System

**Option 1: Python Test Client**
```bash
python test_api.py "What are the symptoms of diabetes?"
```

**Option 2: cURL**
```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is hypertension and how is it treated?"}'
```

**Option 3: Interactive API Docs**
```bash
open http://localhost:8000/docs
```

### Stop Services

```bash
./stop-services.sh
```

## 📦 Installation (Development)

```bash
# Clone repository
git clone <your-repo-url>
cd healthcare-agent

# Install dependencies with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## 🧪 Testing

### Current Test Status

⚠️ **Tests in `test/` directory are currently BROKEN** - they import the old monolithic architecture which has been removed.

### Test Migration Required

**Option 1: Integration Tests (Recommended)**
Test via HTTP API:

```python
import httpx
import pytest

def test_health_check():
    response = httpx.get("http://localhost:8000/health")
    assert response.status_code == 200

def test_query_endpoint():
    response = httpx.post(
        "http://localhost:8000/query",
        json={"query": "What is diabetes?"},
        timeout=60.0
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert len(data["sources_used"]) > 0
```

**Option 2: Unit Tests for Graphs**

```python
from src.agents.agent_graphs import create_pubmed_graph

def test_pubmed_graph():
    graph = create_pubmed_graph()
    result = graph.invoke({"query": "diabetes treatment"})
    
    assert "keywords" in result
    assert "pmids" in result
    assert "article_markdown" in result
```

**Option 3: Mock RemoteGraph**

```python
from unittest.mock import patch
from src.agents.supervisor_tools import search_pubmed_literature

def test_search_pubmed_tool():
    with patch('src.agents.supervisor_tools.pubmed_graph') as mock_graph:
        mock_graph.invoke.return_value = {
            "keywords": ["test"],
            "article_markdown": "Test article"
        }
        
        result = search_pubmed_literature("diabetes")
        assert result["status"] == "success"
```

## 🔧 Development Guide

### Project Structure

```
healthcare-agent/
├── src/
│   ├── agents/
│   │   ├── agent_graphs.py          # LangGraph StateGraph definitions
│   │   ├── pubmed_service.py        # PubMed LangGraphServer
│   │   ├── websearch_service.py     # WebSearch LangGraphServer
│   │   ├── synthesis_service.py     # Synthesis LangGraphServer
│   │   └── supervisor_tools.py      # RemoteGraph tools
│   ├── models/
│   │   └── state.py                 # State type definitions
│   ├── prompts/
│   │   └── prompt.py                # Centralized prompts
│   ├── utils/
│   │   └── pubmed_utils.py          # PubMed API utilities
│   └── supervisor_api.py            # FastAPI supervisor service
├── test/                            # ⚠️ Needs migration
├── examples/                        # Example usage scripts
├── docker-compose.yml               # Service orchestration
├── Dockerfile.*                     # Individual service Dockerfiles
├── test_api.py                      # API test client
├── start-services.sh                # Start all services
├── stop-services.sh                 # Stop all services
└── README.md                        # This file
```

### Environment Variables

**Required:**
- `GEMINI_API_KEY` or `OPENAI_API_KEY` - LLM provider
- `TAVILY_API_KEY` - Web search provider

**Optional:**
- `PUBMED_SERVICE_URL` - Default: `http://pubmed-agent:8001`
- `WEBSEARCH_SERVICE_URL` - Default: `http://websearch-agent:8002`
- `SYNTHESIS_SERVICE_URL` - Default: `http://synthesis-agent:8003`

### Adding a New Agent

1. **Define the graph** in `src/agents/agent_graphs.py`:
```python
from typing import TypedDict
from langgraph.graph import StateGraph

class MyAgentState(TypedDict):
    input_field: str
    output_field: str

def my_node(state: MyAgentState) -> MyAgentState:
    # Process state
    return {"output_field": "result"}

def create_my_agent_graph():
    graph = StateGraph(MyAgentState)
    graph.add_node("my_node", my_node)
    graph.set_entry_point("my_node")
    graph.set_finish_point("my_node")
    return graph.compile()
```

2. **Create service file** `src/agents/my_service.py`:
```python
from langgraph_server import LangGraphServer
from src.agents.agent_graphs import create_my_agent_graph

if __name__ == "__main__":
    graph = create_my_agent_graph()
    server = LangGraphServer(graph=graph, host="0.0.0.0", port=8004)
    import uvicorn
    uvicorn.run(server.app, host="0.0.0.0", port=8004)
```

3. **Add RemoteGraph tool** in `src/agents/supervisor_tools.py`:
```python
from langgraph.pregel.remote import RemoteGraph
from langchain_core.tools import tool

my_graph = RemoteGraph("my_agent", url=os.getenv("MY_SERVICE_URL", "http://localhost:8004"))

@tool
def call_my_agent(query: str) -> Dict[str, Any]:
    """Call my custom agent."""
    result = my_graph.invoke({"input_field": query})
    return {"status": "success", "data": result}
```

4. **Add to Docker Compose** in `docker-compose.yml`:
```yaml
my-agent:
  build:
    context: .
    dockerfile: Dockerfile.myagent
  ports:
    - "8004:8004"
  environment:
    - GEMINI_API_KEY=${GEMINI_API_KEY}
  networks:
    - healthcare-agent-network
```

5. **Create Dockerfile** `Dockerfile.myagent`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install .
COPY src/ src/
EXPOSE 8004
CMD ["python", "-m", "src.agents.my_service"]
```

### Monitoring & Debugging

**View logs:**
```bash
docker-compose logs -f supervisor    # Supervisor logs
docker-compose logs -f pubmed-agent  # PubMed agent logs
docker-compose logs -f              # All services
```

**Check service health:**
```bash
curl http://localhost:8000/health   # Supervisor
curl http://localhost:8001/health   # PubMed (if health endpoint exists)
```

**Restart a single service:**
```bash
docker-compose restart pubmed-agent
```

## 🎯 Quick Commands Reference

### Docker Operations
```bash
./start-services.sh              # Start all services
./stop-services.sh               # Stop and remove containers
docker-compose up -d             # Start in detached mode
docker-compose down              # Stop services
docker-compose logs -f           # Follow all logs
docker-compose restart supervisor # Restart one service
docker-compose ps                # List running services
```

### Testing
```bash
python test_api.py "your query"  # Test with Python client
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "test"}'
open http://localhost:8000/docs  # Interactive API docs
```

### Development
```bash
uv pip install -e .              # Install in editable mode
uv run python test_api.py        # Run with uv
docker-compose build             # Rebuild images
docker-compose up --build        # Rebuild and start
```

## 🌟 Features

- ✅ **True Multi-Agent**: Each agent is an independent microservice
- ✅ **LangGraph Native**: Uses StateGraph, RemoteGraph, LangGraphServer
- ✅ **Type-Safe**: TypedDict states for all graphs
- ✅ **Checkpointing**: Built-in conversation history via SqliteSaver
- ✅ **Distributed**: Docker-based deployment with service discovery
- ✅ **Scalable**: Each service can scale independently
- ✅ **Resilient**: Health checks and dependency management
- ✅ **RESTful**: Standard HTTP API with OpenAPI docs
- ✅ **Source Attribution**: All responses include citations
- ✅ **Medical Disclaimers**: Automatic disclaimer on all responses

## 📝 API Reference

### POST /query

Submit a healthcare query to the system.

**Request:**
```json
{
  "query": "What are the symptoms of diabetes?",
  "thread_id": "optional-conversation-id"
}
```

**Response:**
```json
{
  "response": "Diabetes symptoms include:\n\n1. **Increased thirst**...",
  "sources_used": ["pubmed", "websearch"],
  "timestamp": "2026-07-15T10:30:00Z"
}
```

### GET /health

Check supervisor service health.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-07-15T10:30:00Z"
}
```

## 🔐 Security & Privacy

- ⚠️ **Not for Clinical Use**: This is a research/educational tool
- 🔒 **No PHI Storage**: No personal health information is stored
- 🌐 **API Keys**: Store in `.env`, never commit to version control
- 📋 **Disclaimers**: All responses include medical disclaimers

## 🐛 Troubleshooting

### Services won't start
```bash
# Check Docker is running
docker ps

# Check ports are available
lsof -i :8000 -i :8001 -i :8002 -i :8003

# View error logs
docker-compose logs
```

### API returns 500 errors
```bash
# Check API keys are set
cat .env

# Check supervisor logs
docker-compose logs supervisor

# Restart services
./stop-services.sh && ./start-services.sh
```

### RemoteGraph connection errors
```bash
# Verify all agents are healthy
curl http://localhost:8001/health  # Should return 200
curl http://localhost:8002/health
curl http://localhost:8003/health

# Check network connectivity
docker-compose exec supervisor ping pubmed-agent
```

## 🚧 Known Issues

- ⚠️ Tests in `test/` directory need migration (see Testing section)
- 🔄 SQLite checkpointer (`:memory:`) resets on restart - upgrade to Redis for production
- 📊 No built-in monitoring/metrics yet (TODO: add Prometheus)
- 🔍 No graph visualization yet (TODO: add LangGraph Studio integration)

## 📚 Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph RemoteGraph](https://langchain-ai.github.io/langgraph/cloud/concepts/remote_graph/)
- [PubMed API](https://www.ncbi.nlm.nih.gov/home/develop/api/)
- [Tavily Search API](https://tavily.com/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure tests pass (after migration)
5. Submit a pull request

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Multi-agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) - LLM framework
- [FastAPI](https://fastapi.tiangolo.com/) - REST API framework
- [Docker](https://www.docker.com/) - Containerization

---

**Note**: This project has been refactored from a monolithic supervisor pattern to a true distributed multi-agent system using LangGraph's RemoteGraph. All old architecture files have been removed. See git history for migration details.
