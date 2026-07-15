"""
Healthcare Agent - Distributed Multi-Agent System

NOTE: This project now uses a distributed microservices architecture.
To run the system:

1. Start all services:
   ./start-services.sh

2. Test with the API client:
   python test_api.py "What are the latest treatments for COVID-19?"

3. Or call the API directly:
   curl -X POST http://localhost:8000/query \
        -H "Content-Type: application/json" \
        -d '{"query": "What are the symptoms of diabetes?"}'

The old monolithic workflow has been removed in favor of:
- LangGraph-based agents (src/agents/agent_graphs.py)
- Distributed deployment (docker-compose.yml)
- RemoteGraph communication (src/agents/supervisor_tools.py)
- REST API (src/supervisor_api.py)
"""

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("⚠️  This is now a distributed system.")
    print("   Use ./start-services.sh to launch all services.")
    print("   Use test_api.py to interact with the system.")
    print("="*80)
