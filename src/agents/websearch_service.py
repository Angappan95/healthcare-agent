import logging
import sys

import uvicorn

from src.agents.agent_graphs import create_websearch_graph
from src.agents.service_utils import create_agent_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

graph = create_websearch_graph()
app = create_agent_service("WebSearch Agent Service", "websearch-agent", graph)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8002
    logger.info("Starting WebSearch Agent on port %s", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
