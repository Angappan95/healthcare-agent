import logging
import sys

import uvicorn

from src.agents.agent_graphs import create_synthesis_graph
from src.agents.service_utils import create_agent_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

graph = create_synthesis_graph()
app = create_agent_service("Synthesis Agent Service", "synthesis-agent", graph)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8003
    logger.info("Starting Synthesis Agent on port %s", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
