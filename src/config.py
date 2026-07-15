import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# NCBI BioC API Configuration
NCBI_BIOC_BASE_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pubmed.cgi/BioC_json"
NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

REQUEST_TIMEOUT = 10  # Increased for VPN compatibility
PREVIEW_LENGTH = 200

# Retry configuration for VPN/network resilience
MAX_RETRIES = 2
RETRY_BACKOFF_FACTOR = 2  # Exponential backoff: 1s, 2s, 4s

# LLM Configuration
LLM_REQUEST_TIMEOUT = 30  # Timeout for LLM API requests (seconds)
LLM_MAX_RETRIES = 3  # Number of retry attempts for LLM API calls
LLM_RETRY_BACKOFF_FACTOR = 2  # Exponential backoff: 1s, 2s, 4s, 8s

# LLM Prompt Configuration
ARTICLE_CHAR_LIMIT = 500  # Maximum characters to include per article in LLM prompt

# API Keys (loaded from .env file)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# Tavily Web Search Configuration
TAVILY_MAX_RESULTS = 5
TAVILY_SEARCH_DEPTH = "basic"  # "basic" or "advanced"

# Multi-Agent Configuration
MAX_AGENT_ITERATIONS = 5  # Prevent infinite routing loops
ENABLE_PARALLEL_AGENTS = True  # Allow parallel execution when appropriate
AGENT_TIMEOUT = 30  # Seconds before agent execution times out

# Checkpointing Configuration
CHECKPOINT_DB_PATH = ".langgraph_checkpoints"  # SQLite database for conversation history

# Source Credibility Scores (for synthesis agent)
SOURCE_CREDIBILITY = {
    "pubmed": 1.0,  # Peer-reviewed literature
    "credible_web": 0.8,  # CDC, NIH, Mayo Clinic, etc.
    "general_web": 0.5  # Other web sources
}

# Credible health domains for web search filtering
CREDIBLE_HEALTH_DOMAINS = [
    "cdc.gov",
    "nih.gov",
    "mayoclinic.org",
    "who.int",
    "health.harvard.edu",
    "clevelandclinic.org",
    "webmd.com",
    "medlineplus.gov"
]
