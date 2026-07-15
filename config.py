# NCBI BioC API Configuration
NCBI_BIOC_BASE_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pubmed.cgi/BioC_json"
NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

REQUEST_TIMEOUT = 10  # Increased for VPN compatibility
PREVIEW_LENGTH = 200

# Retry configuration for VPN/network resilience
MAX_RETRIES = 2
RETRY_BACKOFF_FACTOR = 2  # Exponential backoff: 1s, 2s, 4s

# LLM Prompt Configuration
ARTICLE_CHAR_LIMIT = 500  # Maximum characters to include per article in LLM prompt
