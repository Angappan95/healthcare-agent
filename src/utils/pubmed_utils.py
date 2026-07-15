"""
PubMed utilities for searching and extracting articles.

Provides functions to search PubMed and extract article content using NCBI APIs.
"""
import logging
import requests
import xml.etree.ElementTree as ET
import time
from typing import List, Dict, Optional

from src.config import REQUEST_TIMEOUT, NCBI_ESEARCH_URL, NCBI_EFETCH_URL, MAX_RETRIES, RETRY_BACKOFF_FACTOR

logger = logging.getLogger(__name__)

# NCBI recommends identifying your application
HEADERS = {
    'User-Agent': 'PubMedAnalyzer/1.0',
    'Connection': 'close'  # Disable keep-alive for better VPN compatibility
}

def _handle_request_error(error: Exception, context: str = "") -> None:
    """Log request-related errors."""
    if isinstance(error, requests.exceptions.HTTPError):
        status = getattr(error.response, "status_code", "Unknown")
        reason = getattr(error.response, "reason", "Unknown")
        logger.error(f"HTTP {status}: {reason} {context}")
        if status == 404:
            logger.error(f"Resource not found or unavailable {context}")
    elif isinstance(error, requests.exceptions.Timeout):
        logger.error(f"Request timeout after {REQUEST_TIMEOUT}s {context}")
    elif isinstance(error, requests.exceptions.JSONDecodeError):
        logger.error(f"Invalid JSON response {context}")
    else:
        logger.error(f"Request error: {error} {context}")

def _extract_text_from_xml(element) -> Optional[str]:
    """Extract and clean text from an XML element."""
    if element is None:
        return None
    return " ".join(
        text.strip()
        for text in element.itertext()
        if text.strip()
    )

def _extract_ids_from_article(article) -> Dict[str, str]:
    """Extract article identifiers (PMCID, DOI) as case-insensitive dict."""
    return {
        node.attrib.get("IdType", "").lower(): node.text
        for node in article.findall(".//ArticleId")
        if node.text
    }

def _fetch_pmc_full_text(pmcid: str) -> Optional[str]:
    """
    Fetch full text from PMC database for given PMCID.

    Returns full text string on success, None on error.
    """
    try:
        response = requests.get(
            NCBI_EFETCH_URL,
            params={"db": "pmc", "id": pmcid, "retmode": "xml"},
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        body = root.find(".//body")
        return _extract_text_from_xml(body) if body is not None else None
    except (requests.exceptions.RequestException, ET.ParseError) as e:
        logger.warning(f"Failed to fetch PMC text for PMCID {pmcid}: {str(e)}")
        return None

def _build_markdown_for_article(pmid: str, article_data: Dict) -> str:
    """Build markdown formatted string for a single article (optimized for LLM)."""
    lines = [f"## {article_data['title']}\n"]
    
    # Article links for reference
    lines.append(f"**PubMed:** https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n")
    if article_data.get("doi"):
        lines.append(f"**DOI:** https://doi.org/{article_data['doi']}\n")
    
    lines.append("\n")
    
    # Prioritize full text, fallback to abstract
    if article_data.get("full_text"):
        lines.append(f"{article_data['full_text']}\n")
    elif article_data.get("abstract"):
        lines.append(f"{article_data['abstract']}\n")
    else:
        lines.append("No content available.\n")

    return "".join(lines)

def search_articles(query: str, max_results: int = 5) -> List[str]:
    """
    Search PubMed for articles matching the query and return list of PMIDs.

    Args:
        query: Search term/query string
        max_results: Maximum number of results to return (default 5)

    Returns:
        List of PubMed IDs (PMIDs), or empty list on error.
    """
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results
    }

    # Retry logic for VPN/network resilience
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                NCBI_ESEARCH_URL, 
                params=search_params, 
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()

            pmids = response.json().get("esearchresult", {}).get("idlist", [])
            if pmids:
                logger.info(f"Found {len(pmids)} articles for query: '{query}'")
            else:
                logger.warning(f"No articles found for query: '{query}'")
            return pmids

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_BACKOFF_FACTOR ** attempt
                logger.warning(f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
            else:
                _handle_request_error(e, f"for query '{query}' after {MAX_RETRIES} attempts")
                return []

def extract_pubmed_articles_as_markdown(pmids: List[str]) -> str:
    """
    Fetch articles using a list of PMIDs and return as formatted markdown.

    Args:
        pmids: List of PubMed IDs to fetch

    Returns:
        Markdown-formatted string with article details and content.
        Multiple articles are separated by horizontal rules (---).
        Returns empty string if no valid articles found.
    """
    markdown_output = []

    session = requests.Session()
    session.headers.update(HEADERS)

    for pmid in pmids:
        try:
            # Fetch PubMed metadata
            response = session.get(
                NCBI_EFETCH_URL,
                params={"db": "pubmed", "id": pmid, "retmode": "xml"},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            root = ET.fromstring(response.content)
            article = root.find(".//PubmedArticle")

            if article is None:
                error_msg = f"No PubMed article found for PMID {pmid}"
                logger.warning(error_msg)
                markdown_output.append(f"## Error: {error_msg}")
                continue

            # Extract IDs
            ids = _extract_ids_from_article(article)
            pmcid = ids.get("pmc")
            doi = ids.get("doi")

            # Build article data dictionary (optimized - only essential info)
            article_data = {
                "title": _extract_text_from_xml(article.find(".//ArticleTitle")),
                "doi": doi,
                "abstract": _extract_text_from_xml(article.find(".//Abstract")),
                "full_text": _fetch_pmc_full_text(pmcid) if pmcid else None,
            }

            markdown_output.append(_build_markdown_for_article(pmid, article_data))
            logger.info(f"Successfully extracted article for PMID {pmid}")

        except (requests.exceptions.RequestException, ET.ParseError) as e:
            error_msg = f"Error fetching PMID {pmid}: {str(e)}"
            logger.error(error_msg)
            markdown_output.append(f"## {error_msg}")

    result = "\n---\n".join(markdown_output)
    logger.info(f"Extracted markdown for {len([m for m in markdown_output if not m.startswith('## Error')])} articles")
    return result
