import logging
from src.utils.pubmed_utils import extract_pubmed_articles_as_markdown, search_articles

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s"
)
# Ensure root logger captures everything
logging.getLogger().setLevel(logging.INFO)


def test_search_articles_valid_query():
    """Test searching articles with a valid query."""
    result = search_articles("cancer immunotherapy", max_results=5)
    assert isinstance(result, list), "Expected list result"
    assert len(result) > 0, "Expected non-empty list for valid query"
    assert all(isinstance(pmid, str) for pmid in result), "Expected all PMIDs to be strings"


def test_search_articles_invalid_query():
    """Test searching articles with a nonsensical query returns empty list."""
    result = search_articles("xyzabc123notarealterm", max_results=5)
    assert isinstance(result, list), "Expected list result"
    assert len(result) == 0, "Expected empty list for nonsensical query"


def test_extract_pubmed_articles_valid_pmids():
    """Test extraction of articles with valid PMIDs."""
    pmids = ["38743289"]
    result = extract_pubmed_articles_as_markdown(pmids)
    assert isinstance(result, str), "Expected markdown string result"
    assert len(result) > 0, "Expected non-empty markdown for valid PMIDs"
    assert "##" in result, "Expected markdown heading in result"
    assert "PubMed" in result, "Expected PubMed link in result"


def test_extract_pubmed_articles_multiple_pmids():
    """Test extraction with multiple PMIDs."""
    pmids = ["38743289", "38743290"]
    result = extract_pubmed_articles_as_markdown(pmids)
    assert isinstance(result, str), "Expected markdown string result"
    assert len(result) > 0, "Expected non-empty markdown for valid PMIDs"


def test_extract_pubmed_articles_invalid_pmids():
    """Test extraction with invalid PMIDs returns error markdown."""
    pmids = ["99999999999", "88888888888"]
    result = extract_pubmed_articles_as_markdown(pmids)
    assert isinstance(result, str), "Expected markdown string result"
    assert "Error" in result, "Expected error message in result"


def test_search_and_extract_diabetes_articles():
    """Integration test: Search for Diabetes articles and extract as markdown."""
    # Step 1: Search for articles
    pmids = search_articles("Diabetes", max_results=5)
    assert isinstance(pmids, list), "Expected list of PMIDs"
    assert len(pmids) > 0, "Expected to find articles for 'Diabetes'"
    logging.info(f"Found PMIDs: {pmids}")

    # Step 2: Extract articles as markdown
    markdown = extract_pubmed_articles_as_markdown(pmids)
    assert isinstance(markdown, str), "Expected markdown string"
    assert len(markdown) > 0, "Expected non-empty markdown"
    logging.info(f"Found {len(pmids)} articles for 'Diabetes', extracted markdown with {len(markdown)} chars")

    # Step 3: Verify markdown structure
    assert "##" in markdown, "Expected markdown headings"
    assert "PubMed" in markdown, "Expected PubMed link information"
    # Log first 500 chars of result
    preview = markdown[:500].replace('\n', ' ')
    logging.info(f"Markdown preview: {preview}...")
    logging.info("Successfully extracted and verified articles")
