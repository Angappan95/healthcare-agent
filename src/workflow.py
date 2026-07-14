"""LangGraph workflow: PubMed search → keywords → articles → LLM response."""

import logging
from typing import TypedDict, Optional, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from src.utils.pubmed_utils import search_articles, extract_pubmed_articles_as_markdown

logger = logging.getLogger(__name__)

# Lazy-initialize LLM (will be created on first use)
_llm_instance = None

def get_llm():
    """Get or create LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.7)
    return _llm_instance


class WorkflowState(TypedDict):
    """State schema for the PubMed analysis workflow."""
    user_query: str
    extracted_keywords: List[str]
    pmids: List[str]
    article_markdown: str
    response: str


def receive_query(state: WorkflowState) -> WorkflowState:
    """Node 1: Store user query to state."""
    logger.info(f"Received query: {state['user_query']}")
    return state


def extract_keywords(state: WorkflowState) -> WorkflowState:
    """Node 2: Extract keywords from query."""
    # Simple keyword extraction: split and filter stopwords
    stopwords = {'what', 'is', 'are', 'a', 'an', 'the', 'and', 'or', 'of', 'in', 'for', 'to', 'from'}
    words = state['user_query'].lower().split()
    keywords = [w.strip('?.,!') for w in words if w.lower() not in stopwords and len(w) > 2]
    state['extracted_keywords'] = keywords
    logger.info(f"Extracted keywords: {keywords}")
    return state


def search_pubmed_articles(state: WorkflowState) -> WorkflowState:
    """Node 3: Search PubMed for articles using keywords."""
    if not state.get('extracted_keywords'):
        logger.warning("No keywords extracted.")
        state['pmids'] = []
        state['article_markdown'] = "No keywords found."
        return state

    search_query = " OR ".join(state['extracted_keywords'])
    logger.info(f"Searching PubMed: {search_query}")
    pmids = search_articles(search_query, max_results=5)
    state['pmids'] = pmids
    logger.info(f"Found {len(pmids)} articles")

    return state


def extract_article_info(state: WorkflowState) -> WorkflowState:
    """Node 4: Extract article info as markdown."""
    if not state.get('pmids'):
        logger.warning("No PMIDs to extract.")
        state['article_markdown'] = "No articles found."
        return state

    logger.info(f"Extracting articles for {len(state['pmids'])} PMIDs")
    article_markdown = extract_pubmed_articles_as_markdown(state['pmids'])
    state['article_markdown'] = article_markdown
    logger.info("Articles extracted")

    return state


def generate_response(state: WorkflowState) -> WorkflowState:
    """Node 5: Generate response using LLM with query + articles."""
    prompt = f"""Based on the user query and PubMed articles, provide a comprehensive response.

Query: {state['user_query']}
Articles: {state.get('article_markdown', 'No articles found.')[:500]} # Limit to first 500 chars for prompt for testing

Response should:
1. Directly answer the query
2. Reference article information
3. Cite authors and titles
4. Summarize key findings"""

    try:
        logger.info("Generating response...")
        llm = get_llm()
        message = llm.invoke(prompt)
        state['response'] = message.content
        logger.info("Response generated")
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        state['response'] = f"Error: {e}"

    return state


def build_workflow() -> StateGraph:
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(WorkflowState)

    workflow.add_node("receive_query", receive_query)
    workflow.add_node("extract_keywords", extract_keywords)
    workflow.add_node("search_pubmed", search_pubmed_articles)
    workflow.add_node("extract_articles", extract_article_info)
    workflow.add_node("generate_response", generate_response)

    workflow.add_edge(START, "receive_query")
    workflow.add_edge("receive_query", "extract_keywords")
    workflow.add_edge("extract_keywords", "search_pubmed")
    workflow.add_edge("search_pubmed", "extract_articles")
    workflow.add_edge("extract_articles", "generate_response")
    workflow.add_edge("generate_response", END)

    graph = workflow.compile()
    logger.info("Workflow compiled")
    return graph


def run_workflow(user_query: str) -> WorkflowState:
    """Execute the workflow."""
    graph = build_workflow()
    initial_state: WorkflowState = {
        "user_query": user_query,
        "extracted_keywords": [],
        "pmids": [],
        "article_markdown": "",
        "response": ""
    }
    logger.info(f"Starting workflow: {user_query}")
    final_state = graph.invoke(initial_state)
    logger.info("Workflow completed")
    return final_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    result = run_workflow("What are the latest treatments for COVID-19?")
    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"Query: {result['user_query']}\nKeywords: {result['extracted_keywords']}")
    print(f"Articles: {len(result['pmids'])}\nResponse:\n{result['response']}")
