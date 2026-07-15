"""
Centralized Prompt Management for Healthcare Agent.

This module contains all LLM prompts used across the multi-agent workflow system.
Each prompt is defined as a constant or function to enable easy maintenance and updates.
"""

from typing import Dict, List, Any

# ==============================================================================
# KEYWORD EXTRACTION PROMPT (PubMed Agent)
# ==============================================================================

KEYWORD_EXTRACTION_PROMPT = """Extract the 1-2 MOST relevant medical and scientific keywords from the following query for searching PubMed.

Focus on the core medical concept(s):
- Primary medical condition, disease, or symptom
- Key treatment or drug (if mentioned)

Query: {user_query}

Return ONLY 1 or 2 keywords maximum, comma-separated, without any explanation or additional text.
Example: "diabetes, metformin" or "COVID-19" """

def get_keyword_extraction_prompt(user_query: str) -> str:
    """
    Get keyword extraction prompt for PubMed Agent.
    
    Args:
        user_query: The user's query text
        
    Returns:
        Formatted prompt string
    """
    return KEYWORD_EXTRACTION_PROMPT.format(user_query=user_query)

# ==============================================================================
# QUERY ROUTING PROMPT (Supervisor Agent)
# ==============================================================================

QUERY_ROUTING_PROMPT = """You are a medical information routing assistant. Analyze the user's query and decide which information source(s) to use.

Available agents:
1. **pubmed_agent**: Searches peer-reviewed medical literature (PubMed database)
   - Best for: Clinical trials, research studies, medical mechanisms, scientific evidence
   
2. **websearch_agent**: Searches general health websites (CDC, Mayo Clinic, NIH, WebMD)
   - Best for: General health info, symptoms, treatments, drug information, patient guidelines
   
3. **both_agents**: Use both sources for comprehensive information
   - Best for: Complex topics needing both research evidence and practical information

Previous Conversation:
{conversation_context}

Current User Query: "{user_query}"

Note: If this is a follow-up question, consider the conversation context. If the user is asking for clarification or more details about a previous topic, you may be able to use the same routing as before.

Analyze this query and choose ONE of these routing decisions:
- "pubmed_agent" - if query needs peer-reviewed research
- "websearch_agent" - if query needs general health information
- "both_agents" - if query would benefit from both sources

Respond with ONLY the routing decision (one of the three options above), nothing else."""

def get_query_routing_prompt(user_query: str, conversation_context: str) -> str:
    """
    Get query routing prompt for Supervisor Agent.
    
    Args:
        user_query: The user's current query
        conversation_context: Formatted conversation history
        
    Returns:
        Formatted prompt string
    """
    return QUERY_ROUTING_PROMPT.format(
        user_query=user_query,
        conversation_context=conversation_context
    )

# ==============================================================================
# SYNTHESIS PROMPT (Synthesis Agent)
# ==============================================================================

SYNTHESIS_PROMPT = """You are a medical information synthesis assistant. Based on the user's query and information from multiple sources, provide a comprehensive, evidence-based response.

Previous Conversation Context:
{conversation_context}

Current User Query: {user_query}

Note: If this is a follow-up question referencing previous discussion, acknowledge the context and build upon it.

Information Sources ({sources_str}):

{combined_content}

Your response should:
1. Directly answer the user's query (considering any follow-up context)
2. Synthesize information from all sources
3. Prioritize peer-reviewed (PubMed) sources over web sources when they conflict
4. Cite specific sources (e.g., "According to PubMed article by Smith et al..." or "Mayo Clinic reports...")
5. Highlight key findings and consensus across sources
6. Note any conflicts or disagreements between sources
7. Be clear, concise, and evidence-based

Important: This is for informational purposes only and not medical advice."""

def get_synthesis_prompt(
    user_query: str,
    conversation_context: str,
    sources_str: str,
    combined_content: str
) -> str:
    """
    Get synthesis prompt for Synthesis Agent.
    
    Args:
        user_query: The user's current query
        conversation_context: Formatted conversation history
        sources_str: Comma-separated list of sources (e.g., "PubMed, WebSearch")
        combined_content: Combined content from all sources
        
    Returns:
        Formatted prompt string
    """
    return SYNTHESIS_PROMPT.format(
        user_query=user_query,
        conversation_context=conversation_context,
        sources_str=sources_str,
        combined_content=combined_content
    )

# ==============================================================================
# ADDITIONAL PROMPTS (For Future Use)
# ==============================================================================

# Placeholder for additional prompts as the system evolves
# Examples:
# - Query clarification prompts
# - Follow-up question generation prompts
# - Source credibility assessment prompts
# - Medical disclaimer templates
