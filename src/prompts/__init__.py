"""Prompts package - centralized prompt management."""

from src.prompts.prompt import (
    KEYWORD_EXTRACTION_PROMPT,
    QUERY_ROUTING_PROMPT,
    SYNTHESIS_PROMPT,
    get_keyword_extraction_prompt,
    get_query_routing_prompt,
    get_synthesis_prompt,
)

__all__ = [
    'KEYWORD_EXTRACTION_PROMPT',
    'QUERY_ROUTING_PROMPT',
    'SYNTHESIS_PROMPT',
    'get_keyword_extraction_prompt',
    'get_query_routing_prompt',
    'get_synthesis_prompt',
]
