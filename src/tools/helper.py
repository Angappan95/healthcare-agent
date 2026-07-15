"""Helper utilities for the healthcare agent system."""

import logging
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

_llm_instance = None

def get_llm():
    """
    Get a singleton LLM instance.
    
    Returns:
        ChatGoogleGenerativeAI: Configured LLM instance
    """
    global _llm_instance
    if _llm_instance is None:
        api_key = GEMINI_API_KEY or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it in your .env file or environment variables."
            )
        
        _llm_instance = ChatGoogleGenerativeAI(
            model='gemini-3.1-flash-lite',
            temperature=0,
            max_tokens=1000,
            google_api_key=api_key
        )
    return _llm_instance
