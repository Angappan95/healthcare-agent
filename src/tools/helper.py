"""Helper utilities for the healthcare agent system."""

import logging
import os
import time
from typing import Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GEMINI_API_KEY, LLM_REQUEST_TIMEOUT, LLM_MAX_RETRIES, LLM_RETRY_BACKOFF_FACTOR

logger = logging.getLogger(__name__)
_llm_instance = None


def get_llm():
    """Get singleton LLM instance with timeout."""
    global _llm_instance
    if _llm_instance is None:
        api_key = GEMINI_API_KEY or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        _llm_instance = ChatGoogleGenerativeAI(
            model='gemini-3.1-flash-lite',
            temperature=0,
            max_tokens=1000,
            timeout=LLM_REQUEST_TIMEOUT,
            google_api_key=api_key
        )
    return _llm_instance


def call_llm_with_retry(llm: Any, prompt: Any, context: str = "", max_retries: Optional[int] = None) -> Any:
    """Call LLM with exponential backoff retry."""
    max_attempts = max_retries or LLM_MAX_RETRIES
    
    for attempt in range(max_attempts):
        try:
            response = llm.invoke(prompt)
            if attempt > 0:
                logger.info(f"LLM succeeded on attempt {attempt + 1} {context}")
            return response
            
        except Exception as e:
            is_last = attempt == max_attempts - 1
            
            if is_last:
                logger.error(f"LLM failed after {max_attempts} attempts {context}: {e}")
                raise
            
            wait = LLM_RETRY_BACKOFF_FACTOR ** attempt
            logger.warning(f"LLM failed ({type(e).__name__}), retrying in {wait}s {context}")
            time.sleep(wait)
