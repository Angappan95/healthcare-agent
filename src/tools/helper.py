"""Helper utilities for the healthcare agent system."""

import logging
import os
import time
from typing import Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import (
    GEMINI_API_KEY,
    LLM_REQUEST_TIMEOUT, LLM_MAX_RETRIES, LLM_RETRY_BACKOFF_FACTOR,
    PRIMARY_LLM_MODEL, PRIMARY_LLM_PROVIDER,
    FALLBACK_LLM_MODEL, FALLBACK_LLM_PROVIDER
)

logger = logging.getLogger(__name__)
_llm_instance = None
_fallback_llm_instance = None
_using_fallback = False


def get_llm(force_fallback: bool = False):
    """Get singleton LLM instance with automatic fallback support.
    
    Args:
        force_fallback: If True, return fallback LLM directly
    
    Returns:
        LLM instance (primary or fallback)
    """
    global _llm_instance, _fallback_llm_instance, _using_fallback
    
    # If fallback is active or forced, return fallback LLM
    if force_fallback or _using_fallback:
        if _fallback_llm_instance is None:
            _fallback_llm_instance = _create_fallback_llm()
        return _fallback_llm_instance
    
    # Otherwise return primary LLM
    if _llm_instance is None:
        _llm_instance = _create_primary_llm()
    return _llm_instance


def _create_primary_llm():
    """Create primary LLM instance with no auto-retry (for fast fallback)."""
    if PRIMARY_LLM_PROVIDER == "google":
        api_key = GEMINI_API_KEY or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        return ChatGoogleGenerativeAI(
            model=PRIMARY_LLM_MODEL,
            temperature=0,
            max_tokens=1000,
            timeout=LLM_REQUEST_TIMEOUT,
            max_retries=0,  # Disable auto-retry for fast fallback
            google_api_key=api_key
        )
    else:
        raise ValueError(f"Unsupported primary LLM provider: {PRIMARY_LLM_PROVIDER}")


def _create_fallback_llm():
    """Create fallback LLM instance (Gemini 2.0 Flash Lite) with no auto-retry."""
    if FALLBACK_LLM_PROVIDER == "google":
        api_key = GEMINI_API_KEY or os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.warning("GEMINI_API_KEY not found, fallback LLM unavailable")
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        logger.info(f"Initializing fallback LLM: {FALLBACK_LLM_MODEL}")
        return ChatGoogleGenerativeAI(
            model=FALLBACK_LLM_MODEL,
            temperature=0,
            max_tokens=1000,
            timeout=LLM_REQUEST_TIMEOUT,
            max_retries=0,  # Disable auto-retry
            google_api_key=api_key
        )
    else:
        raise ValueError(f"Unsupported fallback LLM provider: {FALLBACK_LLM_PROVIDER}")


def activate_fallback_llm():
    """Switch to fallback LLM globally."""
    global _using_fallback
    if not _using_fallback:
        logger.warning(f"⚠️ Activating fallback LLM: {FALLBACK_LLM_MODEL} ({FALLBACK_LLM_PROVIDER})")
        _using_fallback = True


def is_rate_limit_error(exception: Exception) -> bool:
    """Check if exception is a 429 rate limit error."""
    error_str = str(exception).lower()
    error_type = type(exception).__name__.lower()
    
    return (
        "429" in error_str or
        "too many requests" in error_str or
        "rate limit" in error_str or
        "quota" in error_str or
        "resource_exhausted" in error_str or
        "clienterror" in error_type
    )


def call_llm_with_retry(llm: Any, prompt: Any, context: str = "", max_retries: Optional[int] = None) -> Any:
    """Call LLM with exponential backoff retry and automatic fallback on rate limits.
    
    If a 429 rate limit error occurs, automatically switches to fallback LLM (Gemini 2.0 Flash Lite).
    """
    max_attempts = max_retries or LLM_MAX_RETRIES
    
    for attempt in range(max_attempts):
        try:
            response = llm.invoke(prompt)
            if attempt > 0:
                logger.info(f"LLM succeeded on attempt {attempt + 1} {context}")
            return response
            
        except Exception as e:
            is_last = attempt == max_attempts - 1
            
            # Check if this is a rate limit error
            if is_rate_limit_error(e):
                logger.warning(f"⚠️ Rate limit hit (429): {e}")
                
                # Try fallback LLM immediately
                try:
                    logger.info(f"🔄 Switching to fallback LLM: {FALLBACK_LLM_MODEL}")
                    activate_fallback_llm()
                    fallback_llm = get_llm(force_fallback=True)
                    response = fallback_llm.invoke(prompt)
                    logger.info(f"✅ Fallback LLM succeeded {context}")
                    return response
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback LLM also failed: {fallback_error}")
                    if is_last:
                        raise Exception(
                            f"Both primary and fallback LLMs failed. "
                            f"Primary: {e}, Fallback: {fallback_error}"
                        )
            
            # For non-rate-limit errors, use normal retry logic
            if is_last:
                logger.error(f"LLM failed after {max_attempts} attempts {context}: {e}")
                raise
            
            wait = LLM_RETRY_BACKOFF_FACTOR ** attempt
            logger.warning(f"LLM failed ({type(e).__name__}), retrying in {wait}s {context}")
            time.sleep(wait)
