"""
Demo script to test LLM retry mechanism with exponential backoff.
This script simulates failures and successful retries.
"""
import logging
from unittest.mock import Mock, MagicMock
from src.tools.helper import call_llm_with_retry
from src.config import LLM_MAX_RETRIES, LLM_RETRY_BACKOFF_FACTOR

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_successful_call():
    """Test successful LLM call on first attempt."""
    print("\n=== Test 1: Successful call on first attempt ===")
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "Test response"
    mock_llm.invoke.return_value = mock_response
    
    result = call_llm_with_retry(mock_llm, "test prompt", context="test1")
    assert result.content == "Test response"
    assert mock_llm.invoke.call_count == 1
    print("✓ Success: LLM called once and returned result")

def test_retry_with_eventual_success():
    """Test LLM call that fails twice then succeeds."""
    print("\n=== Test 2: Retry with eventual success (fails 2x, succeeds on 3rd) ===")
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "Success after retries"
    
    # Simulate failures on first two attempts, success on third
    mock_llm.invoke.side_effect = [
        Exception("Timeout error"),
        Exception("Rate limit error"),
        mock_response
    ]
    
    result = call_llm_with_retry(mock_llm, "test prompt", context="test2")
    assert result.content == "Success after retries"
    assert mock_llm.invoke.call_count == 3
    print(f"✓ Success: LLM called {mock_llm.invoke.call_count} times, succeeded on final attempt")

def test_all_retries_exhausted():
    """Test LLM call that fails all retry attempts."""
    print("\n=== Test 3: All retries exhausted (should raise exception) ===")
    mock_llm = Mock()
    mock_llm.invoke.side_effect = Exception("Persistent API error")
    
    try:
        call_llm_with_retry(mock_llm, "test prompt", context="test3")
        assert False, "Should have raised exception"
    except Exception as e:
        assert str(e) == "Persistent API error"
        assert mock_llm.invoke.call_count == LLM_MAX_RETRIES
        print(f"✓ Success: Exception raised after {LLM_MAX_RETRIES} attempts as expected")

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"LLM Retry Configuration Test")
    print(f"Max Retries: {LLM_MAX_RETRIES}")
    print(f"Backoff Factor: {LLM_RETRY_BACKOFF_FACTOR}s (exponential: 1s, 2s, 4s)")
    print(f"{'='*60}")
    
    test_successful_call()
    test_retry_with_eventual_success()
    test_all_retries_exhausted()
    
    print(f"\n{'='*60}")
    print("✓ All retry mechanism tests passed!")
    print(f"{'='*60}\n")
