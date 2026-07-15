"""
Pytest configuration for DeepEval tests.
"""

import pytest
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring running services"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

@pytest.fixture(scope="session", autouse=True)
def setup_deepeval_env():
    """Set up DeepEval environment variables."""
    # Ensure API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set - required for DeepEval metrics")
    
    # Set DeepEval configuration
    os.environ.setdefault("DEEPEVAL_RESULTS_FOLDER", "./eval_results")
    

@pytest.fixture(scope="session")
def check_services_running():
    """Check if healthcare agent services are running."""
    import requests
    supervisor_url = os.getenv("SUPERVISOR_API_URL", "http://localhost:8000")
    
    try:
        response = requests.get(f"{supervisor_url}/health", timeout=5)
        if response.status_code != 200:
            pytest.skip("Healthcare agent services not running. Start services with ./start-services.sh")
    except requests.exceptions.RequestException:
        pytest.skip("Healthcare agent services not running. Start services with ./start-services.sh")
