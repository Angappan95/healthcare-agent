"""
Evaluation Test: Answer Relevancy
Tests if the healthcare agent's responses are relevant to the user's queries.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
import requests
import os

class TestAnswerRelevancy:
    """Test suite for evaluating answer relevancy of the healthcare agent."""
    
    @pytest.fixture
    def supervisor_url(self):
        """Get the supervisor API URL from environment or use default."""
        return os.getenv("SUPERVISOR_API_URL", "http://localhost:8000")
    
    def query_agent(self, query: str, supervisor_url: str) -> str:
        """Helper method to query the healthcare agent."""
        try:
            response = requests.post(
                f"{supervisor_url}/query",
                json={"query": query},
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            return f"Error querying agent: {str(e)}"
    
    @pytest.mark.parametrize(
        "query,expected_keywords",
        [
            (
                "What are the symptoms of diabetes?",
                ["symptoms", "diabetes", "blood sugar", "glucose"]
            ),
            (
                "What are the latest treatments for hypertension?",
                ["treatment", "hypertension", "blood pressure", "medication"]
            ),
            (
                "How does COVID-19 affect the respiratory system?",
                ["COVID-19", "respiratory", "lungs", "breathing"]
            ),
        ]
    )
    def test_medical_query_relevancy(self, query, expected_keywords, supervisor_url):
        """Test that responses to medical queries are relevant."""
        # Query the agent
        actual_output = self.query_agent(query, supervisor_url)
        
        # Create test case
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output
        )
        
        # Define metric with threshold
        metric = AnswerRelevancyMetric(threshold=0.7)
        
        # Assert test passes
        assert_test(test_case, [metric])
    
    def test_comparative_medical_query(self, supervisor_url):
        """Test relevancy for comparative medical queries."""
        query = "What is the difference between type 1 and type 2 diabetes?"
        actual_output = self.query_agent(query, supervisor_url)
        
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output
        )
        
        metric = AnswerRelevancyMetric(threshold=0.7)
        assert_test(test_case, [metric])
    
    def test_treatment_protocol_query(self, supervisor_url):
        """Test relevancy for treatment protocol queries."""
        query = "What is the standard treatment protocol for pneumonia?"
        actual_output = self.query_agent(query, supervisor_url)
        
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output
        )
        
        metric = AnswerRelevancyMetric(threshold=0.7)
        assert_test(test_case, [metric])
    
    def test_prevention_query(self, supervisor_url):
        """Test relevancy for prevention-related queries."""
        query = "How can I prevent cardiovascular disease?"
        actual_output = self.query_agent(query, supervisor_url)
        
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output
        )
        
        metric = AnswerRelevancyMetric(threshold=0.7)
        assert_test(test_case, [metric])
