"""
Evaluation Test: Faithfulness
Tests if the healthcare agent's responses are faithful to the retrieved context
from PubMed and web sources.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
import requests
import os

class TestFaithfulness:
    """Test suite for evaluating faithfulness to source documents."""
    
    @pytest.fixture
    def supervisor_url(self):
        """Get the supervisor API URL from environment or use default."""
        return os.getenv("SUPERVISOR_API_URL", "http://localhost:8000")
    
    def query_agent_with_context(self, query: str, supervisor_url: str) -> tuple:
        """
        Helper method to query the healthcare agent and retrieve context.
        Returns (response, retrieval_context)
        """
        try:
            response = requests.post(
                f"{supervisor_url}/query",
                json={"query": query},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract response and context
            actual_output = data.get("response", "")
            
            # Extract context from metadata if available
            metadata = data.get("metadata", {})
            retrieval_context = []
            
            # Collect PubMed articles
            if "pubmed_articles" in metadata:
                retrieval_context.extend(metadata["pubmed_articles"])
            
            # Collect web search results
            if "web_results" in metadata:
                retrieval_context.extend(metadata["web_results"])
            
            return actual_output, retrieval_context
            
        except Exception as e:
            return f"Error: {str(e)}", []
    
    def test_diabetes_treatment_faithfulness(self, supervisor_url):
        """Test that diabetes treatment information is faithful to sources."""
        query = "What are the current treatment options for type 2 diabetes?"
        actual_output, retrieval_context = self.query_agent_with_context(
            query, supervisor_url
        )
        
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        # Faithfulness should be high (>0.7) to ensure no hallucinations
        metric = FaithfulnessMetric(threshold=0.7)
        assert_test(test_case, [metric])
    
    def test_medication_side_effects_faithfulness(self, supervisor_url):
        """Test faithfulness when reporting medication side effects."""
        query = "What are the side effects of metformin?"
        actual_output, retrieval_context = self.query_agent_with_context(
            query, supervisor_url
        )
        
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        metric = FaithfulnessMetric(threshold=0.7)
        assert_test(test_case, [metric])
    
    def test_research_findings_faithfulness(self, supervisor_url):
        """Test faithfulness when citing research findings."""
        query = "What does recent research say about vitamin D and COVID-19?"
        actual_output, retrieval_context = self.query_agent_with_context(
            query, supervisor_url
        )
        
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        metric = FaithfulnessMetric(threshold=0.7)
        assert_test(test_case, [metric])
    
    def test_clinical_guidelines_faithfulness(self, supervisor_url):
        """Test faithfulness when reporting clinical guidelines."""
        query = "What are the CDC guidelines for blood pressure management?"
        actual_output, retrieval_context = self.query_agent_with_context(
            query, supervisor_url
        )
        
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        # Higher threshold for clinical guidelines (critical information)
        metric = FaithfulnessMetric(threshold=0.8)
        assert_test(test_case, [metric])
    
    def test_dosage_information_faithfulness(self, supervisor_url):
        """Test faithfulness for medication dosage information."""
        query = "What is the typical dosage of aspirin for heart attack prevention?"
        actual_output, retrieval_context = self.query_agent_with_context(
            query, supervisor_url
        )
        
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        # Very high threshold for dosage info (safety critical)
        metric = FaithfulnessMetric(threshold=0.9)
        assert_test(test_case, [metric])
