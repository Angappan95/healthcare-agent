"""
Evaluation Test: Contextual Precision
Tests if relevant context is ranked higher than irrelevant context in retrieval.
"""

import pytest
from deepeval import assert_test
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
import requests
import os

class TestContextualPrecision:
    """Test suite for evaluating contextual precision in document retrieval."""
    
    @pytest.fixture
    def supervisor_url(self):
        """Get the supervisor API URL from environment or use default."""
        return os.getenv("SUPERVISOR_API_URL", "http://localhost:8000")
    
    def query_agent_with_context(self, query: str, supervisor_url: str) -> tuple:
        """
        Helper method to query the healthcare agent and retrieve ordered context.
        Returns (response, retrieval_context, expected_output)
        """
        try:
            response = requests.post(
                f"{supervisor_url}/query",
                json={"query": query},
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            actual_output = data.get("response", "")
            
            # Extract context maintaining order
            metadata = data.get("metadata", {})
            retrieval_context = []
            
            # PubMed articles (usually most relevant for medical queries)
            if "pubmed_articles" in metadata:
                retrieval_context.extend(metadata["pubmed_articles"])
            
            # Web search results
            if "web_results" in metadata:
                retrieval_context.extend(metadata["web_results"])
            
            # Expected output is the actual response for precision evaluation
            expected_output = actual_output
            
            return actual_output, retrieval_context, expected_output
            
        except Exception as e:
            return f"Error: {str(e)}", [], ""
    
    def test_specific_disease_query_precision(self, supervisor_url):
        """Test precision for specific disease queries."""
        query = "What causes Alzheimer's disease?"
        actual_output, retrieval_context, expected_output = \
            self.query_agent_with_context(query, supervisor_url)
        
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context
        )
        
        # High precision expected - relevant docs should be ranked higher
        metric = ContextualPrecisionMetric(threshold=0.7)
        assert_test(test_case, [metric])
    
    def test_drug_interaction_query_precision(self, supervisor_url):
        """Test precision for drug interaction queries."""
        query = "What are the interactions between warfarin and ibuprofen?"
        actual_output, retrieval_context, expected_output = \
            self.query_agent_with_context(query, supervisor_url)
        
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context
        )
        
        metric = ContextualPrecisionMetric(threshold=0.7)
        assert_test(test_case, [metric])
    
    def test_symptom_diagnosis_precision(self, supervisor_url):
        """Test precision for symptom-based diagnostic queries."""
        query = "What conditions can cause chronic fatigue and joint pain?"
        actual_output, retrieval_context, expected_output = \
            self.query_agent_with_context(query, supervisor_url)
        
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context
        )
        
        metric = ContextualPrecisionMetric(threshold=0.6)
        assert_test(test_case, [metric])
    
    def test_treatment_efficacy_precision(self, supervisor_url):
        """Test precision for treatment efficacy queries."""
        query = "How effective is immunotherapy for lung cancer?"
        actual_output, retrieval_context, expected_output = \
            self.query_agent_with_context(query, supervisor_url)
        
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context
        )
        
        metric = ContextualPrecisionMetric(threshold=0.7)
        assert_test(test_case, [metric])
    
    def test_preventive_care_precision(self, supervisor_url):
        """Test precision for preventive care queries."""
        query = "What screening tests are recommended for women over 50?"
        actual_output, retrieval_context, expected_output = \
            self.query_agent_with_context(query, supervisor_url)
        
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context
        )
        
        metric = ContextualPrecisionMetric(threshold=0.7)
        assert_test(test_case, [metric])
