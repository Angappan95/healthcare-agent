"""
Integration tests for the distributed multi-agent system.

These tests require Docker services to be running:
- Start services: ./start-services.sh
- Stop services: ./stop-services.sh

Run with: pytest test/test_integration_api.py -v
Skip if services not running: pytest test/test_integration_api.py -v -m "not requires_services"
"""

import pytest
import httpx
from typing import Dict, Any

pytestmark = pytest.mark.requires_services

class TestSupervisorAPI:
    """Integration tests for the Supervisor REST API."""

    def test_health_check(self, supervisor_url: str):
        """Test supervisor health endpoint."""
        try:
            response = httpx.get(f"{supervisor_url}/health", timeout=5.0)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
        except httpx.ConnectError:
            pytest.skip("Supervisor service not running. Start with ./start-services.sh")

    def test_query_endpoint_simple(self, supervisor_url: str, sample_query: str):
        """Test simple query to supervisor API."""
        try:
            response = httpx.post(
                f"{supervisor_url}/query",
                json={"query": sample_query},
                timeout=60.0
            )
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "response" in data
            assert "sources_used" in data
            assert "timestamp" in data
            
            # Verify response content
            assert len(data["response"]) > 0
            assert isinstance(data["sources_used"], list)
            
        except httpx.ConnectError:
            pytest.skip("Supervisor service not running. Start with ./start-services.sh")

    def test_query_endpoint_with_thread_id(self, supervisor_url: str):
        """Test multi-turn conversation with thread_id."""
        try:
            thread_id = "test-thread-123"
            
            # First query
            response1 = httpx.post(
                f"{supervisor_url}/query",
                json={
                    "query": "What is hypertension?",
                    "thread_id": thread_id
                },
                timeout=60.0
            )
            assert response1.status_code == 200
            data1 = response1.json()
            assert "response" in data1
            
            # Follow-up query (should have context)
            response2 = httpx.post(
                f"{supervisor_url}/query",
                json={
                    "query": "What are its symptoms?",
                    "thread_id": thread_id
                },
                timeout=60.0
            )
            assert response2.status_code == 200
            data2 = response2.json()
            assert "response" in data2
            
        except httpx.ConnectError:
            pytest.skip("Supervisor service not running. Start with ./start-services.sh")

    def test_query_endpoint_invalid_request(self, supervisor_url: str):
        """Test error handling for invalid requests."""
        try:
            response = httpx.post(
                f"{supervisor_url}/query",
                json={},  # Missing required 'query' field
                timeout=5.0
            )
            assert response.status_code == 422  # Validation error
            
        except httpx.ConnectError:
            pytest.skip("Supervisor service not running. Start with ./start-services.sh")

class TestServiceEndpoints:
    """Test individual agent service endpoints (optional)."""

    def test_pubmed_service_available(self, pubmed_service_url: str):
        """Test PubMed service is accessible."""
        try:
            # LangGraphServer may not have a standard health endpoint
            # Just test if the service responds
            response = httpx.get(f"{pubmed_service_url}/", timeout=5.0)
            assert response.status_code in [200, 404, 405]  # Service is responding
        except httpx.ConnectError:
            pytest.skip("PubMed service not running")

    def test_websearch_service_available(self, websearch_service_url: str):
        """Test WebSearch service is accessible."""
        try:
            response = httpx.get(f"{websearch_service_url}/", timeout=5.0)
            assert response.status_code in [200, 404, 405]  # Service is responding
        except httpx.ConnectError:
            pytest.skip("WebSearch service not running")

    def test_synthesis_service_available(self, synthesis_service_url: str):
        """Test Synthesis service is accessible."""
        try:
            response = httpx.get(f"{synthesis_service_url}/", timeout=5.0)
            assert response.status_code in [200, 404, 405]  # Service is responding
        except httpx.ConnectError:
            pytest.skip("Synthesis service not running")

class TestEndToEndScenarios:
    """End-to-end test scenarios."""

    @pytest.mark.slow
    def test_medical_literature_query(self, supervisor_url: str):
        """Test query that should route to PubMed."""
        try:
            response = httpx.post(
                f"{supervisor_url}/query",
                json={"query": "What do clinical trials show about metformin efficacy?"},
                timeout=90.0
            )
            assert response.status_code == 200
            data = response.json()
            
            assert "response" in data
            assert len(data["response"]) > 100  # Substantial response
            assert "sources_used" in data
            # Should include PubMed
            # Note: actual routing depends on supervisor logic
            
        except httpx.ConnectError:
            pytest.skip("Supervisor service not running")

    @pytest.mark.slow
    def test_general_health_query(self, supervisor_url: str):
        """Test query that might route to WebSearch."""
        try:
            response = httpx.post(
                f"{supervisor_url}/query",
                json={"query": "What are common flu symptoms?"},
                timeout=90.0
            )
            assert response.status_code == 200
            data = response.json()
            
            assert "response" in data
            assert len(data["response"]) > 50
            assert "sources_used" in data
            
        except httpx.ConnectError:
            pytest.skip("Supervisor service not running")

    @pytest.mark.slow
    def test_complex_multi_agent_query(self, supervisor_url: str):
        """Test complex query requiring multiple agents."""
        try:
            response = httpx.post(
                f"{supervisor_url}/query",
                json={"query": "What are the latest treatments for type 2 diabetes?"},
                timeout=120.0
            )
            assert response.status_code == 200
            data = response.json()
            
            assert "response" in data
            assert len(data["response"]) > 100
            assert "sources_used" in data
            assert len(data["sources_used"]) >= 1  # At least one source
            
        except httpx.ConnectError:
            pytest.skip("Supervisor service not running")
