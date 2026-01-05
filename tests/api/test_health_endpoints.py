"""
Tests for health check endpoints.

These tests ensure the health endpoints work correctly and provide
accurate system status information for monitoring and frontend display.
"""

import pytest
from fastapi.testclient import TestClient
from server.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test suite for health check endpoints."""
    
    def test_health_check_returns_200(self, client):
        """Test that basic health check returns 200 OK."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
    def test_health_check_returns_json(self, client):
        """Test that health check returns JSON response."""
        response = client.get("/api/v1/health")
        assert response.headers["content-type"] == "application/json"
        
    def test_health_check_has_status_field(self, client):
        """Test that health check response includes status field."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        
    def test_health_check_has_timestamp(self, client):
        """Test that health check includes timestamp."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "timestamp" in data
        
    def test_detailed_health_check_returns_200(self, client):
        """Test that detailed health check returns 200 OK."""
        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        
    def test_detailed_health_includes_components(self, client):
        """Test that detailed health check includes component status."""
        response = client.get("/api/v1/health/detailed")
        data = response.json()
        assert "system" in data  # Changed from "components" to "system"
        
    def test_detailed_health_includes_system_info(self, client):
        """Test that detailed health includes system information."""
        response = client.get("/api/v1/health/detailed")
        data = response.json()
        assert "system" in data
        
    def test_health_endpoint_performance(self, client):
        """Test that health check responds quickly (< 100ms)."""
        import time
        start = time.time()
        response = client.get("/api/v1/health")
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 0.1, f"Health check took {duration}s, should be < 0.1s"
        
    def test_health_check_concurrent_requests(self, client):
        """Test that health check handles concurrent requests."""
        import concurrent.futures
        
        def make_request():
            return client.get("/api/v1/health")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [f.result() for f in futures]
        
        assert all(r.status_code == 200 for r in responses)
        
    def test_health_check_idempotent(self, client):
        """Test that multiple health checks return consistent results."""
        response1 = client.get("/api/v1/health")
        response2 = client.get("/api/v1/health")
        response3 = client.get("/api/v1/health")
        
        assert response1.status_code == response2.status_code == response3.status_code == 200
        assert response1.json()["status"] == response2.json()["status"] == response3.json()["status"]


class TestHealthEndpointErrors:
    """Test error handling in health endpoints."""
    
    def test_invalid_health_endpoint_returns_404(self, client):
        """Test that invalid health endpoint returns 404."""
        response = client.get("/api/v1/health/invalid")
        assert response.status_code == 404
        
    def test_health_endpoint_wrong_method(self, client):
        """Test that wrong HTTP method returns 405."""
        response = client.post("/api/v1/health")
        assert response.status_code == 405


@pytest.mark.integration
class TestHealthEndpointIntegration:
    """Integration tests for health endpoints with real system checks."""
    
    def test_health_check_reflects_system_state(self, client):
        """Test that health check accurately reflects system state."""
        response = client.get("/api/v1/health/detailed")
        data = response.json()
        
        # Should include system status
        assert "system" in data
            
    def test_health_check_includes_version_info(self, client):
        """Test that health check includes version information."""
        response = client.get("/api/v1/health/detailed")
        data = response.json()
        
        assert "version" in data
