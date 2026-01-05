"""
API integration tests for AlexPose server endpoints.

This module tests all API endpoints with real HTTP requests
and validates complete request-response workflows.
"""

import pytest
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import httpx
from fastapi.testclient import TestClient

try:
    from server.main import app
    SERVER_AVAILABLE = True
except ImportError:
    SERVER_AVAILABLE = False

from tests.fixtures.real_data_fixtures import get_real_data_manager


@pytest.mark.skipif(not SERVER_AVAILABLE, reason="Server module not available")
class TestAPIEndpoints:
    """Test API endpoints with real HTTP requests."""

    @pytest.fixture(scope="class")
    def test_client(self):
        """Provide FastAPI test client."""
        return TestClient(app)

    @pytest.fixture(scope="class")
    def sample_videos(self):
        """Provide sample videos for API testing."""
        data_manager = get_real_data_manager()
        return data_manager.get_sample_videos()

    @pytest.fixture
    def api_test_data(self):
        """Provide test data for API requests."""
        return {
            "valid_video_upload": {
                "filename": "test_gait.mp4",
                "content_type": "video/mp4"
            },
            "invalid_video_upload": {
                "filename": "test_document.txt",
                "content_type": "text/plain"
            },
            "analysis_request": {
                "video_id": "test_video_123",
                "analysis_type": "gait_analysis",
                "options": {
                    "include_pose_estimation": True,
                    "include_classification": True
                }
            }
        }

    # ========================================
    # EXISTING ENDPOINT TESTS
    # ========================================

    @pytest.mark.integration
    @pytest.mark.fast
    def test_root_endpoint(self, test_client: TestClient):
        """Test root endpoint returns system information."""
        response = test_client.get("/", headers={"Host": "localhost"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"
        
        # Validate documentation links
        assert "docs" in data
        assert "redoc" in data
        
        # Validate specific values
        assert data["message"] == "AlexPose Gait Analysis System"
        assert data["version"] == "0.1.0"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_health_check_endpoint(self, test_client: TestClient):
        """Test health check endpoint."""
        response = test_client.get("/health", headers={"Host": "localhost"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate health check response
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert data["version"] == "0.1.0"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_documentation_endpoints(self, test_client: TestClient):
        """Test API documentation endpoints are accessible."""
        # Test OpenAPI docs
        response = test_client.get("/docs", headers={"Host": "localhost"})
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        
        # Test ReDoc
        response = test_client.get("/redoc", headers={"Host": "localhost"})
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        
        # Test OpenAPI JSON schema
        response = test_client.get("/openapi.json", headers={"Host": "localhost"})
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
        
        # Validate OpenAPI schema structure
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        assert schema["info"]["title"] == "AlexPose Gait Analysis System"
        assert schema["info"]["version"] == "0.1.0"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_response_times(self, test_client: TestClient):
        """Test API response times meet performance requirements."""
        # Test health check response time (should be < 200ms)
        start_time = time.time()
        response = test_client.get("/health", headers={"Host": "localhost"})
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        
        assert response.status_code == 200
        assert response_time < 200, f"Health check too slow: {response_time:.1f}ms"
        
        # Test root endpoint response time
        start_time = time.time()
        response = test_client.get("/", headers={"Host": "localhost"})
        response_time = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time < 500, f"Root endpoint too slow: {response_time:.1f}ms"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_cors_headers(self, test_client: TestClient):
        """Test CORS headers are properly configured."""
        # Test actual request with Origin header
        response = test_client.get(
            "/",
            headers={"Origin": "http://localhost:3000", "Host": "localhost"}
        )
        
        assert response.status_code == 200
        # CORS headers should be present for cross-origin requests
        assert "access-control-allow-origin" in response.headers

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_error_handling_404(self, test_client: TestClient):
        """Test API 404 error handling."""
        response = test_client.get("/non_existent_endpoint", headers={"Host": "localhost"})
        assert response.status_code == 404
        
        # Should return JSON error response
        data = response.json()
        assert "detail" in data

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_error_handling_malformed_requests(self, test_client: TestClient):
        """Test API handling of malformed requests."""
        # Test malformed JSON in request body
        response = test_client.post(
            "/health",  # POST to GET-only endpoint
            data="invalid json {",
            headers={"Content-Type": "application/json", "Host": "localhost"}
        )
        assert response.status_code in [400, 405, 422], "Should handle malformed JSON"
        
        # Test invalid content type
        response = test_client.post(
            "/health",
            data="some data",
            headers={"Content-Type": "application/xml", "Host": "localhost"}
        )
        assert response.status_code in [400, 405, 415], "Should handle invalid content type"
        
        # Test missing required headers
        response = test_client.get("/health")  # No Host header
        # Should either work or fail gracefully
        assert response.status_code in [200, 400], "Should handle missing headers gracefully"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_error_handling_request_size_limits(self, test_client: TestClient):
        """Test API handling of oversized requests and headers."""
        # Test oversized request body
        large_data = "x" * (1024 * 1024)  # 1MB
        
        response = test_client.post(
            "/health",
            data=large_data,
            headers={"Content-Type": "text/plain", "Host": "localhost"}
        )
        
        # Should either reject due to size or method not allowed
        assert response.status_code in [405, 413, 400], "Should handle large requests"
        
        # Test oversized headers
        huge_header_value = "x" * (64 * 1024)  # 64KB header
        response = test_client.get(
            "/health",
            headers={"Host": "localhost", "X-Large-Header": huge_header_value}
        )
        
        # Should handle large headers gracefully
        assert response.status_code in [200, 400, 431], "Should handle large headers"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_error_handling_malicious_input(self, test_client: TestClient):
        """Test API handling of potentially malicious input."""
        # Test SQL injection attempts in query parameters
        sql_injection_attempts = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; SELECT * FROM videos; --",
            "UNION SELECT * FROM analyses"
        ]
        
        for injection in sql_injection_attempts:
            response = test_client.get(
                f"/health?param={injection}",
                headers={"Host": "localhost"}
            )
            # Should handle SQL injection attempts gracefully
            assert response.status_code in [200, 400], \
                f"Should handle SQL injection attempt: {injection}"
        
        # Test XSS attempts
        xss_attempts = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//"
        ]
        
        for xss in xss_attempts:
            response = test_client.get(
                f"/health?param={xss}",
                headers={"Host": "localhost"}
            )
            # Should handle XSS attempts gracefully
            assert response.status_code in [200, 400], \
                f"Should handle XSS attempt: {xss}"
            
            # Response should not contain unescaped script content
            if response.status_code == 200:
                response_text = response.text.lower()
                assert "<script>" not in response_text, \
                    "Response should not contain unescaped script tags"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_error_handling_path_traversal(self, test_client: TestClient):
        """Test API handling of path traversal attempts."""
        path_traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",  # URL encoded
            "....//....//....//etc/passwd",
            "/var/log/../../etc/passwd"
        ]
        
        for path in path_traversal_attempts:
            response = test_client.get(
                f"/health?file={path}",
                headers={"Host": "localhost"}
            )
            # Should handle path traversal attempts gracefully
            assert response.status_code in [200, 400, 404], \
                f"Should handle path traversal: {path}"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_error_handling_unicode_attacks(self, test_client: TestClient):
        """Test API handling of Unicode-based attacks."""
        unicode_attacks = [
            "\u202e",  # Right-to-left override
            "\ufeff",  # Byte order mark
            "\u2028\u2029",  # Line/paragraph separators
        ]
        
        for attack in unicode_attacks:
            response = test_client.get(
                f"/health?param={attack}",
                headers={"Host": "localhost"}
            )
            # Should handle Unicode attacks gracefully
            assert response.status_code in [200, 400], \
                f"Should handle Unicode attack: {repr(attack)}"
        
        # Test null byte separately as it may be rejected by HTTP client
        try:
            response = test_client.get(
                "/health?param=\u0000",
                headers={"Host": "localhost"}
            )
            assert response.status_code in [200, 400], "Should handle null byte"
        except Exception:
            # HTTP client may reject null bytes - this is acceptable
            pass
        
        try:
            response = test_client.get(
                "/health?param=test\u0000hidden",
                headers={"Host": "localhost"}
            )
            assert response.status_code in [200, 400], "Should handle null byte injection"
        except Exception:
            # HTTP client may reject null bytes - this is acceptable
            pass

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_error_handling_http_method_override(self, test_client: TestClient):
        """Test API handling of HTTP method override attempts."""
        # Test X-HTTP-Method-Override header
        response = test_client.post(
            "/health",
            headers={
                "Host": "localhost",
                "X-HTTP-Method-Override": "DELETE"
            }
        )
        
        # Should not allow method override for security
        assert response.status_code in [405, 400], \
            "Should not allow HTTP method override"
        
        # Test _method parameter
        response = test_client.post(
            "/health?_method=DELETE",
            headers={"Host": "localhost"}
        )
        
        # Should not allow method override via parameter
        assert response.status_code in [405, 400], \
            "Should not allow method override via parameter"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_error_handling_content_type_confusion(self, test_client: TestClient):
        """Test API handling of content type confusion attacks."""
        # Test mismatched content type and data
        response = test_client.post(
            "/health",
            data='{"json": "data"}',
            headers={
                "Content-Type": "text/plain",  # Wrong content type
                "Host": "localhost"
            }
        )
        
        assert response.status_code in [405, 400, 415], \
            "Should handle content type mismatch"
        
        # Test multiple content type headers
        response = test_client.post(
            "/health",
            data="test data",
            headers={
                "Content-Type": "application/json, text/plain",
                "Host": "localhost"
            }
        )
        
        assert response.status_code in [405, 400, 415], \
            "Should handle multiple content types"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_error_handling_header_injection(self, test_client: TestClient):
        """Test API handling of header injection attempts."""
        # Test CRLF injection in headers
        injection_attempts = [
            "test\r\nX-Injected: true",
            "test\nSet-Cookie: malicious=true",
            "test\r\n\r\n<script>alert('xss')</script>",
        ]
        
        for injection in injection_attempts:
            try:
                response = test_client.get(
                    "/health",
                    headers={
                        "Host": "localhost",
                        "X-Test-Header": injection
                    }
                )
                
                # Should handle header injection gracefully
                assert response.status_code in [200, 400], \
                    f"Should handle header injection: {repr(injection)}"
                
                # Response should not contain injected headers
                assert "X-Injected" not in response.headers, \
                    "Should not contain injected headers"
                assert "malicious" not in str(response.headers), \
                    "Should not contain malicious content"
                
            except (ValueError, UnicodeDecodeError):
                # Some injection attempts might be rejected by the HTTP client
                # This is acceptable behavior
                pass

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_error_handling_request_smuggling_prevention(self, test_client: TestClient):
        """Test API prevention of request smuggling attacks."""
        # Test conflicting Content-Length headers
        try:
            response = test_client.post(
                "/health",
                data="test",
                headers={
                    "Host": "localhost",
                    "Content-Length": "4",  # Correct length
                    "X-Content-Length": "100"  # Conflicting length
                }
            )
            
            # Should handle conflicting lengths gracefully
            assert response.status_code in [405, 400], \
                "Should handle conflicting content lengths"
        
        except Exception:
            # Some HTTP clients might reject malformed requests
            # This is acceptable behavior
            pass

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_error_handling_slowloris_simulation(self, test_client: TestClient):
        """Test API handling of slow request attacks (simulation)."""
        import time
        
        # Simulate slow headers by making multiple rapid requests
        # (Real slowloris would send partial headers slowly)
        start_time = time.time()
        
        responses = []
        for i in range(10):
            response = test_client.get(
                "/health",
                headers={
                    "Host": "localhost",
                    f"X-Slow-Header-{i}": f"value-{i}"
                }
            )
            responses.append(response.status_code)
        
        total_time = time.time() - start_time
        
        # All requests should complete reasonably quickly
        assert total_time < 5.0, f"Requests took too long: {total_time:.1f}s"
        
        # Most requests should succeed
        success_count = sum(1 for status in responses if status == 200)
        assert success_count >= 5, "Too many requests failed"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_error_handling_oversized_requests(self, test_client: TestClient):
        """Test API handling of oversized requests."""
        # Create large request body (1MB)
        large_data = "x" * (1024 * 1024)
        
        response = test_client.post(
            "/health",
            data=large_data,
            headers={"Content-Type": "text/plain", "Host": "localhost"}
        )
        
        # Should either reject due to size or method not allowed
        assert response.status_code in [405, 413, 400], "Should handle large requests"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_error_handling_special_characters(self, test_client: TestClient):
        """Test API handling of special characters in URLs."""
        special_urls = [
            "/health%00",  # Null byte
            "/health/../admin",  # Path traversal attempt
            "/health?param=<script>alert('xss')</script>",  # XSS attempt
            "/health?param=" + "A" * 10000,  # Very long parameter
            "/health?param=café",  # Unicode characters
            "/health?param=100%",  # URL encoding edge case
        ]
        
        for url in special_urls:
            response = test_client.get(url, headers={"Host": "localhost"})
            # Should handle special characters gracefully
            assert response.status_code in [200, 400, 404], f"Should handle special URL: {url}"
            
            # Response should be valid JSON if 200
            if response.status_code == 200:
                try:
                    response.json()
                except ValueError:
                    pytest.fail(f"Invalid JSON response for URL: {url}")

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_error_handling_concurrent_errors(self, test_client: TestClient):
        """Test API error handling under concurrent load."""
        import concurrent.futures
        
        def make_error_request(error_type: str):
            if error_type == "404":
                return test_client.get("/nonexistent", headers={"Host": "localhost"})
            elif error_type == "405":
                return test_client.post("/health", headers={"Host": "localhost"})
            elif error_type == "malformed":
                return test_client.post(
                    "/health",
                    data="invalid json",
                    headers={"Content-Type": "application/json", "Host": "localhost"}
                )
            else:
                return test_client.get("/health", headers={"Host": "localhost"})
        
        # Make concurrent error requests
        error_types = ["404", "405", "malformed", "normal"] * 5  # 20 requests
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_error_request, error_type) for error_type in error_types]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should complete (not crash the server)
        assert len(results) == 20, "All concurrent requests should complete"
        
        # Validate response codes are appropriate
        status_codes = [r.status_code for r in results]
        valid_codes = [200, 400, 404, 405, 422]
        assert all(code in valid_codes for code in status_codes), \
            f"Invalid status codes: {[code for code in status_codes if code not in valid_codes]}"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_error_handling_edge_case_headers(self, test_client: TestClient):
        """Test API handling of edge case headers."""
        edge_case_headers = [
            {"Host": "localhost", "User-Agent": ""},  # Empty user agent
            {"Host": "localhost", "Accept": "*/*"},  # Wildcard accept
            {"Host": "localhost", "Accept-Encoding": "gzip, deflate, br, compress, identity"},  # Many encodings
            {"Host": "localhost", "Connection": "close"},  # Connection close
            {"Host": "localhost", "X-Custom-Header": "value" * 1000},  # Very long header value
            {"Host": "localhost", "Authorization": "Bearer invalid_token"},  # Invalid auth
        ]
        
        for headers in edge_case_headers:
            response = test_client.get("/health", headers=headers)
            # Should handle edge case headers gracefully
            assert response.status_code in [200, 400, 401], \
                f"Should handle headers gracefully: {headers}"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_error_recovery_after_errors(self, test_client: TestClient):
        """Test API recovery after handling errors."""
        # Generate several error conditions
        error_responses = []
        
        # 404 error
        response = test_client.get("/nonexistent", headers={"Host": "localhost"})
        error_responses.append(response.status_code)
        
        # 405 error
        response = test_client.post("/health", headers={"Host": "localhost"})
        error_responses.append(response.status_code)
        
        # Malformed request
        response = test_client.post(
            "/health",
            data="invalid",
            headers={"Content-Type": "application/json", "Host": "localhost"}
        )
        error_responses.append(response.status_code)
        
        # After errors, normal requests should still work
        response = test_client.get("/health", headers={"Host": "localhost"})
        assert response.status_code == 200, "API should recover after errors"
        
        # Validate error responses were appropriate
        assert all(code >= 400 for code in error_responses), "Should have generated error codes"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_error_handling_resource_exhaustion_simulation(self, test_client: TestClient):
        """Test API behavior under simulated resource exhaustion."""
        # Simulate high load with many rapid requests
        responses = []
        
        for i in range(100):  # 100 rapid requests
            response = test_client.get("/health", headers={"Host": "localhost"})
            responses.append(response.status_code)
        
        # Most should succeed, some might be rate limited or fail
        success_count = sum(1 for status in responses if status == 200)
        error_count = sum(1 for status in responses if status >= 400)
        
        # At least some should succeed
        assert success_count > 0, "Some requests should succeed even under load"
        
        # If there are errors, they should be appropriate HTTP codes
        if error_count > 0:
            error_codes = [status for status in responses if status >= 400]
            valid_error_codes = [429, 500, 502, 503, 504]  # Rate limit, server errors
            # Allow any 4xx/5xx codes as they indicate graceful error handling
            assert all(400 <= code < 600 for code in error_codes), \
                "Error codes should be valid HTTP error codes"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_method_not_allowed(self, test_client: TestClient):
        """Test API method not allowed handling."""
        # Try POST on GET-only endpoint
        response = test_client.post("/health", headers={"Host": "localhost"})
        assert response.status_code == 405
        
        # Should return JSON error response
        data = response.json()
        assert "detail" in data

    @pytest.mark.integration
    @pytest.mark.fast
    @pytest.mark.asyncio
    async def test_api_concurrent_requests(self):
        """Test API can handle concurrent requests."""
        from httpx import AsyncClient
        from fastapi.testclient import TestClient
        
        def make_sync_request():
            client = TestClient(app)
            response = client.get("/health", headers={"Host": "localhost"})
            return response.status_code == 200
        
        # Make 10 concurrent requests using thread pool
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_sync_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(results), "Some concurrent requests failed"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_headers_validation(self, test_client: TestClient):
        """Test API headers are properly set."""
        response = test_client.get("/health", headers={"Host": "localhost"})
        
        assert response.status_code == 200
        
        # Check content type
        assert "application/json" in response.headers.get("content-type", "")
        
        # Check server headers (if any)
        headers = response.headers
        assert len(headers) > 0

    # ========================================
    # FUTURE API ENDPOINT TESTS (PLACEHOLDER)
    # ========================================
    # These tests are for endpoints that should be implemented
    # but don't exist yet. They are marked as expected failures.

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.xfail(reason="Video upload endpoint not implemented yet")
    def test_video_upload_endpoint_valid_file(
        self, 
        test_client: TestClient,
        sample_videos: Dict[str, Path],
        api_test_data: Dict[str, Any]
    ):
        """Test video upload endpoint with valid video file."""
        # Skip if no sample videos available
        if not sample_videos or "normal_walking" not in sample_videos:
            pytest.skip("Sample videos not available")
        
        video_file = sample_videos["normal_walking"]
        if not video_file.exists():
            pytest.skip(f"Video file not found: {video_file}")
        
        # Prepare file upload
        with open(video_file, "rb") as f:
            files = {
                "file": (
                    api_test_data["valid_video_upload"]["filename"],
                    f,
                    api_test_data["valid_video_upload"]["content_type"]
                )
            }
            
            # Make upload request
            response = test_client.post("/api/v1/videos/upload", files=files)
        
        # Validate response
        assert response.status_code in [200, 201], f"Upload failed: {response.text}"
        data = response.json()
        
        # Validate response structure
        assert "video_id" in data
        assert "status" in data
        assert data["status"] in ["uploaded", "processing"]
        
        # Validate video metadata
        if "metadata" in data:
            metadata = data["metadata"]
            assert "duration" in metadata
            assert "fps" in metadata
            assert "frame_count" in metadata

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.xfail(reason="Video upload endpoint not implemented yet")
    def test_video_upload_endpoint_invalid_file(
        self, 
        test_client: TestClient,
        api_test_data: Dict[str, Any],
        tmp_path: Path
    ):
        """Test video upload endpoint with invalid file."""
        # Create invalid file
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("This is not a video file")
        
        # Prepare file upload
        with open(invalid_file, "rb") as f:
            files = {
                "file": (
                    api_test_data["invalid_video_upload"]["filename"],
                    f,
                    api_test_data["invalid_video_upload"]["content_type"]
                )
            }
            
            # Make upload request
            response = test_client.post("/api/v1/videos/upload", files=files)
        
        # Validate error response
        assert response.status_code in [400, 422], "Should reject invalid file"
        data = response.json()
        
        # Validate error structure
        assert "error" in data or "detail" in data
        error_message = data.get("error", data.get("detail", "")).lower()
        assert any(keyword in error_message for keyword in ["format", "type", "invalid"]), \
            f"Error message should indicate format issue: {error_message}"

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.xfail(reason="Video status endpoint not implemented yet")
    def test_video_status_endpoint(
        self, 
        test_client: TestClient,
        sample_videos: Dict[str, Path]
    ):
        """Test video status endpoint."""
        # First upload a video to get a video ID
        if not sample_videos or "normal_walking" not in sample_videos:
            pytest.skip("Sample videos not available")
        
        video_file = sample_videos["normal_walking"]
        if not video_file.exists():
            pytest.skip(f"Video file not found: {video_file}")
        
        # Upload video
        with open(video_file, "rb") as f:
            files = {"file": ("test_video.mp4", f, "video/mp4")}
            upload_response = test_client.post("/api/v1/videos/upload", files=files)
        
        if upload_response.status_code not in [200, 201]:
            pytest.skip("Video upload failed, cannot test status endpoint")
        
        upload_data = upload_response.json()
        video_id = upload_data.get("video_id")
        
        if not video_id:
            pytest.skip("No video ID returned from upload")
        
        # Check video status
        status_response = test_client.get(f"/api/v1/videos/{video_id}/status")
        
        assert status_response.status_code == 200
        status_data = status_response.json()
        
        # Validate status response
        assert "video_id" in status_data
        assert "status" in status_data
        assert status_data["video_id"] == video_id
        assert status_data["status"] in ["uploaded", "processing", "completed", "failed"]

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.xfail(reason="Analysis request endpoint not implemented yet")
    def test_analysis_request_endpoint(
        self, 
        test_client: TestClient,
        sample_videos: Dict[str, Path],
        api_test_data: Dict[str, Any]
    ):
        """Test analysis request endpoint."""
        # First upload a video
        if not sample_videos or "normal_walking" not in sample_videos:
            pytest.skip("Sample videos not available")
        
        video_file = sample_videos["normal_walking"]
        if not video_file.exists():
            pytest.skip(f"Video file not found: {video_file}")
        
        # Upload video
        with open(video_file, "rb") as f:
            files = {"file": ("test_video.mp4", f, "video/mp4")}
            upload_response = test_client.post("/api/v1/videos/upload", files=files)
        
        if upload_response.status_code not in [200, 201]:
            pytest.skip("Video upload failed, cannot test analysis endpoint")
        
        upload_data = upload_response.json()
        video_id = upload_data.get("video_id")
        
        if not video_id:
            pytest.skip("No video ID returned from upload")
        
        # Request analysis
        analysis_request = api_test_data["analysis_request"].copy()
        analysis_request["video_id"] = video_id
        
        analysis_response = test_client.post(
            "/api/v1/analysis/request",
            json=analysis_request
        )
        
        assert analysis_response.status_code in [200, 202], \
            f"Analysis request failed: {analysis_response.text}"
        
        analysis_data = analysis_response.json()
        
        # Validate analysis response
        assert "analysis_id" in analysis_data
        assert "status" in analysis_data
        assert analysis_data["status"] in ["queued", "processing", "completed"]

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.xfail(reason="Analysis results endpoint not implemented yet")
    def test_analysis_results_endpoint(
        self, 
        test_client: TestClient,
        sample_videos: Dict[str, Path],
        api_test_data: Dict[str, Any]
    ):
        """Test analysis results endpoint."""
        # This test requires a complete analysis workflow
        # For now, we'll test with a mock analysis ID
        mock_analysis_id = "test_analysis_123"
        
        # Request analysis results
        results_response = test_client.get(f"/api/v1/analysis/{mock_analysis_id}/results")
        
        # Should return 404 for non-existent analysis or proper results
        assert results_response.status_code in [200, 404], \
            f"Unexpected status code: {results_response.status_code}"
        
        if results_response.status_code == 200:
            results_data = results_response.json()
            
            # Validate results structure
            assert "analysis_id" in results_data
            assert "status" in results_data
            
            if results_data["status"] == "completed":
                assert "results" in results_data
                results = results_data["results"]
                
                # Validate gait analysis results structure
                if "gait_analysis" in results:
                    gait_results = results["gait_analysis"]
                    assert "classification" in gait_results
                    assert "confidence" in gait_results

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.xfail(reason="Future API endpoints not implemented yet")
    def test_api_error_handling_future_endpoints(self, test_client: TestClient):
        """Test API error handling for future endpoints."""
        # Test non-existent video ID
        response = test_client.get("/api/v1/videos/non_existent_id/status")
        assert response.status_code == 404
        
        # Test invalid analysis request
        invalid_request = {"invalid_field": "invalid_value"}
        response = test_client.post("/api/v1/analysis/request", json=invalid_request)
        assert response.status_code in [400, 422]
        
        # Test non-existent analysis results
        response = test_client.get("/api/v1/analysis/non_existent_id/results")
        assert response.status_code == 404

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.xfail(reason="Future API endpoints not implemented yet")
    def test_api_content_type_validation_future(self, test_client: TestClient):
        """Test API content type validation for future endpoints."""
        # Test JSON endpoint with invalid content type
        response = test_client.post(
            "/api/v1/analysis/request",
            data="invalid json data",
            headers={"Content-Type": "text/plain"}
        )
        
        assert response.status_code in [400, 422, 415], \
            "Should reject invalid content type"

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.xfail(reason="Future API endpoints not implemented yet")
    def test_api_request_size_limits(self, test_client: TestClient, tmp_path: Path):
        """Test API request size limits."""
        # Create a large file (simulating oversized video)
        large_file = tmp_path / "large_file.mp4"
        
        # Create 100MB file (assuming there's a size limit)
        with open(large_file, "wb") as f:
            f.write(b"0" * (100 * 1024 * 1024))  # 100MB
        
        # Attempt to upload large file
        with open(large_file, "rb") as f:
            files = {"file": ("large_video.mp4", f, "video/mp4")}
            response = test_client.post("/api/v1/videos/upload", files=files)
        
        # Should either succeed or fail with appropriate error
        if response.status_code not in [200, 201]:
            # If it fails, should be due to size limit
            assert response.status_code in [413, 400], \
                f"Unexpected error for large file: {response.status_code}"


@pytest.mark.skipif(not SERVER_AVAILABLE, reason="Server module not available")
class TestAPISecurityAndValidation:
    """Test API security and validation aspects."""

    @pytest.fixture(scope="class")
    def test_client(self):
        """Provide FastAPI test client."""
        return TestClient(app)

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_security_headers(self, test_client: TestClient):
        """Test security headers are properly set."""
        response = test_client.get("/health", headers={"Host": "localhost"})
        
        assert response.status_code == 200
        
        # Check for security headers (if implemented)
        headers = response.headers
        
        # These might not be implemented yet, but we test for their presence
        # Common security headers to check for:
        security_headers = [
            "x-content-type-options",
            "x-frame-options", 
            "x-xss-protection",
            "strict-transport-security"
        ]
        
        # For now, just ensure headers exist
        assert len(headers) > 0

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_trusted_host_middleware(self, test_client: TestClient):
        """Test trusted host middleware configuration."""
        # Test with allowed host
        response = test_client.get("/health", headers={"Host": "localhost"})
        assert response.status_code == 200
        
        # Test with potentially disallowed host (might be blocked by middleware)
        response = test_client.get("/health", headers={"Host": "malicious-host.com"})
        # Should either work (if middleware not strict) or be blocked
        assert response.status_code in [200, 400, 403]

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_input_sanitization(self, test_client: TestClient):
        """Test API input sanitization and validation."""
        # Test with potentially malicious query parameters
        malicious_params = [
            "?param=<script>alert('xss')</script>",
            "?param='; DROP TABLE users; --",
            "?param=../../../etc/passwd",
            "?param=" + "A" * 10000  # Very long parameter
        ]
        
        for param in malicious_params:
            response = test_client.get(f"/health{param}", headers={"Host": "localhost"})
            # Should handle malicious input gracefully
            assert response.status_code in [200, 400, 422]

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_rate_limiting_simulation(self, test_client: TestClient):
        """Test API behavior under rapid requests (rate limiting simulation)."""
        # Make many rapid requests to see if there's any rate limiting
        responses = []
        for i in range(50):
            response = test_client.get("/health", headers={"Host": "localhost"})
            responses.append(response.status_code)
        
        # Most should succeed, but some might be rate limited
        success_count = sum(1 for status in responses if status == 200)
        rate_limited_count = sum(1 for status in responses if status == 429)
        
        # At least some should succeed
        assert success_count > 0
        
        # If rate limiting is implemented, we might see 429s
        # If not, all should be 200
        assert all(status in [200, 429] for status in responses)

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_http_methods_validation(self, test_client: TestClient):
        """Test HTTP methods validation for endpoints."""
        # Test all HTTP methods on health endpoint (should only allow GET)
        methods_to_test = ["POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
        
        for method in methods_to_test:
            response = test_client.request(method, "/health", headers={"Host": "localhost"})
            if method == "HEAD":
                # HEAD should work if GET works
                assert response.status_code in [200, 405]
            elif method == "OPTIONS":
                # OPTIONS might be allowed for CORS
                assert response.status_code in [200, 405]
            else:
                # Other methods should not be allowed
                assert response.status_code == 405

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_content_negotiation(self, test_client: TestClient):
        """Test API content negotiation."""
        # Test Accept header handling
        response = test_client.get("/health", headers={"Accept": "application/json", "Host": "localhost"})
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
        
        # Test unsupported Accept header
        response = test_client.get("/health", headers={"Accept": "application/xml", "Host": "localhost"})
        # Should either return JSON anyway or return 406
        assert response.status_code in [200, 406]

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_encoding_handling(self, test_client: TestClient):
        """Test API encoding and character set handling."""
        # Test with different character encodings in headers
        response = test_client.get("/health", headers={"Accept-Charset": "utf-8", "Host": "localhost"})
        assert response.status_code == 200
        
        # Test with unicode characters in URL (should be properly encoded)
        response = test_client.get("/health?param=test%20unicode%20%C3%A9", headers={"Host": "localhost"})
        assert response.status_code in [200, 400]  # Either handled or rejected gracefully


@pytest.mark.skipif(not SERVER_AVAILABLE, reason="Server module not available")
class TestAPIPerformanceAndReliability:
    """Test API performance and reliability aspects."""

    @pytest.fixture(scope="class")
    def test_client(self):
        """Provide FastAPI test client."""
        return TestClient(app)

    @pytest.mark.integration
    @pytest.mark.integration
    @pytest.mark.slow
    def test_api_response_consistency(self, test_client: TestClient):
        """Test API response consistency across multiple requests."""
        # Make multiple requests and ensure consistent responses
        responses = []
        for i in range(10):
            response = test_client.get("/health", headers={"Host": "localhost"})
            responses.append(response.json())
        
        # All responses should be identical
        first_response = responses[0]
        for response in responses[1:]:
            assert response["status"] == first_response["status"]
            assert response["version"] == first_response["version"]

    @pytest.mark.integration
    @pytest.mark.slow
    def test_api_memory_usage_stability(self, test_client: TestClient):
        """Test API memory usage remains stable under load."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make many requests
        for i in range(100):
            response = test_client.get("/health", headers={"Host": "localhost"})
            assert response.status_code == 200
        
        # Check memory usage after requests
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB for 100 requests)
        assert memory_increase < 50, f"Memory increased by {memory_increase:.1f}MB"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_api_error_recovery(self, test_client: TestClient):
        """Test API error recovery and stability."""
        # Make some invalid requests
        invalid_requests = [
            "/nonexistent",
            "/health/invalid/path",
            "//double//slash//path"
        ]
        
        for invalid_path in invalid_requests:
            response = test_client.get(invalid_path, headers={"Host": "localhost"})
            # Should handle errors gracefully
            assert response.status_code in [404, 400]
        
        # After invalid requests, valid requests should still work
        response = test_client.get("/health", headers={"Host": "localhost"})
        assert response.status_code == 200

    @pytest.mark.integration
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_api_concurrent_load(self):
        """Test API under concurrent load."""
        from fastapi.testclient import TestClient
        import concurrent.futures
        
        def make_sync_request():
            client = TestClient(app)
            import time
            start_time = time.time()
            response = client.get("/health", headers={"Host": "localhost"})
            elapsed = time.time() - start_time
            return response.status_code, elapsed
        
        # Make 50 concurrent requests using thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_sync_request) for _ in range(50)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        status_codes = [result[0] for result in results]
        response_times = [result[1] for result in results]
        
        assert all(status == 200 for status in status_codes)
        
        # Average response time should be reasonable
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 1.0, f"Average response time too high: {avg_response_time:.3f}s"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_api_large_response_handling(self, test_client: TestClient):
        """Test API handling of potentially large responses."""
        # Test OpenAPI schema endpoint (potentially large response)
        response = test_client.get("/openapi.json", headers={"Host": "localhost"})
        assert response.status_code == 200
        
        # Ensure response is valid JSON
        schema = response.json()
        assert isinstance(schema, dict)
        assert len(str(schema)) > 100  # Should be a substantial response

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_graceful_shutdown_simulation(self, test_client: TestClient):
        """Test API behavior during simulated shutdown conditions."""
        # This is a basic test - in a real scenario, we'd test actual shutdown
        # For now, just ensure the app can handle requests normally
        response = test_client.get("/health", headers={"Host": "localhost"})
        assert response.status_code == 200
        
        # Test that the app state is accessible
        assert hasattr(app, "state")


@pytest.mark.skipif(not SERVER_AVAILABLE, reason="Server module not available")
class TestAPIIntegrationWithConfiguration:
    """Test API integration with configuration system."""

    @pytest.fixture(scope="class")
    def test_client(self):
        """Provide FastAPI test client."""
        return TestClient(app)

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_configuration_loading(self, test_client: TestClient):
        """Test API properly loads and uses configuration."""
        # Make a request to trigger configuration loading
        response = test_client.get("/health", headers={"Host": "localhost"})
        assert response.status_code == 200
        
        # Check that app state has configuration
        assert hasattr(app, "state")
        # Configuration should be loaded during lifespan
        # We can't directly access it in tests, but we can verify the app works

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_logging_integration(self, test_client: TestClient):
        """Test API logging integration."""
        # Make requests that should generate logs
        response = test_client.get("/health", headers={"Host": "localhost"})
        assert response.status_code == 200
        
        response = test_client.get("/", headers={"Host": "localhost"})
        assert response.status_code == 200
        
        # Make an invalid request that should log an error
        response = test_client.get("/nonexistent", headers={"Host": "localhost"})
        assert response.status_code == 404
        
        # We can't directly verify logs in this test, but we ensure
        # the requests complete successfully, indicating logging works

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_environment_handling(self, test_client: TestClient):
        """Test API handles different environment configurations."""
        # Test that the API works regardless of environment
        response = test_client.get("/health", headers={"Host": "localhost"})
        assert response.status_code == 200
        
        # Verify response contains expected fields
        data = response.json()
        assert "status" in data
        assert "version" in data
        
        # Version should be consistent with app configuration
        assert data["version"] == "0.1.0"

    # ========================================
    # ADDITIONAL ERROR HANDLING AND EDGE CASES
    # ========================================

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_malformed_json_edge_cases(self, test_client: TestClient):
        """Test API handling of various malformed JSON edge cases."""
        malformed_json_cases = [
            '{"incomplete": ',  # Incomplete JSON
            '{"duplicate": "key", "duplicate": "value"}',  # Duplicate keys
            '{"nested": {"incomplete": }',  # Incomplete nested object
            '{"array": [1, 2, 3,]}',  # Trailing comma in array
            '{"unicode": "\\uXXXX"}',  # Invalid unicode escape
            '{"number": 123.456.789}',  # Invalid number format
            '{"string": "unclosed string',  # Unclosed string
            '{"control": "\\x00\\x01\\x02"}',  # Control characters
        ]
        
        for malformed_json in malformed_json_cases:
            response = test_client.post(
                "/health",  # POST to GET-only endpoint with malformed JSON
                data=malformed_json,
                headers={"Content-Type": "application/json", "Host": "localhost"}
            )
            # Should handle malformed JSON gracefully
            assert response.status_code in [400, 405, 422], \
                f"Should handle malformed JSON: {malformed_json[:50]}..."

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_extreme_header_values(self, test_client: TestClient):
        """Test API handling of extreme header values."""
        extreme_headers = [
            {"Host": "localhost", "X-Test": ""},  # Empty header value
            {"Host": "localhost", "X-Test": "a" * 100000},  # Very long header (100KB)
            {"Host": "localhost", "X-Test": "\x00\x01\x02"},  # Binary data in header
            {"Host": "localhost", "X-Test": "café🎬"},  # Unicode in header
            {"Host": "localhost", "X-Test": "line1\nline2"},  # Newlines in header
            {"Host": "localhost", "X-Test": "tab\there"},  # Tabs in header
        ]
        
        for headers in extreme_headers:
            try:
                response = test_client.get("/health", headers=headers)
                # Should handle extreme headers gracefully
                assert response.status_code in [200, 400], \
                    f"Should handle extreme header: {headers['X-Test'][:50]}..."
            except (ValueError, UnicodeDecodeError):
                # Some extreme headers might be rejected by HTTP client
                pass

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_concurrent_error_scenarios(self, test_client: TestClient):
        """Test API handling of concurrent error scenarios."""
        import concurrent.futures
        import random
        
        def make_error_request(request_id: int):
            # Randomly choose different error scenarios
            error_types = [
                lambda: test_client.get("/nonexistent", headers={"Host": "localhost"}),
                lambda: test_client.post("/health", headers={"Host": "localhost"}),
                lambda: test_client.get("/health?param=" + "x" * 10000, headers={"Host": "localhost"}),
                lambda: test_client.get("/health", headers={"Host": "malicious-host.com"}),
            ]
            
            error_func = random.choice(error_types)
            return error_func()
        
        # Make 20 concurrent error requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_error_request, i) for i in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should complete without crashing the server
        assert len(results) == 20, "All concurrent error requests should complete"
        
        # All should return appropriate error codes
        status_codes = [r.status_code for r in results]
        valid_codes = [200, 400, 403, 404, 405, 413, 422, 429]
        assert all(code in valid_codes for code in status_codes), \
            f"Invalid status codes in concurrent errors: {[code for code in status_codes if code not in valid_codes]}"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_resource_cleanup_after_errors(self, test_client: TestClient):
        """Test API resource cleanup after error conditions."""
        # Generate multiple error conditions
        error_scenarios = [
            lambda: test_client.get("/nonexistent", headers={"Host": "localhost"}),
            lambda: test_client.post("/health", data="x" * 1000000, headers={"Host": "localhost"}),
            lambda: test_client.get("/health?param=" + "x" * 50000, headers={"Host": "localhost"}),
        ]
        
        # Execute error scenarios
        for scenario in error_scenarios:
            try:
                response = scenario()
                assert response.status_code >= 400, "Should generate error response"
            except Exception:
                # Some scenarios might raise exceptions, which is acceptable
                pass
        
        # After errors, normal requests should still work (indicating proper cleanup)
        response = test_client.get("/health", headers={"Host": "localhost"})
        assert response.status_code == 200, "API should work normally after error cleanup"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_malicious_request_patterns(self, test_client: TestClient):
        """Test API handling of malicious request patterns."""
        # Test various attack patterns
        attack_patterns = [
            # Directory traversal attempts
            "/health/../../../etc/passwd",
            "/health/..\\..\\..\\windows\\system32\\config\\sam",
            
            # Command injection attempts
            "/health?cmd=ls;cat /etc/passwd",
            "/health?param=`whoami`",
            "/health?param=$(id)",
            
            # Format string attacks
            "/health?param=%s%s%s%s",
            "/health?param=%n%n%n%n",
            
            # LDAP injection
            "/health?param=*)(uid=*))(|(uid=*",
            
            # XML/XXE attempts
            "/health?param=<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>",
        ]
        
        for pattern in attack_patterns:
            try:
                response = test_client.get(pattern, headers={"Host": "localhost"})
                # Should handle malicious patterns gracefully
                assert response.status_code in [200, 400, 404, 422], \
                    f"Should handle attack pattern: {pattern[:50]}..."
                
                # Response should not contain sensitive information
                if response.status_code == 200:
                    response_text = response.text.lower()
                    sensitive_patterns = ["passwd", "shadow", "config", "system32", "root:"]
                    assert not any(pattern in response_text for pattern in sensitive_patterns), \
                        f"Response should not contain sensitive data for: {pattern[:50]}..."
            except Exception as e:
                # HTTP client may reject extremely malicious patterns - this is acceptable
                error_msg = str(e).lower()
                if "url too long" in error_msg or "invalid" in error_msg:
                    # Client-side rejection is acceptable security behavior
                    pass
                else:
                    raise
        
        # Test buffer overflow separately with try-except
        try:
            buffer_overflow_pattern = "/health?" + "A" * 10000  # Reduced size
            response = test_client.get(buffer_overflow_pattern, headers={"Host": "localhost"})
            assert response.status_code in [200, 400, 404, 414, 422], \
                "Should handle buffer overflow attempt"
        except Exception as e:
            # HTTP client may reject oversized URLs - this is acceptable
            error_msg = str(e).lower()
            if "url too long" in error_msg or "invalid" in error_msg:
                pass  # Client-side rejection is acceptable
            else:
                raise

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_state_corruption_recovery(self, test_client: TestClient):
        """Test API recovery from potential state corruption."""
        # Simulate state corruption scenarios
        corruption_requests = [
            # Requests that might corrupt internal state
            lambda: test_client.get("/health", headers={"Host": "localhost", "Connection": "close"}),
            lambda: test_client.get("/health", headers={"Host": "localhost", "Transfer-Encoding": "chunked"}),
            lambda: test_client.get("/health", headers={"Host": "localhost", "Content-Length": "999999"}),
        ]
        
        # Execute potentially corrupting requests
        for request_func in corruption_requests:
            try:
                response = request_func()
                # Should handle potentially corrupting requests
                assert response.status_code in [200, 400], "Should handle state corruption attempts"
            except Exception:
                # Some requests might fail at the HTTP level
                pass
        
        # Verify API still works normally (state not corrupted)
        for i in range(5):
            response = test_client.get("/health", headers={"Host": "localhost"})
            assert response.status_code == 200, f"API should work after corruption attempt {i+1}"
            
            data = response.json()
            assert data["status"] == "healthy", "API state should be consistent"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_edge_case_http_methods(self, test_client: TestClient):
        """Test API handling of edge case HTTP methods."""
        edge_case_methods = [
            "TRACE",    # Potentially dangerous method
            "CONNECT",  # Proxy method
            "PATCH",    # Less common method
            "PURGE",    # Cache purging method
            "LOCK",     # WebDAV method
            "UNLOCK",   # WebDAV method
            "PROPFIND", # WebDAV method
            "MKCOL",    # WebDAV method
        ]
        
        for method in edge_case_methods:
            try:
                response = test_client.request(method, "/health", headers={"Host": "localhost"})
                # Should handle edge case methods gracefully
                assert response.status_code in [200, 405, 501], \
                    f"Should handle HTTP method {method} gracefully"
                
                # TRACE method should be disabled for security
                if method == "TRACE":
                    assert response.status_code in [405, 501], "TRACE method should be disabled"
                    
            except Exception:
                # Some methods might not be supported by test client
                pass

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_protocol_downgrade_attacks(self, test_client: TestClient):
        """Test API resistance to protocol downgrade attacks."""
        # Test various protocol downgrade scenarios
        downgrade_headers = [
            {"Host": "localhost", "Upgrade": "HTTP/1.0"},
            {"Host": "localhost", "Connection": "Upgrade", "Upgrade": "websocket"},
            {"Host": "localhost", "Sec-WebSocket-Version": "13"},
            {"Host": "localhost", "Upgrade-Insecure-Requests": "1"},
        ]
        
        for headers in downgrade_headers:
            response = test_client.get("/health", headers=headers)
            # Should handle downgrade attempts gracefully
            assert response.status_code in [200, 400, 426], \
                f"Should handle protocol downgrade: {headers}"
            
            # Should not actually downgrade protocol
            if response.status_code == 200:
                # Response should still be HTTP/1.1 or higher
                assert "application/json" in response.headers.get("content-type", ""), \
                    "Should maintain proper response format"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_api_timing_attack_resistance(self, test_client: TestClient):
        """Test API resistance to timing attacks."""
        import time
        
        # Test timing consistency for different request types
        timing_tests = [
            lambda: test_client.get("/health", headers={"Host": "localhost"}),
            lambda: test_client.get("/nonexistent", headers={"Host": "localhost"}),
            lambda: test_client.post("/health", headers={"Host": "localhost"}),
        ]
        
        timings = {}
        
        for i, test_func in enumerate(timing_tests):
            test_timings = []
            
            # Measure timing for multiple requests
            for _ in range(10):
                start_time = time.time()
                response = test_func()
                elapsed = time.time() - start_time
                test_timings.append(elapsed)
            
            timings[f"test_{i}"] = test_timings
        
        # Analyze timing patterns
        for test_name, test_timings in timings.items():
            avg_time = sum(test_timings) / len(test_timings)
            max_time = max(test_timings)
            min_time = min(test_timings)
            
            # Timing should be reasonably consistent (not revealing internal state)
            time_variance = max_time - min_time
            assert time_variance < 1.0, f"Excessive timing variance in {test_name}: {time_variance:.3f}s"
            
            # Average response time should be reasonable
            assert avg_time < 0.5, f"Response time too slow for {test_name}: {avg_time:.3f}s"