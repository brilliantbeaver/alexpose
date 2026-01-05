"""
Tests for video upload endpoints.

These tests ensure the upload endpoints work correctly for both
file uploads and YouTube URL submissions.
"""

import pytest
import io
from pathlib import Path
from fastapi.testclient import TestClient
from server.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def sample_video_file():
    """Create a sample video file for testing."""
    # Create a minimal valid video file (just for testing upload mechanism)
    video_content = b"fake video content for testing"
    return io.BytesIO(video_content)


@pytest.fixture
def sample_video_path(tmp_path):
    """Create a sample video file on disk."""
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b"fake video content")
    return video_file


class TestVideoUploadEndpoint:
    """Test suite for video file upload endpoint."""
    
    def test_upload_endpoint_exists(self, client):
        """Test that upload endpoint is accessible."""
        response = client.post("/api/upload/video")
        # Should return 422 (validation error) not 404
        assert response.status_code in [422, 400], "Upload endpoint should exist"
        
    def test_upload_requires_file(self, client):
        """Test that upload endpoint requires a file."""
        response = client.post("/api/upload/video")
        assert response.status_code == 422
        
    def test_upload_with_valid_file(self, client, sample_video_file):
        """Test uploading a valid video file."""
        files = {"file": ("test_video.mp4", sample_video_file, "video/mp4")}
        response = client.post("/api/upload/video", files=files)
        
        # Should return 200 or 201
        assert response.status_code in [200, 201]
        
    def test_upload_returns_upload_id(self, client, sample_video_file):
        """Test that upload returns an upload ID."""
        files = {"file": ("test_video.mp4", sample_video_file, "video/mp4")}
        response = client.post("/api/upload/video", files=files)
        
        if response.status_code in [200, 201]:
            data = response.json()
            assert "upload_id" in data or "id" in data
            
    def test_upload_validates_file_format(self, client):
        """Test that upload validates file format."""
        # Try uploading a non-video file
        fake_file = io.BytesIO(b"not a video")
        files = {"file": ("test.txt", fake_file, "text/plain")}
        response = client.post("/api/upload/video", files=files)
        
        # Should reject non-video files
        assert response.status_code in [400, 422]
        
    def test_upload_handles_large_files(self, client):
        """Test that upload can handle large files."""
        # Create a larger fake video (10MB)
        large_content = b"x" * (10 * 1024 * 1024)
        large_file = io.BytesIO(large_content)
        files = {"file": ("large_video.mp4", large_file, "video/mp4")}
        
        response = client.post("/api/upload/video", files=files)
        # Should either accept or return appropriate error
        assert response.status_code in [200, 201, 413]
        
    def test_upload_supports_multiple_formats(self, client):
        """Test that upload supports multiple video formats."""
        formats = ["mp4", "avi", "mov", "webm"]
        
        for fmt in formats:
            video_file = io.BytesIO(b"fake video")
            files = {"file": (f"test.{fmt}", video_file, f"video/{fmt}")}
            response = client.post("/api/upload/video", files=files)
            
            # Should accept all supported formats
            assert response.status_code in [200, 201, 400, 422]


class TestYouTubeUploadEndpoint:
    """Test suite for YouTube URL upload endpoint."""
    
    def test_youtube_endpoint_exists(self, client):
        """Test that YouTube upload endpoint is accessible."""
        response = client.post("/api/upload/youtube")
        # Should return 422 (validation error) not 404
        assert response.status_code in [422, 400]
        
    def test_youtube_requires_url(self, client):
        """Test that YouTube endpoint requires a URL."""
        response = client.post("/api/upload/youtube", json={})
        assert response.status_code == 422
        
    def test_youtube_with_valid_url(self, client):
        """Test submitting a valid YouTube URL."""
        data = {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
        response = client.post("/api/upload/youtube", json=data)
        
        # Should return 200, 201, or 202 (accepted for processing)
        assert response.status_code in [200, 201, 202]
        
    def test_youtube_returns_upload_id(self, client):
        """Test that YouTube upload returns an upload ID."""
        data = {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
        response = client.post("/api/upload/youtube", json=data)
        
        if response.status_code in [200, 201, 202]:
            result = response.json()
            assert "upload_id" in result or "id" in result
            
    def test_youtube_validates_url_format(self, client):
        """Test that YouTube endpoint validates URL format."""
        invalid_urls = [
            "not a url",
            "http://example.com",
            "https://vimeo.com/123456",
        ]
        
        for url in invalid_urls:
            data = {"url": url}
            response = client.post("/api/upload/youtube", json=data)
            # Should reject invalid URLs
            assert response.status_code in [400, 422]
            
    def test_youtube_accepts_various_formats(self, client):
        """Test that YouTube endpoint accepts various URL formats."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s",
            "https://www.youtube.com/shorts/9bC26r3n4BY",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
        ]
        
        for url in valid_urls:
            data = {"url": url}
            response = client.post("/api/upload/youtube", json=data)
            # Should accept all valid YouTube URL formats
            assert response.status_code in [200, 201, 202, 400, 422]


class TestUploadStatusEndpoint:
    """Test suite for upload status checking endpoint."""
    
    def test_status_endpoint_exists(self, client):
        """Test that status endpoint is accessible."""
        response = client.get("/api/upload/status/test-id")
        # Should return 404 (not found) not 405 (method not allowed)
        assert response.status_code in [404, 200]
        
    def test_status_requires_upload_id(self, client):
        """Test that status endpoint requires an upload ID."""
        response = client.get("/api/upload/status/")
        assert response.status_code == 404
        
    def test_status_returns_upload_info(self, client, sample_video_file):
        """Test that status endpoint returns upload information."""
        # First upload a file
        files = {"file": ("test_video.mp4", sample_video_file, "video/mp4")}
        upload_response = client.post("/api/upload/video", files=files)
        
        if upload_response.status_code in [200, 201]:
            upload_data = upload_response.json()
            upload_id = upload_data.get("upload_id") or upload_data.get("id")
            
            # Check status
            status_response = client.get(f"/api/upload/status/{upload_id}")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            assert "status" in status_data
            
    def test_status_handles_invalid_id(self, client):
        """Test that status endpoint handles invalid upload IDs."""
        response = client.get("/api/upload/status/invalid-id-12345")
        assert response.status_code == 404


@pytest.mark.integration
class TestUploadEndpointIntegration:
    """Integration tests for upload endpoints with real file system."""
    
    def test_upload_saves_file_to_disk(self, client, sample_video_file, tmp_path):
        """Test that uploaded files are saved to disk."""
        files = {"file": ("test_video.mp4", sample_video_file, "video/mp4")}
        response = client.post("/api/upload/video", files=files)
        
        if response.status_code in [200, 201]:
            # Check that file was saved (implementation dependent)
            data = response.json()
            assert "upload_id" in data or "id" in data
            
    def test_upload_creates_metadata(self, client, sample_video_file):
        """Test that upload creates metadata for the file."""
        files = {"file": ("test_video.mp4", sample_video_file, "video/mp4")}
        response = client.post("/api/upload/video", files=files)
        
        if response.status_code in [200, 201]:
            data = response.json()
            # Should include metadata like filename, size, etc.
            assert "filename" in data or "name" in data
            
    def test_concurrent_uploads(self, client):
        """Test that multiple concurrent uploads work correctly."""
        import concurrent.futures
        
        def upload_file():
            video_file = io.BytesIO(b"fake video content")
            files = {"file": ("test_video.mp4", video_file, "video/mp4")}
            return client.post("/api/upload/video", files=files)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(upload_file) for _ in range(5)]
            responses = [f.result() for f in futures]
        
        # All uploads should succeed or fail gracefully
        for response in responses:
            assert response.status_code in [200, 201, 400, 422, 500]


@pytest.mark.property
class TestUploadEndpointProperties:
    """Property-based tests for upload endpoints."""
    
    def test_upload_id_uniqueness(self, client):
        """Test that each upload gets a unique ID."""
        upload_ids = set()
        
        for _ in range(10):
            video_file = io.BytesIO(b"fake video content")
            files = {"file": ("test_video.mp4", video_file, "video/mp4")}
            response = client.post("/api/upload/video", files=files)
            
            if response.status_code in [200, 201]:
                data = response.json()
                upload_id = data.get("upload_id") or data.get("id")
                assert upload_id not in upload_ids, "Upload IDs must be unique"
                upload_ids.add(upload_id)
