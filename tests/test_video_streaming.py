"""
Tests for video streaming functionality.

Tests the video streaming endpoints and range request support.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import shutil

from server.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_video_file(tmp_path):
    """Create a mock video file for testing."""
    video_dir = tmp_path / "youtube"
    video_dir.mkdir()
    
    # Create a small mock video file
    video_path = video_dir / "test_video_id.mp4"
    video_path.write_bytes(b"MOCK_VIDEO_DATA" * 1000)  # 15KB mock file
    
    return video_dir, "test_video_id"


class TestVideoStreaming:
    """Test video streaming endpoints."""
    
    def test_stream_video_full(self, client, mock_video_file, monkeypatch):
        """Test streaming full video without range request."""
        video_dir, video_id = mock_video_file
        
        # Mock the configuration
        class MockConfig:
            class storage:
                youtube_directory = str(video_dir)
        
        class MockConfigManager:
            config = MockConfig()
        
        # Patch the app state
        app.state.config = MockConfigManager()
        
        response = client.get(f"/api/v1/video/stream/{video_id}")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "video/mp4"
        assert "accept-ranges" in response.headers
        assert response.headers["accept-ranges"] == "bytes"
    
    def test_stream_video_with_range(self, client, mock_video_file, monkeypatch):
        """Test streaming video with range request."""
        video_dir, video_id = mock_video_file
        
        class MockConfig:
            class storage:
                youtube_directory = str(video_dir)
        
        class MockConfigManager:
            config = MockConfig()
        
        app.state.config = MockConfigManager()
        
        # Request first 100 bytes
        response = client.get(
            f"/api/v1/video/stream/{video_id}",
            headers={"Range": "bytes=0-99"}
        )
        
        assert response.status_code == 206  # Partial Content
        assert "content-range" in response.headers
        assert response.headers["content-range"].startswith("bytes 0-99/")
    
    def test_stream_video_not_found(self, client, tmp_path, monkeypatch):
        """Test streaming non-existent video."""
        video_dir = tmp_path / "youtube"
        video_dir.mkdir()
        
        class MockConfig:
            class storage:
                youtube_directory = str(video_dir)
        
        class MockConfigManager:
            config = MockConfig()
        
        app.state.config = MockConfigManager()
        
        response = client.get("/api/v1/video/stream/nonexistent_id")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_get_video_info_available(self, client, mock_video_file, monkeypatch):
        """Test getting info for available video."""
        video_dir, video_id = mock_video_file
        
        class MockConfig:
            class storage:
                youtube_directory = str(video_dir)
        
        class MockConfigManager:
            config = MockConfig()
        
        app.state.config = MockConfigManager()
        
        response = client.get(f"/api/v1/video/info/{video_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["available"] is True
        assert data["video_id"] == video_id
        assert "file_size" in data
        assert "stream_url" in data
    
    def test_get_video_info_not_available(self, client, tmp_path, monkeypatch):
        """Test getting info for unavailable video."""
        video_dir = tmp_path / "youtube"
        video_dir.mkdir()
        
        class MockConfig:
            class storage:
                youtube_directory = str(video_dir)
        
        class MockConfigManager:
            config = MockConfig()
        
        app.state.config = MockConfigManager()
        
        response = client.get("/api/v1/video/info/nonexistent_id")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["available"] is False
    
    def test_url_to_video_id_valid(self, client):
        """Test extracting video ID from valid YouTube URL."""
        test_urls = [
            "https://www.youtube.com/watch?v=B5hrxKe2nP8",
            "https://youtu.be/B5hrxKe2nP8",
            "https://www.youtube.com/embed/B5hrxKe2nP8",
            "https://www.youtube.com/shorts/9bC26r3n4BY"
        ]
        
        expected_ids = {
            "https://www.youtube.com/watch?v=B5hrxKe2nP8": "B5hrxKe2nP8",
            "https://youtu.be/B5hrxKe2nP8": "B5hrxKe2nP8",
            "https://www.youtube.com/embed/B5hrxKe2nP8": "B5hrxKe2nP8",
            "https://www.youtube.com/shorts/9bC26r3n4BY": "9bC26r3n4BY"
        }
        
        for url in test_urls:
            response = client.get(f"/api/v1/video/url-to-id?url={url}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["video_id"] == expected_ids[url]
            assert "stream_url" in data
    
    def test_url_to_video_id_invalid(self, client):
        """Test extracting video ID from invalid URL."""
        response = client.get("/api/v1/video/url-to-id?url=https://example.com")
        
        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()


class TestRangeRequests:
    """Test range request handling."""
    
    def test_range_start_only(self, client, mock_video_file, monkeypatch):
        """Test range request with only start byte."""
        video_dir, video_id = mock_video_file
        
        class MockConfig:
            class storage:
                youtube_directory = str(video_dir)
        
        class MockConfigManager:
            config = MockConfig()
        
        app.state.config = MockConfigManager()
        
        response = client.get(
            f"/api/v1/video/stream/{video_id}",
            headers={"Range": "bytes=100-"}
        )
        
        assert response.status_code == 206
        assert "content-range" in response.headers
    
    def test_range_end_only(self, client, mock_video_file, monkeypatch):
        """Test range request with only end byte."""
        video_dir, video_id = mock_video_file
        
        class MockConfig:
            class storage:
                youtube_directory = str(video_dir)
        
        class MockConfigManager:
            config = MockConfig()
        
        app.state.config = MockConfigManager()
        
        response = client.get(
            f"/api/v1/video/stream/{video_id}",
            headers={"Range": "bytes=-100"}
        )
        
        assert response.status_code == 206
    
    def test_multiple_ranges(self, client, mock_video_file, monkeypatch):
        """Test multiple sequential range requests."""
        video_dir, video_id = mock_video_file
        
        class MockConfig:
            class storage:
                youtube_directory = str(video_dir)
        
        class MockConfigManager:
            config = MockConfig()
        
        app.state.config = MockConfigManager()
        
        # Request first chunk
        response1 = client.get(
            f"/api/v1/video/stream/{video_id}",
            headers={"Range": "bytes=0-99"}
        )
        assert response1.status_code == 206
        
        # Request second chunk
        response2 = client.get(
            f"/api/v1/video/stream/{video_id}",
            headers={"Range": "bytes=100-199"}
        )
        assert response2.status_code == 206
        
        # Verify different content
        assert response1.content != response2.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
