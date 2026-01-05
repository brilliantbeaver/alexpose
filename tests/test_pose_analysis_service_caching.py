"""
Test pose analysis service caching to verify performance fix.

This test verifies that the service instance is properly cached
and reused across multiple requests.

Author: AlexPose Team
Date: January 4, 2026
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path


def test_service_caching():
    """Test that service instance is cached and reused."""
    from server.routers.pose_analysis import _get_service, _service_cache
    
    # Clear cache before test
    _service_cache.clear()
    
    # Create mock config manager with proper structure
    mock_config = Mock()
    mock_config.config.storage.cache_directory = 'data/cache'
    mock_config.config.storage.training_directory = 'data/training'
    mock_config.config.storage.data_directory = 'data'
    
    # Mock the PoseAnalysisServiceAPI to avoid full initialization
    with patch('server.routers.pose_analysis.PoseAnalysisServiceAPI') as MockService:
        mock_service_instance = Mock()
        MockService.return_value = mock_service_instance
        
        # First call should create new instance
        service1 = _get_service(mock_config)
        assert service1 is not None
        assert len(_service_cache) == 1
        assert MockService.call_count == 1
        
        # Second call should return same instance
        service2 = _get_service(mock_config)
        assert service2 is service1  # Same object reference
        assert len(_service_cache) == 1  # Still only one cached instance
        assert MockService.call_count == 1  # No additional initialization
        
        # Third call should also return same instance
        service3 = _get_service(mock_config)
        assert service3 is service1
        assert len(_service_cache) == 1
        assert MockService.call_count == 1  # Still only one initialization


def test_service_cache_key():
    """Test that cache key is consistent."""
    from server.routers.pose_analysis import _get_service, _service_cache
    
    _service_cache.clear()
    
    mock_config = Mock()
    mock_config.config.storage.cache_directory = 'data/cache'
    
    with patch('server.routers.pose_analysis.PoseAnalysisServiceAPI') as MockService:
        mock_service_instance = Mock()
        MockService.return_value = mock_service_instance
        
        # Get service multiple times
        for _ in range(5):
            _get_service(mock_config)
        
        # Should only have one cached instance
        assert len(_service_cache) == 1
        assert "default" in _service_cache
        # Should only initialize once
        assert MockService.call_count == 1


def test_service_initialization_count():
    """Test that service is initialized only once."""
    from server.routers.pose_analysis import _get_service, _service_cache
    
    _service_cache.clear()
    
    mock_config = Mock()
    mock_config.config.storage.cache_directory = 'data/cache'
    
    with patch('server.routers.pose_analysis.PoseAnalysisServiceAPI') as MockService:
        mock_service_instance = Mock()
        MockService.return_value = mock_service_instance
        
        # Call _get_service multiple times
        for _ in range(10):
            _get_service(mock_config)
        
        # Should only initialize once
        assert MockService.call_count == 1
        assert len(_service_cache) == 1


def test_cache_isolation():
    """Test that cache is properly isolated between test runs."""
    from server.routers.pose_analysis import _service_cache
    
    # This test verifies cache can be cleared
    _service_cache.clear()
    assert len(_service_cache) == 0
    
    # Add something to cache
    _service_cache["test"] = Mock()
    assert len(_service_cache) == 1
    
    # Clear again
    _service_cache.clear()
    assert len(_service_cache) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
