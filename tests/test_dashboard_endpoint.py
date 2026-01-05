"""
Test dashboard statistics endpoint with real backend.

This test verifies that the statistics endpoint works correctly
with the fixed AnalysisService initialization.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from fastapi.testclient import TestClient

from server.main import app
from ambient.core.config import ConfigurationManager


@pytest.fixture
def test_client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def setup_test_data(tmp_path):
    """Create test analysis data."""
    # Create temporary directories
    analysis_dir = tmp_path / "analysis"
    metadata_dir = analysis_dir / "metadata"
    results_dir = analysis_dir / "results"
    
    metadata_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test analyses
    test_analyses = [
        {
            "analysis_id": "test-001",
            "file_id": "file-001",
            "status": "completed",
            "created_at": "2026-01-01T10:00:00",
            "progress": {"stage": "finished", "percent": 100}
        },
        {
            "analysis_id": "test-002",
            "file_id": "file-002",
            "status": "completed",
            "created_at": "2026-01-01T11:00:00",
            "progress": {"stage": "finished", "percent": 100}
        },
        {
            "analysis_id": "test-003",
            "file_id": "file-003",
            "status": "running",
            "created_at": "2026-01-01T12:00:00",
            "progress": {"stage": "pose_estimation", "percent": 50}
        }
    ]
    
    # Create test results
    test_results = [
        {
            "analysis_id": "test-001",
            "file_id": "file-001",
            "classification": {
                "is_normal": True,
                "confidence": 0.85,
                "explanation": "Normal gait pattern detected",
                "identified_conditions": []
            },
            "frame_count": 300,
            "duration": 10.0,
            "completed_at": "2026-01-01T10:05:00"
        },
        {
            "analysis_id": "test-002",
            "file_id": "file-002",
            "classification": {
                "is_normal": False,
                "confidence": 0.75,
                "explanation": "Abnormal gait pattern detected",
                "identified_conditions": ["Limping", "Reduced stride length"]
            },
            "frame_count": 250,
            "duration": 8.3,
            "completed_at": "2026-01-01T11:05:00"
        }
    ]
    
    # Save metadata files
    for analysis in test_analyses:
        metadata_file = metadata_dir / f"{analysis['analysis_id']}.json"
        with open(metadata_file, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    # Save results files
    for result in test_results:
        results_file = results_dir / f"{result['analysis_id']}.json"
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    return {
        "analysis_dir": analysis_dir,
        "metadata_dir": metadata_dir,
        "results_dir": results_dir
    }


class TestDashboardEndpoint:
    """Test dashboard statistics endpoint."""
    
    def test_statistics_endpoint_structure(self, test_client):
        """Test that statistics endpoint returns correct structure."""
        response = test_client.get("/api/v1/analysis/statistics")
        
        # Should return 200 even with no data
        assert response.status_code == 200
        
        data = response.json()
        assert "success" in data
        assert "statistics" in data
        
        stats = data["statistics"]
        assert "total_analyses" in stats
        assert "normal_patterns" in stats
        assert "abnormal_patterns" in stats
        assert "avg_confidence" in stats
        assert "recent_analyses" in stats
        assert "status_breakdown" in stats
    
    def test_statistics_with_no_data(self, test_client):
        """Test statistics endpoint with no analysis data."""
        response = test_client.get("/api/v1/analysis/statistics")
        
        assert response.status_code == 200
        data = response.json()
        
        stats = data["statistics"]
        assert stats["total_analyses"] == 0
        assert stats["normal_patterns"] == 0
        assert stats["abnormal_patterns"] == 0
        assert stats["avg_confidence"] == 0.0
        assert len(stats["recent_analyses"]) == 0
    
    def test_service_initialization(self):
        """Test that AnalysisService initializes correctly."""
        from server.services.analysis_service import AnalysisService
        
        config_manager = ConfigurationManager()
        
        # This should not raise an error anymore
        service = AnalysisService(config_manager)
        
        # Verify components are initialized
        assert service.pose_factory is not None
        assert service.video_processor is not None
        
        # Verify factory works
        assert hasattr(service.pose_factory, 'create_estimator')
        assert hasattr(service.pose_factory, 'list_available_estimators')
    
    def test_get_dashboard_statistics_method(self):
        """Test get_dashboard_statistics method directly."""
        from server.services.analysis_service import AnalysisService
        
        config_manager = ConfigurationManager()
        service = AnalysisService(config_manager)
        
        # Should return empty statistics without error
        stats = service.get_dashboard_statistics()
        
        assert isinstance(stats, dict)
        assert "total_analyses" in stats
        assert "normal_patterns" in stats
        assert "abnormal_patterns" in stats
        assert "recent_analyses" in stats
        assert stats["total_analyses"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
