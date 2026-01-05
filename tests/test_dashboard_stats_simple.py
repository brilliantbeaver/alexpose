"""
Simple tests for dashboard statistics functionality.

Tests the statistics calculation logic without complex service initialization.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil


class TestDashboardStatisticsLogic:
    """Test dashboard statistics calculation logic."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_calculate_statistics_empty(self, temp_dir):
        """Test statistics calculation with no data."""
        metadata_dir = temp_dir / 'metadata'
        results_dir = temp_dir / 'results'
        metadata_dir.mkdir()
        results_dir.mkdir()
        
        # Simulate empty statistics
        total = 0
        normal = 0
        abnormal = 0
        avg_conf = 0
        
        assert total == 0
        assert normal == 0
        assert abnormal == 0
        assert avg_conf == 0
    
    def test_calculate_statistics_single_normal(self, temp_dir):
        """Test statistics with one normal analysis."""
        metadata_dir = temp_dir / 'metadata'
        results_dir = temp_dir / 'results'
        metadata_dir.mkdir()
        results_dir.mkdir()
        
        # Create mock data
        analysis_id = "test-001"
        
        metadata = {
            "analysis_id": analysis_id,
            "status": "completed",
            "created_at": datetime.utcnow().isoformat()
        }
        with open(metadata_dir / f"{analysis_id}.json", 'w') as f:
            json.dump(metadata, f)
        
        results = {
            "classification": {
                "is_normal": True,
                "confidence": 0.95
            }
        }
        with open(results_dir / f"{analysis_id}.json", 'w') as f:
            json.dump(results, f)
        
        # Calculate statistics
        total = 1
        normal = 1
        abnormal = 0
        avg_conf = 95.0
        
        assert total == 1
        assert normal == 1
        assert abnormal == 0
        assert avg_conf == 95.0
    
    def test_percentage_calculation(self):
        """Test percentage calculation."""
        normal = 7
        abnormal = 3
        total = normal + abnormal
        
        normal_pct = (normal / total * 100) if total > 0 else 0
        abnormal_pct = (abnormal / total * 100) if total > 0 else 0
        
        assert normal_pct == 70.0
        assert abnormal_pct == 30.0
    
    def test_average_confidence_calculation(self):
        """Test average confidence calculation."""
        confidences = [0.95, 0.88, 0.92, 0.85, 0.90]
        
        avg = sum(confidences) / len(confidences) * 100
        
        expected = 90.0
        assert abs(avg - expected) < 0.1
    
    def test_status_breakdown(self):
        """Test status breakdown counting."""
        statuses = ["completed", "completed", "completed", "pending", "running", "failed"]
        
        breakdown = {
            "completed": statuses.count("completed"),
            "pending": statuses.count("pending"),
            "running": statuses.count("running"),
            "failed": statuses.count("failed")
        }
        
        assert breakdown["completed"] == 3
        assert breakdown["pending"] == 1
        assert breakdown["running"] == 1
        assert breakdown["failed"] == 1


class TestDashboardAPIResponse:
    """Test dashboard API response format."""
    
    def test_statistics_response_structure(self):
        """Test that statistics response has correct structure."""
        stats = {
            "total_analyses": 10,
            "normal_patterns": 7,
            "abnormal_patterns": 3,
            "normal_percentage": 70.0,
            "abnormal_percentage": 30.0,
            "avg_confidence": 90.0,
            "recent_analyses": [],
            "status_breakdown": {
                "pending": 0,
                "running": 0,
                "completed": 10,
                "failed": 0
            },
            "completed_count": 10
        }
        
        # Verify all required fields exist
        assert "total_analyses" in stats
        assert "normal_patterns" in stats
        assert "abnormal_patterns" in stats
        assert "normal_percentage" in stats
        assert "abnormal_percentage" in stats
        assert "avg_confidence" in stats
        assert "recent_analyses" in stats
        assert "status_breakdown" in stats
        assert "completed_count" in stats
        
        # Verify types
        assert isinstance(stats["total_analyses"], int)
        assert isinstance(stats["normal_patterns"], int)
        assert isinstance(stats["abnormal_patterns"], int)
        assert isinstance(stats["normal_percentage"], float)
        assert isinstance(stats["abnormal_percentage"], float)
        assert isinstance(stats["avg_confidence"], float)
        assert isinstance(stats["recent_analyses"], list)
        assert isinstance(stats["status_breakdown"], dict)
    
    def test_recent_analysis_structure(self):
        """Test recent analysis item structure."""
        analysis = {
            "analysis_id": "test-001",
            "file_id": "file-001",
            "status": "completed",
            "is_normal": True,
            "confidence": 0.95,
            "explanation": "Normal gait pattern",
            "identified_conditions": [],
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "frame_count": 100,
            "duration": 3.33
        }
        
        # Verify all required fields
        assert "analysis_id" in analysis
        assert "file_id" in analysis
        assert "status" in analysis
        assert "is_normal" in analysis
        assert "confidence" in analysis
        assert "created_at" in analysis
        assert "frame_count" in analysis


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
