"""
Tests for dashboard statistics functionality.

Tests the complete flow from backend statistics calculation to frontend display.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import shutil

from server.services.analysis_service import AnalysisService
from ambient.core.config import ConfigurationManager


class TestDashboardStatistics:
    """Test dashboard statistics calculation."""
    
    @pytest.fixture
    def temp_analysis_dir(self):
        """Create temporary analysis directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def analysis_service(self, temp_analysis_dir):
        """Create analysis service with temporary directory."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.config = Mock()
        config_manager.config.storage = Mock()
        config_manager.config.storage.analysis_directory = str(temp_analysis_dir)
        
        service = AnalysisService(config_manager)
        return service
    
    def create_mock_analysis(
        self,
        service: AnalysisService,
        analysis_id: str,
        status: str = "completed",
        is_normal: bool = True,
        confidence: float = 0.95
    ):
        """Create mock analysis metadata and results."""
        # Create metadata
        metadata = {
            "analysis_id": analysis_id,
            "file_id": f"file_{analysis_id}",
            "status": status,
            "created_at": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
            "progress": {"stage": "finished", "percent": 100}
        }
        
        metadata_file = service.metadata_dir / f"{analysis_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Create results if completed
        if status == "completed":
            results = {
                "analysis_id": analysis_id,
                "file_id": f"file_{analysis_id}",
                "classification": {
                    "is_normal": is_normal,
                    "confidence": confidence,
                    "explanation": "Test explanation",
                    "identified_conditions": [] if is_normal else ["test_condition"]
                },
                "frame_count": 100,
                "duration": 3.33,
                "completed_at": datetime.utcnow().isoformat()
            }
            
            results_file = service.results_dir / f"{analysis_id}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f)
    
    def test_empty_dashboard_statistics(self, analysis_service):
        """Test statistics with no analyses."""
        stats = analysis_service.get_dashboard_statistics()
        
        assert stats['total_analyses'] == 0
        assert stats['normal_patterns'] == 0
        assert stats['abnormal_patterns'] == 0
        assert stats['avg_confidence'] == 0
        assert len(stats['recent_analyses']) == 0
        assert stats['status_breakdown']['completed'] == 0
    
    def test_single_normal_analysis(self, analysis_service):
        """Test statistics with one normal analysis."""
        self.create_mock_analysis(
            analysis_service,
            "test-001",
            status="completed",
            is_normal=True,
            confidence=0.95
        )
        
        stats = analysis_service.get_dashboard_statistics()
        
        assert stats['total_analyses'] == 1
        assert stats['normal_patterns'] == 1
        assert stats['abnormal_patterns'] == 0
        assert stats['normal_percentage'] == 100.0
        assert stats['abnormal_percentage'] == 0.0
        assert stats['avg_confidence'] == 95.0
        assert len(stats['recent_analyses']) == 1
        assert stats['status_breakdown']['completed'] == 1
    
    def test_single_abnormal_analysis(self, analysis_service):
        """Test statistics with one abnormal analysis."""
        self.create_mock_analysis(
            analysis_service,
            "test-002",
            status="completed",
            is_normal=False,
            confidence=0.88
        )
        
        stats = analysis_service.get_dashboard_statistics()
        
        assert stats['total_analyses'] == 1
        assert stats['normal_patterns'] == 0
        assert stats['abnormal_patterns'] == 1
        assert stats['normal_percentage'] == 0.0
        assert stats['abnormal_percentage'] == 100.0
        assert stats['avg_confidence'] == 88.0
    
    def test_mixed_analyses(self, analysis_service):
        """Test statistics with mixed normal and abnormal analyses."""
        # Create 3 normal analyses
        for i in range(3):
            self.create_mock_analysis(
                analysis_service,
                f"normal-{i}",
                status="completed",
                is_normal=True,
                confidence=0.90 + (i * 0.02)
            )
        
        # Create 2 abnormal analyses
        for i in range(2):
            self.create_mock_analysis(
                analysis_service,
                f"abnormal-{i}",
                status="completed",
                is_normal=False,
                confidence=0.85 + (i * 0.03)
            )
        
        stats = analysis_service.get_dashboard_statistics()
        
        assert stats['total_analyses'] == 5
        assert stats['normal_patterns'] == 3
        assert stats['abnormal_patterns'] == 2
        assert stats['normal_percentage'] == 60.0
        assert stats['abnormal_percentage'] == 40.0
        
        # Average confidence: (0.90 + 0.92 + 0.94 + 0.85 + 0.88) / 5 = 0.898
        assert 89.0 <= stats['avg_confidence'] <= 90.0
    
    def test_pending_analyses_not_counted(self, analysis_service):
        """Test that pending analyses don't affect statistics."""
        # Create completed analysis
        self.create_mock_analysis(
            analysis_service,
            "completed-001",
            status="completed",
            is_normal=True,
            confidence=0.95
        )
        
        # Create pending analysis (no results)
        metadata = {
            "analysis_id": "pending-001",
            "file_id": "file_pending",
            "status": "pending",
            "created_at": datetime.utcnow().isoformat()
        }
        metadata_file = analysis_service.metadata_dir / "pending-001.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        stats = analysis_service.get_dashboard_statistics()
        
        assert stats['total_analyses'] == 2  # Both counted in total
        assert stats['normal_patterns'] == 1  # Only completed counted
        assert stats['status_breakdown']['completed'] == 1
        assert stats['status_breakdown']['pending'] == 1
    
    def test_recent_analyses_limit(self, analysis_service):
        """Test that recent analyses are limited to 10."""
        # Create 15 analyses
        for i in range(15):
            self.create_mock_analysis(
                analysis_service,
                f"test-{i:03d}",
                status="completed",
                is_normal=True,
                confidence=0.90
            )
        
        stats = analysis_service.get_dashboard_statistics()
        
        assert stats['total_analyses'] == 15
        assert len(stats['recent_analyses']) == 10  # Limited to 10
    
    def test_status_breakdown(self, analysis_service):
        """Test status breakdown counts."""
        # Create analyses with different statuses
        statuses = [
            ("completed", 5),
            ("pending", 3),
            ("running", 2),
            ("failed", 1)
        ]
        
        for status, count in statuses:
            for i in range(count):
                if status == "completed":
                    self.create_mock_analysis(
                        analysis_service,
                        f"{status}-{i}",
                        status=status,
                        is_normal=True,
                        confidence=0.90
                    )
                else:
                    # Just create metadata for non-completed
                    metadata = {
                        "analysis_id": f"{status}-{i}",
                        "file_id": f"file_{status}_{i}",
                        "status": status,
                        "created_at": datetime.utcnow().isoformat()
                    }
                    metadata_file = analysis_service.metadata_dir / f"{status}-{i}.json"
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f)
        
        stats = analysis_service.get_dashboard_statistics()
        
        assert stats['total_analyses'] == 11
        assert stats['status_breakdown']['completed'] == 5
        assert stats['status_breakdown']['pending'] == 3
        assert stats['status_breakdown']['running'] == 2
        assert stats['status_breakdown']['failed'] == 1
    
    def test_confidence_calculation(self, analysis_service):
        """Test average confidence calculation."""
        confidences = [0.95, 0.88, 0.92, 0.85, 0.90]
        
        for i, conf in enumerate(confidences):
            self.create_mock_analysis(
                analysis_service,
                f"test-{i}",
                status="completed",
                is_normal=True,
                confidence=conf
            )
        
        stats = analysis_service.get_dashboard_statistics()
        
        expected_avg = sum(confidences) / len(confidences) * 100
        assert abs(stats['avg_confidence'] - expected_avg) < 0.1
    
    def test_percentage_calculation(self, analysis_service):
        """Test normal/abnormal percentage calculation."""
        # Create 7 normal, 3 abnormal
        for i in range(7):
            self.create_mock_analysis(
                analysis_service,
                f"normal-{i}",
                status="completed",
                is_normal=True,
                confidence=0.90
            )
        
        for i in range(3):
            self.create_mock_analysis(
                analysis_service,
                f"abnormal-{i}",
                status="completed",
                is_normal=False,
                confidence=0.85
            )
        
        stats = analysis_service.get_dashboard_statistics()
        
        assert stats['normal_percentage'] == 70.0
        assert stats['abnormal_percentage'] == 30.0
    
    def test_recent_analyses_content(self, analysis_service):
        """Test that recent analyses contain correct information."""
        self.create_mock_analysis(
            analysis_service,
            "test-001",
            status="completed",
            is_normal=False,
            confidence=0.88
        )
        
        stats = analysis_service.get_dashboard_statistics()
        recent = stats['recent_analyses'][0]
        
        assert recent['analysis_id'] == "test-001"
        assert recent['file_id'] == "file_test-001"
        assert recent['status'] == "completed"
        assert recent['is_normal'] is False
        assert recent['confidence'] == 0.88
        assert recent['frame_count'] == 100
        assert recent['duration'] == 3.33
        assert 'created_at' in recent
        assert 'completed_at' in recent


class TestDashboardStatisticsEdgeCases:
    """Test edge cases for dashboard statistics."""
    
    @pytest.fixture
    def temp_analysis_dir(self):
        """Create temporary analysis directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def analysis_service(self, temp_analysis_dir):
        """Create analysis service with temporary directory."""
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.config = Mock()
        config_manager.config.storage = Mock()
        config_manager.config.storage.analysis_directory = str(temp_analysis_dir)
        
        service = AnalysisService(config_manager)
        return service
    
    def test_zero_confidence(self, analysis_service):
        """Test handling of zero confidence."""
        # Create analysis with zero confidence
        metadata = {
            "analysis_id": "test-001",
            "file_id": "file_001",
            "status": "completed",
            "created_at": datetime.utcnow().isoformat()
        }
        metadata_file = analysis_service.metadata_dir / "test-001.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        results = {
            "analysis_id": "test-001",
            "classification": {
                "is_normal": True,
                "confidence": 0.0,  # Zero confidence
                "explanation": "Test",
                "identified_conditions": []
            },
            "frame_count": 100,
            "completed_at": datetime.utcnow().isoformat()
        }
        results_file = analysis_service.results_dir / "test-001.json"
        with open(results_file, 'w') as f:
            json.dump(results, f)
        
        stats = analysis_service.get_dashboard_statistics()
        
        # Zero confidence should not be included in average
        assert stats['avg_confidence'] == 0.0
    
    def test_missing_classification(self, analysis_service):
        """Test handling of missing classification data."""
        metadata = {
            "analysis_id": "test-001",
            "file_id": "file_001",
            "status": "completed",
            "created_at": datetime.utcnow().isoformat()
        }
        metadata_file = analysis_service.metadata_dir / "test-001.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        results = {
            "analysis_id": "test-001",
            "classification": {},  # Empty classification
            "frame_count": 100,
            "completed_at": datetime.utcnow().isoformat()
        }
        results_file = analysis_service.results_dir / "test-001.json"
        with open(results_file, 'w') as f:
            json.dump(results, f)
        
        stats = analysis_service.get_dashboard_statistics()
        
        # Should handle gracefully
        assert stats['total_analyses'] == 1
        assert stats['normal_patterns'] == 0
        assert stats['abnormal_patterns'] == 0
    
    def test_corrupted_metadata_file(self, analysis_service):
        """Test handling of corrupted metadata files."""
        # Create corrupted metadata file
        metadata_file = analysis_service.metadata_dir / "corrupted.json"
        with open(metadata_file, 'w') as f:
            f.write("{ invalid json")
        
        # Should not crash
        stats = analysis_service.get_dashboard_statistics()
        assert stats is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
