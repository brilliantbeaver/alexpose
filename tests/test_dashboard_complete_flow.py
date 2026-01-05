"""
Complete end-to-end test for unified dashboard.

This test verifies the complete flow from data creation to dashboard display.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock

from server.services.analysis_service import AnalysisService
from server.services.gavd_service import GAVDService
from ambient.core.config import ConfigurationManager


class TestCompleteDashboardFlow:
    """Test complete dashboard data flow."""
    
    def test_complete_flow_with_both_data_types(self, tmp_path):
        """Test complete flow with both GAVD and gait analysis data."""
        # Setup directories
        analysis_dir = tmp_path / "analysis"
        training_dir = tmp_path / "training"
        
        analysis_metadata_dir = analysis_dir / "metadata"
        analysis_results_dir = analysis_dir / "results"
        gavd_metadata_dir = training_dir / "gavd" / "metadata"
        
        analysis_metadata_dir.mkdir(parents=True, exist_ok=True)
        analysis_results_dir.mkdir(parents=True, exist_ok=True)
        gavd_metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock config
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.config = Mock()
        config_manager.config.storage = Mock()
        config_manager.config.storage.analysis_directory = str(analysis_dir)
        config_manager.config.storage.training_directory = str(training_dir)
        
        # Create services
        analysis_service = AnalysisService(config_manager)
        gavd_service = GAVDService(config_manager)
        
        # Step 1: Create GAVD dataset
        gavd_metadata = {
            "dataset_id": "gavd-001",
            "original_filename": "test_dataset.csv",
            "status": "completed",
            "uploaded_at": "2026-01-03T10:00:00",
            "processing_completed_at": "2026-01-03T10:05:00",
            "total_sequences_processed": 5,
            "total_frames_processed": 150,
            "sequence_count": 5,
            "row_count": 150
        }
        gavd_service.save_dataset_metadata("gavd-001", gavd_metadata)
        
        # Step 2: Create gait analysis
        gait_metadata = {
            "analysis_id": "gait-001",
            "file_id": "file-001",
            "status": "completed",
            "created_at": "2026-01-03T11:00:00",
            "progress": {"stage": "finished", "percent": 100}
        }
        
        gait_results = {
            "analysis_id": "gait-001",
            "file_id": "file-001",
            "classification": {
                "is_normal": False,
                "confidence": 0.85,
                "explanation": "Abnormal gait detected",
                "identified_conditions": ["Limping"]
            },
            "frame_count": 300,
            "duration": 10.0,
            "completed_at": "2026-01-03T11:05:00"
        }
        
        with open(analysis_metadata_dir / "gait-001.json", 'w') as f:
            json.dump(gait_metadata, f)
        
        with open(analysis_results_dir / "gait-001.json", 'w') as f:
            json.dump(gait_results, f)
        
        # Step 3: Get statistics from both services
        gait_stats = analysis_service.get_dashboard_statistics()
        gavd_datasets = gavd_service.list_datasets(limit=10, offset=0)
        
        # Step 4: Verify GAVD data
        assert len(gavd_datasets) == 1
        assert gavd_datasets[0]["dataset_id"] == "gavd-001"
        assert gavd_datasets[0]["status"] == "completed"
        assert gavd_datasets[0]["total_sequences_processed"] == 5
        
        # Step 5: Verify gait analysis data
        assert gait_stats["total_analyses"] == 1
        assert gait_stats["abnormal_patterns"] == 1
        assert gait_stats["normal_patterns"] == 0
        assert len(gait_stats["recent_analyses"]) == 1
        
        # Step 6: Simulate unified statistics aggregation
        gavd_status_counts = {
            "uploaded": 0,
            "processing": 0,
            "completed": 0,
            "error": 0
        }
        
        for dataset in gavd_datasets:
            status = dataset.get("status", "unknown")
            if status in gavd_status_counts:
                gavd_status_counts[status] += 1
        
        # Step 7: Create combined recent analyses
        combined_recent = []
        
        # Add GAVD items
        for dataset in gavd_datasets:
            combined_recent.append({
                "type": "gavd_dataset",
                "dataset_id": dataset["dataset_id"],
                "filename": dataset["original_filename"],
                "status": dataset["status"],
                "completed_at": dataset.get("processing_completed_at"),
                "total_sequences_processed": dataset.get("total_sequences_processed", 0),
                "total_frames_processed": dataset.get("total_frames_processed", 0)
            })
        
        # Add gait analysis items
        for gait_item in gait_stats["recent_analyses"]:
            combined_recent.append({
                "type": "gait_analysis",
                **gait_item
            })
        
        # Sort by date
        combined_recent.sort(
            key=lambda x: x.get("completed_at") or "",
            reverse=True
        )
        
        # Step 8: Verify combined statistics
        total_analyses = gait_stats["total_analyses"] + len(gavd_datasets)
        
        assert total_analyses == 2
        assert len(combined_recent) == 2
        assert gavd_status_counts["completed"] == 1
        
        # Step 9: Verify recent analyses have both types
        types = [item["type"] for item in combined_recent]
        assert "gavd_dataset" in types
        assert "gait_analysis" in types
        
        # Step 10: Verify sorting (most recent first)
        # Gait analysis completed at 11:05, GAVD at 10:05
        assert combined_recent[0]["type"] == "gait_analysis"
        assert combined_recent[1]["type"] == "gavd_dataset"
        
        print("\n✅ Complete flow test passed!")
        print(f"   - Created 1 GAVD dataset")
        print(f"   - Created 1 gait analysis")
        print(f"   - Combined statistics: {total_analyses} total analyses")
        print(f"   - Recent analyses sorted correctly")
    
    def test_empty_state(self, tmp_path):
        """Test dashboard with no data."""
        # Setup directories
        analysis_dir = tmp_path / "analysis"
        training_dir = tmp_path / "training"
        
        analysis_metadata_dir = analysis_dir / "metadata"
        gavd_metadata_dir = training_dir / "gavd" / "metadata"
        
        analysis_metadata_dir.mkdir(parents=True, exist_ok=True)
        gavd_metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock config
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.config = Mock()
        config_manager.config.storage = Mock()
        config_manager.config.storage.analysis_directory = str(analysis_dir)
        config_manager.config.storage.training_directory = str(training_dir)
        
        # Create services
        analysis_service = AnalysisService(config_manager)
        gavd_service = GAVDService(config_manager)
        
        # Get statistics
        gait_stats = analysis_service.get_dashboard_statistics()
        gavd_datasets = gavd_service.list_datasets(limit=10, offset=0)
        
        # Verify empty state
        assert gait_stats["total_analyses"] == 0
        assert len(gavd_datasets) == 0
        assert len(gait_stats["recent_analyses"]) == 0
        
        print("\n✅ Empty state test passed!")
    
    def test_only_gavd_data(self, tmp_path):
        """Test dashboard with only GAVD data."""
        # Setup directories
        analysis_dir = tmp_path / "analysis"
        training_dir = tmp_path / "training"
        
        analysis_metadata_dir = analysis_dir / "metadata"
        gavd_metadata_dir = training_dir / "gavd" / "metadata"
        
        analysis_metadata_dir.mkdir(parents=True, exist_ok=True)
        gavd_metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock config
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.config = Mock()
        config_manager.config.storage = Mock()
        config_manager.config.storage.analysis_directory = str(analysis_dir)
        config_manager.config.storage.training_directory = str(training_dir)
        
        # Create services
        analysis_service = AnalysisService(config_manager)
        gavd_service = GAVDService(config_manager)
        
        # Create only GAVD data
        gavd_metadata = {
            "dataset_id": "gavd-001",
            "original_filename": "test.csv",
            "status": "completed",
            "uploaded_at": "2026-01-03T10:00:00",
            "total_sequences_processed": 3,
            "total_frames_processed": 90
        }
        gavd_service.save_dataset_metadata("gavd-001", gavd_metadata)
        
        # Get statistics
        gait_stats = analysis_service.get_dashboard_statistics()
        gavd_datasets = gavd_service.list_datasets(limit=10, offset=0)
        
        # Verify
        assert gait_stats["total_analyses"] == 0
        assert len(gavd_datasets) == 1
        assert len(gait_stats["recent_analyses"]) == 0
        
        # Combined total should be 1
        total = gait_stats["total_analyses"] + len(gavd_datasets)
        assert total == 1
        
        print("\n✅ GAVD-only test passed!")
    
    def test_multiple_gavd_datasets(self, tmp_path):
        """Test dashboard with multiple GAVD datasets."""
        # Setup directories
        training_dir = tmp_path / "training"
        gavd_metadata_dir = training_dir / "gavd" / "metadata"
        gavd_metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock config
        config_manager = Mock(spec=ConfigurationManager)
        config_manager.config = Mock()
        config_manager.config.storage = Mock()
        config_manager.config.storage.training_directory = str(training_dir)
        
        # Create service
        gavd_service = GAVDService(config_manager)
        
        # Create multiple datasets with different statuses
        datasets = [
            {"dataset_id": "gavd-001", "status": "completed"},
            {"dataset_id": "gavd-002", "status": "completed"},
            {"dataset_id": "gavd-003", "status": "processing"},
            {"dataset_id": "gavd-004", "status": "uploaded"},
            {"dataset_id": "gavd-005", "status": "error"}
        ]
        
        for dataset in datasets:
            metadata = {
                **dataset,
                "original_filename": f"{dataset['dataset_id']}.csv",
                "uploaded_at": "2026-01-03T10:00:00"
            }
            gavd_service.save_dataset_metadata(dataset["dataset_id"], metadata)
        
        # Get datasets
        gavd_datasets = gavd_service.list_datasets(limit=10, offset=0)
        
        # Count by status
        status_counts = {
            "uploaded": 0,
            "processing": 0,
            "completed": 0,
            "error": 0
        }
        
        for dataset in gavd_datasets:
            status = dataset.get("status", "unknown")
            if status in status_counts:
                status_counts[status] += 1
        
        # Verify
        assert len(gavd_datasets) == 5
        assert status_counts["completed"] == 2
        assert status_counts["processing"] == 1
        assert status_counts["uploaded"] == 1
        assert status_counts["error"] == 1
        
        print("\n✅ Multiple GAVD datasets test passed!")
        print(f"   - Total: {len(gavd_datasets)}")
        print(f"   - Completed: {status_counts['completed']}")
        print(f"   - Processing: {status_counts['processing']}")
        print(f"   - Uploaded: {status_counts['uploaded']}")
        print(f"   - Error: {status_counts['error']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
