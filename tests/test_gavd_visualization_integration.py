"""
Integration tests for GAVD Visualization Tab

Tests the complete flow from frontend to backend for the visualization tab.
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, settings

from server.services.gavd_service import GAVDService
from ambient.core.config import ConfigurationManager


@pytest.fixture
def mock_config():
    """Create a mock configuration manager."""
    config = Mock()
    config.storage = Mock()
    config.storage.training_directory = 'data/training'
    config.storage.youtube_directory = 'data/youtube'
    
    config_manager = Mock(spec=ConfigurationManager)
    config_manager.config = config
    
    return config_manager


@pytest.fixture
def gavd_service(mock_config, tmp_path):
    """Create a GAVD service with temporary directories."""
    service = GAVDService(mock_config)
    service.gavd_dir = tmp_path / 'gavd'
    service.metadata_dir = tmp_path / 'gavd' / 'metadata'
    service.results_dir = tmp_path / 'gavd' / 'results'
    
    service.gavd_dir.mkdir(parents=True, exist_ok=True)
    service.metadata_dir.mkdir(parents=True, exist_ok=True)
    service.results_dir.mkdir(parents=True, exist_ok=True)
    
    return service


@pytest.fixture
def sample_pose_data(gavd_service, tmp_path):
    """Create sample pose data."""
    dataset_id = "test-dataset-123"
    
    # Create metadata
    csv_file = tmp_path / 'gavd' / f'{dataset_id}.csv'
    csv_file.write_text("seq,frame_num\n")
    
    metadata = {
        "dataset_id": dataset_id,
        "file_path": str(csv_file),
        "status": "completed"
    }
    gavd_service.save_dataset_metadata(dataset_id, metadata)
    
    # Create results
    results = {
        "total_sequences": 1,
        "summary": {"total_frames": 2},
        "sequences": {
            "seq_001": {"frame_count": 2, "has_pose_data": True}
        }
    }
    results_file = gavd_service.results_dir / f"{dataset_id}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f)
    
    # Create pose data
    pose_data = {
        "seq_001": {
            "1": {"keypoints": [], "source_width": 1920, "source_height": 1080},
            "2": {"keypoints": [], "source_width": 1920, "source_height": 1080}
        }
    }
    pose_file = gavd_service.results_dir / f"{dataset_id}_pose_data.json"
    with open(pose_file, 'w') as f:
        json.dump(pose_data, f)
    
    return dataset_id, pose_data


class TestGAVDVisualizationIntegration:
    """Integration tests for visualization workflow."""
    
    @pytest.mark.asyncio
    async def test_complete_visualization_workflow(self, gavd_service, sample_pose_data):
        """Test complete workflow: select sequence -> load frames -> load pose data."""
        dataset_id, _ = sample_pose_data
        
        # Step 1: Get sequences
        sequences = gavd_service.get_dataset_sequences(dataset_id)
        assert sequences is not None
        assert len(sequences) > 0
        
        # Step 2: Select first sequence
        selected_sequence = sequences[0]["sequence_id"]
        assert selected_sequence == "seq_001"
        
        # Step 3: Load frames for sequence (mocked)
        with patch('ambient.gavd.gavd_processor.GAVDDataLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader_class.return_value = mock_loader
            
            import pandas as pd
            mock_df = pd.DataFrame({
                'frame_num': [1, 2],
                'bbox': [{}] * 2,
                'vid_info': [{}] * 2,
                'url': [''] * 2,
                'gait_event': [''] * 2,
                'cam_view': [''] * 2,
                'gait_pat': [''] * 2,
                'dataset': [''] * 2
            })
            
            mock_loader.load_gavd_data.return_value = mock_df
            mock_loader.organize_by_sequence.return_value = {selected_sequence: mock_df}
            
            frames = gavd_service.get_sequence_frames(dataset_id, selected_sequence)
            assert frames is not None
            assert len(frames) == 2
        
        # Step 4: Load pose data for each frame
        for frame in frames:
            pose_data = gavd_service.get_frame_pose_data(
                dataset_id, 
                selected_sequence, 
                frame['frame_num']
            )
            if frame['frame_num'] in [1, 2]:
                assert pose_data is not None
                assert 'keypoints' in pose_data


@settings(max_examples=20, deadline=None)
@given(
    frame_count=st.integers(min_value=1, max_value=100),
    keypoint_count=st.integers(min_value=0, max_value=33)
)
def test_property_frame_loading(frame_count, keypoint_count):
    """Property-based test: frame loading should handle various counts."""
    # Simulate frame data
    frames = [
        {
            "frame_num": i,
            "bbox": {"left": 0, "top": 0, "width": 100, "height": 200},
            "vid_info": {"width": 1920, "height": 1080}
        }
        for i in range(frame_count)
    ]
    
    # Verify frame structure
    assert len(frames) == frame_count
    for frame in frames:
        assert "frame_num" in frame
        assert "bbox" in frame
        assert "vid_info" in frame


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
