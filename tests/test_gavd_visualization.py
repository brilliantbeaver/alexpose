"""
Tests for GAVD visualization features.

Tests bounding box overlay, pose keypoint visualization, and frame navigation.
"""

import pytest
from pathlib import Path
import json
import tempfile

from server.services.gavd_service import GAVDService
from ambient.core.config import ConfigurationManager


@pytest.fixture
def gavd_service(tmp_path):
    """Create GAVD service with temporary directories."""
    # Create mock configuration
    class MockConfig:
        class storage:
            training_directory = str(tmp_path / "training")
            youtube_directory = str(tmp_path / "youtube")
    
    class MockConfigManager:
        config = MockConfig()
    
    service = GAVDService(MockConfigManager())
    return service


@pytest.fixture
def mock_dataset(gavd_service, tmp_path):
    """Create mock dataset with frames and pose data."""
    dataset_id = "test_dataset_123"
    
    # Create mock CSV file
    csv_content = """seq,frame_num,cam_view,gait_event,dataset,gait_pat,bbox,vid_info,id,url
test_seq_001,100,front,heel_strike,GAVD,normal,"{""left"":100,""top"":50,""width"":200,""height"":400}","{""width"":1280,""height"":720}",test001,https://www.youtube.com/watch?v=test123
test_seq_001,101,front,toe_off,GAVD,normal,"{""left"":102,""top"":51,""width"":200,""height"":400}","{""width"":1280,""height"":720}",test002,https://www.youtube.com/watch?v=test123
"""
    
    csv_path = gavd_service.gavd_dir / f"{dataset_id}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(csv_content)
    
    # Create metadata
    metadata = {
        "dataset_id": dataset_id,
        "original_filename": "test.csv",
        "file_path": str(csv_path),
        "status": "completed",
        "row_count": 2,
        "sequence_count": 1
    }
    gavd_service.save_dataset_metadata(dataset_id, metadata)
    
    # Create mock pose data
    pose_data = {
        "test_seq_001": {
            "100": [
                {"x": 150.0, "y": 100.0, "confidence": 0.95, "keypoint_id": 0},
                {"x": 155.0, "y": 120.0, "confidence": 0.92, "keypoint_id": 1},
                {"x": 160.0, "y": 140.0, "confidence": 0.88, "keypoint_id": 2}
            ],
            "101": [
                {"x": 152.0, "y": 102.0, "confidence": 0.94, "keypoint_id": 0},
                {"x": 157.0, "y": 122.0, "confidence": 0.91, "keypoint_id": 1},
                {"x": 162.0, "y": 142.0, "confidence": 0.87, "keypoint_id": 2}
            ]
        }
    }
    
    pose_file = gavd_service.results_dir / f"{dataset_id}_pose_data.json"
    pose_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pose_file, 'w') as f:
        json.dump(pose_data, f)
    
    return dataset_id, "test_seq_001"


class TestBoundingBoxVisualization:
    """Test bounding box overlay functionality."""
    
    def test_get_sequence_frames_with_bbox(self, gavd_service, mock_dataset):
        """Test retrieving frames with bounding box data."""
        dataset_id, sequence_id = mock_dataset
        
        frames = gavd_service.get_sequence_frames(dataset_id, sequence_id)
        
        assert frames is not None
        assert len(frames) == 2
        
        # Check first frame
        frame1 = frames[0]
        assert frame1["frame_num"] == 100
        assert "bbox" in frame1
        assert frame1["bbox"]["left"] == 100
        assert frame1["bbox"]["top"] == 50
        assert frame1["bbox"]["width"] == 200
        assert frame1["bbox"]["height"] == 400
    
    def test_bbox_scaling_calculation(self):
        """Test bounding box scaling for different resolutions."""
        # Original bbox at 1280x720
        bbox = {"left": 100, "top": 50, "width": 200, "height": 400}
        annotation_width = 1280
        annotation_height = 720
        
        # Video is actually 640x360 (half resolution)
        video_width = 640
        video_height = 360
        
        scale_x = video_width / annotation_width
        scale_y = video_height / annotation_height
        
        scaled_bbox = {
            "left": bbox["left"] * scale_x,
            "top": bbox["top"] * scale_y,
            "width": bbox["width"] * scale_x,
            "height": bbox["height"] * scale_y
        }
        
        assert scaled_bbox["left"] == 50
        assert scaled_bbox["top"] == 25
        assert scaled_bbox["width"] == 100
        assert scaled_bbox["height"] == 200
    
    def test_bbox_no_scaling_needed(self):
        """Test bounding box when no scaling is needed."""
        bbox = {"left": 100, "top": 50, "width": 200, "height": 400}
        annotation_width = 1280
        annotation_height = 720
        video_width = 1280
        video_height = 720
        
        scale_x = video_width / annotation_width
        scale_y = video_height / annotation_height
        
        assert scale_x == 1.0
        assert scale_y == 1.0


class TestPoseVisualization:
    """Test pose keypoint overlay functionality."""
    
    def test_get_frame_pose_data(self, gavd_service, mock_dataset):
        """Test retrieving pose keypoints for a frame."""
        dataset_id, sequence_id = mock_dataset
        
        pose_data = gavd_service.get_frame_pose_data(dataset_id, sequence_id, 100)
        
        assert pose_data is not None
        
        # New format returns dict with keypoints and source dimensions
        assert 'keypoints' in pose_data
        keypoints = pose_data['keypoints']
        assert len(keypoints) == 3
        
        # Check first keypoint
        kp1 = keypoints[0]
        assert kp1["x"] == 150.0
        assert kp1["y"] == 100.0
        assert kp1["confidence"] == 0.95
        assert kp1["keypoint_id"] == 0
    
    def test_pose_data_multiple_frames(self, gavd_service, mock_dataset):
        """Test pose data for multiple frames."""
        dataset_id, sequence_id = mock_dataset
        
        pose_100 = gavd_service.get_frame_pose_data(dataset_id, sequence_id, 100)
        pose_101 = gavd_service.get_frame_pose_data(dataset_id, sequence_id, 101)
        
        assert pose_100 is not None
        assert pose_101 is not None
        
        # New format returns dict with keypoints
        kp_100 = pose_100['keypoints']
        kp_101 = pose_101['keypoints']
        
        assert kp_100 != kp_101
        
        # Verify keypoints moved
        assert kp_100[0]["x"] != kp_101[0]["x"]
        assert kp_100[0]["y"] != kp_101[0]["y"]
    
    def test_pose_data_not_found(self, gavd_service, mock_dataset):
        """Test retrieving pose data for non-existent frame."""
        dataset_id, sequence_id = mock_dataset
        
        pose_data = gavd_service.get_frame_pose_data(dataset_id, sequence_id, 999)
        
        assert pose_data is None
    
    def test_pose_keypoint_confidence_filtering(self):
        """Test filtering keypoints by confidence threshold."""
        keypoints = [
            {"x": 100, "y": 100, "confidence": 0.95, "keypoint_id": 0},
            {"x": 110, "y": 110, "confidence": 0.25, "keypoint_id": 1},
            {"x": 120, "y": 120, "confidence": 0.85, "keypoint_id": 2},
            {"x": 130, "y": 130, "confidence": 0.15, "keypoint_id": 3}
        ]
        
        threshold = 0.3
        filtered = [kp for kp in keypoints if kp["confidence"] > threshold]
        
        assert len(filtered) == 2
        assert filtered[0]["keypoint_id"] == 0
        assert filtered[1]["keypoint_id"] == 2


class TestFrameNavigation:
    """Test frame navigation and seeking."""
    
    def test_frame_to_time_conversion(self):
        """Test converting frame number to video time."""
        fps = 30
        frame_num = 100  # 1-based
        
        # Convert to 0-based
        frame_index = frame_num - 1
        time_seconds = frame_index / fps
        
        assert time_seconds == 99 / 30
        assert time_seconds == pytest.approx(3.3, rel=0.01)
    
    def test_time_to_frame_conversion(self):
        """Test converting video time to frame number."""
        fps = 30
        time_seconds = 3.3
        
        # Convert to frame index (0-based)
        frame_index = int(time_seconds * fps)
        # Convert to frame number (1-based)
        frame_num = frame_index + 1
        
        assert frame_num == 100
    
    def test_frame_range_validation(self, gavd_service, mock_dataset):
        """Test frame range validation."""
        dataset_id, sequence_id = mock_dataset
        
        frames = gavd_service.get_sequence_frames(dataset_id, sequence_id)
        
        assert frames is not None
        assert len(frames) > 0
        
        # Get frame range
        min_frame = min(f["frame_num"] for f in frames)
        max_frame = max(f["frame_num"] for f in frames)
        
        assert min_frame == 100
        assert max_frame == 101
        
        # Test frame in range
        assert 100 >= min_frame and 100 <= max_frame
        assert 101 >= min_frame and 101 <= max_frame
        
        # Test frame out of range
        assert not (99 >= min_frame and 99 <= max_frame)
        assert not (102 >= min_frame and 102 <= max_frame)


class TestVisualizationIntegration:
    """Test integration of visualization features."""
    
    def test_complete_frame_data(self, gavd_service, mock_dataset):
        """Test retrieving complete frame data with bbox and pose."""
        dataset_id, sequence_id = mock_dataset
        
        # Get frames
        frames = gavd_service.get_sequence_frames(dataset_id, sequence_id)
        assert frames is not None
        
        frame = frames[0]
        
        # Get pose data
        pose_data = gavd_service.get_frame_pose_data(
            dataset_id, sequence_id, frame["frame_num"]
        )
        
        # Verify complete data
        assert "bbox" in frame
        assert "vid_info" in frame
        assert "url" in frame
        assert pose_data is not None
        assert len(pose_data) > 0
    
    def test_visualization_data_consistency(self, gavd_service, mock_dataset):
        """Test consistency of visualization data across frames."""
        dataset_id, sequence_id = mock_dataset
        
        frames = gavd_service.get_sequence_frames(dataset_id, sequence_id)
        
        for frame in frames:
            # All frames should have required fields
            assert "frame_num" in frame
            assert "bbox" in frame
            assert "vid_info" in frame
            assert "url" in frame
            
            # Bbox should have required fields
            assert "left" in frame["bbox"]
            assert "top" in frame["bbox"]
            assert "width" in frame["bbox"]
            assert "height" in frame["bbox"]
            
            # Vid info should have dimensions
            assert "width" in frame["vid_info"]
            assert "height" in frame["vid_info"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
