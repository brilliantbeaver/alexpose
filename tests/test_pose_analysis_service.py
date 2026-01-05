"""
Unit tests for pose analysis service.

Tests the PoseAnalysisServiceAPI class with various scenarios including:
- Normal operation
- Error handling
- Caching behavior
- Edge cases

Author: AlexPose Team
Date: January 4, 2026
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from server.services.pose_analysis_service import PoseAnalysisServiceAPI


class TestPoseAnalysisServiceAPI:
    """Test suite for PoseAnalysisServiceAPI"""
    
    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock configuration manager"""
        config = Mock()
        config.config = Mock()
        config.config.storage = Mock()
        
        # Set all storage paths to strings (not Mocks)
        config.config.storage.cache_directory = str(tmp_path / 'cache')
        config.config.storage.training_directory = str(tmp_path / 'training')
        config.config.storage.youtube_directory = str(tmp_path / 'youtube')
        config.config.storage.data_directory = str(tmp_path / 'data')
        
        return config
    
    @pytest.fixture
    def service(self, mock_config):
        """Create service instance with temporary cache directory"""
        return PoseAnalysisServiceAPI(mock_config)
    
    @pytest.fixture
    def sample_pose_sequence(self):
        """Create sample pose sequence data"""
        return [
            {
                "frame_num": i,
                "keypoints": [
                    {"x": 100 + i, "y": 200 + i, "confidence": 0.9, "keypoint_id": j}
                    for j in range(17)  # COCO_17 format
                ]
            }
            for i in range(30)  # 30 frames
        ]
    
    def test_initialization(self, service):
        """Test service initialization"""
        assert service is not None
        assert service.analyzer is not None
        assert service.gavd_service is not None
        assert service.cache_dir.exists()
    
    def test_get_sequence_analysis_no_pose_data(self, service):
        """Test analysis when no pose data is available"""
        with patch.object(service, '_load_pose_sequence', return_value=[]):
            result = service.get_sequence_analysis('dataset1', 'seq1')
            
            assert result is not None
            assert 'error' in result
            assert result['error'] == 'no_pose_data'
    
    def test_get_sequence_analysis_success(self, service, sample_pose_sequence):
        """Test successful analysis"""
        with patch.object(service, '_load_pose_sequence', return_value=sample_pose_sequence):
            result = service.get_sequence_analysis('dataset1', 'seq1', use_cache=False)
            
            assert result is not None
            assert 'error' not in result
            assert 'sequence_info' in result or 'metadata' in result
            assert 'features' in result
            assert 'summary' in result
    
    def test_get_sequence_analysis_with_cache(self, service, sample_pose_sequence):
        """Test caching behavior"""
        with patch.object(service, '_load_pose_sequence', return_value=sample_pose_sequence):
            # First call - should analyze and cache
            result1 = service.get_sequence_analysis('dataset1', 'seq1', use_cache=True)
            assert result1 is not None
            
            # Second call - should use cache
            with patch.object(service.analyzer, 'analyze_gait_sequence') as mock_analyze:
                result2 = service.get_sequence_analysis('dataset1', 'seq1', use_cache=True)
                assert result2 is not None
                # Analyzer should not be called (using cache)
                mock_analyze.assert_not_called()
    
    def test_get_sequence_analysis_force_refresh(self, service, sample_pose_sequence):
        """Test force refresh bypasses cache"""
        with patch.object(service, '_load_pose_sequence', return_value=sample_pose_sequence):
            # First call - cache result
            result1 = service.get_sequence_analysis('dataset1', 'seq1', use_cache=True)
            assert result1 is not None
            
            # Second call with force_refresh - should re-analyze
            with patch.object(service.analyzer, 'analyze_gait_sequence') as mock_analyze:
                mock_analyze.return_value = {"test": "data"}
                result2 = service.get_sequence_analysis(
                    'dataset1', 'seq1', 
                    use_cache=True, 
                    force_refresh=True
                )
                # Analyzer should be called
                mock_analyze.assert_called_once()
    
    def test_get_sequence_features(self, service, sample_pose_sequence):
        """Test getting features only"""
        with patch.object(service, '_load_pose_sequence', return_value=sample_pose_sequence):
            result = service.get_sequence_features('dataset1', 'seq1')
            
            assert result is not None
            assert 'features' in result
            assert 'dataset_id' in result
            assert 'sequence_id' in result
    
    def test_get_sequence_cycles(self, service, sample_pose_sequence):
        """Test getting gait cycles only"""
        with patch.object(service, '_load_pose_sequence', return_value=sample_pose_sequence):
            result = service.get_sequence_cycles('dataset1', 'seq1')
            
            assert result is not None
            assert 'gait_cycles' in result
            assert 'dataset_id' in result
            assert 'sequence_id' in result
    
    def test_get_sequence_symmetry(self, service, sample_pose_sequence):
        """Test getting symmetry analysis only"""
        with patch.object(service, '_load_pose_sequence', return_value=sample_pose_sequence):
            result = service.get_sequence_symmetry('dataset1', 'seq1')
            
            assert result is not None
            assert 'symmetry_analysis' in result
            assert 'dataset_id' in result
            assert 'sequence_id' in result
    
    def test_invalid_inputs(self, service):
        """Test validation of invalid inputs"""
        with pytest.raises(ValueError):
            service.get_sequence_analysis('', 'seq1')
        
        with pytest.raises(ValueError):
            service.get_sequence_analysis('dataset1', '')
        
        with pytest.raises(ValueError):
            service.get_sequence_analysis(None, 'seq1')
    
    def test_clear_cache_specific_sequence(self, service, sample_pose_sequence):
        """Test clearing cache for specific sequence"""
        with patch.object(service, '_load_pose_sequence', return_value=sample_pose_sequence):
            # Create cache
            service.get_sequence_analysis('dataset1', 'seq1', use_cache=True)
            
            # Clear cache
            deleted = service.clear_cache('dataset1', 'seq1')
            assert deleted == 1
            
            # Verify cache is cleared
            cached = service._get_cached_analysis('dataset1', 'seq1')
            assert cached is None
    
    def test_clear_cache_dataset(self, service, sample_pose_sequence):
        """Test clearing cache for entire dataset"""
        with patch.object(service, '_load_pose_sequence', return_value=sample_pose_sequence):
            # Create multiple caches
            service.get_sequence_analysis('dataset1', 'seq1', use_cache=True)
            service.get_sequence_analysis('dataset1', 'seq2', use_cache=True)
            
            # Clear dataset cache
            deleted = service.clear_cache('dataset1')
            assert deleted == 2
    
    def test_clear_all_cache(self, service, sample_pose_sequence):
        """Test clearing all cache"""
        with patch.object(service, '_load_pose_sequence', return_value=sample_pose_sequence):
            # Create multiple caches
            service.get_sequence_analysis('dataset1', 'seq1', use_cache=True)
            service.get_sequence_analysis('dataset2', 'seq1', use_cache=True)
            
            # Clear all cache
            deleted = service.clear_cache()
            assert deleted == 2
    
    def test_get_cache_stats(self, service, sample_pose_sequence):
        """Test getting cache statistics"""
        with patch.object(service, '_load_pose_sequence', return_value=sample_pose_sequence):
            # Create some cache
            service.get_sequence_analysis('dataset1', 'seq1', use_cache=True)
            
            # Get stats
            stats = service.get_cache_stats()
            assert 'total_files' in stats
            assert 'total_size_bytes' in stats
            assert stats['total_files'] >= 1
    
    def test_load_pose_sequence_with_gavd_service(self, service):
        """Test loading pose sequence from GAVD service"""
        # Mock GAVD service methods
        mock_frames = [
            {"frame_num": 1},
            {"frame_num": 2},
            {"frame_num": 3}
        ]
        
        mock_pose_data = {
            'keypoints': [
                {"x": 100, "y": 200, "confidence": 0.9, "keypoint_id": i}
                for i in range(17)
            ],
            'source_width': 640,
            'source_height': 480
        }
        
        with patch.object(service.gavd_service, 'get_sequence_frames', return_value=mock_frames):
            with patch.object(service.gavd_service, 'get_frame_pose_data', return_value=mock_pose_data):
                result = service._load_pose_sequence('dataset1', 'seq1')
                
                assert len(result) == 3
                assert all('keypoints' in frame for frame in result)
                assert all('frame_num' in frame for frame in result)
    
    def test_load_pose_sequence_old_format(self, service):
        """Test loading pose sequence with old format (list instead of dict)"""
        mock_frames = [{"frame_num": 1}]
        
        # Old format: just a list of keypoints
        mock_pose_data = [
            {"x": 100, "y": 200, "confidence": 0.9, "keypoint_id": i}
            for i in range(17)
        ]
        
        with patch.object(service.gavd_service, 'get_sequence_frames', return_value=mock_frames):
            with patch.object(service.gavd_service, 'get_frame_pose_data', return_value=mock_pose_data):
                result = service._load_pose_sequence('dataset1', 'seq1')
                
                assert len(result) == 1
                assert 'keypoints' in result[0]
    
    def test_load_pose_sequence_partial_data(self, service):
        """Test loading pose sequence when some frames don't have pose data"""
        mock_frames = [
            {"frame_num": 1},
            {"frame_num": 2},
            {"frame_num": 3}
        ]
        
        mock_pose_data = {
            'keypoints': [
                {"x": 100, "y": 200, "confidence": 0.9, "keypoint_id": i}
                for i in range(17)
            ]
        }
        
        # Mock to return pose data for only some frames
        def mock_get_pose_data(dataset_id, sequence_id, frame_num):
            if frame_num == 2:
                return None  # No pose data for frame 2
            return mock_pose_data
        
        with patch.object(service.gavd_service, 'get_sequence_frames', return_value=mock_frames):
            with patch.object(service.gavd_service, 'get_frame_pose_data', side_effect=mock_get_pose_data):
                result = service._load_pose_sequence('dataset1', 'seq1')
                
                # Should only have 2 frames (1 and 3)
                assert len(result) == 2
                assert result[0]['frame_num'] == 1
                assert result[1]['frame_num'] == 3
    
    def test_analysis_performance_metadata(self, service, sample_pose_sequence):
        """Test that performance metadata is included in results"""
        with patch.object(service, '_load_pose_sequence', return_value=sample_pose_sequence):
            result = service.get_sequence_analysis('dataset1', 'seq1', use_cache=False)
            
            assert 'performance' in result
            assert 'analysis_time_seconds' in result['performance']
            assert 'frames_per_second' in result['performance']
            assert result['performance']['analysis_time_seconds'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
