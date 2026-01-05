"""
Comprehensive unit tests for Frame and FrameSequence classes.

Tests the flexible frame data model with various data sources and formats,
focusing on real functionality with minimal mocking.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from tests.conftest import skip_if_no_opencv, skip_if_no_ambient
from tests.utils.test_helpers import FileManager, PerformanceProfiler
from tests.utils.assertions import assert_frame_sequence_valid, AssertionHelpers

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ambient.core.frame import Frame, FrameSequence
    AMBIENT_AVAILABLE = True
except ImportError:
    AMBIENT_AVAILABLE = False


@skip_if_no_ambient()
class TestFrameInitialization:
    """Test Frame object initialization and basic properties."""
    
    def test_frame_init_with_minimal_data(self):
        """Test Frame initialization with minimal required data."""
        frame = Frame()
        
        assert frame.data is None
        assert frame.source_type == "array"
        assert frame.metadata == {}
        assert frame.format == 'RGB'
        assert frame.lazy_load is False
        assert frame.is_loaded is False
    
    def test_frame_init_with_numpy_array(self):
        """Test Frame initialization with numpy array data."""
        test_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        frame = Frame(
            data=test_data,
            source_type="array",
            metadata={"source": "test"},
            format='RGB',
            lazy_load=False
        )
        
        assert np.array_equal(frame.data, test_data)
        assert frame.source_type == "array"
        assert frame.metadata["source"] == "test"
        assert frame.format == 'RGB'
        assert frame.lazy_load is False
        assert frame.is_loaded is True  # Should be loaded since lazy_load=False and data is provided
    
    def test_frame_init_with_file_path(self, tmp_path):
        """Test Frame initialization with file path."""
        test_file = tmp_path / "test_image.jpg"
        test_file.write_bytes(b"fake image data")
        
        frame = Frame(
            data=str(test_file),
            source_type="file",
            metadata={"source_path": str(test_file)},
            format='RGB',
            lazy_load=True
        )
        
        assert frame.data == str(test_file)
        assert frame.source_type == "file"
        assert frame.metadata["source_path"] == str(test_file)
        assert frame.format == 'RGB'
        assert frame.lazy_load is True
        assert frame.is_loaded is False  # Not loaded yet due to lazy_load=True
    
    def test_frame_init_with_video_data(self, tmp_path):
        """Test Frame initialization with video data."""
        test_video = tmp_path / "test_video.mp4"
        test_video.write_bytes(b"fake video data")
        
        video_data = {"video_path": str(test_video), "frame_index": 10}
        
        frame = Frame(
            data=video_data,
            source_type="video",
            metadata={"frame_index": 10, "timestamp": 0.333},
            format='RGB',
            lazy_load=True
        )
        
        assert frame.data == video_data
        assert frame.source_type == "video"
        assert frame.metadata["frame_index"] == 10
        assert frame.metadata["timestamp"] == 0.333
        assert frame.format == 'RGB'
        assert frame.lazy_load is True
        assert frame.is_loaded is False  # Not loaded yet
    
    def test_frame_init_with_url(self):
        """Test Frame initialization with URL."""
        test_url = "https://example.com/image.jpg"
        
        frame = Frame(
            data=test_url,
            source_type="url",
            metadata={"source_url": test_url, "timeout": 30},
            format='RGB',
            lazy_load=True
        )
        
        assert frame.data == test_url
        assert frame.source_type == "url"
        assert frame.metadata["source_url"] == test_url
        assert frame.metadata["timeout"] == 30
        assert frame.format == 'RGB'
        assert frame.lazy_load is True
        assert frame.is_loaded is False  # Not loaded yet


@skip_if_no_ambient()
class TestFrameProperties:
    """Test Frame properties and computed attributes."""
    
    def test_is_loaded_property(self):
        """Test is_loaded property correctly identifies loaded frames."""
        # Frame without data
        frame = Frame()
        assert frame.is_loaded is False
        
        # Frame with data but lazy_load=True (not loaded initially)
        test_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame_lazy = Frame(data=test_data, lazy_load=True)
        assert frame_lazy.is_loaded is False
        
        # Frame with data and lazy_load=False (loaded immediately)
        frame_loaded = Frame(data=test_data, lazy_load=False)
        assert frame_loaded.is_loaded is True
        
        # Frame after unloading (only works with lazy_load=True)
        frame_lazy_loaded = Frame(data=test_data, lazy_load=True)
        frame_lazy_loaded.load()  # Load it first
        assert frame_lazy_loaded.is_loaded is True
        
        frame_lazy_loaded.unload()  # Now unload it
        assert frame_lazy_loaded.is_loaded is False
    
    def test_shape_property_with_loaded_data(self):
        """Test shape property when data is loaded."""
        test_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame = Frame(data=test_data, lazy_load=False)
        
        assert frame.shape == (480, 640, 3)
    
    def test_shape_property_with_metadata(self):
        """Test shape property when using metadata without loaded data."""
        frame = Frame(
            metadata={"shape": (720, 1280, 3)},
            lazy_load=True
        )
        
        assert frame.shape == (720, 1280, 3)
    
    def test_shape_property_no_data_no_metadata(self):
        """Test shape property when no data or metadata available."""
        frame = Frame()
        assert frame.shape is None


@skip_if_no_ambient()
@skip_if_no_opencv()
class TestFrameDataLoading:
    """Test Frame data loading from various sources."""
    
    def test_load_from_numpy_array(self):
        """Test loading frame data from numpy array."""
        test_data = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        frame = Frame(data=test_data, lazy_load=False)
        
        loaded_data = frame.load()
        
        assert np.array_equal(loaded_data, test_data)
        assert frame.is_loaded is True
        assert frame.shape == (240, 320, 3)
    
    def test_load_from_file_path(self, sample_image_file):
        """Test loading frame data from image file."""
        frame = Frame(
            data=str(sample_image_file),
            source_type="file",
            lazy_load=True
        )
        
        # Mock cv2.imread to avoid actual file I/O complexity
        with patch('cv2.imread') as mock_imread, \
             patch('cv2.cvtColor') as mock_cvtcolor:
            
            # Mock image data (BGR format from cv2.imread)
            mock_bgr_data = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            mock_rgb_data = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            
            mock_imread.return_value = mock_bgr_data
            mock_cvtcolor.return_value = mock_rgb_data
            
            loaded_data = frame.load()
            
            # Verify cv2 functions were called correctly
            mock_imread.assert_called_once_with(str(sample_image_file))
            mock_cvtcolor.assert_called_once_with(mock_bgr_data, cv2.COLOR_BGR2RGB)
            
            assert np.array_equal(loaded_data, mock_rgb_data)
            assert frame.is_loaded is True
    
    def test_load_from_nonexistent_file(self, tmp_path):
        """Test loading from non-existent file raises appropriate error."""
        nonexistent_file = tmp_path / "does_not_exist.jpg"
        frame = Frame(
            data=str(nonexistent_file),
            source_type="file",
            lazy_load=True
        )
        
        with pytest.raises(Exception):  # Should raise FrameError or similar
            frame.load()
    
    def test_load_already_loaded_frame(self):
        """Test loading already loaded frame returns existing data."""
        test_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame = Frame(data=test_data, lazy_load=False)
        
        # First load (should already be loaded)
        loaded_data1 = frame.load()
        
        # Second load should return same data
        loaded_data2 = frame.load()
        
        assert np.array_equal(loaded_data1, loaded_data2)
        assert np.array_equal(loaded_data1, test_data)
    
    def test_unload_frame_data(self):
        """Test unloading frame data frees memory."""
        test_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame = Frame(data=test_data, lazy_load=True)
        
        # Load the frame first
        frame.load()
        assert frame.is_loaded is True
        
        frame.unload()
        
        assert frame.is_loaded is False
        # Original data reference should still exist
        assert frame.data is not None


@skip_if_no_ambient()
class TestFrameFormatConversion:
    """Test Frame format conversion methods."""
    
    def test_to_array_method(self):
        """Test conversion to array using to_array method."""
        test_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame = Frame(data=test_data, format='RGB', lazy_load=False)
        
        array_data = frame.to_array()
        
        assert np.array_equal(array_data, test_data)
        assert frame.is_loaded is True
    
    def test_save_frame_to_file(self, tmp_path):
        """Test saving frame to file."""
        test_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame = Frame(data=test_data, lazy_load=False)
        
        output_path = tmp_path / "saved_frame.jpg"
        
        with patch('cv2.cvtColor') as mock_cvtcolor, \
             patch('cv2.imwrite') as mock_imwrite:
            
            mock_bgr_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            mock_cvtcolor.return_value = mock_bgr_data
            mock_imwrite.return_value = True
            
            frame.save(output_path)
            
            # Check that cv2.cvtColor was called with correct parameters
            assert mock_cvtcolor.call_count == 1
            call_args = mock_cvtcolor.call_args
            assert np.array_equal(call_args[0][0], test_data)
            assert call_args[0][1] == cv2.COLOR_RGB2BGR
            
            # Check that cv2.imwrite was called
            mock_imwrite.assert_called_once_with(str(output_path), mock_bgr_data)
    
    def test_copy_frame(self):
        """Test creating a copy of a frame."""
        test_data = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        original_frame = Frame(
            data=test_data, 
            format='RGB', 
            metadata={"test": "value"},
            lazy_load=False
        )
        
        copied_frame = original_frame.copy()
        
        # Should be different objects
        assert copied_frame is not original_frame
        
        # But should have same data and format
        assert np.array_equal(copied_frame.to_array(), original_frame.to_array())
        assert copied_frame.format == original_frame.format
        
        # Original metadata should be preserved (copy may add additional metadata)
        assert "test" in copied_frame.metadata
        assert copied_frame.metadata["test"] == "value"
        
        # Modifying copy shouldn't affect original
        copied_frame.metadata["new_key"] = "new_value"
        assert "new_key" not in original_frame.metadata


@skip_if_no_ambient()
class TestFrameSequence:
    """Test FrameSequence functionality."""
    
    def test_frame_sequence_initialization(self):
        """Test FrameSequence initialization."""
        frames = [
            Frame(metadata={"frame_number": 0}),
            Frame(metadata={"frame_number": 1}),
            Frame(metadata={"frame_number": 2})
        ]
        
        sequence = FrameSequence(
            sequence_id="test_sequence",
            frames=frames,
            metadata={"fps": 30.0, "source": "test"}
        )
        
        assert sequence.sequence_id == "test_sequence"
        assert len(sequence.frames) == 3
        assert len(sequence) == 3
        assert sequence.metadata["fps"] == 30.0
        assert sequence.metadata["source"] == "test"
    
    def test_frame_sequence_indexing(self):
        """Test frame sequence indexing and slicing."""
        test_data1 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        test_data2 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        test_data3 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        frames = [
            Frame(data=test_data1, lazy_load=False),
            Frame(data=test_data2, lazy_load=False),
            Frame(data=test_data3, lazy_load=False)
        ]
        
        sequence = FrameSequence(sequence_id="indexing_test", frames=frames)
        
        # Test single frame access
        frame_1 = sequence[1]
        assert np.array_equal(frame_1.to_array(), test_data2)
        
        # Test slice access
        sub_sequence = sequence[0:2]
        assert isinstance(sub_sequence, FrameSequence)
        assert len(sub_sequence) == 2
        assert np.array_equal(sub_sequence[0].to_array(), test_data1)
        assert np.array_equal(sub_sequence[1].to_array(), test_data2)
    
    def test_frame_sequence_load_all(self):
        """Test loading all frames in sequence."""
        test_data1 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        test_data2 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        frames = [
            Frame(data=test_data1, lazy_load=True),  # Not loaded initially
            Frame(data=test_data2, lazy_load=True)   # Not loaded initially
        ]
        
        sequence = FrameSequence(sequence_id="load_all_test", frames=frames)
        
        # Initially, frames should not be loaded (lazy_load=True)
        assert not any(frame.is_loaded for frame in sequence.frames)
        
        # Load all frames
        loaded_arrays = sequence.load_all()
        
        # Now all frames should be loaded
        assert all(frame.is_loaded for frame in sequence.frames)
        assert len(loaded_arrays) == 2
        assert np.array_equal(loaded_arrays[0], test_data1)
        assert np.array_equal(loaded_arrays[1], test_data2)
    
    def test_frame_sequence_unload_all(self):
        """Test unloading all frames in sequence."""
        test_data1 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        test_data2 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        frames = [
            Frame(data=test_data1, lazy_load=True),  # Use lazy_load=True so unload works
            Frame(data=test_data2, lazy_load=True)   # Use lazy_load=True so unload works
        ]
        
        # Load the frames first
        for frame in frames:
            frame.load()
        
        sequence = FrameSequence(sequence_id="unload_all_test", frames=frames)
        
        # Initially loaded
        assert all(frame.is_loaded for frame in sequence.frames)
        
        # Unload all
        sequence.unload_all()
        
        assert all(not frame.is_loaded for frame in sequence.frames)
    
    def test_frame_sequence_append_extend(self):
        """Test appending and extending frame sequences."""
        test_data1 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        test_data2 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        test_data3 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        sequence = FrameSequence(sequence_id="append_test")
        
        # Test append
        frame1 = Frame(data=test_data1, lazy_load=False)
        sequence.append(frame1)
        assert len(sequence) == 1
        
        # Test extend
        frames_to_add = [
            Frame(data=test_data2, lazy_load=False),
            Frame(data=test_data3, lazy_load=False)
        ]
        sequence.extend(frames_to_add)
        assert len(sequence) == 3
        
        # Verify data integrity
        assert np.array_equal(sequence[0].to_array(), test_data1)
        assert np.array_equal(sequence[1].to_array(), test_data2)
        assert np.array_equal(sequence[2].to_array(), test_data3)


@skip_if_no_ambient()
class TestFrameSequenceValidation:
    """Test FrameSequence validation and integrity checks."""
    
    def test_frame_sequence_basic_validation(self):
        """Test that frame sequence validates basic properties."""
        # Valid sequence with frames
        test_data1 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        test_data2 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        frames = [
            Frame(data=test_data1, metadata={"frame_number": 0, "timestamp": 0.0}),
            Frame(data=test_data2, metadata={"frame_number": 1, "timestamp": 0.033})
        ]
        
        valid_sequence = FrameSequence(sequence_id="valid_test", frames=frames)
        assert_frame_sequence_valid(valid_sequence)
    
    def test_empty_frame_sequence(self):
        """Test handling of empty frame sequence."""
        empty_sequence = FrameSequence(sequence_id="empty_test", frames=[])
        
        assert len(empty_sequence) == 0
        
        # Empty sequence should fail validation
        with pytest.raises(AssertionError):
            assert_frame_sequence_valid(empty_sequence)
    
    def test_frame_sequence_memory_stats(self):
        """Test memory statistics for frame sequence."""
        test_data1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_data2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        frames = [
            Frame(data=test_data1, lazy_load=False),  # Loaded
            Frame(data=test_data2, lazy_load=True)    # Not loaded
        ]
        
        sequence = FrameSequence(sequence_id="memory_stats_test", frames=frames)
        
        stats = sequence.get_memory_stats()
        
        assert stats['sequence_id'] == "memory_stats_test"
        assert stats['total_frames'] == 2
        assert stats['loaded_frames'] == 1  # Only first frame is loaded
        assert stats['total_memory_mb'] > 0  # Should have some memory usage


@pytest.mark.performance
@skip_if_no_ambient()
class TestFramePerformance:
    """Test Frame and FrameSequence performance characteristics."""
    
    def test_frame_loading_performance(self):
        """Test frame loading performance with different data sizes."""
        profiler = PerformanceProfiler()
        
        # Test different image sizes
        sizes = [(100, 100), (640, 480), (1920, 1080)]
        
        for width, height in sizes:
            test_data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            frame = Frame(data=test_data, lazy_load=True)
            
            with profiler.profile(f"load_{width}x{height}"):
                loaded_data = frame.load()
            
            assert np.array_equal(loaded_data, test_data)
        
        # Verify performance scales reasonably
        small_time = profiler.get_metrics("load_100x100")["execution_time"]
        large_time = profiler.get_metrics("load_1920x1080")["execution_time"]
        
        # Large image shouldn't take more than 100x longer than small image
        # (this is a very generous bound for in-memory operations)
        # Add small epsilon to handle zero execution times
        epsilon = 0.001
        assert large_time < (small_time + epsilon) * 100
    
    def test_frame_sequence_memory_management(self):
        """Test memory management in frame sequences."""
        profiler = PerformanceProfiler()
        
        # Create sequence with multiple frames
        frames = []
        for i in range(10):
            test_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame = Frame(data=test_data, lazy_load=True)
            frames.append(frame)
        
        sequence = FrameSequence(sequence_id="memory_test", frames=frames)
        
        with profiler.profile("load_all_frames"):
            sequence.load_all()
        
        with profiler.profile("unload_all_frames"):
            sequence.unload_all()
        
        # Verify all frames are unloaded
        assert all(not frame.is_loaded for frame in sequence.frames)
        
        # Memory operations should be fast
        load_time = profiler.get_metrics("load_all_frames")["execution_time"]
        unload_time = profiler.get_metrics("unload_all_frames")["execution_time"]
        
        # These should complete quickly (< 1 second for 10 frames)
        assert load_time < 1.0
        assert unload_time < 1.0
    
    def test_frame_sequence_batch_processing(self):
        """Test batch processing performance."""
        profiler = PerformanceProfiler()
        
        # Create sequence with frames
        frames = []
        for i in range(20):
            test_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            frame = Frame(data=test_data, lazy_load=True)
            frames.append(frame)
        
        sequence = FrameSequence(sequence_id="batch_test", frames=frames)
        
        def simple_processor(batch_data, start_idx):
            """Simple processor that just returns batch size."""
            return len(batch_data)
        
        with profiler.profile("batch_processing"):
            results = sequence.process_in_batches(simple_processor, batch_size=5)
        
        # Should have 4 batches (20 frames / 5 per batch)
        assert len(results) == 4
        assert all(result == 5 for result in results)
        
        # Batch processing should be efficient
        batch_time = profiler.get_metrics("batch_processing")["execution_time"]
        assert batch_time < 2.0  # Should complete in under 2 seconds


@skip_if_no_ambient()
class TestFrameErrorHandling:
    """Test Frame error handling and edge cases."""
    
    def test_frame_invalid_data_source(self):
        """Test frame with no valid data source."""
        frame = Frame()  # No data provided
        
        with pytest.raises(Exception):  # Should raise FrameError or similar
            frame.load()
    
    def test_frame_invalid_format_conversion(self):
        """Test frame format conversion with invalid data."""
        frame = Frame()  # No data provided
        
        # Try to convert without loaded data
        with pytest.raises(Exception):  # Should raise FrameError or similar
            frame.to_array()
    
    def test_frame_sequence_with_mixed_frame_types(self):
        """Test frame sequence with frames from different sources."""
        test_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        frames = [
            Frame(data=test_data, lazy_load=False),
            Frame(data="fake_file.jpg", source_type="file", lazy_load=True),
            Frame(data="https://example.com/image.jpg", source_type="url", lazy_load=True)
        ]
        
        sequence = FrameSequence(sequence_id="mixed_test", frames=frames)
        
        assert len(sequence) == 3
        assert sequence.frames[0].is_loaded is True
        assert sequence.frames[1].is_loaded is False
        assert sequence.frames[2].is_loaded is False
    
    def test_frame_resize_functionality(self):
        """Test frame resizing functionality."""
        test_data = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        frame = Frame(data=test_data, lazy_load=False)
        
        with patch('cv2.resize') as mock_resize:
            resized_data = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
            mock_resize.return_value = resized_data
            
            resized_frame = frame.resize(150, 100)
            
            # Check that cv2.resize was called with correct parameters
            assert mock_resize.call_count == 1
            call_args = mock_resize.call_args
            assert np.array_equal(call_args[0][0], test_data)
            assert call_args[0][1] == (150, 100)
            
            assert np.array_equal(resized_frame.to_array(), resized_data)
            assert resized_frame.metadata["original_shape"] == (200, 300, 3)
            assert resized_frame.metadata["resized_to"] == (100, 150, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])