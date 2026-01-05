"""
Tests for Frame lazy loading and memory management capabilities.

This module tests the enhanced memory management features including:
- Lazy loading of frame data
- Automatic memory cleanup
- LRU cache management
- Memory usage tracking
- Batch processing with memory optimization
"""

import numpy as np
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from ambient.core.frame import (
    Frame, 
    FrameSequence, 
    MemoryOptimizedFrameSequence,
    TemporaryMemorySettings,
    get_global_memory_stats,
    set_global_memory_threshold,
    force_global_cleanup,
    _memory_manager
)


class TestFrameLazyLoading:
    """Test lazy loading functionality in Frame objects."""
    
    def test_lazy_loading_enabled_by_default(self):
        """Test that frames created with lazy_load=True don't load data immediately."""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        
        try:
            # Create a simple test image
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            # Mock cv2.imread to return our test image
            with patch('ambient.core.frame.cv2') as mock_cv2:
                mock_cv2.imread.return_value = test_image
                mock_cv2.cvtColor.return_value = test_image
                
                # Create frame with lazy loading
                frame = Frame.from_file(temp_path, lazy_load=True)
                
                # Frame should not be loaded initially
                assert not frame.is_loaded
                assert frame.memory_usage_mb == 0.0
                
                # Access data should trigger loading
                data = frame.load()
                assert frame.is_loaded
                assert frame.memory_usage_mb > 0.0
                assert data.shape == (100, 100, 3)
        
        finally:
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()
    
    def test_eager_loading_for_arrays(self):
        """Test that frames created from arrays are loaded immediately."""
        test_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        frame = Frame.from_array(test_array)
        
        # Should be loaded immediately for array data
        assert frame.is_loaded
        assert frame.memory_usage_mb > 0.0
        np.testing.assert_array_equal(frame.to_array(), test_array)
    
    def test_unload_functionality(self):
        """Test that frames can be unloaded to free memory."""
        test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Create frame with lazy loading enabled
        frame = Frame.from_array(test_array)
        frame.lazy_load = True  # Enable lazy loading for unloading
        
        # Should be loaded initially
        assert frame.is_loaded
        initial_memory = frame.memory_usage_mb
        assert initial_memory > 0.0
        
        # Unload frame
        frame.unload()
        
        # Should be unloaded
        assert not frame.is_loaded
        assert frame.memory_usage_mb == 0.0
        
        # Should be able to reload
        reloaded_data = frame.load()
        assert frame.is_loaded
        np.testing.assert_array_equal(reloaded_data, test_array)


class TestMemoryManager:
    """Test the global memory manager functionality."""
    
    def test_memory_stats_tracking(self):
        """Test that memory manager tracks frame statistics correctly."""
        # Clear any existing frames
        force_global_cleanup()
        
        # Create some test frames
        frames = []
        for i in range(5):
            test_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            frame = Frame.from_array(test_array)
            frames.append(frame)
        
        # Check memory stats
        stats = get_global_memory_stats()
        assert stats['loaded_frames'] == 5
        assert stats['total_memory_mb'] > 0.0
    
    def test_memory_threshold_enforcement(self):
        """Test that memory manager enforces memory thresholds."""
        # Set a very low threshold for testing
        original_threshold = get_global_memory_stats()['memory_threshold_mb']
        
        try:
            set_global_memory_threshold(1)  # 1MB threshold
            
            # Create frames that exceed threshold
            frames = []
            for i in range(10):
                # Create large frames to exceed threshold
                test_array = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
                frame = Frame.from_array(test_array)
                frame.lazy_load = True  # Enable unloading
                frames.append(frame)
            
            # Some frames should be automatically unloaded
            loaded_count = sum(1 for frame in frames if frame.is_loaded)
            assert loaded_count < len(frames)
            
        finally:
            # Restore original threshold
            set_global_memory_threshold(original_threshold)
    
    def test_temporary_memory_settings(self):
        """Test temporary memory settings context manager."""
        original_stats = get_global_memory_stats()
        
        with TemporaryMemorySettings(threshold_mb=500, max_frames=50):
            stats = get_global_memory_stats()
            assert stats['memory_threshold_mb'] == 500
            assert stats['max_loaded_frames'] == 50
        
        # Settings should be restored
        restored_stats = get_global_memory_stats()
        assert restored_stats['memory_threshold_mb'] == original_stats['memory_threshold_mb']
        assert restored_stats['max_loaded_frames'] == original_stats['max_loaded_frames']


class TestFrameSequenceMemoryManagement:
    """Test memory management in FrameSequence objects."""
    
    def test_batch_processing(self):
        """Test batch processing with memory management."""
        # Create a sequence of frames
        frames = []
        for i in range(20):
            test_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            frame = Frame.from_array(test_array)
            frame.lazy_load = True
            frames.append(frame)
        
        sequence = FrameSequence(frames=frames, lazy_load=True)
        sequence.set_batch_size(5)
        
        # Process in batches
        def simple_processor(batch_data, start_idx):
            return len(batch_data)
        
        results = sequence.process_in_batches(
            simple_processor, 
            batch_size=5, 
            unload_after_processing=True
        )
        
        # Should have processed all frames in batches
        assert len(results) == 4  # 20 frames / 5 batch_size = 4 batches
        assert all(result == 5 for result in results)
        
        # Most frames should be unloaded after processing
        loaded_count = sum(1 for frame in frames if frame.is_loaded)
        assert loaded_count < len(frames)
    
    def test_smart_preloading(self):
        """Test smart preloading around current position."""
        # Create sequence with lazy loading
        frames = []
        for i in range(10):
            test_array = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
            frame = Frame.from_array(test_array)
            frame.lazy_load = True
            frame.unload()  # Start unloaded
            frames.append(frame)
        
        sequence = FrameSequence(frames=frames, lazy_load=True)
        sequence.set_preload_window(2)
        
        # Access frame in middle
        frame_5 = sequence[5]
        
        # Frames around position 5 should be preloaded
        for i in range(3, 8):  # Window of 2 around position 5
            if i < len(frames):
                assert frames[i].is_loaded, f"Frame {i} should be preloaded"
        
        # Frames outside window should not be loaded
        assert not frames[0].is_loaded
        assert not frames[9].is_loaded
    
    def test_memory_optimization(self):
        """Test memory optimization functionality."""
        # Create sequence with frames
        frames = []
        for i in range(10):
            test_array = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)
            frame = Frame.from_array(test_array)
            frame.lazy_load = True
            frames.append(frame)
        
        sequence = FrameSequence(frames=frames, lazy_load=True)
        
        # Access some frames to create access patterns
        sequence[0].load()
        time.sleep(0.01)  # Small delay to differentiate access times
        sequence[5].load()
        time.sleep(0.01)
        sequence[9].load()
        
        # Optimize memory to keep only 2 recent frames
        sequence.optimize_memory(keep_recent=2)
        
        # Should keep only the 2 most recently accessed frames
        loaded_count = sum(1 for frame in frames if frame.is_loaded)
        assert loaded_count <= 2
    
    def test_memory_stats(self):
        """Test memory statistics reporting."""
        frames = []
        for i in range(5):
            test_array = np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)
            frame = Frame.from_array(test_array)
            frames.append(frame)
        
        sequence = FrameSequence(frames=frames, sequence_id="test_sequence")
        
        stats = sequence.get_memory_stats()
        
        assert stats['sequence_id'] == "test_sequence"
        assert stats['total_frames'] == 5
        assert stats['loaded_frames'] == 5
        assert stats['total_memory_mb'] > 0.0
        assert stats['memory_efficiency'] == 1.0  # All frames loaded


class TestMemoryOptimizedFrameSequence:
    """Test the memory-optimized frame sequence."""
    
    def test_auto_optimization(self):
        """Test automatic memory optimization."""
        frames = []
        for i in range(20):
            test_array = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
            frame = Frame.from_array(test_array)
            frame.lazy_load = True
            frames.append(frame)
        
        sequence = MemoryOptimizedFrameSequence(frames=frames, lazy_load=True)
        sequence.enable_auto_optimization(enabled=True, interval=5)
        
        # Access frames to trigger auto-optimization
        for i in range(10):
            sequence[i % len(frames)]
        
        # Some optimization should have occurred
        stats = sequence.get_memory_stats()
        assert stats['loaded_frames'] < stats['total_frames']
    
    def test_disable_auto_optimization(self):
        """Test disabling automatic optimization."""
        frames = []
        for i in range(10):
            test_array = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
            frame = Frame.from_array(test_array)
            frames.append(frame)
        
        sequence = MemoryOptimizedFrameSequence(frames=frames)
        sequence.enable_auto_optimization(enabled=False)
        
        # Access all frames
        for i in range(len(frames)):
            sequence[i]
        
        # All frames should still be loaded (no auto-optimization)
        stats = sequence.get_memory_stats()
        assert stats['loaded_frames'] == stats['total_frames']


class TestFrameAccessStats:
    """Test frame access statistics tracking."""
    
    def test_access_count_tracking(self):
        """Test that frame access counts are tracked correctly."""
        test_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        frame = Frame.from_array(test_array)
        
        # Initial access count should be 1 (from creation)
        stats = frame.access_stats
        assert stats['access_count'] >= 1
        
        # Access frame multiple times
        for _ in range(5):
            frame.load()
        
        # Access count should increase
        updated_stats = frame.access_stats
        assert updated_stats['access_count'] > stats['access_count']
        assert updated_stats['last_access_time'] >= stats['last_access_time']
    
    def test_memory_usage_reporting(self):
        """Test that memory usage is reported correctly."""
        test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame = Frame.from_array(test_array)
        
        # Should report memory usage
        assert frame.memory_usage_mb > 0.0
        
        # Memory usage should match expected size
        expected_size_mb = test_array.nbytes / (1024 * 1024)
        assert abs(frame.memory_usage_mb - expected_size_mb) < 0.01


if __name__ == "__main__":
    pytest.main([__file__])