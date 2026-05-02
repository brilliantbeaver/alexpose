"""
Tests for FrameBuffer implementation.

This module tests the circular frame buffer used for realtime processing,
including memory management, frame retrieval, and statistics tracking.
"""

import time
import pytest
import numpy as np
from hypothesis import given, strategies as st

from ambient.realtime.frame_buffer import FrameBuffer
from ambient.realtime.interfaces import RealtimeFrame


@pytest.fixture
def frame_buffer():
    """Create a frame buffer for testing."""
    return FrameBuffer(max_size=10, max_memory_mb=1, auto_cleanup=True)


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    return RealtimeFrame(
        data=np.zeros((480, 640, 3), dtype=np.uint8),
        timestamp=time.time(),
        frame_id=1,
        metadata={'width': 640, 'height': 480, 'channels': 3}
    )


class TestFrameBufferBasics:
    """Test basic frame buffer operations."""
    
    def test_initialization(self):
        """Test frame buffer initialization."""
        buffer = FrameBuffer(max_size=20, max_memory_mb=5)
        
        assert buffer.max_size == 20
        assert buffer.max_memory_bytes == 5 * 1024 * 1024
        assert buffer.is_empty()
        assert not buffer.is_full()
    
    def test_add_frame(self, frame_buffer, sample_frame):
        """Test adding a frame to buffer."""
        frame_buffer.add_frame(sample_frame)
        
        assert not frame_buffer.is_empty()
        assert len(frame_buffer._frames) == 1
        
        stats = frame_buffer.get_buffer_stats()
        assert stats['frames_added'] == 1
        assert stats['current_size'] == 1
    
    def test_get_latest_frame(self, frame_buffer, sample_frame):
        """Test retrieving the latest frame."""
        frame_buffer.add_frame(sample_frame)
        
        latest = frame_buffer.get_latest_frame()
        assert latest is not None
        assert latest.frame_id == sample_frame.frame_id
    
    def test_get_latest_frame_empty(self, frame_buffer):
        """Test getting latest frame from empty buffer."""
        latest = frame_buffer.get_latest_frame()
        assert latest is None
    
    def test_clear(self, frame_buffer, sample_frame):
        """Test clearing the buffer."""
        frame_buffer.add_frame(sample_frame)
        assert not frame_buffer.is_empty()
        
        frame_buffer.clear()
        assert frame_buffer.is_empty()
        assert frame_buffer.get_latest_frame() is None


class TestFrameBufferCapacity:
    """Test frame buffer capacity management."""
    
    def test_max_size_enforcement(self, frame_buffer):
        """Test that buffer respects max_size."""
        # Add more frames than max_size
        for i in range(15):
            frame = RealtimeFrame(
                data=np.zeros((100, 100, 3), dtype=np.uint8),
                timestamp=time.time(),
                frame_id=i,
                metadata={}
            )
            frame_buffer.add_frame(frame)
        
        # Should only keep max_size frames
        assert len(frame_buffer._frames) == frame_buffer.max_size
        
        # Should have dropped oldest frames
        stats = frame_buffer.get_buffer_stats()
        assert stats['frames_dropped'] == 5
    
    def test_circular_buffer_behavior(self, frame_buffer):
        """Test circular buffer behavior."""
        # Fill buffer
        for i in range(frame_buffer.max_size):
            frame = RealtimeFrame(
                data=np.zeros((100, 100, 3), dtype=np.uint8),
                timestamp=time.time(),
                frame_id=i,
                metadata={}
            )
            frame_buffer.add_frame(frame)
        
        assert frame_buffer.is_full()
        
        # Add one more frame
        new_frame = RealtimeFrame(
            data=np.zeros((100, 100, 3), dtype=np.uint8),
            timestamp=time.time(),
            frame_id=999,
            metadata={}
        )
        frame_buffer.add_frame(new_frame)
        
        # Should still be at max size
        assert len(frame_buffer._frames) == frame_buffer.max_size
        
        # Latest frame should be the new one
        latest = frame_buffer.get_latest_frame()
        assert latest.frame_id == 999


class TestFrameSequence:
    """Test frame sequence retrieval."""
    
    def test_get_frame_sequence(self, frame_buffer):
        """Test getting a sequence of frames."""
        # Add multiple frames
        for i in range(5):
            frame = RealtimeFrame(
                data=np.zeros((100, 100, 3), dtype=np.uint8),
                timestamp=time.time(),
                frame_id=i,
                metadata={}
            )
            frame_buffer.add_frame(frame)
        
        # Get last 3 frames
        sequence = frame_buffer.get_frame_sequence(3)
        assert len(sequence) == 3
        assert sequence[-1].frame_id == 4  # Most recent
        assert sequence[0].frame_id == 2   # Oldest in sequence
    
    def test_get_frame_sequence_more_than_available(self, frame_buffer):
        """Test requesting more frames than available."""
        # Add 3 frames
        for i in range(3):
            frame = RealtimeFrame(
                data=np.zeros((100, 100, 3), dtype=np.uint8),
                timestamp=time.time(),
                frame_id=i,
                metadata={}
            )
            frame_buffer.add_frame(frame)
        
        # Request 10 frames
        sequence = frame_buffer.get_frame_sequence(10)
        assert len(sequence) == 3  # Should return all available
    
    def test_get_frame_sequence_empty(self, frame_buffer):
        """Test getting sequence from empty buffer."""
        sequence = frame_buffer.get_frame_sequence(5)
        assert len(sequence) == 0


class TestFrameRetrieval:
    """Test frame retrieval by ID and time range."""
    
    def test_get_frame_by_id(self, frame_buffer):
        """Test retrieving frame by ID."""
        # Add frames with specific IDs
        for i in [10, 20, 30]:
            frame = RealtimeFrame(
                data=np.zeros((100, 100, 3), dtype=np.uint8),
                timestamp=time.time(),
                frame_id=i,
                metadata={}
            )
            frame_buffer.add_frame(frame)
        
        # Retrieve specific frame
        frame = frame_buffer.get_frame_by_id(20)
        assert frame is not None
        assert frame.frame_id == 20
    
    def test_get_frame_by_id_not_found(self, frame_buffer):
        """Test retrieving non-existent frame."""
        frame = frame_buffer.get_frame_by_id(999)
        assert frame is None
    
    def test_get_frames_in_time_range(self, frame_buffer):
        """Test retrieving frames by time range."""
        start_time = time.time()
        
        # Add frames with different timestamps
        for i in range(5):
            frame = RealtimeFrame(
                data=np.zeros((100, 100, 3), dtype=np.uint8),
                timestamp=start_time + i * 0.1,
                frame_id=i,
                metadata={}
            )
            frame_buffer.add_frame(frame)
            time.sleep(0.01)
        
        # Get frames in middle range
        frames = frame_buffer.get_frames_in_time_range(
            start_time + 0.1,
            start_time + 0.3
        )
        
        assert len(frames) >= 2  # Should get frames 1, 2, 3


class TestMemoryManagement:
    """Test memory management and cleanup."""
    
    def test_memory_tracking(self, frame_buffer):
        """Test memory usage tracking."""
        initial_memory = frame_buffer.get_memory_usage_mb()
        assert initial_memory == 0.0
        
        # Add a frame
        frame = RealtimeFrame(
            data=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=time.time(),
            frame_id=1,
            metadata={}
        )
        frame_buffer.add_frame(frame)
        
        # Memory should increase
        current_memory = frame_buffer.get_memory_usage_mb()
        assert current_memory > 0.0
    
    def test_memory_cleanup(self):
        """Test automatic memory cleanup."""
        # Create buffer with very small memory limit
        buffer = FrameBuffer(max_size=100, max_memory_mb=0.5, auto_cleanup=True)
        
        # Add frames until memory limit is reached
        for i in range(20):
            frame = RealtimeFrame(
                data=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp=time.time(),
                frame_id=i,
                metadata={}
            )
            buffer.add_frame(frame)
        
        # Should have triggered cleanup
        stats = buffer.get_buffer_stats()
        assert stats['memory_cleanups'] > 0


class TestBufferStatistics:
    """Test buffer statistics tracking."""
    
    def test_statistics_tracking(self, frame_buffer):
        """Test that statistics are tracked correctly."""
        # Add some frames
        for i in range(5):
            frame = RealtimeFrame(
                data=np.zeros((100, 100, 3), dtype=np.uint8),
                timestamp=time.time(),
                frame_id=i,
                metadata={}
            )
            frame_buffer.add_frame(frame)
        
        stats = frame_buffer.get_buffer_stats()
        
        assert stats['frames_added'] == 5
        assert stats['current_size'] == 5
        assert stats['max_size'] == 10
        assert stats['average_frame_size_kb'] > 0
    
    def test_memory_usage_percent(self, frame_buffer):
        """Test memory usage percentage calculation."""
        stats = frame_buffer.get_buffer_stats()
        
        assert 'memory_usage_percent' in stats
        assert 0 <= stats['memory_usage_percent'] <= 100


@pytest.mark.property
class TestFrameBufferProperties:
    """Property-based tests for frame buffer."""
    
    @given(
        max_size=st.integers(min_value=1, max_value=100),
        num_frames=st.integers(min_value=0, max_value=150)
    )
    def test_buffer_size_invariant(self, max_size, num_frames):
        """Test that buffer never exceeds max_size."""
        buffer = FrameBuffer(max_size=max_size, max_memory_mb=100)
        
        for i in range(num_frames):
            frame = RealtimeFrame(
                data=np.zeros((10, 10, 3), dtype=np.uint8),
                timestamp=time.time(),
                frame_id=i,
                metadata={}
            )
            buffer.add_frame(frame)
        
        assert len(buffer._frames) <= max_size
    
    @given(
        num_frames=st.integers(min_value=1, max_value=50)
    )
    def test_latest_frame_is_most_recent(self, num_frames):
        """Test that latest frame is always the most recently added."""
        buffer = FrameBuffer(max_size=100)
        
        last_frame_id = None
        for i in range(num_frames):
            frame = RealtimeFrame(
                data=np.zeros((10, 10, 3), dtype=np.uint8),
                timestamp=time.time(),
                frame_id=i,
                metadata={}
            )
            buffer.add_frame(frame)
            last_frame_id = i
        
        latest = buffer.get_latest_frame()
        assert latest is not None
        assert latest.frame_id == last_frame_id


@pytest.mark.integration
class TestFrameBufferIntegration:
    """Integration tests for frame buffer."""
    
    def test_realistic_usage_scenario(self):
        """Test realistic usage scenario with video frames."""
        buffer = FrameBuffer(max_size=30, max_memory_mb=10)
        
        # Simulate 60 frames at 30 FPS (2 seconds of video)
        for i in range(60):
            frame = RealtimeFrame(
                data=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                timestamp=time.time(),
                frame_id=i,
                metadata={'width': 640, 'height': 480}
            )
            buffer.add_frame(frame)
            time.sleep(0.001)  # Simulate frame timing
        
        # Should have kept last 30 frames
        assert len(buffer._frames) == 30
        
        # Latest frame should be frame 59
        latest = buffer.get_latest_frame()
        assert latest.frame_id == 59
        
        # Get last 10 frames
        sequence = buffer.get_frame_sequence(10)
        assert len(sequence) == 10
        assert sequence[-1].frame_id == 59
        assert sequence[0].frame_id == 50
        
        # Check statistics
        stats = buffer.get_buffer_stats()
        assert stats['frames_added'] == 60
        assert stats['frames_dropped'] == 30
        assert stats['current_size'] == 30
