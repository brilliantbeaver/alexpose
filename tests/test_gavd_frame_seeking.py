"""
Comprehensive tests for GAVD frame seeking functionality.

This module tests the critical behavior that GAVD frame numbers are ABSOLUTE
positions in the original YouTube video, not relative offsets.

Key invariants tested:
1. seek_time = frame_num / fps (ABSOLUTE, not relative)
2. Playback stays within annotated frame range
3. Overlays only render for annotated frames
4. Frame synchronization is accurate during playback
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, example
from typing import List, Dict, Tuple, Optional
import math


# ============================================================================
# Core Frame Seeking Logic (mirrors frontend implementation)
# ============================================================================

def calculate_seek_time(frame_num: int, fps: int) -> float:
    """
    Calculate the video seek time for a GAVD frame number.
    
    CRITICAL: GAVD frame_num is an ABSOLUTE frame number in the YouTube video.
    To seek to frame 1757, we calculate: time = 1757 / fps
    
    This is NOT: time = (frame_num - first_frame) / fps
    """
    return frame_num / fps


def find_closest_frame_index(
    frames: List[Dict],
    absolute_frame_num: int
) -> int:
    """Find the index of the closest annotated frame to a given absolute frame number."""
    if not frames:
        return -1
    
    closest_index = 0
    min_diff = float('inf')
    
    for i, frame in enumerate(frames):
        diff = abs(frame["frame_num"] - absolute_frame_num)
        if diff < min_diff:
            min_diff = diff
            closest_index = i
    
    return closest_index


def is_frame_in_annotated_range(
    frames: List[Dict],
    absolute_frame_num: int
) -> bool:
    """Check if an absolute frame number is within the annotated range."""
    if not frames:
        return False
    
    first_frame = frames[0]["frame_num"]
    last_frame = frames[-1]["frame_num"]
    
    return first_frame <= absolute_frame_num <= last_frame


def video_time_to_absolute_frame(video_time: float, fps: int) -> int:
    """Convert video time to absolute frame number."""
    return round(video_time * fps)


# ============================================================================
# Property-Based Tests
# ============================================================================

class TestSeekTimeCalculation:
    """Property-based tests for seek time calculation."""
    
    @given(
        frame_num=st.integers(min_value=0, max_value=10_000_000),
        fps=st.sampled_from([24, 25, 30, 60])
    )
    @settings(max_examples=100)
    def test_seek_time_is_absolute(self, frame_num: int, fps: int):
        """Property: seek time should be frame_num / fps (absolute)."""
        seek_time = calculate_seek_time(frame_num, fps)
        
        assert seek_time == frame_num / fps
        assert seek_time >= 0
    
    @given(
        first_frame=st.integers(min_value=1000, max_value=100_000),
        frame_count=st.integers(min_value=1, max_value=1000),
        fps=st.sampled_from([24, 30, 60])
    )
    @settings(max_examples=50)
    def test_seek_time_not_relative(self, first_frame: int, frame_count: int, fps: int):
        """Property: seek time should NOT be based on relative offset."""
        frames = [{"frame_num": first_frame + i} for i in range(frame_count)]
        
        for i, frame in enumerate(frames):
            absolute_time = calculate_seek_time(frame["frame_num"], fps)
            relative_time = i / fps  # WRONG calculation
            
            # Absolute time should be much larger than relative time
            # (unless first_frame is 0, which we exclude)
            assert absolute_time > relative_time
    
    @given(
        frame_num=st.integers(min_value=0, max_value=10_000_000),
        fps=st.sampled_from([24, 25, 30, 60])
    )
    @settings(max_examples=100)
    def test_seek_time_roundtrip(self, frame_num: int, fps: int):
        """Property: frame -> time -> frame should be consistent."""
        seek_time = calculate_seek_time(frame_num, fps)
        recovered_frame = video_time_to_absolute_frame(seek_time, fps)
        
        assert recovered_frame == frame_num
    
    @given(
        frame_num=st.integers(min_value=1, max_value=10_000_000),
        fps=st.sampled_from([24, 25, 30, 60])
    )
    @settings(max_examples=50)
    def test_seek_time_monotonic(self, frame_num: int, fps: int):
        """Property: higher frame numbers should have higher seek times."""
        time1 = calculate_seek_time(frame_num, fps)
        time2 = calculate_seek_time(frame_num + 1, fps)
        
        assert time2 > time1


class TestAnnotatedRangeDetection:
    """Property-based tests for annotated range detection."""
    
    @given(
        first_frame=st.integers(min_value=100, max_value=10_000),
        frame_count=st.integers(min_value=10, max_value=500)
    )
    @settings(max_examples=50)
    def test_frames_in_range_detected(self, first_frame: int, frame_count: int):
        """Property: all annotated frames should be detected as in range."""
        frames = [{"frame_num": first_frame + i} for i in range(frame_count)]
        
        for frame in frames:
            assert is_frame_in_annotated_range(frames, frame["frame_num"])
    
    @given(
        first_frame=st.integers(min_value=100, max_value=10_000),
        frame_count=st.integers(min_value=10, max_value=500),
        offset=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=50)
    def test_frames_before_range_not_detected(self, first_frame: int, frame_count: int, offset: int):
        """Property: frames before annotated range should not be detected."""
        frames = [{"frame_num": first_frame + i} for i in range(frame_count)]
        
        before_frame = first_frame - offset
        assert not is_frame_in_annotated_range(frames, before_frame)
    
    @given(
        first_frame=st.integers(min_value=100, max_value=10_000),
        frame_count=st.integers(min_value=10, max_value=500),
        offset=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=50)
    def test_frames_after_range_not_detected(self, first_frame: int, frame_count: int, offset: int):
        """Property: frames after annotated range should not be detected."""
        frames = [{"frame_num": first_frame + i} for i in range(frame_count)]
        
        last_frame = frames[-1]["frame_num"]
        after_frame = last_frame + offset
        assert not is_frame_in_annotated_range(frames, after_frame)


class TestClosestFrameFinding:
    """Property-based tests for finding closest annotated frame."""
    
    @given(
        first_frame=st.integers(min_value=100, max_value=10_000),
        frame_count=st.integers(min_value=10, max_value=500)
    )
    @settings(max_examples=50)
    def test_exact_match_found(self, first_frame: int, frame_count: int):
        """Property: exact frame matches should return correct index."""
        frames = [{"frame_num": first_frame + i} for i in range(frame_count)]
        
        for i, frame in enumerate(frames):
            found_index = find_closest_frame_index(frames, frame["frame_num"])
            assert found_index == i
    
    @given(
        first_frame=st.integers(min_value=100, max_value=10_000),
        frame_count=st.integers(min_value=10, max_value=500)
    )
    @settings(max_examples=50)
    def test_closest_frame_is_valid(self, first_frame: int, frame_count: int):
        """Property: closest frame index should always be valid."""
        frames = [{"frame_num": first_frame + i} for i in range(frame_count)]
        
        # Test with various target frames
        for target in [first_frame - 10, first_frame + frame_count // 2, first_frame + frame_count + 10]:
            found_index = find_closest_frame_index(frames, target)
            assert 0 <= found_index < len(frames)
    
    @given(
        first_frame=st.integers(min_value=100, max_value=10_000),
        frame_count=st.integers(min_value=10, max_value=500),
        target_offset=st.integers(min_value=-100, max_value=600)
    )
    @settings(max_examples=100)
    def test_closest_frame_minimizes_distance(self, first_frame: int, frame_count: int, target_offset: int):
        """Property: found frame should minimize distance to target."""
        frames = [{"frame_num": first_frame + i} for i in range(frame_count)]
        target = first_frame + target_offset
        
        found_index = find_closest_frame_index(frames, target)
        found_distance = abs(frames[found_index]["frame_num"] - target)
        
        # Verify no other frame is closer
        for i, frame in enumerate(frames):
            other_distance = abs(frame["frame_num"] - target)
            assert found_distance <= other_distance


class TestPlaybackBehavior:
    """Property-based tests for playback behavior."""
    
    @given(
        first_frame=st.integers(min_value=1000, max_value=5000),
        frame_count=st.integers(min_value=50, max_value=200),
        fps=st.sampled_from([24, 30, 60])
    )
    @settings(max_examples=30)
    def test_playback_stays_in_range(self, first_frame: int, frame_count: int, fps: int):
        """Property: playback should stay within annotated frame range."""
        frames = [{"frame_num": first_frame + i} for i in range(frame_count)]
        
        last_frame = frames[-1]["frame_num"]
        
        # Simulate playback
        for i in range(frame_count):
            current_frame = frames[i]["frame_num"]
            video_time = calculate_seek_time(current_frame, fps)
            
            # Verify we're in range
            absolute_frame = video_time_to_absolute_frame(video_time, fps)
            assert is_frame_in_annotated_range(frames, absolute_frame)
    
    @given(
        first_frame=st.integers(min_value=1000, max_value=5000),
        frame_count=st.integers(min_value=50, max_value=200),
        fps=st.sampled_from([24, 30, 60])
    )
    @settings(max_examples=30)
    def test_playback_stops_at_end(self, first_frame: int, frame_count: int, fps: int):
        """Property: playback should stop at last annotated frame."""
        frames = [{"frame_num": first_frame + i} for i in range(frame_count)]
        
        last_frame = frames[-1]["frame_num"]
        
        # Simulate going past the end
        past_end_time = calculate_seek_time(last_frame + 10, fps)
        past_end_frame = video_time_to_absolute_frame(past_end_time, fps)
        
        # Should detect we're past the range
        assert not is_frame_in_annotated_range(frames, past_end_frame)
        
        # Should find last frame as closest
        closest = find_closest_frame_index(frames, past_end_frame)
        assert closest == len(frames) - 1


# ============================================================================
# Specific Regression Tests
# ============================================================================

class TestRegressionCases:
    """Specific regression tests for known issues."""
    
    def test_frame_1757_seeks_to_correct_time(self):
        """Regression: Frame 1757 should seek to ~58.57s, not 0s."""
        frame_num = 1757
        fps = 30
        
        seek_time = calculate_seek_time(frame_num, fps)
        
        # Should be ~58.57 seconds
        assert seek_time == pytest.approx(58.567, rel=0.01)
        
        # Should NOT be 0
        assert seek_time != 0.0
    
    def test_frame_2136_seeks_to_correct_time(self):
        """Regression: Frame 2136 should seek to ~71.2s."""
        frame_num = 2136
        fps = 30
        
        seek_time = calculate_seek_time(frame_num, fps)
        
        assert seek_time == pytest.approx(71.2, rel=0.01)
    
    def test_512_frame_sequence_starting_at_1757(self):
        """Regression: 512 frames starting at 1757 should work correctly."""
        frames = [{"frame_num": 1757 + i} for i in range(512)]
        fps = 30
        
        # First frame
        first_time = calculate_seek_time(frames[0]["frame_num"], fps)
        assert first_time == pytest.approx(58.567, rel=0.01)
        
        # Last frame (1757 + 511 = 2268)
        last_time = calculate_seek_time(frames[-1]["frame_num"], fps)
        assert last_time == pytest.approx(75.6, rel=0.01)
        
        # Duration of annotated segment
        duration = last_time - first_time
        assert duration == pytest.approx(17.033, rel=0.01)  # ~17 seconds
    
    def test_overlay_not_shown_for_frame_0(self):
        """Regression: Overlay should not show for frame 0 when annotations start at 1757."""
        frames = [{"frame_num": 1757 + i} for i in range(512)]
        
        # Frame 0 is not in annotated range
        assert not is_frame_in_annotated_range(frames, 0)
        
        # Frame 1000 is not in annotated range
        assert not is_frame_in_annotated_range(frames, 1000)
        
        # Frame 1757 IS in annotated range
        assert is_frame_in_annotated_range(frames, 1757)
    
    def test_playback_from_middle_frame(self):
        """Regression: Starting playback from middle frame should work."""
        frames = [{"frame_num": 1757 + i} for i in range(512)]
        fps = 30
        
        # Start at frame 380 (frame_num = 2137)
        start_index = 380
        start_frame = frames[start_index]["frame_num"]
        
        assert start_frame == 2137
        
        # Seek time should be correct
        seek_time = calculate_seek_time(start_frame, fps)
        assert seek_time == pytest.approx(71.233, rel=0.01)
        
        # Should be able to find this frame
        found_index = find_closest_frame_index(frames, start_frame)
        assert found_index == start_index


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_frames_list(self):
        """Edge case: empty frames list."""
        frames: List[Dict] = []
        
        assert find_closest_frame_index(frames, 1000) == -1
        assert not is_frame_in_annotated_range(frames, 1000)
    
    def test_single_frame(self):
        """Edge case: single frame in list."""
        frames = [{"frame_num": 1757}]
        fps = 30
        
        seek_time = calculate_seek_time(frames[0]["frame_num"], fps)
        assert seek_time == pytest.approx(58.567, rel=0.01)
        
        assert find_closest_frame_index(frames, 1757) == 0
        assert find_closest_frame_index(frames, 1000) == 0
        assert find_closest_frame_index(frames, 2000) == 0
    
    def test_non_consecutive_frames(self):
        """Edge case: non-consecutive frame numbers (gaps)."""
        frames = [
            {"frame_num": 1757},
            {"frame_num": 1760},  # Gap
            {"frame_num": 1770},  # Larger gap
        ]
        
        # Frame 1758 should find 1757 as closest
        assert find_closest_frame_index(frames, 1758) == 0
        
        # Frame 1765 should find 1760 or 1770 (equidistant)
        found = find_closest_frame_index(frames, 1765)
        assert found in [1, 2]
    
    def test_very_large_frame_numbers(self):
        """Edge case: very large frame numbers."""
        frame_num = 10_000_000
        fps = 30
        
        seek_time = calculate_seek_time(frame_num, fps)
        
        # Should be ~333,333 seconds (~92 hours)
        assert seek_time == pytest.approx(333333.33, rel=0.01)
    
    def test_frame_zero(self):
        """Edge case: frame number 0."""
        frame_num = 0
        fps = 30
        
        seek_time = calculate_seek_time(frame_num, fps)
        assert seek_time == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
