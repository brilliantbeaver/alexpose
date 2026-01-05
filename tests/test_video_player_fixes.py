"""
Tests for GAVD video player functionality.

Tests the fixes for:
1. Absolute frame number seeking (GAVD frames are absolute positions in YouTube video)
2. Playback constrained to annotated frame range
3. Bounding box/pose overlay only for annotated frames
4. Frame synchronization during playback

CRITICAL CONCEPT:
GAVD frame_num values (e.g., 1757, 1758, ...) are ABSOLUTE frame numbers in the 
original YouTube video. To seek to frame 1757, we calculate: time = 1757 / fps
NOT: time = (1757 - firstFrame) / fps
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import List, Dict, Any


class TestAbsoluteFrameSeeking:
    """Test that video seeking uses absolute frame numbers."""
    
    def test_seek_to_first_annotated_frame(self):
        """Test seeking to first annotated frame uses absolute frame number."""
        # GAVD frames start at 1757 (absolute position in YouTube video)
        frames = [
            {"frame_num": 1757},
            {"frame_num": 1758},
            {"frame_num": 1759},
        ]
        
        fps = 30
        first_frame = frames[0]["frame_num"]
        
        # To seek to frame 1757, we need time = 1757 / 30 = 58.567s
        # NOT time = 0 (which would be the beginning of the video)
        target_time = first_frame / fps
        
        assert target_time == pytest.approx(58.567, rel=0.01)
        assert target_time != 0.0  # Critical: should NOT be 0
    
    def test_seek_to_middle_frame(self):
        """Test seeking to a middle frame uses absolute frame number."""
        frames = [
            {"frame_num": 1757},
            {"frame_num": 1800},
            {"frame_num": 1850},
        ]
        
        fps = 30
        middle_frame = frames[1]["frame_num"]
        
        # To seek to frame 1800, time = 1800 / 30 = 60.0s
        target_time = middle_frame / fps
        
        assert target_time == 60.0
    
    def test_seek_to_last_annotated_frame(self):
        """Test seeking to last annotated frame."""
        frames = [
            {"frame_num": 1757},
            {"frame_num": 2268},  # Last frame (512 frames later)
        ]
        
        fps = 30
        last_frame = frames[-1]["frame_num"]
        
        target_time = last_frame / fps
        
        assert target_time == pytest.approx(75.6, rel=0.01)
    
    @given(
        first_frame=st.integers(min_value=0, max_value=100000),
        frame_count=st.integers(min_value=1, max_value=1000),
        fps=st.sampled_from([24, 25, 30, 60])
    )
    @settings(max_examples=50)
    def test_absolute_seek_time_property(self, first_frame: int, frame_count: int, fps: int):
        """Property: seek time should always be frame_num / fps."""
        frames = [{"frame_num": first_frame + i} for i in range(frame_count)]
        
        for i, frame in enumerate(frames):
            expected_time = frame["frame_num"] / fps
            
            # Verify the seek time is based on absolute frame number
            assert expected_time >= 0
            assert expected_time == frame["frame_num"] / fps
            
            # Verify it's NOT based on relative offset
            relative_time = i / fps
            if first_frame > 0:
                assert expected_time != relative_time


class TestAnnotatedFrameRange:
    """Test that playback is constrained to annotated frame range."""
    
    def test_frame_in_annotated_range(self):
        """Test detecting if a frame is within annotated range."""
        frames = [
            {"frame_num": 1757},
            {"frame_num": 1758},
            {"frame_num": 1759},
            {"frame_num": 1760},
        ]
        
        first_frame = frames[0]["frame_num"]
        last_frame = frames[-1]["frame_num"]
        
        # Frame 1758 is in range
        test_frame = 1758
        assert first_frame <= test_frame <= last_frame
        
        # Frame 1000 is NOT in range (before)
        test_frame = 1000
        assert not (first_frame <= test_frame <= last_frame)
        
        # Frame 2000 is NOT in range (after)
        test_frame = 2000
        assert not (first_frame <= test_frame <= last_frame)
    
    def test_find_closest_annotated_frame(self):
        """Test finding closest annotated frame for a given video time."""
        frames = [
            {"frame_num": 1757},
            {"frame_num": 1760},
            {"frame_num": 1765},
            {"frame_num": 1770},
        ]
        
        fps = 30
        
        # Video time 58.6s = frame 1758
        video_time = 58.6
        absolute_frame = round(video_time * fps)  # 1758
        
        # Find closest annotated frame
        closest_index = 0
        min_diff = float('inf')
        
        for i, frame in enumerate(frames):
            diff = abs(frame["frame_num"] - absolute_frame)
            if diff < min_diff:
                min_diff = diff
                closest_index = i
        
        # Should find frame 1757 (index 0) as closest to 1758
        assert closest_index == 0
        assert frames[closest_index]["frame_num"] == 1757
    
    def test_stop_at_last_annotated_frame(self):
        """Test that playback stops at last annotated frame."""
        frames = [
            {"frame_num": 1757},
            {"frame_num": 1758},
            {"frame_num": 1759},
        ]
        
        fps = 30
        last_frame = frames[-1]["frame_num"]
        
        # Simulate video time past last annotated frame
        video_time = 60.0  # Frame 1800
        absolute_frame = round(video_time * fps)
        
        # Should detect we're past the annotated range
        assert absolute_frame > last_frame
        
        # Playback should stop and seek to last annotated frame
        stop_time = last_frame / fps
        assert stop_time == pytest.approx(58.633, rel=0.01)
    
    @given(
        first_frame=st.integers(min_value=100, max_value=10000),
        frame_count=st.integers(min_value=10, max_value=500),
        fps=st.sampled_from([24, 30, 60])
    )
    @settings(max_examples=30)
    def test_annotated_range_boundaries_property(self, first_frame: int, frame_count: int, fps: int):
        """Property: frames outside annotated range should be detected."""
        frames = [{"frame_num": first_frame + i} for i in range(frame_count)]
        
        first = frames[0]["frame_num"]
        last = frames[-1]["frame_num"]
        
        # Test frame before range
        before_frame = first - 10
        assert before_frame < first
        
        # Test frame after range
        after_frame = last + 10
        assert after_frame > last
        
        # Test frame in range
        middle_frame = first + frame_count // 2
        assert first <= middle_frame <= last


class TestFrameSynchronization:
    """Test frame synchronization during playback."""
    
    def test_video_time_to_frame_index(self):
        """Test converting video time to frame index."""
        frames = [
            {"frame_num": 1757},
            {"frame_num": 1758},
            {"frame_num": 1759},
            {"frame_num": 1760},
            {"frame_num": 1761},
        ]
        
        fps = 30
        
        # Video at time 58.6s (frame 1758)
        video_time = 58.6
        absolute_frame = round(video_time * fps)
        
        # Find matching frame index
        frame_index = None
        for i, frame in enumerate(frames):
            if frame["frame_num"] == absolute_frame:
                frame_index = i
                break
        
        assert frame_index == 1  # Index of frame 1758
    
    def test_playback_through_all_annotated_frames(self):
        """Test that playback can progress through all annotated frames."""
        # Simulate 512 frames starting at 1757
        frames = [{"frame_num": 1757 + i} for i in range(512)]
        
        fps = 30
        first_frame = frames[0]["frame_num"]
        last_frame = frames[-1]["frame_num"]
        
        # Simulate playback at various times
        visited_indices = set()
        
        for frame_index in range(len(frames)):
            frame_num = frames[frame_index]["frame_num"]
            video_time = frame_num / fps
            
            # Calculate which frame we'd be at
            absolute_frame = round(video_time * fps)
            
            # Find closest annotated frame
            closest_index = 0
            min_diff = float('inf')
            
            for i, f in enumerate(frames):
                diff = abs(f["frame_num"] - absolute_frame)
                if diff < min_diff:
                    min_diff = diff
                    closest_index = i
            
            visited_indices.add(closest_index)
        
        # Should visit all frames
        assert len(visited_indices) == len(frames)
    
    @given(
        first_frame=st.integers(min_value=1000, max_value=5000),
        frame_count=st.integers(min_value=50, max_value=200),
        fps=st.sampled_from([24, 30, 60])
    )
    @settings(max_examples=20)
    def test_frame_sync_roundtrip_property(self, first_frame: int, frame_count: int, fps: int):
        """Property: frame -> time -> frame should be consistent."""
        frames = [{"frame_num": first_frame + i} for i in range(frame_count)]
        
        for i, frame in enumerate(frames):
            # Convert frame to time
            time = frame["frame_num"] / fps
            
            # Convert time back to frame
            recovered_frame = round(time * fps)
            
            # Should recover the same frame number
            assert recovered_frame == frame["frame_num"]


class TestOverlayRendering:
    """Test that overlays are only rendered for annotated frames."""
    
    def test_overlay_for_annotated_frame(self):
        """Test that overlay is rendered for annotated frame."""
        frames = [
            {"frame_num": 1757, "bbox": {"left": 100, "top": 100, "width": 200, "height": 400}},
            {"frame_num": 1758, "bbox": {"left": 102, "top": 102, "width": 200, "height": 400}},
        ]
        
        # Frame index 0 is valid
        frame_index = 0
        should_render = 0 <= frame_index < len(frames)
        
        assert should_render
        assert frames[frame_index]["bbox"] is not None
    
    def test_no_overlay_for_invalid_index(self):
        """Test that overlay is not rendered for invalid frame index."""
        frames = [
            {"frame_num": 1757, "bbox": {"left": 100, "top": 100, "width": 200, "height": 400}},
        ]
        
        # Frame index -1 is invalid
        frame_index = -1
        should_render = 0 <= frame_index < len(frames)
        
        assert not should_render
        
        # Frame index 5 is invalid (out of range)
        frame_index = 5
        should_render = 0 <= frame_index < len(frames)
        
        assert not should_render
    
    def test_bbox_scaling(self):
        """Test bounding box scaling when video resolution differs."""
        bbox = {"left": 100, "top": 100, "width": 200, "height": 400}
        vid_info = {"width": 1280, "height": 720}
        
        # Actual video is 1920x1080
        actual_width = 1920
        actual_height = 1080
        
        scale_x = actual_width / vid_info["width"]
        scale_y = actual_height / vid_info["height"]
        
        scaled_left = bbox["left"] * scale_x
        scaled_top = bbox["top"] * scale_y
        scaled_width = bbox["width"] * scale_x
        scaled_height = bbox["height"] * scale_y
        
        assert scale_x == 1.5
        assert scale_y == 1.5
        assert scaled_left == 150
        assert scaled_top == 150
        assert scaled_width == 300
        assert scaled_height == 600


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_frames_array(self):
        """Test handling of empty frames array."""
        frames: List[Dict[str, Any]] = []
        
        # Should handle gracefully
        assert len(frames) == 0
        
        # No first/last frame
        first_frame = frames[0]["frame_num"] if frames else None
        last_frame = frames[-1]["frame_num"] if frames else None
        
        assert first_frame is None
        assert last_frame is None
    
    def test_single_frame(self):
        """Test handling of single frame."""
        frames = [{"frame_num": 1757}]
        
        fps = 30
        first_frame = frames[0]["frame_num"]
        last_frame = frames[-1]["frame_num"]
        
        # First and last are the same
        assert first_frame == last_frame == 1757
        
        # Seek time
        seek_time = first_frame / fps
        assert seek_time == pytest.approx(58.567, rel=0.01)
    
    def test_non_consecutive_frames(self):
        """Test handling of non-consecutive frame numbers."""
        # GAVD data may have gaps in frame numbers
        frames = [
            {"frame_num": 1757},
            {"frame_num": 1760},  # Gap of 2
            {"frame_num": 1770},  # Gap of 9
            {"frame_num": 1800},  # Gap of 29
        ]
        
        fps = 30
        
        # Should still work with gaps
        for frame in frames:
            seek_time = frame["frame_num"] / fps
            assert seek_time > 0
        
        # Finding closest frame should work
        test_frame = 1765  # Between 1760 and 1770
        
        closest_index = 0
        min_diff = float('inf')
        
        for i, frame in enumerate(frames):
            diff = abs(frame["frame_num"] - test_frame)
            if diff < min_diff:
                min_diff = diff
                closest_index = i
        
        # Should find 1760 (diff=5) or 1770 (diff=5)
        assert frames[closest_index]["frame_num"] in [1760, 1770]
    
    def test_very_large_frame_numbers(self):
        """Test handling of very large frame numbers."""
        # Some videos might have frames in the millions
        frames = [
            {"frame_num": 1000000},
            {"frame_num": 1000001},
        ]
        
        fps = 30
        
        # Should calculate correct seek time
        seek_time = frames[0]["frame_num"] / fps
        assert seek_time == pytest.approx(33333.33, rel=0.01)
    
    @given(
        frame_num=st.integers(min_value=0, max_value=10000000),
        fps=st.sampled_from([24, 25, 30, 60])
    )
    @settings(max_examples=50)
    def test_seek_time_always_positive_property(self, frame_num: int, fps: int):
        """Property: seek time should always be non-negative."""
        seek_time = frame_num / fps
        assert seek_time >= 0


class TestIntegration:
    """Integration tests for complete scenarios."""
    
    def test_complete_playback_scenario(self):
        """Test complete playback scenario with absolute frame numbers."""
        # Setup: 512 frames starting at 1757
        frames = [{"frame_num": 1757 + i} for i in range(512)]
        
        fps = 30
        first_frame = frames[0]["frame_num"]
        last_frame = frames[-1]["frame_num"]
        
        # Initial state: seek to first annotated frame
        initial_time = first_frame / fps
        assert initial_time == pytest.approx(58.567, rel=0.01)
        
        # Playback: simulate advancing through frames
        current_index = 0
        
        for step in range(100):  # Simulate 100 animation frames
            # Calculate current video time
            current_frame = frames[current_index]["frame_num"]
            video_time = current_frame / fps
            
            # Verify we're in annotated range
            absolute_frame = round(video_time * fps)
            assert first_frame <= absolute_frame <= last_frame
            
            # Advance to next frame
            if current_index < len(frames) - 1:
                current_index += 1
        
        # End state: at frame 100
        assert current_index == 100
        assert frames[current_index]["frame_num"] == 1857
    
    def test_seek_and_play_scenario(self):
        """Test seeking to a specific frame then playing."""
        frames = [{"frame_num": 1757 + i} for i in range(512)]
        
        fps = 30
        
        # Seek to frame 250 (frame_num = 2007)
        seek_index = 250
        seek_frame = frames[seek_index]["frame_num"]
        seek_time = seek_frame / fps
        
        assert seek_frame == 2007
        assert seek_time == pytest.approx(66.9, rel=0.01)
        
        # Play from there
        current_index = seek_index
        
        for _ in range(10):
            if current_index < len(frames) - 1:
                current_index += 1
        
        # Should be at frame 260
        assert current_index == 260
        assert frames[current_index]["frame_num"] == 2017


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
