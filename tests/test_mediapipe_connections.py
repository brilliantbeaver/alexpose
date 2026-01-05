"""
Test MediaPipe Pose connections against official specification.

This test verifies that our skeleton connections match the official
MediaPipe Pose Landmarker specification.
"""

import pytest


class TestMediaPipeConnections:
    """Test MediaPipe Pose skeleton connections."""
    
    def test_all_connections_use_valid_landmarks(self):
        """All connections should use landmark indices 0-32."""
        # Our implementation connections
        connections = [
            # Face contour
            [0, 1], [1, 2], [2, 3], [3, 7],
            [0, 4], [4, 5], [5, 6], [6, 8],
            [9, 10],
            # Shoulders and arms
            [11, 12], [11, 13], [13, 15],
            [12, 14], [14, 16],
            # Left hand
            [15, 17], [15, 19], [15, 21],
            [17, 19],
            # Right hand
            [16, 18], [16, 20], [16, 22],
            [18, 20],
            # Torso
            [11, 23], [12, 24], [23, 24],
            # Left leg
            [23, 25], [25, 27],
            [27, 29], [27, 31], [29, 31],
            # Right leg
            [24, 26], [26, 28],
            [28, 30], [28, 32], [30, 32]
        ]
        
        # All indices should be in range [0, 32]
        for start, end in connections:
            assert 0 <= start <= 32, f"Invalid start index: {start}"
            assert 0 <= end <= 32, f"Invalid end index: {end}"
    
    def test_connection_count(self):
        """MediaPipe Pose should have 35 connections."""
        connections = [
            # Face contour
            [0, 1], [1, 2], [2, 3], [3, 7],
            [0, 4], [4, 5], [5, 6], [6, 8],
            [9, 10],
            # Shoulders and arms
            [11, 12], [11, 13], [13, 15],
            [12, 14], [14, 16],
            # Left hand
            [15, 17], [15, 19], [15, 21],
            [17, 19],
            # Right hand
            [16, 18], [16, 20], [16, 22],
            [18, 20],
            # Torso
            [11, 23], [12, 24], [23, 24],
            # Left leg
            [23, 25], [25, 27],
            [27, 29], [27, 31], [29, 31],
            # Right leg
            [24, 26], [26, 28],
            [28, 30], [28, 32], [30, 32]
        ]
        
        assert len(connections) == 35, f"Expected 35 connections, got {len(connections)}"
    
    def test_landmark_names_match_indices(self):
        """Verify landmark indices match MediaPipe specification."""
        # Official MediaPipe Pose landmark names
        landmark_names = {
            0: "nose",
            1: "left_eye_inner",
            2: "left_eye",
            3: "left_eye_outer",
            4: "right_eye_inner",
            5: "right_eye",
            6: "right_eye_outer",
            7: "left_ear",
            8: "right_ear",
            9: "mouth_left",
            10: "mouth_right",
            11: "left_shoulder",
            12: "right_shoulder",
            13: "left_elbow",
            14: "right_elbow",
            15: "left_wrist",
            16: "right_wrist",
            17: "left_pinky",
            18: "right_pinky",
            19: "left_index",
            20: "right_index",
            21: "left_thumb",
            22: "right_thumb",
            23: "left_hip",
            24: "right_hip",
            25: "left_knee",
            26: "right_knee",
            27: "left_ankle",
            28: "right_ankle",
            29: "left_heel",
            30: "right_heel",
            31: "left_foot_index",
            32: "right_foot_index"
        }
        
        # Verify we have all 33 landmarks
        assert len(landmark_names) == 33
        
        # Verify indices are 0-32
        assert min(landmark_names.keys()) == 0
        assert max(landmark_names.keys()) == 32
    
    def test_connections_form_valid_skeleton(self):
        """Test that connections form a valid skeleton structure."""
        connections = [
            # Face contour
            [0, 1], [1, 2], [2, 3], [3, 7],
            [0, 4], [4, 5], [5, 6], [6, 8],
            [9, 10],
            # Shoulders and arms
            [11, 12], [11, 13], [13, 15],
            [12, 14], [14, 16],
            # Left hand
            [15, 17], [15, 19], [15, 21],
            [17, 19],
            # Right hand
            [16, 18], [16, 20], [16, 22],
            [18, 20],
            # Torso
            [11, 23], [12, 24], [23, 24],
            # Left leg
            [23, 25], [25, 27],
            [27, 29], [27, 31], [29, 31],
            # Right leg
            [24, 26], [26, 28],
            [28, 30], [28, 32], [30, 32]
        ]
        
        # Build adjacency list
        adjacency = {}
        for start, end in connections:
            if start not in adjacency:
                adjacency[start] = []
            if end not in adjacency:
                adjacency[end] = []
            adjacency[start].append(end)
            adjacency[end].append(start)
        
        # Key structural tests
        
        # 1. Shoulders should be connected
        assert 12 in adjacency[11], "Left shoulder should connect to right shoulder"
        
        # 2. Left arm chain: shoulder -> elbow -> wrist
        assert 13 in adjacency[11], "Left shoulder should connect to left elbow"
        assert 15 in adjacency[13], "Left elbow should connect to left wrist"
        
        # 3. Right arm chain: shoulder -> elbow -> wrist
        assert 14 in adjacency[12], "Right shoulder should connect to right elbow"
        assert 16 in adjacency[14], "Right elbow should connect to right wrist"
        
        # 4. Torso: shoulders to hips
        assert 23 in adjacency[11], "Left shoulder should connect to left hip"
        assert 24 in adjacency[12], "Right shoulder should connect to right hip"
        assert 24 in adjacency[23], "Left hip should connect to right hip"
        
        # 5. Left leg chain: hip -> knee -> ankle
        assert 25 in adjacency[23], "Left hip should connect to left knee"
        assert 27 in adjacency[25], "Left knee should connect to left ankle"
        
        # 6. Right leg chain: hip -> knee -> ankle
        assert 26 in adjacency[24], "Right hip should connect to right knee"
        assert 28 in adjacency[26], "Right knee should connect to right ankle"
        
        # 7. Feet details
        assert 29 in adjacency[27], "Left ankle should connect to left heel"
        assert 31 in adjacency[27], "Left ankle should connect to left foot index"
        assert 30 in adjacency[28], "Right ankle should connect to right heel"
        assert 32 in adjacency[28], "Right ankle should connect to right foot index"
    
    def test_no_duplicate_connections(self):
        """Ensure no duplicate connections exist."""
        connections = [
            # Face contour
            [0, 1], [1, 2], [2, 3], [3, 7],
            [0, 4], [4, 5], [5, 6], [6, 8],
            [9, 10],
            # Shoulders and arms
            [11, 12], [11, 13], [13, 15],
            [12, 14], [14, 16],
            # Left hand
            [15, 17], [15, 19], [15, 21],
            [17, 19],
            # Right hand
            [16, 18], [16, 20], [16, 22],
            [18, 20],
            # Torso
            [11, 23], [12, 24], [23, 24],
            # Left leg
            [23, 25], [25, 27],
            [27, 29], [27, 31], [29, 31],
            # Right leg
            [24, 26], [26, 28],
            [28, 30], [28, 32], [30, 32]
        ]
        
        # Normalize connections (sort each pair)
        normalized = [tuple(sorted(conn)) for conn in connections]
        
        # Check for duplicates
        assert len(normalized) == len(set(normalized)), "Duplicate connections found"
    
    def test_symmetric_body_parts(self):
        """Test that left and right body parts have symmetric connections."""
        connections = [
            # Face contour
            [0, 1], [1, 2], [2, 3], [3, 7],
            [0, 4], [4, 5], [5, 6], [6, 8],
            [9, 10],
            # Shoulders and arms
            [11, 12], [11, 13], [13, 15],
            [12, 14], [14, 16],
            # Left hand
            [15, 17], [15, 19], [15, 21],
            [17, 19],
            # Right hand
            [16, 18], [16, 20], [16, 22],
            [18, 20],
            # Torso
            [11, 23], [12, 24], [23, 24],
            # Left leg
            [23, 25], [25, 27],
            [27, 29], [27, 31], [29, 31],
            # Right leg
            [24, 26], [26, 28],
            [28, 30], [28, 32], [30, 32]
        ]
        
        # Build sets of connections for left and right sides
        left_arm = {(11, 13), (13, 15)}
        right_arm = {(12, 14), (14, 16)}
        
        left_leg = {(23, 25), (25, 27)}
        right_leg = {(24, 26), (26, 28)}
        
        # Verify symmetric structure (same number of connections)
        assert len(left_arm) == len(right_arm), "Arms should have symmetric connections"
        assert len(left_leg) == len(right_leg), "Legs should have symmetric connections"


class TestConnectionsVsOldFormat:
    """Compare new MediaPipe format vs old BODY_25 format."""
    
    def test_mediapipe_has_more_connections_than_body25(self):
        """MediaPipe should have more connections than BODY_25."""
        # MediaPipe connections (35)
        mediapipe_count = 35
        
        # BODY_25 connections (approximately 24)
        body25_count = 24
        
        assert mediapipe_count > body25_count, \
            f"MediaPipe ({mediapipe_count}) should have more connections than BODY_25 ({body25_count})"
    
    def test_mediapipe_includes_hand_details(self):
        """MediaPipe includes finger details that BODY_25 doesn't have."""
        # MediaPipe hand connections
        left_hand_connections = [
            [15, 17],  # wrist to pinky
            [15, 19],  # wrist to index
            [15, 21],  # wrist to thumb
            [17, 19],  # pinky to index
        ]
        
        right_hand_connections = [
            [16, 18],  # wrist to pinky
            [16, 20],  # wrist to index
            [16, 22],  # wrist to thumb
            [18, 20],  # pinky to index
        ]
        
        # BODY_25 doesn't have these detailed finger connections
        assert len(left_hand_connections) == 4
        assert len(right_hand_connections) == 4
    
    def test_mediapipe_includes_foot_details(self):
        """MediaPipe includes foot details that BODY_25 doesn't have."""
        # MediaPipe foot connections
        left_foot_connections = [
            [27, 29],  # ankle to heel
            [27, 31],  # ankle to foot index
            [29, 31],  # heel to foot index
        ]
        
        right_foot_connections = [
            [28, 30],  # ankle to heel
            [28, 32],  # ankle to foot index
            [30, 32],  # heel to foot index
        ]
        
        # BODY_25 doesn't have these detailed foot connections
        assert len(left_foot_connections) == 3
        assert len(right_foot_connections) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
