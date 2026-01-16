"""
Pose Keypoint Data Structures

This module provides robust, type-safe data structures for representing pose keypoints
and keypoint sets. It follows SOLID principles with clear separation of concerns:

- Keypoint: Immutable representation of a single pose keypoint
- KeypointSet: Collection of keypoints with metadata about the pose system
- KeypointSchema: Defines the structure and semantics of different pose systems
- KeypointFormat: Enum for supported pose estimation formats

Design Principles:
- Immutability: Keypoint objects are immutable for thread safety
- Type Safety: Strong typing with validation
- Extensibility: Easy to add new pose formats
- Interoperability: Conversion to/from common formats (dict, numpy, pandas)
- Performance: Efficient storage and access patterns
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd


class KeypointFormat(Enum):
    """
    Supported pose keypoint formats.
    
    Each format defines a specific set of body landmarks with
    different numbers of keypoints and semantic meanings.
    """
    MEDIAPIPE_33 = "mediapipe_33"  # MediaPipe BlazePose 33 landmarks
    COCO_17 = "coco_17"  # COCO 17 keypoints
    OPENPOSE_25 = "openpose_25"  # OpenPose 25 body keypoints
    ALPHAPOSE_26 = "alphapose_26"  # AlphaPose 26 keypoints
    HALPE_136 = "halpe_136"  # Halpe full body 136 keypoints
    CUSTOM = "custom"  # User-defined format
    
    @property
    def expected_count(self) -> Optional[int]:
        """Get the expected number of keypoints for this format."""
        counts = {
            KeypointFormat.MEDIAPIPE_33: 33,
            KeypointFormat.COCO_17: 17,
            KeypointFormat.OPENPOSE_25: 25,
            KeypointFormat.ALPHAPOSE_26: 26,
            KeypointFormat.HALPE_136: 136,
            KeypointFormat.CUSTOM: None,
        }
        return counts.get(self)


@dataclass(frozen=True)
class Keypoint:
    """
    Immutable representation of a single pose keypoint.
    
    A keypoint represents a specific anatomical landmark (e.g., left elbow, nose)
    with its spatial coordinates and confidence metrics.
    
    Attributes:
        id: Unique identifier within the keypoint set (0-indexed)
        name: Semantic name of the landmark (e.g., "LEFT_ELBOW", "NOSE")
        x: X coordinate in pixels (image space)
        y: Y coordinate in pixels (image space)
        z: Depth coordinate (optional, format-dependent)
        confidence: Overall confidence score [0.0, 1.0]
        visibility: Visibility score [0.0, 1.0] - whether landmark is visible
        presence: Presence score [0.0, 1.0] - whether landmark is present in frame
        x_normalized: X coordinate normalized to [0.0, 1.0] range
        y_normalized: Y coordinate normalized to [0.0, 1.0] range
    
    Design Notes:
        - Frozen dataclass ensures immutability
        - All coordinates stored as floats for precision
        - Normalized coordinates enable resolution-independent processing
        - Separate confidence, visibility, and presence for fine-grained quality assessment
    """
    id: int
    name: str
    x: float
    y: float
    z: float = 0.0
    confidence: float = 1.0
    visibility: float = 1.0
    presence: float = 1.0
    x_normalized: float = 0.0
    y_normalized: float = 0.0
    
    def __post_init__(self):
        """Validate keypoint data after initialization."""
        # Validate ID
        if self.id < 0:
            raise ValueError(f"Keypoint ID must be non-negative, got {self.id}")
        
        # Validate name
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"Keypoint name must be a non-empty string, got {self.name}")
        
        # Validate confidence scores (0.0 to 1.0)
        for score_name, score_value in [
            ("confidence", self.confidence),
            ("visibility", self.visibility),
            ("presence", self.presence),
        ]:
            if not 0.0 <= score_value <= 1.0:
                raise ValueError(
                    f"{score_name} must be in range [0.0, 1.0], got {score_value}"
                )
        
        # Validate normalized coordinates (0.0 to 1.0)
        if not 0.0 <= self.x_normalized <= 1.0:
            raise ValueError(
                f"x_normalized must be in range [0.0, 1.0], got {self.x_normalized}"
            )
        if not 0.0 <= self.y_normalized <= 1.0:
            raise ValueError(
                f"y_normalized must be in range [0.0, 1.0], got {self.y_normalized}"
            )
    
    @property
    def position(self) -> Tuple[float, float]:
        """Get 2D position as (x, y) tuple."""
        return (self.x, self.y)
    
    @property
    def position_3d(self) -> Tuple[float, float, float]:
        """Get 3D position as (x, y, z) tuple."""
        return (self.x, self.y, self.z)
    
    @property
    def normalized_position(self) -> Tuple[float, float]:
        """Get normalized 2D position as (x_norm, y_norm) tuple."""
        return (self.x_normalized, self.y_normalized)
    
    @property
    def is_visible(self) -> bool:
        """Check if keypoint is considered visible (visibility > 0.5)."""
        return self.visibility > 0.5
    
    @property
    def is_present(self) -> bool:
        """Check if keypoint is considered present (presence > 0.5)."""
        return self.presence > 0.5
    
    @property
    def is_reliable(self) -> bool:
        """Check if keypoint is reliable (confidence > 0.5)."""
        return self.confidence > 0.5
    
    def to_dict(self) -> Dict[str, Union[int, float, str]]:
        """
        Convert keypoint to dictionary representation.
        
        Returns:
            Dictionary with all keypoint attributes
        """
        return {
            'id': self.id,
            'name': self.name,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'confidence': self.confidence,
            'visibility': self.visibility,
            'presence': self.presence,
            'x_normalized': self.x_normalized,
            'y_normalized': self.y_normalized,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Keypoint':
        """
        Create keypoint from dictionary representation.
        
        Args:
            data: Dictionary with keypoint attributes
            
        Returns:
            Keypoint instance
        """
        return cls(
            id=data['id'],
            name=data['name'],
            x=data['x'],
            y=data['y'],
            z=data.get('z', 0.0),
            confidence=data.get('confidence', 1.0),
            visibility=data.get('visibility', 1.0),
            presence=data.get('presence', 1.0),
            x_normalized=data.get('x_normalized', 0.0),
            y_normalized=data.get('y_normalized', 0.0),
        )
    
    def distance_to(self, other: 'Keypoint') -> float:
        """
        Calculate Euclidean distance to another keypoint.
        
        Args:
            other: Another keypoint
            
        Returns:
            Euclidean distance in pixels
        """
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def distance_3d_to(self, other: 'Keypoint') -> float:
        """
        Calculate 3D Euclidean distance to another keypoint.
        
        Args:
            other: Another keypoint
            
        Returns:
            3D Euclidean distance
        """
        return np.sqrt(
            (self.x - other.x)**2 + 
            (self.y - other.y)**2 + 
            (self.z - other.z)**2
        )


@dataclass
class KeypointSet:
    """
    Collection of keypoints representing a complete pose detection.
    
    A KeypointSet represents all detected landmarks for a single person
    in a single frame, along with metadata about the detection format
    and quality.
    
    Attributes:
        keypoints: List of Keypoint objects
        format: KeypointFormat enum indicating the pose system used
        frame_width: Width of the source frame in pixels
        frame_height: Height of the source frame in pixels
        timestamp: Optional timestamp (frame number or time in seconds)
        person_id: Optional identifier for multi-person tracking
        metadata: Optional additional metadata
    
    Design Notes:
        - Mutable to allow efficient batch operations
        - Validates keypoint count against format expectations
        - Provides high-level operations (filtering, statistics, conversion)
        - Supports both single-person and multi-person scenarios
    """
    keypoints: List[Keypoint]
    format: KeypointFormat
    frame_width: int
    frame_height: int
    timestamp: Optional[float] = None
    person_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate keypoint set after initialization."""
        # Validate frame dimensions
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError(
                f"Frame dimensions must be positive, got {self.frame_width}x{self.frame_height}"
            )
        
        # Validate keypoint count for known formats
        expected_count = self.format.expected_count
        if expected_count is not None and len(self.keypoints) != expected_count:
            raise ValueError(
                f"Format {self.format.value} expects {expected_count} keypoints, "
                f"got {len(self.keypoints)}"
            )
        
        # Validate keypoint IDs are sequential
        for i, kp in enumerate(self.keypoints):
            if kp.id != i:
                raise ValueError(
                    f"Keypoint IDs must be sequential starting from 0, "
                    f"expected ID {i} but got {kp.id}"
                )
    
    def __len__(self) -> int:
        """Get number of keypoints."""
        return len(self.keypoints)
    
    def __getitem__(self, index: Union[int, str]) -> Keypoint:
        """
        Get keypoint by index or name.
        
        Args:
            index: Integer index or string name
            
        Returns:
            Keypoint at the specified index or with the specified name
        """
        if isinstance(index, int):
            return self.keypoints[index]
        elif isinstance(index, str):
            for kp in self.keypoints:
                if kp.name == index:
                    return kp
            raise KeyError(f"No keypoint with name '{index}'")
        else:
            raise TypeError(f"Index must be int or str, got {type(index)}")
    
    def __iter__(self):
        """Iterate over keypoints."""
        return iter(self.keypoints)
    
    @property
    def visible_keypoints(self) -> List[Keypoint]:
        """Get list of visible keypoints (visibility > 0.5)."""
        return [kp for kp in self.keypoints if kp.is_visible]
    
    @property
    def reliable_keypoints(self) -> List[Keypoint]:
        """Get list of reliable keypoints (confidence > 0.5)."""
        return [kp for kp in self.keypoints if kp.is_reliable]
    
    @property
    def avg_confidence(self) -> float:
        """Calculate average confidence across all keypoints."""
        if not self.keypoints:
            return 0.0
        return np.mean([kp.confidence for kp in self.keypoints])
    
    @property
    def avg_visibility(self) -> float:
        """Calculate average visibility across all keypoints."""
        if not self.keypoints:
            return 0.0
        return np.mean([kp.visibility for kp in self.keypoints])
    
    @property
    def detection_quality(self) -> float:
        """
        Calculate overall detection quality score.
        
        Combines confidence, visibility, and presence into a single metric.
        """
        if not self.keypoints:
            return 0.0
        
        scores = [
            (kp.confidence + kp.visibility + kp.presence) / 3.0
            for kp in self.keypoints
        ]
        return np.mean(scores)
    
    def get_keypoint_by_name(self, name: str) -> Optional[Keypoint]:
        """
        Get keypoint by name.
        
        Args:
            name: Keypoint name
            
        Returns:
            Keypoint with the specified name, or None if not found
        """
        for kp in self.keypoints:
            if kp.name == name:
                return kp
        return None
    
    def filter_by_confidence(self, min_confidence: float = 0.5) -> 'KeypointSet':
        """
        Create a new KeypointSet with only high-confidence keypoints.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            New KeypointSet with filtered keypoints
        """
        filtered = [kp for kp in self.keypoints if kp.confidence >= min_confidence]
        return KeypointSet(
            keypoints=filtered,
            format=KeypointFormat.CUSTOM,  # Filtered set is custom format
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            timestamp=self.timestamp,
            person_id=self.person_id,
            metadata={**self.metadata, 'filtered': True, 'min_confidence': min_confidence}
        )
    
    def to_numpy(self) -> np.ndarray:
        """
        Convert keypoints to numpy array.
        
        Returns:
            Array of shape (N, 10) with columns:
            [id, x, y, z, confidence, visibility, presence, x_norm, y_norm, name_hash]
        """
        if not self.keypoints:
            return np.empty((0, 10))
        
        data = []
        for kp in self.keypoints:
            data.append([
                kp.id,
                kp.x,
                kp.y,
                kp.z,
                kp.confidence,
                kp.visibility,
                kp.presence,
                kp.x_normalized,
                kp.y_normalized,
                hash(kp.name) % 1000000,  # Name hash for reference
            ])
        return np.array(data)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert keypoints to pandas DataFrame.
        
        Returns:
            DataFrame with one row per keypoint
        """
        if not self.keypoints:
            return pd.DataFrame()
        
        data = [kp.to_dict() for kp in self.keypoints]
        df = pd.DataFrame(data)
        
        # Add metadata columns
        if self.timestamp is not None:
            df['timestamp'] = self.timestamp
        if self.person_id is not None:
            df['person_id'] = self.person_id
        df['format'] = self.format.value
        
        return df
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """
        Convert keypoints to list of dictionaries.
        
        Returns:
            List of keypoint dictionaries
        """
        return [kp.to_dict() for kp in self.keypoints]
    
    @classmethod
    def from_dict_list(
        cls,
        data: List[Dict[str, Any]],
        format: KeypointFormat,
        frame_width: int,
        frame_height: int,
        timestamp: Optional[float] = None,
        person_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'KeypointSet':
        """
        Create KeypointSet from list of dictionaries.
        
        Args:
            data: List of keypoint dictionaries
            format: KeypointFormat enum
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            timestamp: Optional timestamp
            person_id: Optional person identifier
            metadata: Optional metadata dictionary
            
        Returns:
            KeypointSet instance
        """
        keypoints = [Keypoint.from_dict(kp_dict) for kp_dict in data]
        return cls(
            keypoints=keypoints,
            format=format,
            frame_width=frame_width,
            frame_height=frame_height,
            timestamp=timestamp,
            person_id=person_id,
            metadata=metadata or {}
        )
    
    @classmethod
    def from_mediapipe(
        cls,
        landmarks: List[Any],
        frame_width: int,
        frame_height: int,
        landmark_names: List[str],
        timestamp: Optional[float] = None,
        person_id: Optional[int] = None
    ) -> 'KeypointSet':
        """
        Create KeypointSet from MediaPipe landmarks.
        
        Args:
            landmarks: MediaPipe landmark list
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            landmark_names: List of landmark names
            timestamp: Optional timestamp
            person_id: Optional person identifier
            
        Returns:
            KeypointSet instance
        """
        keypoints = []
        for i, landmark in enumerate(landmarks):
            name = landmark_names[i] if i < len(landmark_names) else f"LANDMARK_{i}"
            keypoint = Keypoint(
                id=i,
                name=name,
                x=landmark.x * frame_width,
                y=landmark.y * frame_height,
                z=landmark.z,
                confidence=landmark.visibility,
                visibility=landmark.visibility,
                presence=landmark.presence,
                x_normalized=landmark.x,
                y_normalized=landmark.y
            )
            keypoints.append(keypoint)
        
        return cls(
            keypoints=keypoints,
            format=KeypointFormat.MEDIAPIPE_33,
            frame_width=frame_width,
            frame_height=frame_height,
            timestamp=timestamp,
            person_id=person_id
        )


@dataclass
class KeypointSchema:
    """
    Defines the structure and semantics of a keypoint format.
    
    A schema describes the expected keypoints, their names, semantic
    groupings, and connections for a specific pose estimation system.
    
    Attributes:
        format: KeypointFormat enum
        keypoint_names: Ordered list of keypoint names
        connections: List of (start_idx, end_idx) tuples defining skeleton
        semantic_groups: Dictionary grouping keypoints by body part
        description: Human-readable description of the format
    
    Design Notes:
        - Enables validation and visualization
        - Supports multiple pose formats
        - Extensible for custom formats
    """
    format: KeypointFormat
    keypoint_names: List[str]
    connections: List[Tuple[int, int]] = field(default_factory=list)
    semantic_groups: Dict[str, List[int]] = field(default_factory=dict)
    description: str = ""
    
    def __post_init__(self):
        """Validate schema after initialization."""
        expected_count = self.format.expected_count
        if expected_count is not None and len(self.keypoint_names) != expected_count:
            raise ValueError(
                f"Format {self.format.value} expects {expected_count} keypoints, "
                f"got {len(self.keypoint_names)} names"
            )
    
    def validate_keypoint_set(self, keypoint_set: KeypointSet) -> bool:
        """
        Validate that a KeypointSet conforms to this schema.
        
        Args:
            keypoint_set: KeypointSet to validate
            
        Returns:
            True if valid, False otherwise
        """
        if keypoint_set.format != self.format:
            return False
        
        if len(keypoint_set) != len(self.keypoint_names):
            return False
        
        for i, kp in enumerate(keypoint_set.keypoints):
            if kp.name != self.keypoint_names[i]:
                return False
        
        return True


# ============================================================================
# Predefined Schemas
# ============================================================================

# MediaPipe BlazePose 33 landmarks
MEDIAPIPE_33_NAMES = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
    'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
    'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
    'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
    'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
    'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
    'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]

MEDIAPIPE_33_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 9), (0, 10), (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

MEDIAPIPE_33_GROUPS = {
    'face': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'left_arm': [11, 13, 15, 17, 19, 21],
    'right_arm': [12, 14, 16, 18, 20, 22],
    'torso': [11, 12, 23, 24],
    'left_leg': [23, 25, 27, 29, 31],
    'right_leg': [24, 26, 28, 30, 32],
}

MEDIAPIPE_33_SCHEMA = KeypointSchema(
    format=KeypointFormat.MEDIAPIPE_33,
    keypoint_names=MEDIAPIPE_33_NAMES,
    connections=MEDIAPIPE_33_CONNECTIONS,
    semantic_groups=MEDIAPIPE_33_GROUPS,
    description="MediaPipe BlazePose 33 landmarks for full body pose estimation"
)


# Schema registry for easy lookup
KEYPOINT_SCHEMAS: Dict[KeypointFormat, KeypointSchema] = {
    KeypointFormat.MEDIAPIPE_33: MEDIAPIPE_33_SCHEMA,
}


def get_schema(format: KeypointFormat) -> Optional[KeypointSchema]:
    """
    Get the schema for a specific keypoint format.
    
    Args:
        format: KeypointFormat enum
        
    Returns:
        KeypointSchema if available, None otherwise
    """
    return KEYPOINT_SCHEMAS.get(format)
