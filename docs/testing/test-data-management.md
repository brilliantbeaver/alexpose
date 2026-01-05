# Test Data Management Guide

## Overview

Effective test data management is crucial for reliable, maintainable testing. This guide covers strategies for managing test data in the AlexPose system, with emphasis on using real data while maintaining test reliability and performance.

## Test Data Philosophy

### Core Principles

1. **Real Data Priority**: Use actual video files, pose sequences, and gait data over synthetic data (target: >70% real data usage)
2. **Data Versioning**: Version control test data to ensure reproducible tests
3. **Efficient Storage**: Use appropriate storage strategies for different data types
4. **Data Isolation**: Ensure tests don't interfere with each other's data
5. **Performance Optimization**: Balance data realism with test execution speed

### Data Hierarchy

```
tests/
├── fixtures/                    # Test data fixtures and managers
│   ├── real_data_fixtures.py   # Real data management
│   ├── synthetic_fixtures.py   # Synthetic data generation
│   └── model_fixtures.py       # ML model mocks
├── data/                       # Test-specific data files
│   ├── videos/                 # Sample video files
│   ├── poses/                  # Pose sequence data
│   ├── features/               # Gait feature samples
│   └── configs/                # Test configurations
└── utils/
    ├── data_generators.py      # Synthetic data generators
    └── data_validators.py      # Data validation utilities
```

## Real Data Management

### RealDataManager Class

The `RealDataManager` provides centralized access to real test data:

```python
from tests.fixtures.real_data_fixtures import RealDataManager

# Initialize data manager
data_manager = RealDataManager()

# Get sample videos
videos = data_manager.get_sample_videos()
normal_video = videos["normal_walking"]
abnormal_video = videos["abnormal_gait"]

# Get GAVD test subset
gavd_data = data_manager.get_gavd_test_subset()
normal_samples = gavd_data["normal_samples"]
abnormal_samples = gavd_data["abnormal_samples"]
```

### Video Test Data

#### Sample Video Categories

1. **Normal Walking**: Typical gait patterns for baseline testing
2. **Abnormal Gait**: Various pathological gait patterns
3. **Multiple Subjects**: Videos with multiple people for detection testing
4. **Edge Cases**: Challenging scenarios (poor lighting, occlusion, etc.)

#### Video Data Structure

```python
# Expected video data structure
sample_videos = {
    "normal_walking": Path("data/test_data/videos/normal_walking_sample.mp4"),
    "abnormal_gait": Path("data/test_data/videos/abnormal_gait_sample.mp4"),
    "multiple_subjects": Path("data/test_data/videos/multiple_subjects.mp4"),
    "short_clip": Path("data/test_data/videos/short_clip_5s.mp4"),
    "long_clip": Path("data/test_data/videos/long_clip_60s.mp4"),
    "poor_quality": Path("data/test_data/videos/poor_quality.mp4"),
    "high_resolution": Path("data/test_data/videos/high_res_1080p.mp4")
}
```

#### Video Data Validation

```python
def validate_video_data(video_path: Path) -> Dict[str, Any]:
    """Validate video file for testing."""
    import cv2
    
    if not video_path.exists():
        return {"valid": False, "error": "File does not exist"}
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"valid": False, "error": "Cannot open video file"}
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        "valid": True,
        "frame_count": frame_count,
        "fps": fps,
        "resolution": (width, height),
        "duration": duration,
        "file_size": video_path.stat().st_size
    }
```

### GAVD Dataset Integration

#### GAVD Test Subset Management

```python
class GAVDTestManager:
    """Manage GAVD dataset subset for testing."""
    
    def __init__(self, data_dir: Path = Path("data/test_datasets")):
        self.data_dir = data_dir
        self.gavd_file = data_dir / "gavd_test_subset.csv"
    
    def load_test_subset(self) -> Dict[str, Any]:
        """Load curated GAVD test subset."""
        if self.gavd_file.exists():
            return self._load_real_gavd_data()
        else:
            return self._create_synthetic_gavd_subset()
    
    def _load_real_gavd_data(self) -> Dict[str, Any]:
        """Load real GAVD data from CSV file."""
        import pandas as pd
        
        df = pd.read_csv(self.gavd_file)
        
        # Validate required columns
        required_columns = ['subject_id', 'label', 'stride_time', 'cadence', 'stride_length']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Separate by classification
        normal_samples = df[df['label'] == 'normal'].to_dict('records')
        abnormal_samples = df[df['label'] == 'abnormal'].to_dict('records')
        
        return {
            "normal_samples": normal_samples,
            "abnormal_samples": abnormal_samples,
            "metadata": {
                "total_samples": len(df),
                "normal_count": len(normal_samples),
                "abnormal_count": len(abnormal_samples),
                "source": "real_gavd_data"
            }
        }
    
    def _create_synthetic_gavd_subset(self) -> Dict[str, Any]:
        """Create synthetic GAVD-like data for testing."""
        import numpy as np
        
        # Generate realistic synthetic data based on GAVD statistics
        normal_samples = []
        for i in range(50):
            normal_samples.append({
                'subject_id': f'normal_{i:03d}',
                'stride_time': np.random.normal(1.2, 0.1),
                'cadence': np.random.normal(115, 10),
                'stride_length': np.random.normal(1.4, 0.15),
                'step_width': np.random.normal(0.1, 0.02),
                'foot_angle': np.random.normal(15, 5),
                'label': 'normal'
            })
        
        abnormal_samples = []
        for i in range(30):
            abnormal_samples.append({
                'subject_id': f'abnormal_{i:03d}',
                'stride_time': np.random.normal(1.6, 0.2),
                'cadence': np.random.normal(85, 15),
                'stride_length': np.random.normal(1.0, 0.2),
                'step_width': np.random.normal(0.15, 0.03),
                'foot_angle': np.random.normal(25, 8),
                'label': 'abnormal'
            })
        
        return {
            "normal_samples": normal_samples,
            "abnormal_samples": abnormal_samples,
            "metadata": {
                "total_samples": 80,
                "normal_count": 50,
                "abnormal_count": 30,
                "source": "synthetic_data"
            }
        }
```

### Pose Sequence Data

#### Pose Data Structure

```python
# MediaPipe pose landmark structure
pose_landmark = {
    'x': float,          # Normalized x coordinate (0-1)
    'y': float,          # Normalized y coordinate (0-1)
    'z': float,          # Depth coordinate
    'confidence': float, # Detection confidence (0-1)
    'visibility': float  # Visibility score (0-1)
}

# Complete pose sequence
pose_sequence = {
    'subject_id': str,
    'video_id': str,
    'frame_rate': float,
    'total_frames': int,
    'landmarks': [
        {
            'frame_number': int,
            'timestamp': float,
            'pose_landmarks': [pose_landmark] * 33  # 33 MediaPipe landmarks
        }
    ]
}
```

#### Pose Data Fixtures

```python
@pytest.fixture(scope="session")
def sample_pose_sequences():
    """Provide sample pose sequences for testing."""
    data_manager = RealDataManager()
    
    # Try to load real pose data
    pose_data_dir = Path("data/test_data/poses")
    if pose_data_dir.exists():
        return load_real_pose_sequences(pose_data_dir)
    else:
        return generate_synthetic_pose_sequences()

def load_real_pose_sequences(data_dir: Path) -> Dict[str, Any]:
    """Load real pose sequence data."""
    sequences = {}
    
    for pose_file in data_dir.glob("*.json"):
        with open(pose_file, 'r') as f:
            sequence_data = json.load(f)
            sequences[pose_file.stem] = sequence_data
    
    return sequences

def generate_synthetic_pose_sequences() -> Dict[str, Any]:
    """Generate synthetic pose sequences for testing."""
    from tests.utils.data_generators import generate_realistic_pose_sequence
    
    return {
        "normal_walking": generate_realistic_pose_sequence(
            duration=5.0, frame_rate=30.0, gait_pattern="normal"
        ),
        "abnormal_gait": generate_realistic_pose_sequence(
            duration=5.0, frame_rate=30.0, gait_pattern="abnormal"
        ),
        "multiple_subjects": generate_realistic_pose_sequence(
            duration=3.0, frame_rate=30.0, num_subjects=2
        )
    }
```

## Synthetic Data Generation

### Data Generation Strategies

#### Realistic Pose Sequence Generation

```python
import numpy as np
from typing import List, Dict, Any

def generate_realistic_pose_sequence(
    duration: float = 5.0,
    frame_rate: float = 30.0,
    gait_pattern: str = "normal",
    num_subjects: int = 1
) -> Dict[str, Any]:
    """Generate realistic pose sequence data."""
    
    num_frames = int(duration * frame_rate)
    landmarks_per_frame = []
    
    for frame_idx in range(num_frames):
        timestamp = frame_idx / frame_rate
        frame_landmarks = []
        
        for subject_idx in range(num_subjects):
            # Generate pose landmarks for this subject
            subject_landmarks = generate_subject_pose(
                timestamp, gait_pattern, subject_idx
            )
            frame_landmarks.extend(subject_landmarks)
        
        landmarks_per_frame.append({
            'frame_number': frame_idx,
            'timestamp': timestamp,
            'pose_landmarks': frame_landmarks
        })
    
    return {
        'duration': duration,
        'frame_rate': frame_rate,
        'total_frames': num_frames,
        'num_subjects': num_subjects,
        'gait_pattern': gait_pattern,
        'landmarks': landmarks_per_frame
    }

def generate_subject_pose(
    timestamp: float,
    gait_pattern: str,
    subject_idx: int = 0
) -> List[Dict[str, float]]:
    """Generate pose landmarks for a single subject."""
    
    # Base pose template (standing position)
    base_pose = get_base_pose_template()
    
    # Apply gait cycle animation
    gait_phase = (timestamp * 2.0) % 2.0  # 2-second gait cycle
    
    if gait_pattern == "normal":
        animated_pose = apply_normal_gait_animation(base_pose, gait_phase)
    elif gait_pattern == "abnormal":
        animated_pose = apply_abnormal_gait_animation(base_pose, gait_phase)
    else:
        animated_pose = base_pose
    
    # Add subject-specific offset
    if subject_idx > 0:
        animated_pose = apply_subject_offset(animated_pose, subject_idx)
    
    # Add realistic noise and confidence values
    return add_realistic_noise(animated_pose)

def get_base_pose_template() -> List[Dict[str, float]]:
    """Get base pose template for MediaPipe landmarks."""
    # MediaPipe pose landmark indices and approximate positions
    landmarks = []
    
    # Define base positions for 33 landmarks
    base_positions = [
        # Head landmarks (0-10)
        (0.5, 0.1), (0.48, 0.12), (0.52, 0.12), (0.46, 0.14), (0.54, 0.14),
        (0.44, 0.16), (0.56, 0.16), (0.42, 0.18), (0.58, 0.18), (0.48, 0.2), (0.52, 0.2),
        
        # Upper body landmarks (11-22)
        (0.45, 0.3), (0.55, 0.3), (0.4, 0.4), (0.6, 0.4), (0.35, 0.5), (0.65, 0.5),
        (0.38, 0.45), (0.62, 0.45), (0.36, 0.48), (0.64, 0.48), (0.34, 0.52), (0.66, 0.52),
        
        # Lower body landmarks (23-32)
        (0.45, 0.6), (0.55, 0.6), (0.44, 0.8), (0.56, 0.8), (0.43, 0.9), (0.57, 0.9),
        (0.42, 0.95), (0.58, 0.95), (0.41, 0.98), (0.59, 0.98), (0.4, 1.0)
    ]
    
    for i, (x, y) in enumerate(base_positions):
        landmarks.append({
            'x': x,
            'y': y,
            'z': np.random.normal(0, 0.1),  # Random depth
            'confidence': np.random.uniform(0.8, 1.0),
            'visibility': np.random.uniform(0.9, 1.0)
        })
    
    return landmarks

def apply_normal_gait_animation(
    base_pose: List[Dict[str, float]], 
    gait_phase: float
) -> List[Dict[str, float]]:
    """Apply normal gait cycle animation to pose."""
    animated_pose = base_pose.copy()
    
    # Simulate walking motion
    # Left leg forward phase (0-1), right leg forward phase (1-2)
    left_leg_phase = gait_phase if gait_phase < 1.0 else 2.0 - gait_phase
    right_leg_phase = 1.0 - left_leg_phase
    
    # Apply leg movements (simplified)
    # Left hip (23), knee (25), ankle (27)
    animated_pose[23]['y'] += 0.05 * np.sin(left_leg_phase * np.pi)
    animated_pose[25]['y'] += 0.1 * np.sin(left_leg_phase * np.pi)
    animated_pose[27]['y'] += 0.02 * np.sin(left_leg_phase * np.pi)
    
    # Right hip (24), knee (26), ankle (28)
    animated_pose[24]['y'] += 0.05 * np.sin(right_leg_phase * np.pi)
    animated_pose[26]['y'] += 0.1 * np.sin(right_leg_phase * np.pi)
    animated_pose[28]['y'] += 0.02 * np.sin(right_leg_phase * np.pi)
    
    # Add arm swing
    # Left shoulder (11), elbow (13), wrist (15)
    arm_swing = 0.03 * np.sin(gait_phase * np.pi)
    animated_pose[11]['x'] += arm_swing
    animated_pose[13]['x'] += arm_swing * 1.5
    animated_pose[15]['x'] += arm_swing * 2
    
    # Right shoulder (12), elbow (14), wrist (16)
    animated_pose[12]['x'] -= arm_swing
    animated_pose[14]['x'] -= arm_swing * 1.5
    animated_pose[16]['x'] -= arm_swing * 2
    
    return animated_pose

def apply_abnormal_gait_animation(
    base_pose: List[Dict[str, float]], 
    gait_phase: float
) -> List[Dict[str, float]]:
    """Apply abnormal gait pattern animation."""
    # Start with normal gait
    animated_pose = apply_normal_gait_animation(base_pose, gait_phase)
    
    # Add abnormal characteristics
    # Asymmetric stride length
    if gait_phase < 1.0:  # Left leg forward
        # Shorter left stride
        animated_pose[23]['y'] *= 0.7
        animated_pose[25]['y'] *= 0.7
    else:  # Right leg forward
        # Normal right stride
        pass
    
    # Add limp (reduced weight bearing on left leg)
    if gait_phase < 1.0:
        # Reduce left leg loading
        animated_pose[23]['y'] += 0.02
        animated_pose[25]['y'] += 0.03
    
    # Add trunk lean
    trunk_lean = 0.02 * np.sin(gait_phase * np.pi)
    for i in range(11, 17):  # Upper body landmarks
        animated_pose[i]['x'] += trunk_lean
    
    return animated_pose

def add_realistic_noise(pose: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Add realistic noise and confidence values to pose data."""
    noisy_pose = []
    
    for landmark in pose:
        # Add small amount of noise to coordinates
        noisy_landmark = {
            'x': landmark['x'] + np.random.normal(0, 0.01),
            'y': landmark['y'] + np.random.normal(0, 0.01),
            'z': landmark['z'] + np.random.normal(0, 0.02),
            'confidence': max(0.0, min(1.0, landmark['confidence'] + np.random.normal(0, 0.05))),
            'visibility': max(0.0, min(1.0, landmark['visibility'] + np.random.normal(0, 0.03)))
        }
        
        # Ensure coordinates stay within valid ranges
        noisy_landmark['x'] = max(0.0, min(1.0, noisy_landmark['x']))
        noisy_landmark['y'] = max(0.0, min(1.0, noisy_landmark['y']))
        
        noisy_pose.append(noisy_landmark)
    
    return noisy_pose
```

#### Gait Feature Generation

```python
def generate_realistic_gait_features(
    pattern: str = "normal",
    num_samples: int = 100
) -> List[Dict[str, float]]:
    """Generate realistic gait feature data."""
    
    features = []
    
    for i in range(num_samples):
        if pattern == "normal":
            feature_set = {
                'stride_time': np.random.normal(1.2, 0.1),
                'cadence': np.random.normal(115, 10),
                'stride_length': np.random.normal(1.4, 0.15),
                'step_width': np.random.normal(0.1, 0.02),
                'foot_angle': np.random.normal(15, 5),
                'double_support_time': np.random.normal(0.24, 0.03),
                'swing_time': np.random.normal(0.38, 0.04),
                'stance_time': np.random.normal(0.62, 0.04)
            }
        elif pattern == "abnormal":
            feature_set = {
                'stride_time': np.random.normal(1.6, 0.2),
                'cadence': np.random.normal(85, 15),
                'stride_length': np.random.normal(1.0, 0.2),
                'step_width': np.random.normal(0.15, 0.03),
                'foot_angle': np.random.normal(25, 8),
                'double_support_time': np.random.normal(0.35, 0.05),
                'swing_time': np.random.normal(0.32, 0.06),
                'stance_time': np.random.normal(0.68, 0.06)
            }
        else:
            # Random features
            feature_set = {
                'stride_time': np.random.uniform(0.8, 2.0),
                'cadence': np.random.uniform(60, 180),
                'stride_length': np.random.uniform(0.8, 2.5),
                'step_width': np.random.uniform(0.05, 0.3),
                'foot_angle': np.random.uniform(-30, 30),
                'double_support_time': np.random.uniform(0.15, 0.45),
                'swing_time': np.random.uniform(0.25, 0.45),
                'stance_time': np.random.uniform(0.55, 0.75)
            }
        
        # Add derived features
        feature_set['step_time'] = feature_set['stride_time'] / 2
        feature_set['velocity'] = feature_set['stride_length'] / feature_set['stride_time']
        feature_set['asymmetry_index'] = np.random.uniform(0.9, 1.1)
        
        # Add sample metadata
        feature_set['sample_id'] = f"{pattern}_{i:03d}"
        feature_set['pattern'] = pattern
        
        features.append(feature_set)
    
    return features
```

## Data Storage and Versioning

### Git LFS Integration

For large test data files (videos, large datasets):

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "data/test_data/videos/*.mp4"
git lfs track "data/test_data/poses/*.json"
git lfs track "*.h5"  # Model files

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for test data"
```

### Data Versioning Strategy

```python
class TestDataVersionManager:
    """Manage test data versions and compatibility."""
    
    def __init__(self, data_dir: Path = Path("data/test_data")):
        self.data_dir = data_dir
        self.version_file = data_dir / "data_version.json"
    
    def get_current_version(self) -> Dict[str, Any]:
        """Get current test data version information."""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                return json.load(f)
        else:
            return self.create_initial_version()
    
    def create_initial_version(self) -> Dict[str, Any]:
        """Create initial version file."""
        version_info = {
            "version": "1.0.0",
            "created": datetime.now().isoformat(),
            "description": "Initial test data version",
            "data_files": self.catalog_data_files(),
            "checksums": self.calculate_checksums()
        }
        
        with open(self.version_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        return version_info
    
    def catalog_data_files(self) -> Dict[str, List[str]]:
        """Catalog all test data files by category."""
        catalog = {
            "videos": [str(f.relative_to(self.data_dir)) for f in self.data_dir.glob("videos/*.mp4")],
            "poses": [str(f.relative_to(self.data_dir)) for f in self.data_dir.glob("poses/*.json")],
            "features": [str(f.relative_to(self.data_dir)) for f in self.data_dir.glob("features/*.csv")],
            "configs": [str(f.relative_to(self.data_dir)) for f in self.data_dir.glob("configs/*.yaml")]
        }
        return catalog
    
    def calculate_checksums(self) -> Dict[str, str]:
        """Calculate checksums for data integrity verification."""
        import hashlib
        
        checksums = {}
        for file_path in self.data_dir.rglob("*"):
            if file_path.is_file() and file_path.name != "data_version.json":
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    checksums[str(file_path.relative_to(self.data_dir))] = file_hash
        
        return checksums
    
    def verify_data_integrity(self) -> Dict[str, Any]:
        """Verify test data integrity using checksums."""
        version_info = self.get_current_version()
        stored_checksums = version_info.get("checksums", {})
        
        current_checksums = self.calculate_checksums()
        
        missing_files = set(stored_checksums.keys()) - set(current_checksums.keys())
        new_files = set(current_checksums.keys()) - set(stored_checksums.keys())
        modified_files = []
        
        for file_path in set(stored_checksums.keys()) & set(current_checksums.keys()):
            if stored_checksums[file_path] != current_checksums[file_path]:
                modified_files.append(file_path)
        
        return {
            "integrity_check": len(missing_files) == 0 and len(modified_files) == 0,
            "missing_files": list(missing_files),
            "new_files": list(new_files),
            "modified_files": modified_files,
            "total_files": len(current_checksums)
        }
```

## Test Data Fixtures

### Pytest Fixtures for Data Management

```python
import pytest
from pathlib import Path
from tests.fixtures.real_data_fixtures import RealDataManager

@pytest.fixture(scope="session")
def real_data_manager():
    """Provide real data manager for tests."""
    return RealDataManager()

@pytest.fixture(scope="session")
def sample_gait_videos(real_data_manager):
    """Provide real gait video samples for testing."""
    videos = real_data_manager.get_sample_videos()
    
    # Validate that required videos exist
    required_videos = ["normal_walking", "abnormal_gait"]
    for video_name in required_videos:
        if video_name not in videos or not videos[video_name].exists():
            pytest.skip(f"Required test video not available: {video_name}")
    
    return videos

@pytest.fixture(scope="session")
def gavd_test_subset(real_data_manager):
    """Provide curated GAVD dataset subset for consistent testing."""
    return real_data_manager.get_gavd_test_subset()

@pytest.fixture
def synthetic_pose_sequence():
    """Generate synthetic but realistic pose sequences."""
    from tests.utils.data_generators import generate_realistic_pose_sequence
    return generate_realistic_pose_sequence(
        duration=5.0,
        frame_rate=30.0,
        gait_pattern="normal"
    )

@pytest.fixture
def synthetic_gait_features():
    """Generate synthetic gait features for testing."""
    from tests.utils.data_generators import generate_realistic_gait_features
    return generate_realistic_gait_features(pattern="normal", num_samples=10)

@pytest.fixture
def test_data_cleanup():
    """Cleanup test data after test execution."""
    created_files = []
    
    def register_file(file_path: Path):
        created_files.append(file_path)
    
    yield register_file
    
    # Cleanup
    for file_path in created_files:
        if file_path.exists():
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                import shutil
                shutil.rmtree(file_path)
```

### Parameterized Test Data

```python
import pytest

# Parameterize tests with different data sets
@pytest.mark.parametrize("video_type,expected_classification", [
    ("normal_walking", "normal"),
    ("abnormal_gait", "abnormal"),
    ("multiple_subjects", "multiple")
])
def test_video_classification(sample_gait_videos, video_type, expected_classification):
    """Test video classification with different video types."""
    video_path = sample_gait_videos[video_type]
    
    # Process video
    result = classify_video(video_path)
    
    assert result["classification"] == expected_classification

# Parameterize with synthetic data
@pytest.mark.parametrize("gait_pattern,expected_features", [
    ("normal", {"stride_time": (1.0, 1.4), "cadence": (100, 130)}),
    ("abnormal", {"stride_time": (1.4, 1.8), "cadence": (70, 100)})
])
def test_gait_feature_extraction(gait_pattern, expected_features):
    """Test gait feature extraction with different patterns."""
    from tests.utils.data_generators import generate_realistic_pose_sequence
    
    pose_sequence = generate_realistic_pose_sequence(
        duration=10.0, frame_rate=30.0, gait_pattern=gait_pattern
    )
    
    features = extract_gait_features(pose_sequence)
    
    for feature_name, (min_val, max_val) in expected_features.items():
        assert min_val <= features[feature_name] <= max_val
```

## Performance Optimization

### Lazy Loading

```python
class LazyTestDataLoader:
    """Lazy loading for expensive test data."""
    
    def __init__(self):
        self._video_cache = {}
        self._pose_cache = {}
    
    def get_video(self, video_name: str) -> Path:
        """Get video with lazy loading."""
        if video_name not in self._video_cache:
            video_path = self._load_video(video_name)
            self._video_cache[video_name] = video_path
        
        return self._video_cache[video_name]
    
    def get_pose_sequence(self, sequence_name: str) -> Dict[str, Any]:
        """Get pose sequence with lazy loading."""
        if sequence_name not in self._pose_cache:
            pose_data = self._load_pose_sequence(sequence_name)
            self._pose_cache[sequence_name] = pose_data
        
        return self._pose_cache[sequence_name]
    
    def _load_video(self, video_name: str) -> Path:
        """Load video file."""
        # Implementation for loading video
        pass
    
    def _load_pose_sequence(self, sequence_name: str) -> Dict[str, Any]:
        """Load pose sequence data."""
        # Implementation for loading pose data
        pass
```

### Data Caching

```python
import functools
import pickle
from pathlib import Path

def cache_test_data(cache_dir: Path = Path(".test_cache")):
    """Decorator to cache expensive test data generation."""
    cache_dir.mkdir(exist_ok=True)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            cache_file = cache_dir / f"{cache_key}.pkl"
            
            # Try to load from cache
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    # Cache corrupted, regenerate
                    pass
            
            # Generate data and cache it
            result = func(*args, **kwargs)
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except Exception:
                # Caching failed, but return result anyway
                pass
            
            return result
        
        return wrapper
    return decorator

# Usage
@cache_test_data()
def generate_expensive_test_data(duration: float, complexity: str) -> Dict[str, Any]:
    """Generate expensive test data with caching."""
    # Expensive data generation logic
    return complex_data_generation(duration, complexity)
```

## Data Validation

### Test Data Validation Framework

```python
from typing import Protocol, Any, List
from dataclasses import dataclass

class DataValidator(Protocol):
    """Protocol for test data validators."""
    
    def validate(self, data: Any) -> 'ValidationResult':
        """Validate test data and return result."""
        ...

@dataclass
class ValidationResult:
    """Result of data validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any] = None

class VideoDataValidator:
    """Validator for video test data."""
    
    def validate(self, video_path: Path) -> ValidationResult:
        """Validate video file for testing."""
        errors = []
        warnings = []
        metadata = {}
        
        # Check file existence
        if not video_path.exists():
            errors.append(f"Video file does not exist: {video_path}")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        # Check file size
        file_size = video_path.stat().st_size
        if file_size == 0:
            errors.append("Video file is empty")
        elif file_size > 100 * 1024 * 1024:  # 100MB
            warnings.append(f"Large video file: {file_size / 1024 / 1024:.1f}MB")
        
        metadata['file_size'] = file_size
        
        # Validate video properties
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                errors.append("Cannot open video file")
                return ValidationResult(valid=False, errors=errors, warnings=warnings)
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            # Validate properties
            if frame_count == 0:
                errors.append("Video has no frames")
            elif frame_count < 30:  # Less than 1 second at 30fps
                warnings.append(f"Short video: {frame_count} frames")
            
            if fps <= 0:
                errors.append("Invalid frame rate")
            elif fps < 15:
                warnings.append(f"Low frame rate: {fps:.1f} fps")
            
            if width < 640 or height < 480:
                warnings.append(f"Low resolution: {width}x{height}")
            
            metadata.update({
                'frame_count': frame_count,
                'fps': fps,
                'resolution': (width, height),
                'duration': frame_count / fps if fps > 0 else 0
            })
            
        except Exception as e:
            errors.append(f"Error validating video properties: {str(e)}")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )

class PoseDataValidator:
    """Validator for pose sequence data."""
    
    def validate(self, pose_data: Dict[str, Any]) -> ValidationResult:
        """Validate pose sequence data."""
        errors = []
        warnings = []
        metadata = {}
        
        # Check required fields
        required_fields = ['landmarks', 'frame_rate', 'total_frames']
        for field in required_fields:
            if field not in pose_data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        # Validate landmarks structure
        landmarks = pose_data['landmarks']
        if not isinstance(landmarks, list):
            errors.append("Landmarks must be a list")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        if len(landmarks) == 0:
            errors.append("No landmark data found")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)
        
        # Validate individual frames
        for i, frame_data in enumerate(landmarks):
            if 'pose_landmarks' not in frame_data:
                errors.append(f"Frame {i}: Missing pose_landmarks")
                continue
            
            pose_landmarks = frame_data['pose_landmarks']
            if len(pose_landmarks) != 33:  # MediaPipe has 33 landmarks
                warnings.append(f"Frame {i}: Expected 33 landmarks, got {len(pose_landmarks)}")
            
            # Validate landmark structure
            for j, landmark in enumerate(pose_landmarks):
                required_keys = ['x', 'y', 'confidence']
                for key in required_keys:
                    if key not in landmark:
                        errors.append(f"Frame {i}, landmark {j}: Missing {key}")
                
                # Validate coordinate ranges
                if 'x' in landmark and not (0 <= landmark['x'] <= 1):
                    warnings.append(f"Frame {i}, landmark {j}: x coordinate out of range: {landmark['x']}")
                
                if 'y' in landmark and not (0 <= landmark['y'] <= 1):
                    warnings.append(f"Frame {i}, landmark {j}: y coordinate out of range: {landmark['y']}")
                
                if 'confidence' in landmark and not (0 <= landmark['confidence'] <= 1):
                    warnings.append(f"Frame {i}, landmark {j}: confidence out of range: {landmark['confidence']}")
        
        metadata.update({
            'frame_count': len(landmarks),
            'expected_frame_count': pose_data['total_frames'],
            'frame_rate': pose_data['frame_rate']
        })
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )

# Usage in tests
def test_with_data_validation(sample_gait_videos):
    """Test with automatic data validation."""
    validator = VideoDataValidator()
    
    for video_name, video_path in sample_gait_videos.items():
        result = validator.validate(video_path)
        
        if not result.valid:
            pytest.fail(f"Invalid test data {video_name}: {result.errors}")
        
        if result.warnings:
            for warning in result.warnings:
                print(f"Warning for {video_name}: {warning}")
```

## Best Practices

### Do's

✅ **Prioritize real data** over synthetic data when possible
✅ **Version control test data** for reproducibility
✅ **Validate test data** before using in tests
✅ **Use appropriate storage strategies** for different data types
✅ **Cache expensive data generation** to improve test performance
✅ **Clean up test data** after test execution
✅ **Document data sources** and generation methods
✅ **Use lazy loading** for large datasets
✅ **Implement data integrity checks** with checksums

### Don'ts

❌ **Don't commit large binary files** to Git without LFS
❌ **Don't use production data** in tests without anonymization
❌ **Don't hardcode file paths** - use relative paths and fixtures
❌ **Don't ignore data validation failures** - they indicate real issues
❌ **Don't generate data in every test** - use fixtures and caching
❌ **Don't leave test data** scattered across the filesystem
❌ **Don't use overly complex synthetic data** that's hard to debug
❌ **Don't skip data cleanup** - it can cause test interference

## Troubleshooting

### Common Issues

1. **Missing Test Data**: Use fallback to synthetic data generation
2. **Large File Performance**: Implement lazy loading and caching
3. **Data Corruption**: Use checksums for integrity verification
4. **Version Conflicts**: Implement data version management
5. **Storage Limitations**: Use Git LFS for large files

### Debugging Data Issues

```python
def debug_test_data_issues():
    """Debug common test data issues."""
    data_manager = RealDataManager()
    
    # Check data availability
    try:
        videos = data_manager.get_sample_videos()
        print(f"Available videos: {list(videos.keys())}")
        
        for name, path in videos.items():
            if path.exists():
                size = path.stat().st_size / 1024 / 1024  # MB
                print(f"  {name}: {size:.1f}MB")
            else:
                print(f"  {name}: MISSING")
    
    except Exception as e:
        print(f"Error loading videos: {e}")
    
    # Check GAVD data
    try:
        gavd_data = data_manager.get_gavd_test_subset()
        metadata = gavd_data.get('metadata', {})
        print(f"GAVD data: {metadata}")
    
    except Exception as e:
        print(f"Error loading GAVD data: {e}")
    
    # Validate data integrity
    version_manager = TestDataVersionManager()
    integrity_result = version_manager.verify_data_integrity()
    print(f"Data integrity: {integrity_result}")
```

Effective test data management is crucial for reliable testing. By following these guidelines and using the provided tools, you can ensure that your tests have access to high-quality, realistic data while maintaining good performance and reliability.