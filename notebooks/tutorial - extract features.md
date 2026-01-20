# Tutorial: Extract Gait Features

This notebook demonstrates the core feature extraction pipeline for gait analysis using the AlexPose platform.

## What It Does

The notebook walks through extracting and analyzing gait features from the GAVD (Gait Abnormality Video Dataset):

1. **Load GAVD Data** - Imports clinical annotations and organizes video sequences by ID
2. **Visualize Frames** - Displays individual frames from gait sequences with bounding boxes
3. **Pose Estimation** - Extracts 33 body keypoints per frame using MediaPipe's BlazePose model
4. **Joint Angle Calculation** - Computes joint angles (hip, knee, ankle, etc.) from keypoints across all frames
5. **Statistical Analysis** - Calculates mean joint angles and other statistics for each joint

## Key Components Used

- `GAVDDataLoader` - Loads and organizes GAVD dataset
- `SequenceKeypointExtractor` - Extracts pose keypoints from video frames
- `get_joint_angles()` - Computes biomechanical joint angles
- Visualization utilities for frames and keypoints

## Output

The notebook produces joint angle statistics that serve as input features for downstream gait classification and abnormality detection.
