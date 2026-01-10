# CSV Parser for Pose Estimation Data

> **Theodore Mui**<br/>
> _Sunday, July 27, 2025 12:30:46 AM_

This module provides a robust Python CSV parser that can handle CSV files containing dictionary objects in certain fields, specifically designed for pose estimation data used in the Chain of Causality fall risk detection system.

## Features

- **Dictionary Field Parsing**: Automatically parses dictionary strings into Python dictionaries
- **Multiple Format Support**: Handles both JSON and Python literal string formats
- **Auto-Detection**: Can automatically detect fields containing dictionary objects
- **OpenPose Integration**: Specialized parser for OpenPose CSV files
- **Error Handling**: Robust error handling for malformed data
- **Pandas Alternative**: Optional pandas-based implementation for complex CSV files

## Quick Start

### Basic Usage

```python
import sys
from pathlib import Path

# Add the theodore directory to the Python path
project_root = Path(__file__).parent
theodore_path = project_root / "theodore"
sys.path.insert(0, str(theodore_path))

from ambient.utils.csv_parser import parse_csv_with_dicts

# Parse CSV with dictionary fields
rows = parse_csv_with_dicts('pose_data.csv', dict_fields=['bbox', 'vid_info', 'pose_data'])

# Access parsed data
for row in rows:
    bbox = row['bbox']  # Now a dictionary, not a string
    print(f"Bounding box: x={bbox['x']}, y={bbox['y']}")
```

### Auto-Detection

```python
# Let the parser automatically detect dictionary fields
rows = parse_csv_with_dicts('pose_data.csv')  # No dict_fields specified
```

### OpenPose CSV Parsing

```python
from ambient.utils.csv_parser import parse_openpose_csv

# Parse OpenPose format CSV
pose_data = parse_openpose_csv('openpose_output.csv')

# Access keypoint data
for frame in pose_data:
    keypoints = frame['pose_keypoints_2d']  # List of keypoint dictionaries
    for kp in keypoints:
        print(f"Keypoint: x={kp['x']}, y={kp['y']}, confidence={kp['confidence']}")
```

## CSV Format Examples

### General CSV with Dictionary Fields

```csv
id,bbox,vid_info,pose_data,confidence
1,"{'x': 100, 'y': 200, 'width': 50, 'height': 100}","{'fps': 30, 'duration': 60}","{'keypoints': [1, 2, 3]}",0.95
2,"{'x': 150, 'y': 250, 'width': 60, 'height': 120}","{'fps': 25, 'duration': 45}","{'keypoints': [4, 5, 6]}",0.87
```

### OpenPose CSV Format

```csv
frame,person_id,pose_keypoints_2d,face_keypoints_2d,hand_left_keypoints_2d,hand_right_keypoints_2d
1,0,"100.5,200.3,0.9,150.2,250.1,0.8,200.0,300.0,0.7","","",""
2,0,"110.5,210.3,0.85,160.2,260.1,0.75,210.0,310.0,0.65","","",""
```

## Integration with Fall Risk Detection

The `gait_processor.py` file demonstrates how to integrate the CSV parser with the Chain of Causality fall risk detection system:

```python
from tests.gait_processor import GaitDataProcessor

# Initialize processor
processor = GaitDataProcessor()

# Process gait video data
results = processor.process_gait_video('openpose_data.csv')

# Access results
print(f"Tinetti Gait Score: {results['tinetti_gait_score']}/12")
print(f"Fall Risk Level: {results['fall_risk_level']}")
print(f"Fall Risk Score: {results['fall_risk_score']}")
```

## API Reference

### `parse_csv_with_dicts(file_path, dict_fields=None)`

Parse a CSV file containing dictionary objects.

**Parameters:**
- `file_path` (str): Path to the CSV file
- `dict_fields` (List[str], optional): List of field names containing dictionaries

**Returns:**
- `List[Dict[str, Any]]`: List of dictionaries representing CSV rows

### `parse_openpose_csv(file_path)`

Parse OpenPose format CSV files.

**Parameters:**
- `file_path` (str): Path to OpenPose CSV file

**Returns:**
- `List[Dict[str, Any]]`: List of pose data frames with parsed keypoints

### `parse_csv_with_pandas(file_path, dict_fields=None)`

Alternative implementation using pandas for complex CSV files.

**Parameters:**
- `file_path` (str): Path to the CSV file
- `dict_fields` (List[str], optional): List of field names containing dictionaries

**Returns:**
- `List[Dict[str, Any]]`: List of dictionaries representing CSV rows

## Testing

Run the test suite to verify functionality:

```bash
# From the project root
python theodore/tests/test_csv_parser.py

# Or from the theodore directory
cd theodore
python tests/test_csv_parser.py
```

This will test:
- General CSV parsing with dictionary fields
- JSON format CSV parsing
- OpenPose CSV parsing
- Error handling with malformed data

The test files are located in `theodore/data/` and include:
- `test_data.csv` - General CSV with dictionary fields
- `test_json_data.csv` - JSON format CSV
- `test_openpose.csv` - OpenPose format CSV
- `malformed_data.csv` - Malformed data for error testing
- `sample_gait_data.csv` - Sample gait data for pose processing

## Error Handling

The parser handles various error conditions:

- **File not found**: Raises `FileNotFoundError`
- **Malformed dictionary strings**: Returns original string value
- **Invalid CSV format**: Raises descriptive exception
- **Missing fields**: Gracefully handles missing columns

## Performance Considerations

- **Memory usage**: Processes CSV files row by row to minimize memory usage
- **Large files**: Use pandas implementation for very large CSV files
- **Real-time processing**: Suitable for real-time pose estimation data processing

## Dependencies

- `csv` (standard library)
- `json` (standard library)
- `ast` (standard library)
- `pandas` (optional, for pandas implementation)
- `numpy` (optional, for pose data processing)

## Use Cases

1. **OpenPose Data Processing**: Parse OpenPose output CSV files for gait analysis
2. **MediaPipe Integration**: Handle MediaPipe pose estimation results
3. **YOLO-Pose Data**: Process YOLO pose estimation outputs
4. **AlphaPose Results**: Parse AlphaPose CSV outputs
5. **General Pose Data**: Handle any pose estimation CSV with dictionary fields

## Contributing

When adding new pose estimation model support:

1. Add specialized parsing function if needed
2. Update test suite with new format examples
3. Document the new format in this README
4. Ensure compatibility with existing fall risk detection pipeline

## Installation

The CSV parser is part of the ambient package and can be imported as:

```python
from ambient.utils.csv_parser import parse_csv_with_dicts
```

Make sure to add the `theodore` directory to your Python path as shown in the examples above.
