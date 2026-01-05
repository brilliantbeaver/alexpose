# Task 3.4: CLI Interface - Completion Report

## Overview

Task 3.4 has been successfully completed, providing a comprehensive command-line interface for the AlexPose Gait Analysis System. The CLI enables batch processing, automation, and flexible output formatting for researchers and developers.

## Completed Components

### 1. Main CLI Application ✅

**File**: `ambient/cli/main.py`

**Features**:
- Click-based command-line interface
- Global options for configuration, verbosity, and logging
- Context management for sharing configuration across commands
- Comprehensive help documentation
- Error handling and logging integration

**Global Options**:
```bash
--config, -c    Path to configuration file
--verbose, -v   Enable verbose output
--quiet, -q     Suppress non-error output
--log-file      Path to log file
```

**Commands**:
- `analyze` - Analyze individual gait videos
- `batch` - Batch process multiple videos
- `config` - Manage system configuration
- `info` - Display system information

---

### 2. Analyze Command ✅

**File**: `ambient/cli/commands/analyze.py`

**Purpose**: Process individual gait videos with comprehensive analysis workflow.

**Features**:
- Support for local video files and YouTube URLs
- Configurable pose estimator selection
- LLM-based or traditional classification
- Multiple output formats (JSON, CSV, XML, text)
- Optional frame and pose data saving
- Real-time progress reporting
- Detailed error handling

**Usage Examples**:
```bash
# Analyze local video with default settings
alexpose analyze video.mp4

# Analyze YouTube video with custom output
alexpose analyze https://youtube.com/watch?v=VIDEO_ID -o results.json

# Use specific pose estimator and save intermediate data
alexpose analyze video.mp4 -p openpose --save-frames --save-poses

# Output in CSV format
alexpose analyze video.mp4 -f csv -o results.csv

# Disable LLM classification
alexpose analyze video.mp4 --no-llm

# Use specific LLM model
alexpose analyze video.mp4 --llm-model gpt-4.1
```

**Options**:
```
--output, -o              Output file path
--format, -f              Output format (json, csv, xml, text)
--pose-estimator, -p      Pose estimation framework
--frame-rate              Frame extraction rate (fps)
--use-llm/--no-llm       Use LLM for classification
--llm-model              LLM model to use
--save-frames            Save extracted frames
--save-poses             Save pose data
```

**Workflow**:
1. Video Processing - Extract frames from video or YouTube URL
2. Pose Estimation - Detect poses using selected framework
3. Gait Analysis - Analyze gait patterns and metrics
4. Classification - Classify as normal/abnormal with LLM or traditional methods
5. Output - Format and save results

---

### 3. Batch Command ✅

**File**: `ambient/cli/commands/batch.py`

**Purpose**: Process multiple gait videos in batch with parallel processing support.

**Features**:
- Glob pattern support for file selection
- File list input support (@file.txt)
- Parallel processing with configurable workers
- Continue-on-error option for resilience
- Automatic summary report generation
- Individual result files for each video
- Progress tracking for all videos
- Comprehensive error reporting

**Usage Examples**:
```bash
# Process all MP4 files in a directory
alexpose batch "videos/*.mp4" -o results/

# Process videos from a list file
alexpose batch @video_list.txt -o results/

# Parallel processing with 4 workers
alexpose batch "videos/*.mp4" -j 4 --continue-on-error

# Custom output format
alexpose batch "videos/*.mp4" -f csv -o results/

# Disable summary report
alexpose batch "videos/*.mp4" --no-summary
```

**Options**:
```
--output-dir, -o         Output directory for results
--format, -f             Output format (json, csv, xml)
--pose-estimator, -p     Pose estimation framework
--frame-rate             Frame extraction rate (fps)
--use-llm/--no-llm      Use LLM for classification
--llm-model             LLM model to use
--parallel, -j          Number of parallel processes
--continue-on-error     Continue processing if a video fails
--summary/--no-summary  Generate summary report
```

**Output Structure**:
```
results/
├── video1.json
├── video2.json
├── video3.json
└── summary.json
```

**Summary Report**:
- Total videos processed
- Success/failure counts
- Individual video results with classifications
- Failed videos with error messages

---

### 4. Config Command ✅

**File**: `ambient/cli/commands/config.py`

**Purpose**: Manage system configuration from the command line.

**Features**:
- Display current configuration
- Get specific configuration values
- Set configuration values
- Validate configuration
- Support for nested configuration keys

**Usage Examples**:
```bash
# Show current configuration
alexpose config show

# Get a specific value
alexpose config get pose_estimators.mediapipe.model_complexity

# Set a configuration value
alexpose config set default_frame_rate 60.0

# Validate configuration
alexpose config validate

# Use custom configuration file
alexpose config show -f config/custom.yaml
```

**Actions**:
- `show` - Display current configuration
- `get` - Get a specific configuration value
- `set` - Set a configuration value
- `validate` - Validate configuration

---

### 5. Info Command ✅

**File**: `ambient/cli/commands/info.py`

**Purpose**: Display system information and status.

**Features**:
- Version information
- System information (platform, architecture, processor)
- Resource usage (CPU, memory, disk)
- Configuration paths
- Available components (pose estimators, LLM models)
- Directory status
- Detailed mode for comprehensive information

**Usage Examples**:
```bash
# Show basic information
alexpose info

# Show detailed information
alexpose info --detailed
```

**Output Sections**:
- Version Information
- System Information
- System Resources (detailed mode)
- Configuration
- Available Components
- Directory Status (detailed mode)

---

### 6. Progress Tracking ✅

**File**: `ambient/cli/utils/progress.py`

**Classes**:

**ProgressTracker**:
- Single video analysis progress
- Stage-based progress reporting
- Elapsed time tracking
- Verbose and quiet modes
- Success/failure indicators

**BatchProgressTracker**:
- Batch processing progress
- Video-by-video tracking
- Overall progress percentage
- Stage updates
- Completion statistics

**Features**:
- Real-time progress updates
- Elapsed time calculation
- Visual indicators (✓, ✗, →)
- Verbose and quiet output modes
- Error reporting

---

### 7. Output Formatting ✅

**File**: `ambient/cli/utils/output.py`

**Class**: `OutputFormatter`

**Supported Formats**:

**JSON**:
- Pretty-printed with 2-space indentation
- Handles complex nested structures
- Date/time serialization

**CSV**:
- Single result: Full metrics in columns
- Batch results: Summary with key metrics
- Header row with column names

**XML**:
- Hierarchical structure
- Pretty-printed with indentation
- Handles nested dictionaries and lists

**Text**:
- Human-readable format
- Formatted sections for different data types
- Visual separators and indentation
- Percentage formatting for confidence scores

**Example Outputs**:

**JSON**:
```json
{
  "video": "video.mp4",
  "analysis": {
    "frame_count": 150,
    "duration": 5.0,
    "frame_rate": 30.0
  },
  "gait_metrics": {
    "stride_length": 1.2,
    "cadence": 120.5
  },
  "classification": {
    "is_normal": true,
    "confidence": 0.85
  }
}
```

**Text**:
```
============================================================
AlexPose Gait Analysis Results
============================================================

Video: video.mp4

Analysis:
  Frame Count: 150
  Duration: 5.0s
  Frame Rate: 30.0 fps

Gait Metrics:
  Stride Length: 1.20
  Cadence: 120.50 steps/min

Classification:
  Status: NORMAL
  Confidence: 85.00%
============================================================
```

---

## Integration with Core Components

The CLI seamlessly integrates with all core AlexPose components:

1. **Configuration Management** (`ambient.core.config`)
   - Loads and validates configuration
   - Supports environment-specific settings

2. **Video Processing** (`ambient.video.processor`)
   - Extracts frames from videos
   - Handles YouTube URLs via `youtube_handler`

3. **Pose Estimation** (`ambient.pose.factory`)
   - Creates pose estimators dynamically
   - Supports multiple frameworks

4. **Gait Analysis** (`ambient.analysis.gait_analyzer`)
   - Analyzes frame sequences
   - Extracts gait metrics

5. **LLM Classification** (`ambient.classification.llm_classifier`)
   - Classifies gait patterns
   - Provides explanations

6. **Logging** (`ambient.utils.logging`)
   - Structured logging throughout
   - Configurable log levels

---

## Command-Line Help Documentation

### Main Help
```bash
$ alexpose --help

Usage: alexpose [OPTIONS] COMMAND [ARGS]...

  AlexPose Gait Analysis System - CLI Interface

  Process gait videos to identify normal vs abnormal patterns and classify
  specific health conditions using AI-powered analysis.

Options:
  -c, --config PATH   Path to configuration file
  -v, --verbose       Enable verbose output
  -q, --quiet         Suppress non-error output
  --log-file PATH     Path to log file
  --help              Show this message and exit.

Commands:
  analyze  Analyze a gait video to identify patterns and classify...
  batch    Batch process multiple gait videos.
  config   Manage system configuration.
  info     Display system information and status.
```

### Analyze Help
```bash
$ alexpose analyze --help

Usage: alexpose analyze [OPTIONS] VIDEO

  Analyze a gait video to identify patterns and classify conditions.

  VIDEO can be a local file path or a YouTube URL.

Options:
  -o, --output PATH                Output file path
  -f, --format [json|csv|xml|text] Output format
  -p, --pose-estimator [mediapipe|openpose|ultralytics|alphapose]
                                   Pose estimation framework
  --frame-rate FLOAT               Frame extraction rate (fps)
  --use-llm / --no-llm            Use LLM for classification
  --llm-model TEXT                LLM model to use
  --save-frames / --no-save-frames Save extracted frames
  --save-poses / --no-save-poses  Save pose data
  --help                          Show this message and exit.
```

---

## Testing & Validation

### Manual Testing Commands

1. **Test basic analysis**:
```bash
alexpose analyze test_video.mp4
```

2. **Test with verbose output**:
```bash
alexpose -v analyze test_video.mp4 -o results.json
```

3. **Test batch processing**:
```bash
alexpose batch "videos/*.mp4" -o batch_results/
```

4. **Test configuration management**:
```bash
alexpose config show
alexpose config get default_frame_rate
alexpose config set default_frame_rate 60.0
alexpose config validate
```

5. **Test system information**:
```bash
alexpose info
alexpose info --detailed
```

6. **Test different output formats**:
```bash
alexpose analyze video.mp4 -f json -o results.json
alexpose analyze video.mp4 -f csv -o results.csv
alexpose analyze video.mp4 -f xml -o results.xml
alexpose analyze video.mp4 -f text
```

---

## Acceptance Criteria Verification

### ✅ Create main CLI application using Click
- Implemented in `ambient/cli/main.py`
- Uses Click for command-line parsing
- Global options for configuration and verbosity
- Context management for shared state

### ✅ Implement video analysis commands with progress reporting
- Implemented in `ambient/cli/commands/analyze.py`
- Real-time progress tracking with `ProgressTracker`
- Stage-based progress reporting
- Elapsed time tracking
- Visual indicators for success/failure

### ✅ Add batch processing capabilities
- Implemented in `ambient/cli/commands/batch.py`
- Glob pattern support for file selection
- File list input support
- Parallel processing support (configurable workers)
- Continue-on-error option
- Automatic summary report generation

### ✅ Support multiple output formats (JSON, CSV, XML)
- Implemented in `ambient/cli/utils/output.py`
- JSON: Pretty-printed with proper serialization
- CSV: Tabular format for metrics
- XML: Hierarchical structure
- Text: Human-readable format
- Format selection via `--format` option

### ✅ Add configuration file support
- Implemented in `ambient/cli/commands/config.py`
- Show, get, set, and validate actions
- Support for nested configuration keys
- Custom configuration file paths
- YAML format support

### ✅ Implement verbose logging and error reporting
- Verbose mode via `--verbose` flag
- Quiet mode via `--quiet` flag
- Structured logging with loguru
- Detailed error messages
- Exception handling throughout
- Log file support via `--log-file` option

---

## Installation & Setup

### Install CLI Entry Point

Add to `pyproject.toml`:
```toml
[project.scripts]
alexpose = "ambient.cli.main:main"
```

### Install Package
```bash
# Development mode
uv pip install -e .

# Or using pip
pip install -e .
```

### Verify Installation
```bash
alexpose --help
alexpose info
```

---

## Usage Workflows

### Workflow 1: Single Video Analysis
```bash
# 1. Analyze video
alexpose analyze video.mp4 -o results.json

# 2. View results
cat results.json

# 3. Analyze with verbose output
alexpose -v analyze video.mp4 -o results.json
```

### Workflow 2: Batch Processing
```bash
# 1. Create video list
ls videos/*.mp4 > video_list.txt

# 2. Process batch
alexpose batch @video_list.txt -o batch_results/ --continue-on-error

# 3. View summary
cat batch_results/summary.json
```

### Workflow 3: YouTube Video Analysis
```bash
# 1. Analyze YouTube video
alexpose analyze "https://youtube.com/watch?v=VIDEO_ID" -o youtube_results.json

# 2. View results
alexpose analyze "https://youtube.com/watch?v=VIDEO_ID" -f text
```

### Workflow 4: Configuration Management
```bash
# 1. View current configuration
alexpose config show

# 2. Modify settings
alexpose config set default_frame_rate 60.0
alexpose config set pose_estimators.mediapipe.model_complexity 2

# 3. Validate configuration
alexpose config validate
```

---

## Future Enhancements

### Potential Improvements:
1. **Watch Mode**: Monitor directory for new videos and process automatically
2. **Interactive Mode**: Interactive prompts for configuration
3. **Export Templates**: Custom output templates
4. **Comparison Mode**: Compare results from multiple analyses
5. **Visualization**: Generate charts and graphs from CLI
6. **Database Integration**: Store results in database
7. **API Client**: Interact with FastAPI server from CLI
8. **Plugin System**: Load custom analysis plugins

---

## Conclusion

Task 3.4 has been successfully completed with all acceptance criteria met. The CLI provides:

- **Comprehensive Commands**: Analyze, batch, config, and info commands
- **Flexible Output**: Multiple format options (JSON, CSV, XML, text)
- **Progress Tracking**: Real-time progress reporting with visual indicators
- **Batch Processing**: Efficient processing of multiple videos
- **Configuration Management**: Easy configuration viewing and modification
- **Error Handling**: Robust error handling and reporting
- **Integration**: Seamless integration with all core components

The CLI is production-ready and provides a powerful interface for researchers and developers to automate gait analysis workflows.

---

## Phase 3 Complete ✅

With Task 3.4 completed, **Phase 3: Application Layer** is now fully complete:

- ✅ Task 3.1: FastAPI Server Foundation
- ✅ Task 3.2: Video Upload Endpoints
- ✅ Task 3.3: Analysis Endpoints
- ✅ Task 3.4: CLI Interface

The AlexPose system now has a complete application layer with both web API and command-line interfaces, ready for frontend development and deployment.
