# Quick Start Guide

Get up and running with AlexPose gait analysis in minutes.

## Prerequisites

- AlexPose installed (see [Installation Guide](installation.md))
- API keys configured (OpenAI and/or Google)
- Sample video file or YouTube URL

## 1. Verify Installation

First, ensure everything is working:

```bash
# Activate environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Validate configuration
python scripts/validate_config.py
```

You should see:
```
✅ Configuration validation passed
```

## 2. Prepare Sample Data

### Option A: Use Sample Video File

Place a video file in the `data/videos/` directory:

```bash
# Create directory if it doesn't exist
mkdir -p data/videos

# Copy your video file
cp /path/to/your/video.mp4 data/videos/sample_gait.mp4
```

### Option B: Use YouTube URL

You can analyze videos directly from YouTube:

```
https://www.youtube.com/watch?v=example-gait-video
```

## 3. Basic Analysis

### Command Line Analysis

```bash
# Analyze a local video file
python -m ambient.cli.analyze \
  --input data/videos/sample_gait.mp4 \
  --output data/analysis/results.json \
  --estimator mediapipe \
  --classify

# Analyze a YouTube video
python -m ambient.cli.analyze \
  --input "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output data/analysis/youtube_results.json \
  --estimator mediapipe \
  --classify
```

### Python API

```python
from ambient.core.gait_analyzer import EnhancedGaitAnalyzer
from ambient.classification.llm_classifier import LLMClassifier

# Initialize components
analyzer = EnhancedGaitAnalyzer()
classifier = LLMClassifier()

# Analyze video
video_path = "data/videos/sample_gait.mp4"
results = analyzer.analyze_video(video_path)

# Classify gait pattern
classification = classifier.classify_gait(results['features'])

print(f"Normal gait: {classification['is_normal']}")
print(f"Confidence: {classification['normal_abnormal_confidence']:.2f}")

if not classification['is_normal']:
    conditions = classification['identified_conditions']
    print(f"Identified conditions: {[c['condition'] for c in conditions]}")
```

## 4. Web Interface (Optional)

If you have the frontend installed:

```bash
# Start the API server
python -m ambient.server.main &

# Start the frontend (in another terminal)
cd frontend
npm run dev
```

Then open http://localhost:3000 in your browser.

## 5. Understanding Results

### Analysis Output Structure

```json
{
  "video_info": {
    "duration": 10.5,
    "fps": 30.0,
    "total_frames": 315
  },
  "pose_estimation": {
    "estimator": "mediapipe",
    "keypoints_detected": 315,
    "confidence_mean": 0.87
  },
  "gait_analysis": {
    "gait_cycles": [
      {
        "cycle_id": 0,
        "start_frame": 45,
        "end_frame": 78,
        "duration_seconds": 1.1,
        "foot": "left"
      }
    ],
    "features": {
      "cadence_steps_per_minute": 96.5,
      "step_length_mean": 0.65,
      "stance_swing_ratio": 1.8,
      "left_right_symmetry": 0.92
    }
  },
  "classification": {
    "is_normal": false,
    "normal_abnormal_confidence": 0.85,
    "identified_conditions": [
      {
        "condition": "Mild Limp",
        "confidence": 0.78,
        "explanation": "Asymmetric gait pattern with reduced stance phase on left side"
      }
    ]
  }
}
```

### Key Metrics

| Metric | Normal Range | Description |
|--------|--------------|-------------|
| Cadence | 100-130 steps/min | Walking speed |
| Step Length | 0.6-0.8m | Distance per step |
| Stance/Swing Ratio | 1.5-2.0 | Time on ground vs in air |
| Symmetry Index | 0.9-1.0 | Left-right balance |

## 6. Common Use Cases

### Batch Processing

Process multiple videos:

```bash
# Process all videos in a directory
for video in data/videos/*.mp4; do
  python -m ambient.cli.analyze \
    --input "$video" \
    --output "data/analysis/$(basename "$video" .mp4)_results.json" \
    --estimator mediapipe \
    --classify
done
```

### Custom Configuration

Create a custom analysis configuration:

```python
from ambient.core.config import ConfigurationManager

# Load configuration
config_manager = ConfigurationManager()

# Modify settings for your use case
config_manager.config.gait_analysis.gait_cycle_detection_method = "combined"
config_manager.config.classification.llm.model = "gpt-5-mini"  # Cost-effective

# Use modified configuration
analyzer = EnhancedGaitAnalyzer(config=config_manager.config)
```

### Export Results

Export analysis results in different formats:

```python
import json
import pandas as pd

# Load results
with open('data/analysis/results.json', 'r') as f:
    results = json.load(f)

# Export to CSV
features_df = pd.DataFrame([results['gait_analysis']['features']])
features_df.to_csv('data/exports/gait_features.csv', index=False)

# Export classification summary
classification_summary = {
    'video_file': 'sample_gait.mp4',
    'is_normal': results['classification']['is_normal'],
    'confidence': results['classification']['normal_abnormal_confidence'],
    'conditions': ', '.join([c['condition'] for c in results['classification']['identified_conditions']])
}

summary_df = pd.DataFrame([classification_summary])
summary_df.to_csv('data/exports/classification_summary.csv', index=False)
```

## 7. Troubleshooting

### Common Issues

#### Video Processing Errors

```bash
# Check video format
ffprobe data/videos/sample_gait.mp4

# Convert if needed
ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4
```

#### Pose Estimation Failures

```python
# Test pose estimator
from ambient.pose.mediapipe_estimator import MediaPipeEstimator

estimator = MediaPipeEstimator()
# Check if initialization succeeds
print("MediaPipe estimator ready")
```

#### Classification Errors

```bash
# Check API keys
env | grep API_KEY

# Test API connection
python -c "
from ambient.classification.llm_classifier import LLMClassifier
classifier = LLMClassifier()
print('LLM classifier ready')
"
```

#### Low Confidence Results

If you're getting low confidence scores:

1. **Check video quality**: Ensure clear view of full body
2. **Verify lighting**: Good lighting improves pose detection
3. **Check duration**: Longer videos (10+ seconds) give better results
4. **Review angle**: Side view or 45-degree angle works best

### Performance Optimization

#### For Faster Processing

```python
# Use lighter pose estimator
config.pose_estimation.default_estimator = "mediapipe"

# Reduce video resolution
config.video_processing.max_resolution = "720p"

# Use cost-effective LLM model
config.classification.llm.model = "gpt-5-nano"
```

#### For Better Accuracy

```python
# Use more sophisticated pose estimator
config.pose_estimation.default_estimator = "alphapose"

# Use flagship LLM model
config.classification.llm.model = "gpt-5.2"

# Enable multimodal analysis
config.classification.llm.multimodal_enabled = True
```

## 8. Next Steps

### Learn More

- **[Configuration Guide](configuration.md)**: Customize system behavior
- **[Analysis Documentation](../analysis/)**: Understand the analysis pipeline
- **[API Reference](../api/)**: Integrate with your applications
- **[Development Guide](../development/)**: Contribute to the project

### Advanced Features

- **Batch Processing**: Process multiple videos automatically
- **Custom Models**: Train your own classification models
- **API Integration**: Build applications using the REST API
- **Real-time Analysis**: Process live video streams

### Community

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share use cases
- **Documentation**: Contribute to documentation improvements

## Example Workflows

### Clinical Assessment

```python
# Clinical workflow example
def clinical_assessment(patient_video):
    # Analyze gait
    results = analyzer.analyze_video(patient_video)
    
    # Generate clinical report
    report = {
        'patient_id': 'P001',
        'assessment_date': datetime.now().isoformat(),
        'gait_metrics': results['gait_analysis']['features'],
        'classification': results['classification'],
        'recommendations': generate_recommendations(results)
    }
    
    return report
```

### Research Study

```python
# Research workflow example
def research_analysis(video_directory):
    results = []
    
    for video_file in Path(video_directory).glob('*.mp4'):
        # Process each video
        analysis = analyzer.analyze_video(str(video_file))
        
        # Extract key metrics for research
        metrics = {
            'subject_id': video_file.stem,
            'cadence': analysis['gait_analysis']['features']['cadence_steps_per_minute'],
            'symmetry': analysis['gait_analysis']['features']['left_right_symmetry'],
            'classification': analysis['classification']['is_normal']
        }
        
        results.append(metrics)
    
    # Save research dataset
    df = pd.DataFrame(results)
    df.to_csv('research_dataset.csv', index=False)
    
    return df
```

You're now ready to start analyzing gait patterns with AlexPose! 🚀