"""
Analyze command for processing individual gait videos.
"""

import click
from pathlib import Path
import asyncio
from typing import Optional

from ambient.cli.utils.progress import ProgressTracker
from ambient.cli.utils.output import OutputFormatter
from ambient.video.processor import VideoProcessor
from ambient.pose.factory import PoseEstimatorFactory
from ambient.analysis.gait_analyzer import GaitAnalyzer
from ambient.classification.llm_classifier import LLMClassifier


@click.command()
@click.argument('video', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'csv', 'xml', 'text']), default='json', help='Output format')
@click.option('--pose-estimator', '-p', type=click.Choice(['mediapipe', 'openpose', 'ultralytics', 'alphapose']), default='mediapipe', help='Pose estimation framework')
@click.option('--frame-rate', type=float, default=30.0, help='Frame extraction rate (fps)')
@click.option('--use-llm/--no-llm', default=True, help='Use LLM for classification')
@click.option('--llm-model', type=str, default='gpt-4o-mini', help='LLM model to use')
@click.option('--save-frames/--no-save-frames', default=False, help='Save extracted frames')
@click.option('--save-poses/--no-save-poses', default=False, help='Save pose data')
@click.pass_context
def analyze(ctx, video, output, format, pose_estimator, frame_rate, use_llm, llm_model, save_frames, save_poses):
    """
    Analyze a gait video to identify patterns and classify conditions.
    
    VIDEO can be a local file path or a YouTube URL.
    
    Examples:
    
        # Analyze local video with default settings
        alexpose analyze video.mp4
        
        # Analyze YouTube video with custom output
        alexpose analyze https://youtube.com/watch?v=VIDEO_ID -o results.json
        
        # Use specific pose estimator and save intermediate data
        alexpose analyze video.mp4 -p openpose --save-frames --save-poses
        
        # Output in CSV format
        alexpose analyze video.mp4 -f csv -o results.csv
    """
    config_manager = ctx.obj['config']
    logger = ctx.obj['logger']
    verbose = ctx.obj['verbose']
    
    logger.info(f"Starting analysis of: {video}")
    
    # Initialize progress tracker
    progress = ProgressTracker(verbose=verbose)
    
    try:
        # Step 1: Video processing
        progress.start_stage("Video Processing", "Extracting frames from video")
        
        video_processor = VideoProcessor(config_manager)
        video_path = Path(video)
        
        # Check if it's a YouTube URL
        if video.startswith('http'):
            from ambient.video.youtube_handler import YouTubeHandler
            youtube_handler = YouTubeHandler(str(config_manager.config.storage.youtube_directory))
            
            progress.update("Downloading YouTube video")
            result = youtube_handler.download_video(video)
            
            if not result['success']:
                raise click.ClickException(f"Failed to download YouTube video: {result.get('error')}")
            
            video_path = Path(result['video_path'])
            progress.update(f"Downloaded to: {video_path}")
        
        # Extract frames
        progress.update(f"Extracting frames at {frame_rate} fps")
        frame_sequence = video_processor.extract_frames(
            video_path=video_path,
            frame_rate=frame_rate
        )
        
        progress.complete_stage(f"Extracted {len(frame_sequence.frames)} frames")
        
        # Save frames if requested
        if save_frames:
            frames_dir = Path(output).parent / "frames" if output else Path("frames")
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            progress.update(f"Saving frames to {frames_dir}")
            for i, frame in enumerate(frame_sequence.frames):
                frame.save(frames_dir / f"frame_{i:04d}.jpg")
        
        # Step 2: Pose estimation
        progress.start_stage("Pose Estimation", f"Detecting poses using {pose_estimator}")
        
        pose_factory = PoseEstimatorFactory(config_manager)
        estimator = pose_factory.create_estimator(pose_estimator)
        
        for i, frame in enumerate(frame_sequence.frames):
            progress.update(f"Processing frame {i+1}/{len(frame_sequence.frames)}")
            keypoints = estimator.estimate_pose(frame)
            frame.pose_landmarks = keypoints
        
        progress.complete_stage(f"Detected poses in {len(frame_sequence.frames)} frames")
        
        # Save poses if requested
        if save_poses:
            import json
            poses_file = Path(output).parent / "poses.json" if output else Path("poses.json")
            
            poses_data = {
                "video": str(video_path),
                "frame_count": len(frame_sequence.frames),
                "poses": [
                    {
                        "frame": i,
                        "keypoints": [
                            {"x": kp.x, "y": kp.y, "confidence": kp.confidence, "name": kp.name}
                            for kp in frame.pose_landmarks
                        ] if frame.pose_landmarks else []
                    }
                    for i, frame in enumerate(frame_sequence.frames)
                ]
            }
            
            with open(poses_file, 'w') as f:
                json.dump(poses_data, f, indent=2)
            
            progress.update(f"Saved pose data to {poses_file}")
        
        # Step 3: Gait analysis
        progress.start_stage("Gait Analysis", "Analyzing gait patterns")
        
        gait_analyzer = GaitAnalyzer(config_manager)
        gait_metrics = gait_analyzer.analyze_sequence(frame_sequence)
        
        progress.complete_stage("Gait analysis completed")
        
        # Step 4: Classification
        progress.start_stage("Classification", "Classifying gait patterns")
        
        if use_llm:
            progress.update(f"Using LLM model: {llm_model}")
            llm_classifier = LLMClassifier(config_manager)
            
            # Run async classification
            classification_result = asyncio.run(
                llm_classifier.classify_gait(
                    gait_metrics=gait_metrics,
                    frame_sequence=frame_sequence,
                    model=llm_model
                )
            )
        else:
            progress.update("Using traditional classification")
            # Implement traditional classification fallback
            from dataclasses import dataclass
            
            @dataclass
            class SimpleClassification:
                is_normal: bool
                confidence: float
                explanation: str
                identified_conditions: list
            
            classification_result = SimpleClassification(
                is_normal=True,
                confidence=0.5,
                explanation="Traditional classification - analysis pending",
                identified_conditions=[]
            )
        
        progress.complete_stage("Classification completed")
        
        # Step 5: Format and output results
        progress.start_stage("Output", "Formatting results")
        
        results = {
            "video": str(video_path),
            "analysis": {
                "frame_count": len(frame_sequence.frames),
                "duration": frame_sequence.duration,
                "frame_rate": frame_rate,
                "pose_estimator": pose_estimator
            },
            "gait_metrics": {
                "stride_length": getattr(gait_metrics, 'stride_length', 0),
                "stride_time": getattr(gait_metrics, 'stride_time', 0),
                "cadence": getattr(gait_metrics, 'cadence', 0),
                "step_width": getattr(gait_metrics, 'step_width', 0),
                "symmetry_index": getattr(gait_metrics, 'symmetry_index', 0)
            },
            "classification": {
                "is_normal": getattr(classification_result, 'is_normal', None),
                "confidence": getattr(classification_result, 'confidence', 0),
                "explanation": getattr(classification_result, 'explanation', ''),
                "conditions": getattr(classification_result, 'identified_conditions', [])
            }
        }
        
        # Format output
        formatter = OutputFormatter()
        formatted_output = formatter.format(results, format)
        
        # Save or print results
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(formatted_output)
            
            progress.complete_stage(f"Results saved to: {output_path}")
            
            if verbose:
                click.echo("\n" + formatted_output)
        else:
            progress.complete_stage("Analysis completed")
            click.echo("\n" + formatted_output)
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        progress.fail(str(e))
        logger.error(f"Analysis failed: {str(e)}")
        raise click.ClickException(str(e))
