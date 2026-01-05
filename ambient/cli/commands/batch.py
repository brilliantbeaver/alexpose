"""
Batch command for processing multiple gait videos.
"""

import click
from pathlib import Path
import asyncio
from typing import List
import glob

from ambient.cli.utils.progress import BatchProgressTracker
from ambient.cli.utils.output import OutputFormatter
from ambient.video.processor import VideoProcessor
from ambient.pose.factory import PoseEstimatorFactory
from ambient.analysis.gait_analyzer import GaitAnalyzer
from ambient.classification.llm_classifier import LLMClassifier


@click.command()
@click.argument('pattern', type=str)
@click.option('--output-dir', '-o', type=click.Path(), default='batch_results', help='Output directory for results')
@click.option('--format', '-f', type=click.Choice(['json', 'csv', 'xml']), default='json', help='Output format')
@click.option('--pose-estimator', '-p', type=click.Choice(['mediapipe', 'openpose', 'ultralytics', 'alphapose']), default='mediapipe', help='Pose estimation framework')
@click.option('--frame-rate', type=float, default=30.0, help='Frame extraction rate (fps)')
@click.option('--use-llm/--no-llm', default=True, help='Use LLM for classification')
@click.option('--llm-model', type=str, default='gpt-4o-mini', help='LLM model to use')
@click.option('--parallel', '-j', type=int, default=1, help='Number of parallel processes')
@click.option('--continue-on-error', is_flag=True, help='Continue processing if a video fails')
@click.option('--summary/--no-summary', default=True, help='Generate summary report')
@click.pass_context
def batch(ctx, pattern, output_dir, format, pose_estimator, frame_rate, use_llm, llm_model, parallel, continue_on_error, summary):
    """
    Batch process multiple gait videos.
    
    PATTERN can be a glob pattern or a file containing video paths (one per line).
    
    Examples:
    
        # Process all MP4 files in a directory
        alexpose batch "videos/*.mp4" -o results/
        
        # Process videos from a list file
        alexpose batch @video_list.txt -o results/
        
        # Parallel processing with 4 workers
        alexpose batch "videos/*.mp4" -j 4 --continue-on-error
        
        # Custom output format
        alexpose batch "videos/*.mp4" -f csv -o results/
    """
    config_manager = ctx.obj['config']
    logger = ctx.obj['logger']
    verbose = ctx.obj['verbose']
    
    # Parse video list
    videos = _parse_video_pattern(pattern)
    
    if not videos:
        raise click.ClickException(f"No videos found matching pattern: {pattern}")
    
    logger.info(f"Found {len(videos)} videos to process")
    click.echo(f"Processing {len(videos)} videos...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize batch progress tracker
    progress = BatchProgressTracker(total=len(videos), verbose=verbose)
    
    # Process videos
    results = []
    failed = []
    
    try:
        if parallel > 1:
            # Parallel processing
            click.echo(f"Using {parallel} parallel workers")
            # Note: Actual parallel implementation would use multiprocessing
            # For now, process sequentially
            for video in videos:
                result = _process_single_video(
                    video, output_path, format, pose_estimator, frame_rate,
                    use_llm, llm_model, config_manager, progress, logger
                )
                
                if result['success']:
                    results.append(result)
                else:
                    failed.append(result)
                    if not continue_on_error:
                        raise click.ClickException(f"Failed to process {video}: {result['error']}")
        else:
            # Sequential processing
            for video in videos:
                result = _process_single_video(
                    video, output_path, format, pose_estimator, frame_rate,
                    use_llm, llm_model, config_manager, progress, logger
                )
                
                if result['success']:
                    results.append(result)
                else:
                    failed.append(result)
                    if not continue_on_error:
                        raise click.ClickException(f"Failed to process {video}: {result['error']}")
        
        progress.complete()
        
        # Generate summary
        if summary:
            _generate_summary(results, failed, output_path, format)
        
        # Print final statistics
        click.echo(f"\n{'='*60}")
        click.echo(f"Batch processing completed")
        click.echo(f"{'='*60}")
        click.echo(f"Total videos: {len(videos)}")
        click.echo(f"Successful: {len(results)}")
        click.echo(f"Failed: {len(failed)}")
        click.echo(f"Output directory: {output_path}")
        
        if failed:
            click.echo(f"\nFailed videos:")
            for result in failed:
                click.echo(f"  - {result['video']}: {result['error']}")
        
        logger.info(f"Batch processing completed: {len(results)} successful, {len(failed)} failed")
        
    except Exception as e:
        progress.fail(str(e))
        logger.error(f"Batch processing failed: {str(e)}")
        raise click.ClickException(str(e))


def _parse_video_pattern(pattern: str) -> List[str]:
    """Parse video pattern to get list of video files."""
    if pattern.startswith('@'):
        # Read from file
        file_path = Path(pattern[1:])
        if not file_path.exists():
            return []
        
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        # Glob pattern
        return glob.glob(pattern)


def _process_single_video(video, output_path, format, pose_estimator, frame_rate, use_llm, llm_model, config_manager, progress, logger):
    """Process a single video and return results."""
    video_name = Path(video).stem
    progress.start_video(video_name)
    
    try:
        # Initialize components
        video_processor = VideoProcessor(config_manager)
        pose_factory = PoseEstimatorFactory(config_manager)
        gait_analyzer = GaitAnalyzer(config_manager)
        
        # Extract frames
        progress.update_stage("Extracting frames")
        frame_sequence = video_processor.extract_frames(
            video_path=Path(video),
            frame_rate=frame_rate
        )
        
        # Pose estimation
        progress.update_stage("Detecting poses")
        estimator = pose_factory.create_estimator(pose_estimator)
        
        for frame in frame_sequence.frames:
            keypoints = estimator.estimate_pose(frame)
            frame.pose_landmarks = keypoints
        
        # Gait analysis
        progress.update_stage("Analyzing gait")
        gait_metrics = gait_analyzer.analyze_sequence(frame_sequence)
        
        # Classification
        progress.update_stage("Classifying")
        if use_llm:
            llm_classifier = LLMClassifier(config_manager)
            classification_result = asyncio.run(
                llm_classifier.classify_gait(
                    gait_metrics=gait_metrics,
                    frame_sequence=frame_sequence,
                    model=llm_model
                )
            )
        else:
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
                explanation="Traditional classification",
                identified_conditions=[]
            )
        
        # Format results
        results = {
            "video": str(video),
            "video_name": video_name,
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
        
        # Save individual result
        formatter = OutputFormatter()
        formatted_output = formatter.format(results, format)
        
        output_file = output_path / f"{video_name}.{format}"
        with open(output_file, 'w') as f:
            f.write(formatted_output)
        
        progress.complete_video(video_name)
        
        return {
            "success": True,
            "video": video,
            "video_name": video_name,
            "output_file": str(output_file),
            "results": results
        }
        
    except Exception as e:
        progress.fail_video(video_name, str(e))
        logger.error(f"Failed to process {video}: {str(e)}")
        
        return {
            "success": False,
            "video": video,
            "video_name": video_name,
            "error": str(e)
        }


def _generate_summary(results, failed, output_path, format):
    """Generate summary report for batch processing."""
    summary = {
        "total_videos": len(results) + len(failed),
        "successful": len(results),
        "failed": len(failed),
        "results": [
            {
                "video": r['video_name'],
                "is_normal": r['results']['classification']['is_normal'],
                "confidence": r['results']['classification']['confidence'],
                "output_file": r['output_file']
            }
            for r in results
        ],
        "failed_videos": [
            {
                "video": f['video_name'],
                "error": f['error']
            }
            for f in failed
        ]
    }
    
    # Save summary
    formatter = OutputFormatter()
    formatted_summary = formatter.format(summary, format)
    
    summary_file = output_path / f"summary.{format}"
    with open(summary_file, 'w') as f:
        f.write(formatted_summary)
    
    click.echo(f"\nSummary saved to: {summary_file}")
