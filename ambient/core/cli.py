"""
Command Line Interface handler for the Ambient gait analysis system.

This module provides functionality for parsing command line arguments and
managing CLI interactions.

@Theodore Mui
Monday, July 28, 2025 12:30:00 AM
"""

import argparse
from pathlib import Path
from typing import List, Optional


# Define ValidationError locally to avoid circular imports
class ValidationError(Exception):
    """Exception raised for validation errors."""

    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class CLIHandler:
    """
    Handles command line interface operations for the Ambient system.

    This class manages argument parsing, validation, and provides a clean
    interface for CLI operations.
    """

    def __init__(self):
        """Initialize the CLI handler."""
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """
        Create the argument parser with all available options.

        Returns:
            Configured argument parser
        """
        parser = argparse.ArgumentParser(
            description="Run Gemini gait analysis on one or many bottom-view videos using cached file references.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Analyze all videos (batch processing)
  python -m ambient.analysis.gait_main --config config/OAW/OAW-pose.yaml --all

  # Analyze specific record
  python -m ambient.analysis.gait_main --config config/OAW/OAW-pose.yaml --record OAW05

  # Analyze specific video file
  python -m ambient.analysis.gait_main --config config/OAW/OAW-pose.yaml --video path/to/video.mp4

  # List available videos
  python -m ambient.analysis.gait_main --config config/OAW/OAW-pose.yaml --list-videos

  # Validate configuration
  python -m ambient.analysis.gait_main --config config/OAW/OAW-pose.yaml --validate

  # Show help
  python -m ambient.analysis.gait_main --help
            """,
        )

        parser.add_argument(
            "--config",
            "-c",
            help="Path to a YAML configuration file.",
            default=None,
            metavar="CONFIG_PATH",
        )

        parser.add_argument(
            "--record",
            "-r",
            help="Record ID like OAW09. If given, the script analyses that participant.",
            default=None,
            metavar="RECORD_ID",
        )

        parser.add_argument(
            "--video",
            "-v",
            help="Full path to a single bottom-view video to analyse.",
            default=None,
            metavar="VIDEO_PATH",
        )

        parser.add_argument(
            "--all",
            action="store_true",
            help="Process all videos in the input directory (batch processing).",
        )

        parser.add_argument(
            "--list-videos",
            action="store_true",
            help="List all available videos and exit.",
        )

        parser.add_argument(
            "--validate", action="store_true", help="Validate configuration and exit."
        )

        parser.add_argument(
            "--verbose", "-V", action="store_true", help="Enable verbose output."
        )

        return parser

    def parse_arguments(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse command line arguments.

        Args:
            args: List of arguments to parse, defaults to sys.argv[1:]

        Returns:
            Parsed arguments namespace

        Raises:
            SystemExit: If argument parsing fails or help is requested
        """
        return self.parser.parse_args(args)

    def validate_arguments(self, args: argparse.Namespace) -> bool:
        """
        Validate parsed arguments.

        Args:
            args: Parsed arguments namespace

        Returns:
            True if arguments are valid

        Raises:
            ValidationError: If arguments are invalid
        """
        # Check for mutually exclusive arguments
        if args.record and args.video:
            raise ValidationError(
                "Cannot specify both --record and --video",
                "Use either --record to analyze a specific participant or --video to analyze a specific file",
            )

        if args.record and args.all:
            raise ValidationError(
                "Cannot specify both --record and --all",
                "Use either --record to analyze a specific participant or --all to analyze all videos",
            )

        if args.video and args.all:
            raise ValidationError(
                "Cannot specify both --video and --all",
                "Use either --video to analyze a specific file or --all to analyze all videos",
            )

        # Validate video path if provided
        if args.video:
            video_path = Path(args.video)
            if not video_path.exists():
                raise ValidationError(
                    f"Video file does not exist: {args.video}",
                    "Please check the file path and try again",
                )

            if not video_path.suffix.lower() == ".mp4":
                raise ValidationError(
                    f"Video file must be an MP4 file: {args.video}",
                    "Please provide a valid MP4 file",
                )

        # Validate record ID if provided
        if args.record:
            if not args.record.startswith("OAW"):
                raise ValidationError(
                    f"Record ID must start with 'OAW': {args.record}",
                    "Please provide a valid record ID like OAW05",
                )

        return True

    def get_video_paths(self, args: argparse.Namespace, videos_dir: Path) -> List[Path]:
        """
        Get the list of video paths to process based on arguments.

        Args:
            args: Parsed arguments namespace
            videos_dir: Directory containing video files

        Returns:
            List of video paths to process
        """
        if args.video:
            # Single video file
            return [Path(args.video)]

        elif args.record:
            # Specific record
            pattern = videos_dir / f"{args.record}-bottom.mp4"
            if pattern.exists():
                return [pattern]
            else:
                return []

        elif args.all:
            # All bottom-view videos (explicit batch processing)
            return list(videos_dir.glob("*-bottom.mp4"))

        else:
            # Default behavior: all bottom-view videos (for backward compatibility)
            return list(videos_dir.glob("*-bottom.mp4"))

    def print_help(self) -> None:
        """Print help information."""
        self.parser.print_help()

    def print_version(self) -> None:
        """Print version information."""
        print("Theodore Gait Analysis System v1.0.0")
        print("Author: Theodore Mui")
        print("Date: 2025-01-27")

    def print_video_list(self, videos_dir: Path) -> None:
        """
        Print a list of all available videos.

        Args:
            videos_dir: Directory containing video files
        """
        if not videos_dir.exists():
            print(f"Videos directory does not exist: {videos_dir}")
            return

        videos = list(videos_dir.glob("*-bottom.mp4"))

        if not videos:
            print(f"No bottom-view videos found in: {videos_dir}")
            return

        print(f"Found {len(videos)} bottom-view videos:")
        print("-" * 50)

        for video in sorted(videos):
            record_id = video.stem.replace("-bottom", "")
            print(f"  {record_id}: {video.name}")

        print("-" * 50)

    def print_configuration_summary(self, config_manager) -> None:
        """
        Print a summary of the current configuration.

        Args:
            config_manager: Configuration manager instance
        """
        try:
            print("Configuration Summary:")
            print("-" * 30)
            print(f"Videos Directory: {config_manager.get_videos_directory()}")
            print(f"OpenPose Directory: {config_manager.get_openpose_directory()}")
            print(f"Gemini Configured: {config_manager.is_gemini_configured()}")

            # Count videos
            videos_dir = config_manager.get_videos_directory()
            video_count = len(list(videos_dir.glob("*-bottom.mp4")))
            print(f"Available Videos: {video_count}")

        except Exception as e:
            print(f"Error getting configuration summary: {str(e)}")
