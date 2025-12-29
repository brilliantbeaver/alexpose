"""
Main application orchestrator for the Ambient gait analysis system.

This module provides the main application class that coordinates all components
and serves as the primary entry point for the system.

@Theodore Mui
Monday, July 28, 2025 12:30:00 AM
"""

import sys
from pathlib import Path
from typing import Optional

from ambient.analysis.gait_analyzer import GaitAnalyzer, GeminiAnalyzer
from ambient.core import (
    CLIHandler,
    EnvConfigurationManager,
    OutputManager,
    YAMLConfigurationManager,
)
from ambient.exceptions import AmbientError, ConfigurationError, ValidationError

from .upload_manager import AmbientGeminiFileManager


class GaitAnalysisApplication:
    """
    Main application orchestrator for the Ambient gait analysis system.

    This class coordinates all components and provides the main entry point
    for the gait analysis system.
    """

    def __init__(self, config_manager=None):
        """
        Initialize the application with all required components.

        Args:
            config_manager: Configuration manager instance. If None, creates a default one.
        """
        try:
            # Initialize configuration manager
            if config_manager is None:
                self.config_manager = YAMLConfigurationManager()
            else:
                self.config_manager = config_manager

            # Initialize file manager with cache config from YAML if available
            cache_config = None
            if hasattr(self.config_manager, "get_config_value"):
                cache_config = self.config_manager.get_config_value(
                    "gemini_cache_config"
                )
            self.file_manager = AmbientGeminiFileManager(cache_config)

            # Initialize output manager with output_dir from YAML if available
            output_dir = "outputs"
            if hasattr(self.config_manager, "get_config_value"):
                yaml_output_dir = self.config_manager.get_config_value("output_dir")
                if yaml_output_dir:
                    output_dir = yaml_output_dir
            self.output_manager = OutputManager(base_output_dir=output_dir)

            self.cli_handler = CLIHandler()

            # Initialize analyzer if Gemini is configured
            self.analyzer = None
            self.gait_analyzer = None

            if self.config_manager.is_gemini_configured():
                api_key = self.config_manager.get_gemini_api_key()

                # Get model and temperature from YAML config if available
                vlm_model = "gemini-2.5-pro"
                temperature = 0.0
                if hasattr(self.config_manager, "get_config_value"):
                    yaml_model = self.config_manager.get_config_value("vlm_model")
                    if yaml_model:
                        vlm_model = yaml_model
                    yaml_temperature = self.config_manager.get_config_value(
                        "temperature"
                    )
                    if yaml_temperature is not None:
                        temperature = yaml_temperature

                self.analyzer = GeminiAnalyzer(
                    api_key,
                    self.file_manager,
                    vlm_model,
                    temperature,
                    self.config_manager,
                )
                self.gait_analyzer = GaitAnalyzer(
                    self.config_manager,
                    self.analyzer,
                    self.file_manager,
                    self.output_manager,
                )

        except Exception as e:
            print(f"Failed to initialize application: {str(e)}", flush=True)
            sys.exit(1)

    def run(self, args: Optional[list] = None) -> int:
        """
        Run the main application.

        Args:
            args: Command line arguments, defaults to sys.argv[1:]

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            # Parse command line arguments
            parsed_args = self.cli_handler.parse_arguments(args)

            # Handle special commands
            if hasattr(parsed_args, "list_videos") and parsed_args.list_videos:
                self._handle_list_videos()
                return 0

            if hasattr(parsed_args, "validate") and parsed_args.validate:
                self._handle_validate_configuration()
                return 0

            # Validate arguments
            self.cli_handler.validate_arguments(parsed_args)

            # Validate configuration
            if not self.config_manager.validate_configuration():
                print(
                    "Configuration validation failed. Please check your settings.",
                    flush=True,
                )
                return 1

            # Check if Gemini is configured
            if not self.config_manager.is_gemini_configured():
                print(
                    "WARNING: Gemini is not properly configured. Analysis will be skipped.",
                    flush=True,
                )
                return 1

            # Get video paths to process
            videos_dir = self.config_manager.get_videos_directory()
            video_paths = self.cli_handler.get_video_paths(parsed_args, videos_dir)

            if not video_paths:
                print("No matching videos found.", flush=True)
                return 0

            # Process videos
            success_count = 0
            total_count = len(video_paths)

            print(f"Processing {total_count} video(s)...", flush=True)

            for video_path in video_paths:
                try:
                    # Extract record ID from filename
                    base_name = video_path.name
                    if not base_name.endswith("-bottom.mp4"):
                        print(
                            f"Skipping {video_path.name} (does not follow naming pattern)",
                            flush=True,
                        )
                        continue

                    record_id = base_name.replace("-bottom.mp4", "")
                    if self.gait_analyzer.analyze_single_video(
                        str(video_path), record_id
                    ):
                        success_count += 1

                except Exception as e:
                    print(f"ERROR processing {video_path.name}: {str(e)}", flush=True)

            # Print summary
            print(
                f"\nAnalysis complete: {success_count}/{total_count} videos processed successfully.",
                flush=True,
            )

            return 0 if success_count > 0 else 1

        except ValidationError as e:
            print(f"Validation error: {e.message}", flush=True)
            if e.details:
                print(f"Details: {e.details}", flush=True)
            return 1

        except ConfigurationError as e:
            print(f"Configuration error: {e.message}", flush=True)
            if e.details:
                print(f"Details: {e.details}", flush=True)
            return 1

        except AmbientError as e:
            print(f"Ambient error: {e.message}", flush=True)
            if e.details:
                print(f"Details: {e.details}", flush=True)
            return 1

        except KeyboardInterrupt:
            print("\nOperation cancelled by user.", flush=True)
            return 1

        except Exception as e:
            print(f"Unexpected error: {str(e)}", flush=True)
            return 1

    def _handle_list_videos(self) -> None:
        """Handle the --list-videos command."""
        try:
            videos_dir = self.config_manager.get_videos_directory()
            self.cli_handler.print_video_list(videos_dir)
        except Exception as e:
            print(f"Error listing videos: {str(e)}", flush=True)

    def _handle_validate_configuration(self) -> None:
        """Handle the --validate command."""
        try:
            self.cli_handler.print_configuration_summary(self.config_manager)

            if self.config_manager.validate_configuration():
                print("\n✓ Configuration is valid!", flush=True)
            else:
                print("\n✗ Configuration validation failed!", flush=True)

        except Exception as e:
            print(f"Error validating configuration: {str(e)}", flush=True)

    def analyze_single_video(
        self, video_path: str, record_id: Optional[str] = None
    ) -> bool:
        """
        Analyze a single video file.

        Args:
            video_path: Path to the video file
            record_id: Optional record ID

        Returns:
            True if analysis was successful
        """
        if not self.gait_analyzer:
            print("Gemini is not configured. Cannot perform analysis.", flush=True)
            return False

        return self.gait_analyzer.analyze_single_video(video_path, record_id)

    def analyze_all_videos(self) -> None:
        """Analyze all available videos."""
        if not self.gait_analyzer:
            print("Gemini is not configured. Cannot perform analysis.", flush=True)
            return

        self.gait_analyzer.analyze_all_videos()

    def get_status(self) -> dict:
        """
        Get the current status of the application.

        Returns:
            Dictionary containing status information
        """
        try:
            return {
                "gemini_configured": self.config_manager.is_gemini_configured(),
                "videos_directory": str(self.config_manager.get_videos_directory()),
                "openpose_directory": str(self.config_manager.get_openpose_directory()),
                "output_summary": self.output_manager.get_output_summary(),
            }
        except Exception as e:
            return {"error": str(e)}


def main():
    """Main entry point for the application."""
    app = GaitAnalysisApplication()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
