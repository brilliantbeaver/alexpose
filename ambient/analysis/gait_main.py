##############################################################################
# gait_main.py
#
# Analyse video files using Gemini AI for gait analysis.
#
# This program is used to analyse video files using Gemini AI for gait analysis.
# It uses the AmbientGeminiFileManager to upload the video and CSV files to Gemini
# and then uses the Gemini API to analyse the video.
#
# @Theodore Mui
# Monday, July 28, 2025 12:30:00 AM
##############################################################################

import sys
from pathlib import Path

from ambient.core.cli import CLIHandler
from ambient.core.configuration import EnvConfigurationManager, YAMLConfigurationManager

from .gait_app import GaitAnalysisApplication


def parse_arguments():
    """Parse command line arguments using CLIHandler."""
    cli_handler = CLIHandler()
    return cli_handler.parse_arguments()


def main():
    """
    Main entry point for the analysis video script.

    This function initializes the application and runs it with the provided
    command line arguments.
    """
    args = parse_arguments()

    # Initialize configuration manager based on arguments
    if args.config:
        # Use YAML configuration
        config_manager = YAMLConfigurationManager(Path(args.config))
        print(f"Using YAML configuration: {args.config}", flush=True)
    else:
        # Use environment-based configuration (backward compatibility)
        config_manager = EnvConfigurationManager()
        print("Using environment-based configuration", flush=True)

    # Create and run application with the appropriate configuration
    app = GaitAnalysisApplication(config_manager)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
