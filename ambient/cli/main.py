"""
Main CLI application for AlexPose Gait Analysis System.

Provides command-line interface for video analysis, batch processing,
and system management using Click.
"""

import click
from pathlib import Path
from loguru import logger
import sys

from ambient.core.config import ConfigurationManager
from ambient.utils.logging import setup_logging, create_component_logger
from .commands.analyze import analyze
from .commands.batch import batch
from .commands.config import config_cmd
from .commands.info import info


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-error output')
@click.option('--log-file', type=click.Path(), help='Path to log file')
@click.pass_context
def cli(ctx, config, verbose, quiet, log_file):
    """
    AlexPose Gait Analysis System - CLI Interface
    
    Process gait videos to identify normal vs abnormal patterns and classify
    specific health conditions using AI-powered analysis.
    
    Examples:
    
        # Analyze a single video
        alexpose analyze video.mp4
        
        # Analyze a YouTube video
        alexpose analyze https://youtube.com/watch?v=VIDEO_ID
        
        # Batch process multiple videos
        alexpose batch videos/*.mp4 --output results/
        
        # Show system information
        alexpose info
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Set up logging
    log_level = "DEBUG" if verbose else "WARNING" if quiet else "INFO"
    log_dir = Path(log_file).parent if log_file else Path("logs")
    
    setup_logging(
        log_level=log_level,
        log_dir=str(log_dir),
        format_type="structured"
    )
    
    cli_logger = create_component_logger("cli")
    
    # Load configuration
    try:
        if config:
            config_manager = ConfigurationManager(config_dir=Path(config).parent)
        else:
            config_manager = ConfigurationManager()
        
        ctx.obj['config'] = config_manager
        ctx.obj['verbose'] = verbose
        ctx.obj['quiet'] = quiet
        ctx.obj['logger'] = cli_logger
        
        if verbose:
            cli_logger.info("Configuration loaded successfully")
            
    except Exception as e:
        cli_logger.error(f"Failed to load configuration: {str(e)}")
        sys.exit(1)


# Register commands
cli.add_command(analyze)
cli.add_command(batch)
cli.add_command(config_cmd)
cli.add_command(info)


def main():
    """Entry point for the CLI application."""
    cli(obj={})


if __name__ == '__main__':
    main()
