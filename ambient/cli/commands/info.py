"""
Info command for displaying system information.
"""

import click
import sys
import platform
from pathlib import Path
import psutil


@click.command()
@click.option('--detailed', '-d', is_flag=True, help='Show detailed system information')
@click.pass_context
def info(ctx, detailed):
    """
    Display system information and status.
    
    Examples:
    
        # Show basic information
        alexpose info
        
        # Show detailed information
        alexpose info --detailed
    """
    config_manager = ctx.obj['config']
    
    click.echo("AlexPose Gait Analysis System")
    click.echo("=" * 60)
    
    # Version information
    click.echo("\nVersion Information:")
    click.echo(f"  AlexPose Version: 0.1.0")
    click.echo(f"  Python Version: {sys.version.split()[0]}")
    
    # System information
    click.echo("\nSystem Information:")
    click.echo(f"  Platform: {platform.system()} {platform.release()}")
    click.echo(f"  Architecture: {platform.machine()}")
    click.echo(f"  Processor: {platform.processor()}")
    
    if detailed:
        # Detailed system metrics
        click.echo("\nSystem Resources:")
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        click.echo(f"  CPU Usage: {cpu_percent}%")
        click.echo(f"  CPU Cores: {cpu_count}")
        
        # Memory
        memory = psutil.virtual_memory()
        click.echo(f"  Memory Total: {memory.total / (1024**3):.2f} GB")
        click.echo(f"  Memory Available: {memory.available / (1024**3):.2f} GB")
        click.echo(f"  Memory Usage: {memory.percent}%")
        
        # Disk
        disk = psutil.disk_usage('/')
        click.echo(f"  Disk Total: {disk.total / (1024**3):.2f} GB")
        click.echo(f"  Disk Free: {disk.free / (1024**3):.2f} GB")
        click.echo(f"  Disk Usage: {disk.percent}%")
    
    # Configuration
    click.echo("\nConfiguration:")
    click.echo(f"  Config Directory: {config_manager.config_dir}")
    click.echo(f"  Data Directory: {config_manager.config.storage.data_directory}")
    click.echo(f"  Logs Directory: {config_manager.config.storage.logs_directory}")
    
    # Available components
    click.echo("\nAvailable Components:")
    
    # Check pose estimators
    try:
        from ambient.pose.factory import PoseEstimatorFactory
        factory = PoseEstimatorFactory(config_manager)
        estimators = factory.list_available()
        click.echo(f"  Pose Estimators: {', '.join(estimators)}")
    except Exception as e:
        click.echo(f"  Pose Estimators: Error loading ({str(e)})")
    
    # Check LLM models
    try:
        llm_config = config_manager.config.classification.llm
        models = llm_config.available_models
        click.echo(f"  LLM Models: {', '.join(models)}")
    except Exception as e:
        click.echo(f"  LLM Models: Not configured")
    
    # Directory status
    if detailed:
        click.echo("\nDirectory Status:")
        
        directories = {
            "Data": config_manager.config.storage.data_directory,
            "Videos": config_manager.config.storage.videos_directory,
            "YouTube": config_manager.config.storage.youtube_directory,
            "Analysis": config_manager.config.storage.analysis_directory,
            "Models": config_manager.config.storage.models_directory,
            "Logs": config_manager.config.storage.logs_directory
        }
        
        for name, path in directories.items():
            dir_path = Path(path)
            exists = "✓" if dir_path.exists() else "✗"
            click.echo(f"  {name}: {exists} {path}")
    
    click.echo("\n" + "=" * 60)
