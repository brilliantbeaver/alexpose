"""
Progress tracking utilities for CLI operations.
"""

import click
from datetime import datetime
from typing import Optional


class ProgressTracker:
    """
    Progress tracker for single video analysis.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.current_stage = None
        self.start_time = None
    
    def start_stage(self, stage: str, message: str = ""):
        """Start a new processing stage."""
        self.current_stage = stage
        self.start_time = datetime.now()
        
        if self.verbose:
            click.echo(f"\n[{stage}] {message}")
        else:
            click.echo(f"[{stage}] {message}", nl=False)
    
    def update(self, message: str):
        """Update progress with a message."""
        if self.verbose:
            click.echo(f"  → {message}")
        else:
            click.echo(".", nl=False)
    
    def complete_stage(self, message: str = ""):
        """Complete the current stage."""
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            if self.verbose:
                click.echo(f"  ✓ {message} ({elapsed:.2f}s)")
            else:
                click.echo(f" ✓ ({elapsed:.2f}s)")
        else:
            if self.verbose:
                click.echo(f"  ✓ {message}")
            else:
                click.echo(" ✓")
        
        self.current_stage = None
        self.start_time = None
    
    def fail(self, error: str):
        """Mark the operation as failed."""
        if self.verbose:
            click.echo(f"  ✗ Failed: {error}", err=True)
        else:
            click.echo(f" ✗ {error}", err=True)


class BatchProgressTracker:
    """
    Progress tracker for batch video processing.
    """
    
    def __init__(self, total: int, verbose: bool = False):
        self.total = total
        self.verbose = verbose
        self.current = 0
        self.current_video = None
        self.start_time = datetime.now()
    
    def start_video(self, video_name: str):
        """Start processing a video."""
        self.current += 1
        self.current_video = video_name
        
        if self.verbose:
            click.echo(f"\n[{self.current}/{self.total}] Processing: {video_name}")
        else:
            click.echo(f"[{self.current}/{self.total}] {video_name}...", nl=False)
    
    def update_stage(self, stage: str):
        """Update the current processing stage."""
        if self.verbose:
            click.echo(f"  → {stage}")
        else:
            click.echo(".", nl=False)
    
    def complete_video(self, video_name: str):
        """Complete processing a video."""
        if self.verbose:
            click.echo(f"  ✓ Completed: {video_name}")
        else:
            click.echo(" ✓")
    
    def fail_video(self, video_name: str, error: str):
        """Mark a video as failed."""
        if self.verbose:
            click.echo(f"  ✗ Failed: {video_name} - {error}", err=True)
        else:
            click.echo(f" ✗ ({error})", err=True)
    
    def complete(self):
        """Complete batch processing."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        click.echo(f"\nBatch processing completed in {elapsed:.2f}s")
        click.echo(f"Processed {self.current}/{self.total} videos")
    
    def fail(self, error: str):
        """Mark batch processing as failed."""
        click.echo(f"\nBatch processing failed: {error}", err=True)
