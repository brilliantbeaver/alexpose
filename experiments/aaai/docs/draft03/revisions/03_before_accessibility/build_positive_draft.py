"""Compatibility entry point for the standalone manuscript builder."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('build_draft.py')),run_name='__main__')
