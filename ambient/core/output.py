"""
Output management for the Ambient gait analysis system.

This module provides functionality for saving analysis results, raw responses,
and managing output directories.

The OutputManager creates the following directory structure:
    {base_output_dir}/
    ├── outputs/           # Analysis text files (*_analysis.txt)
    └── raw/              # Raw Gemini responses (*_raw.txt)

@Theodore Mui
Monday, July 28, 2025 12:30:00 AM
"""

import json
import os
from pathlib import Path
from typing import Any, Dict


# Define OutputError locally to avoid circular imports
class OutputError(Exception):
    """Exception raised for output errors."""

    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


from .interfaces import IOutputManager


class OutputManager(IOutputManager):
    """
    Manages output operations for the Ambient gait analysis system.

    This class handles saving analysis results, raw responses, and managing
    output directory structure.

    Directory Structure:
        {base_output_dir}/
        ├── outputs/           # Analysis text files (*_analysis.txt)
        └── raw/              # Raw Gemini responses (*_raw.txt)

    File Naming Convention:
        - Analysis files: {record_id}_analysis.txt (e.g., OAW01_analysis.txt)
        - Raw files: {record_id}_raw.txt (e.g., OAW01_raw.txt)
    """

    def __init__(self, base_output_dir: str = None):
        """
        Initialize the output manager.

        Args:
            base_output_dir: Base directory for outputs, defaults to script directory.
                           This should be the exact path from YAML configuration
                           (e.g., "outputs/ambient_pose" or "outputs/ambient_video").

        Directory Structure Created:
            {base_output_dir}/
            ├── outputs/           # Analysis text files
            └── raw/              # Raw Gemini responses
        """
        if base_output_dir is None:
            base_output_dir = os.path.dirname(os.path.dirname(__file__))

        # Use the exact directory provided (respects YAML output_dir configuration)
        self.base_dir = Path(base_output_dir)
        self.raw_dir = self.base_dir / "raw"
        self.txt_dir = self.base_dir / "outputs"

        # Create directories
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure that all required output directories exist."""
        try:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            self.txt_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise OutputError(f"Failed to create output directories: {str(e)}")

    def save_raw_response(self, record_id: str, response: Any) -> Path:
        """
        Save the raw response to a file in the raw/ subdirectory.

        Args:
            record_id: The record ID for the analysis
            response: The raw response object from Gemini

        Returns:
            Path to the saved raw response file (in raw/ subdirectory)

        Raises:
            OutputError: If saving fails

        File Location:
            {base_dir}/raw/{record_id}_raw.txt
        """
        try:
            raw_path = self.raw_dir / f"{record_id}_raw.txt"

            # Convert response to string
            try:
                if hasattr(response, "to_dict"):
                    raw_str = json.dumps(response.to_dict(), indent=2)
                else:
                    raw_str = str(response)
            except Exception:
                raw_str = str(response)

            # Write to file
            with open(raw_path, "w") as f_raw:
                f_raw.write(raw_str)

            return raw_path

        except Exception as e:
            raise OutputError(f"Failed to save raw response for {record_id}: {str(e)}")

    def save_analysis_text(self, record_id: str, text: str) -> Path:
        """
        Save the analysis text to a file in the outputs/ subdirectory.

        Args:
            record_id: The record ID for the analysis
            text: The analysis text to save

        Returns:
            Path to the saved analysis file (in outputs/ subdirectory)

        Raises:
            OutputError: If saving fails

        File Location:
            {base_dir}/outputs/{record_id}_analysis.txt
        """
        try:
            out_path = self.txt_dir / f"{record_id}_analysis.txt"

            with open(out_path, "w") as f_out:
                f_out.write(text)

            return out_path

        except Exception as e:
            raise OutputError(f"Failed to save analysis text for {record_id}: {str(e)}")

    def get_output_directories(self) -> Dict[str, Path]:
        """
        Get the output directory paths.

        Returns:
            Dictionary mapping directory names to paths:
            - "base": Main output directory (e.g., outputs/ambient_pose)
            - "raw": Raw files directory (e.g., outputs/ambient_pose/raw)
            - "outputs": Analysis files directory (e.g., outputs/ambient_pose/outputs)
        """
        return {"base": self.base_dir, "raw": self.raw_dir, "outputs": self.txt_dir}

    def list_output_files(self) -> Dict[str, list]:
        """
        List all output files organized by type.

        Returns:
            Dictionary mapping file types to lists of file paths:
            - "raw": List of raw response file paths
            - "outputs": List of analysis file paths
        """
        try:
            raw_files = (
                list(self.raw_dir.glob("*.txt")) if self.raw_dir.exists() else []
            )
            output_files = (
                list(self.txt_dir.glob("*.txt")) if self.txt_dir.exists() else []
            )

            return {
                "raw": [str(f) for f in raw_files],
                "outputs": [str(f) for f in output_files],
            }
        except Exception as e:
            raise OutputError(f"Failed to list output files: {str(e)}")

    def clear_outputs(self) -> None:
        """
        Clear all output files and recreate directories.

        This removes all files from both raw/ and outputs/ subdirectories
        and recreates the directory structure.

        Raises:
            OutputError: If clearing fails
        """
        try:
            import shutil

            if self.base_dir.exists():
                shutil.rmtree(self.base_dir)

            # Recreate directories
            self._ensure_directories()

        except Exception as e:
            raise OutputError(f"Failed to clear outputs: {str(e)}")

    def get_output_summary(self) -> Dict[str, Any]:
        """
        Get a summary of output files and directories.

        Returns:
            Dictionary containing output summary information:
            - "directories": Mapping of directory names to paths
            - "files": Dictionary of file lists by type
            - "total_raw_files": Count of raw response files
            - "total_output_files": Count of analysis files
        """
        try:
            dirs = self.get_output_directories()
            files = self.list_output_files()

            return {
                "directories": {name: str(path) for name, path in dirs.items()},
                "files": files,
                "total_raw_files": len(files.get("raw", [])),
                "total_output_files": len(files.get("outputs", [])),
            }
        except Exception as e:
            raise OutputError(f"Failed to get output summary: {str(e)}")
