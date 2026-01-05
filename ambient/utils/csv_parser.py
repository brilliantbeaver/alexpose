"""
CSV Parser for Pose Estimation Data

This module provides a robust Python CSV parser that can handle CSV files containing
dictionary objects in certain fields, specifically designed for pose estimation data
used in the Chain of Causality fall risk detection system.

@Theodore Mui
Monday, July 28, 2025 12:30:00 AM
"""

import ast
import csv
import json
from typing import Any, Dict, List, Union

import pandas as pd
from loguru import logger


def parse_csv_with_dicts(
    file_path: str, dict_fields: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Read a CSV file that contains dictionary objects in certain fields and return a list of rows.

    Args:
        file_path (str): Path to the CSV file
        dict_fields (List[str], optional): List of field names that contain dictionary objects.
                                         If None, will attempt to auto-detect dictionary fields.

    Returns:
        List[Dict[str, Any]]: List of dictionaries, where each dictionary represents a row
                             with dictionary fields properly parsed

    Example:
        # For a CSV with columns: id, bbox, vid_info, pose_data
        # where bbox and vid_info contain dictionary strings
        rows = parse_csv_with_dicts('pose_data.csv', dict_fields=['bbox', 'vid_info'])
    """

    def try_parse_dict(value: str) -> Union[Dict, str]:
        """
        Attempt to parse a string as a dictionary, return original string if parsing fails.

        Args:
            value (str): String value that might be a dictionary

        Returns:
            Union[Dict, str]: Parsed dictionary or original string
        """
        if not value or value.strip() == "":
            return value

        # Try JSON parsing first (most common for API responses)
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try ast.literal_eval for Python literal strings
        try:
            result = ast.literal_eval(value)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass

        # Return original value if parsing fails
        return value

    def auto_detect_dict_fields(first_row: Dict[str, str]) -> List[str]:
        """
        Auto-detect fields that likely contain dictionary objects.

        Args:
            first_row (Dict[str, str]): First row of the CSV

        Returns:
            List[str]: List of field names that likely contain dictionaries
        """
        dict_fields = []
        for field, value in first_row.items():
            if value and value.strip().startswith("{") and value.strip().endswith("}"):
                dict_fields.append(field)
        return dict_fields

    try:
        # Read the CSV file
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        if not rows:
            return []

        # Auto-detect dict fields if not provided
        if dict_fields is None:
            dict_fields = auto_detect_dict_fields(rows[0])

        # Parse dictionary fields in each row
        parsed_rows = []
        for row in rows:
            parsed_row = {}
            for field, value in row.items():
                if field in dict_fields:
                    parsed_row[field] = try_parse_dict(value)
                else:
                    parsed_row[field] = value
            parsed_rows.append(parsed_row)

        return parsed_rows

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {str(e)}")


def parse_csv_with_pandas(
    file_path: str, dict_fields: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Alternative implementation using pandas for better handling of complex CSV files.

    Args:
        file_path (str): Path to the CSV file
        dict_fields (List[str], optional): List of field names that contain dictionary objects

    Returns:
        List[Dict[str, Any]]: List of dictionaries representing CSV rows
    """

    def parse_dict_column(series):
        """Parse a pandas series that contains dictionary strings."""

        def safe_parse(value):
            if pd.isna(value) or value == "":
                return value
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                try:
                    return ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    return value

        return series.apply(safe_parse)

    try:
        # Read CSV with pandas
        df = pd.read_csv(file_path)

        # Parse dictionary fields
        if dict_fields:
            for field in dict_fields:
                if field in df.columns:
                    df[field] = parse_dict_column(df[field])

        # Convert to list of dictionaries
        return df.to_dict("records")

    except Exception as e:
        raise Exception(f"Error reading CSV file with pandas: {str(e)}")


def parse_openpose_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    Specialized function for parsing OpenPose CSV files with pose keypoints.

    Args:
        file_path (str): Path to OpenPose CSV file

    Returns:
        List[Dict[str, Any]]: List of pose data frames
    """

    def parse_keypoints(keypoint_str: str) -> List[Dict[str, float]]:
        """
        Parse OpenPose keypoint string into list of keypoint dictionaries.
        OpenPose format: x1,y1,conf1,x2,y2,conf2,...
        """
        if not keypoint_str or keypoint_str.strip() == "":
            return []

        try:
            # Split by comma and convert to floats
            values = [float(x.strip()) for x in keypoint_str.split(",")]

            # Group into keypoints (x, y, confidence)
            keypoints = []
            for i in range(0, len(values), 3):
                if i + 2 < len(values):
                    keypoints.append(
                        {
                            "x": values[i],
                            "y": values[i + 1],
                            "confidence": values[i + 2],
                        }
                    )

            return keypoints
        except (ValueError, IndexError):
            return []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        parsed_rows = []
        for row in rows:
            parsed_row = {}
            for field, value in row.items():
                if "keypoints" in field.lower() or "pose" in field.lower():
                    parsed_row[field] = parse_keypoints(value)
                else:
                    # Try to convert numeric fields
                    try:
                        parsed_row[field] = float(value) if "." in value else int(value)
                    except (ValueError, TypeError):
                        parsed_row[field] = value

            parsed_rows.append(parsed_row)

        return parsed_rows

    except Exception as e:
        raise Exception(f"Error parsing OpenPose CSV: {str(e)}")


# Example usage and testing functions
def example_usage():
    """
    Example usage of the CSV parser function.
    """
    # Example CSV content (you would save this as a file)
    example_csv_content = """id,bbox,vid_info,pose_data,confidence
1,"{'x': 100, 'y': 200, 'width': 50, 'height': 100}","{'fps': 30, 'duration': 60}","{'keypoints': [1, 2, 3]}",0.95
2,"{'x': 150, 'y': 250, 'width': 60, 'height': 120}","{'fps': 25, 'duration': 45}","{'keypoints': [4, 5, 6]}",0.87
3,"{'x': 200, 'y': 300, 'width': 70, 'height': 140}","{'fps': 30, 'duration': 90}","{'keypoints': [7, 8, 9]}",0.92"""

    # Save example CSV
    with open("example_pose_data.csv", "w", newline="", encoding="utf-8") as file:
        file.write(example_csv_content)

    # Parse the CSV
    try:
        # Method 1: Using the main function
        rows = parse_csv_with_dicts(
            "example_pose_data.csv", dict_fields=["bbox", "vid_info", "pose_data"]
        )

        logger.info("Parsed CSV rows:")
        for i, row in enumerate(rows):
            logger.info(f"Row {i+1}:")
            for key, value in row.items():
                logger.info(f"  {key}: {value} (type: {type(value).__name__})")

        # Method 2: Using pandas
        rows_pandas = parse_csv_with_pandas(
            "example_pose_data.csv", dict_fields=["bbox", "vid_info", "pose_data"]
        )

        logger.info(f"Parsed {len(rows_pandas)} rows using pandas method")

        # Access dictionary fields
        first_row = rows[0]
        bbox = first_row["bbox"]  # This is now a dictionary
        logger.info(f"First row bbox: {bbox}")
        logger.info(f"Bbox x coordinate: {bbox['x']}")

    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    # Run example usage
    example_usage()
