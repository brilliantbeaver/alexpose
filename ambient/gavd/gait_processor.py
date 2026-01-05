#!/usr/bin/env python3
"""
Pose Data Processor for Fall Risk Detection

This module provides comprehensive pose estimation data processing capabilities
for the Chain of Causal Modalities fall risk detection system. It integrates with
OpenPose, MediaPipe, and other pose estimation outputs to extract gait features
and assess fall risk using the Tinetti POMA scoring system.

@Theodore Mui
Monday, July 28, 2025 12:30:00 AM
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

# Add the toplevel directory to the Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ambient.utils.csv_parser import parse_csv_with_dicts, parse_openpose_csv


def get_data_path(filename: str) -> str:
    """
    Get the full path to a data file in the tests/data directory.

    Args:
        filename (str): Name of the data file

    Returns:
        str: Full path to the data file
    """
    # Get the current file's directory
    current_file = Path(__file__)

    # Navigate to the tests/data directory
    # From ambient/pose/gait_processor.py -> tests/data/
    tests_data_dir = current_file.parent.parent.parent / "tests" / "data"
    file_path = tests_data_dir / filename

    # If not found, check in the original data directory as fallback
    if not file_path.exists():
        data_dir = project_root / "data"
        file_path = data_dir / filename

        # If not found, check in data/tests directory
        if not file_path.exists():
            file_path = data_dir / "tests" / filename

    return str(file_path)


class GaitDataProcessor:
    """
    Process pose estimation data for fall risk assessment.

    This class integrates with the Chain of Causality approach using OpenPose data
    to extract gait features and assess fall risk using the Tinetti POMA scoring system.

    Attributes:
        gait_features (Dict): Extracted gait features from pose data
        tinetti_scores (Dict): Calculated Tinetti POMA scores
    """

    def __init__(self):
        """Initialize the GaitDataProcessor."""
        self.gait_features = {}
        self.tinetti_scores = {}

    def load_pose_data(
        self, csv_file_path: str, data_type: str = "openpose"
    ) -> List[Dict[str, Any]]:
        """
        Load pose estimation data from CSV file.

        Args:
            csv_file_path (str): Path to the CSV file
            data_type (str): Type of pose data ("openpose", "general", "mediapipe")

        Returns:
            List[Dict[str, Any]]: Parsed pose data

        Raises:
            FileNotFoundError: If the CSV file is not found
            Exception: If there's an error parsing the CSV file

        Example:
            >>> processor = GaitDataProcessor()
            >>> pose_data = processor.load_pose_data('openpose_output.csv', 'openpose')
            >>> print(f"Loaded {len(pose_data)} frames")
        """
        if data_type == "openpose":
            return parse_openpose_csv(csv_file_path)
        else:
            # For general CSV with dictionary fields
            dict_fields = ["bbox", "vid_info", "pose_data", "metadata"]
            return parse_csv_with_dicts(csv_file_path, dict_fields=dict_fields)

    def extract_gait_features(self, pose_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract gait features aligned with Tinetti POMA criteria.

        This method processes pose keypoints to extract biomechanical features
        that are relevant for fall risk assessment, including step length,
        trunk sway, and gait symmetry.

        Args:
            pose_data (List[Dict[str, Any]]): Parsed pose data from load_pose_data()

        Returns:
            Dict[str, Any]: Extracted gait features with summary statistics

        Example:
            >>> pose_data = processor.load_pose_data('gait_data.csv')
            >>> features = processor.extract_gait_features(pose_data)
            >>> print(f"Average step length: {features['step_length_mean']:.2f}")
        """
        features = {
            "step_length": [],
            "step_symmetry": [],
            "gait_velocity": [],
            "trunk_sway": [],
            "path_deviation": [],
            "step_continuity": [],
        }

        # Process each frame
        for i, frame in enumerate(pose_data):
            if "pose_keypoints_2d" in frame and frame["pose_keypoints_2d"]:
                keypoints = frame["pose_keypoints_2d"]

                # Extract relevant joints (OpenPose BODY_25 format)
                # Hip joints (8, 11), Knee joints (9, 12), Ankle joints (10, 13)
                if len(keypoints) >= 14:
                    left_hip = keypoints[8] if len(keypoints) > 8 else None
                    right_hip = keypoints[11] if len(keypoints) > 11 else None
                    left_knee = keypoints[9] if len(keypoints) > 9 else None
                    right_knee = keypoints[12] if len(keypoints) > 12 else None
                    left_ankle = keypoints[10] if len(keypoints) > 10 else None
                    right_ankle = keypoints[13] if len(keypoints) > 13 else None

                    # Calculate step length (distance between feet)
                    if left_ankle and right_ankle:
                        step_length = np.sqrt(
                            (left_ankle["x"] - right_ankle["x"]) ** 2
                            + (left_ankle["y"] - right_ankle["y"]) ** 2
                        )
                        features["step_length"].append(step_length)

                    # Calculate trunk sway (hip movement)
                    if left_hip and right_hip:
                        hip_center_x = (left_hip["x"] + right_hip["x"]) / 2
                        if i > 0 and "pose_keypoints_2d" in pose_data[i - 1]:
                            prev_keypoints = pose_data[i - 1]["pose_keypoints_2d"]
                            if len(prev_keypoints) > 11:
                                prev_left_hip = prev_keypoints[8]
                                prev_right_hip = prev_keypoints[11]
                                prev_hip_center_x = (
                                    prev_left_hip["x"] + prev_right_hip["x"]
                                ) / 2
                                trunk_sway = abs(hip_center_x - prev_hip_center_x)
                                features["trunk_sway"].append(trunk_sway)

        # Calculate summary statistics
        summary_features = {}
        for feature_name, values in features.items():
            if values:
                summary_features[f"{feature_name}_mean"] = np.mean(values)
                summary_features[f"{feature_name}_std"] = np.std(values)
                summary_features[f"{feature_name}_max"] = np.max(values)
                summary_features[f"{feature_name}_min"] = np.min(values)

        return summary_features

    def calculate_tinetti_gait_score(
        self, gait_features: Dict[str, Any]
    ) -> Tuple[int, Dict[str, int]]:
        """
        Calculate Tinetti POMA gait score based on extracted features.

        The Tinetti POMA (Performance Oriented Mobility Assessment) is a
        validated clinical tool for assessing balance and gait. This method
        automates the scoring process based on extracted biomechanical features.

        Args:
            gait_features (Dict[str, Any]): Extracted gait features from extract_gait_features()

        Returns:
            Tuple[int, Dict[str, int]]: Total gait score (0-12) and individual component scores

        Example:
            >>> features = processor.extract_gait_features(pose_data)
            >>> total_score, components = processor.calculate_tinetti_gait_score(features)
            >>> print(f"Tinetti Gait Score: {total_score}/12")
        """
        component_scores = {
            "gait_initiation": 0,
            "step_length_height": 0,
            "step_symmetry": 0,
            "step_continuity": 0,
            "path_deviation": 0,
            "trunk_sway": 0,
            "walking_stance": 0,
        }

        # Gait Initiation (0-1 points)
        # Check for hesitation in first few steps
        if gait_features.get("step_length_mean", 0) > 0:
            component_scores["gait_initiation"] = 1

        # Step Length and Height (0-2 points)
        step_length_mean = gait_features.get("step_length_mean", 0)
        if step_length_mean > 50:  # Normal step length
            component_scores["step_length_height"] = 2
        elif step_length_mean > 30:  # Reduced step length
            component_scores["step_length_height"] = 1

        # Step Symmetry (0-1 points)
        step_symmetry_std = gait_features.get("step_length_std", 0)
        if step_symmetry_std < 10:  # Symmetric steps
            component_scores["step_symmetry"] = 1

        # Step Continuity (0-1 points)
        if gait_features.get("step_length_std", 0) < 15:  # Continuous steps
            component_scores["step_continuity"] = 1

        # Path Deviation (0-2 points)
        trunk_sway_mean = gait_features.get("trunk_sway_mean", 0)
        if trunk_sway_mean < 5:  # Straight path
            component_scores["path_deviation"] = 2
        elif trunk_sway_mean < 10:  # Slight deviation
            component_scores["path_deviation"] = 1

        # Trunk Sway (0-2 points)
        if trunk_sway_mean < 3:  # Minimal sway
            component_scores["trunk_sway"] = 2
        elif trunk_sway_mean < 8:  # Moderate sway
            component_scores["trunk_sway"] = 1

        # Walking Stance (0-1 points)
        # Simplified assessment - could be enhanced with more detailed analysis
        component_scores["walking_stance"] = 1

        total_score = sum(component_scores.values())
        return total_score, component_scores

    def assess_fall_risk(self, tinetti_score: int) -> Tuple[str, float]:
        """
        Assess fall risk based on Tinetti POMA score.

        This method categorizes fall risk into three levels based on the
        Tinetti POMA score, which is a validated clinical assessment tool.

        Args:
            tinetti_score (int): Tinetti score. If ≤ 12, treated as gait-only (0–12);
                                 otherwise assumed full POMA (0–28).

        Returns:
            Tuple[str, float]: Risk level ("Low Risk", "Moderate Risk", "High Risk")
                              and risk score (0.0-1.0)

        Example:
            >>> total_score, _ = processor.calculate_tinetti_gait_score(features)
            >>> risk_level, risk_score = processor.assess_fall_risk(total_score)
            >>> print(f"Fall Risk: {risk_level} (Score: {risk_score:.2f})")
        """
        # Determine scale: gait-only (0–12) or full (0–28)
        max_score = 12 if tinetti_score <= 12 else 28
        # Continuous risk mapping (higher score -> lower risk)
        risk_score = max(0.0, min(1.0, 1.0 - (tinetti_score / max_score)))
        # Discrete level bands (tunable):
        if risk_score >= 2.0 / 3.0:
            level = "High Risk"
        elif risk_score >= 1.0 / 3.0:
            level = "Moderate Risk"
        else:
            level = "Low Risk"
        return level, float(risk_score)

    def process_gait_video(self, csv_file_path: str) -> Dict[str, Any]:
        """
        Complete processing pipeline for gait video analysis.

        This is the main method that orchestrates the entire gait analysis
        pipeline from CSV data to fall risk assessment.

        Args:
            csv_file_path (str): Path to pose estimation CSV file

        Returns:
            Dict[str, Any]: Complete analysis results including:
                - pose_data_frames: Number of frames analyzed
                - gait_features: Extracted biomechanical features
                - tinetti_gait_score: Total Tinetti POMA score
                - tinetti_component_scores: Individual component scores
                - fall_risk_level: Categorized risk level
                - fall_risk_score: Numerical risk score
                - analysis_summary: Summary statistics and key metrics

        Example:
            >>> processor = GaitDataProcessor()
            >>> results = processor.process_gait_video('gait_data.csv')
            >>> print(f"Fall Risk: {results['fall_risk_level']}")
            >>> print(f"Tinetti Score: {results['tinetti_gait_score']}/12")
        """
        # Load pose data
        pose_data = self.load_pose_data(csv_file_path, "openpose")

        # Extract gait features
        gait_features = self.extract_gait_features(pose_data)

        # Calculate Tinetti scores
        gait_score, component_scores = self.calculate_tinetti_gait_score(gait_features)

        # Assess fall risk
        risk_level, risk_score = self.assess_fall_risk(gait_score)

        # Compile results
        results = {
            "pose_data_frames": len(pose_data),
            "gait_features": gait_features,
            "tinetti_gait_score": gait_score,
            "tinetti_component_scores": component_scores,
            "fall_risk_level": risk_level,
            "fall_risk_score": risk_score,
            "analysis_summary": {
                "total_frames_analyzed": len(pose_data),
                "key_gait_metrics": {
                    "average_step_length": gait_features.get("step_length_mean", 0),
                    "step_symmetry_variability": gait_features.get(
                        "step_length_std", 0
                    ),
                    "trunk_stability": gait_features.get("trunk_sway_mean", 0),
                },
            },
        }

        return results


class GaitSequenceAnalyzer:
    """
    Analyzes individual gait sequences for fall risk assessment.

    This class follows the Single Responsibility Principle by focusing solely
    on analyzing individual gait sequences. It provides a clean interface
    for sequence analysis with configurable output options.

    The class follows the Open/Closed Principle by allowing extension of
    analysis methods without modifying existing code, and the Dependency
    Inversion Principle by depending on abstractions (GaitDataProcessor)
    rather than concrete implementations.

    Attributes:
        processor (GaitDataProcessor): Pose data processor for feature extraction
        verbose (bool): Whether to print detailed analysis information
        include_metadata (bool): Whether to include metadata in results
    """

    def __init__(
        self,
        processor: Optional["GaitDataProcessor"] = None,
        verbose: bool = True,
        include_metadata: bool = True,
    ):
        """
        Initialize the gait sequence analyzer.

        Args:
            processor (Optional[GaitDataProcessor]): Pose data processor instance.
                If None, a new instance will be created.
            verbose (bool): Whether to print detailed analysis information
            include_metadata (bool): Whether to include metadata in results
        """
        # Import here to avoid circular import
        self.processor = processor or GaitDataProcessor()
        self.verbose = verbose
        self.include_metadata = include_metadata

    def analyze_sequence(
        self, seq_id: str, pose_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze a single gait sequence for fall risk assessment.

        This method performs a comprehensive analysis of a gait sequence,
        including feature extraction, Tinetti scoring, and fall risk assessment.
        It follows the Single Responsibility Principle by focusing on sequence
        analysis and delegating specific tasks to appropriate components.

        Args:
            seq_id (str): Unique identifier for the gait sequence
            pose_data (List[Dict[str, Any]]): Pose data for the sequence.
                Each frame should contain pose keypoints and optional metadata.

        Returns:
            Dict[str, Any]: Comprehensive analysis results including:
                - sequence_id: Unique identifier for the sequence
                - frames_analyzed: Number of frames processed
                - gait_features: Extracted biomechanical features
                - tinetti_gait_score: Total Tinetti POMA score (0-12)
                - tinetti_component_scores: Individual component scores
                - fall_risk_level: Categorized risk level
                - fall_risk_score: Numerical risk score (0.0-1.0)
                - metadata: Sequence metadata (if include_metadata=True)

        Raises:
            ValueError: If pose_data is empty or invalid
            Exception: If analysis fails due to data issues

        Example:
            >>> analyzer = GaitSequenceAnalyzer(verbose=True)
            >>> results = analyzer.analyze_sequence("seq_001", pose_data)
            >>> print(f"Fall Risk: {results['fall_risk_level']}")
        """
        if not pose_data:
            raise ValueError("Pose data cannot be empty")

        if self.verbose:
            logger.info(f"Analyzing Sequence: {seq_id}")
            logger.info(f"Frames: {len(pose_data)}")

        try:
            # Extract gait features using the pose processor
            features = self.processor.extract_gait_features(pose_data)

            # Calculate Tinetti scores for gait assessment
            gait_score, component_scores = self.processor.calculate_tinetti_gait_score(
                features
            )

            # Assess fall risk based on Tinetti score
            risk_level, risk_score = self.processor.assess_fall_risk(gait_score)

            # Compile comprehensive results
            results = self._compile_analysis_results(
                seq_id,
                pose_data,
                features,
                gait_score,
                component_scores,
                risk_level,
                risk_score,
            )

            if self.verbose:
                logger.info(f"Tinetti Score: {gait_score}/12")
                logger.info(f"Fall Risk: {risk_level} ({risk_score:.2f})")

            return results

        except Exception as e:
            if self.verbose:
                logger.error(f"Error analyzing sequence {seq_id}: {str(e)}")
            raise

    def _compile_analysis_results(
        self,
        seq_id: str,
        pose_data: List[Dict[str, Any]],
        features: Dict[str, Any],
        gait_score: int,
        component_scores: Dict[str, int],
        risk_level: str,
        risk_score: float,
    ) -> Dict[str, Any]:
        """
        Compile analysis results into a structured format.

        This private method follows the Single Responsibility Principle by
        focusing solely on result compilation and formatting.

        Args:
            seq_id (str): Sequence identifier
            pose_data (List[Dict[str, Any]]): Original pose data
            features (Dict[str, Any]): Extracted gait features
            gait_score (int): Total Tinetti gait score
            component_scores (Dict[str, int]): Individual component scores
            risk_level (str): Categorized risk level
            risk_score (float): Numerical risk score

        Returns:
            Dict[str, Any]: Structured analysis results
        """
        results = {
            "sequence_id": seq_id,
            "frames_analyzed": len(pose_data),
            "gait_features": features,
            "tinetti_gait_score": gait_score,
            "tinetti_component_scores": component_scores,
            "fall_risk_level": risk_level,
            "fall_risk_score": risk_score,
        }

        # Add metadata if requested and available
        if self.include_metadata and pose_data:
            metadata = self._extract_metadata(pose_data[0])
            if metadata:
                results["metadata"] = metadata

        return results

    def _extract_metadata(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from frame data.

        This private method safely extracts metadata from the first frame
        of a sequence, handling cases where metadata might not be present.

        Args:
            frame_data (Dict[str, Any]): Frame data containing optional metadata

        Returns:
            Dict[str, Any]: Extracted metadata or empty dict if not available
        """
        if not frame_data:
            return {}

        # Look for metadata in common locations
        metadata_sources = ["gavd_metadata", "metadata", "info"]

        for source in metadata_sources:
            if source in frame_data and frame_data[source]:
                return frame_data[source]

        # If no structured metadata found, extract basic info
        basic_metadata = {}
        for key in ["seq", "gait_pat", "cam_view", "gait_event", "dataset"]:
            if key in frame_data:
                basic_metadata[key] = frame_data[key]

        return basic_metadata

    def analyze_multiple_sequences(
        self,
        sequences_data: Dict[str, List[Dict[str, Any]]],
        max_sequences: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Analyze multiple gait sequences in batch.

        This method provides batch processing capabilities for multiple sequences,
        following the Open/Closed Principle by allowing different processing
        strategies while maintaining a consistent interface.

        Args:
            sequences_data (Dict[str, List[Dict[str, Any]]]): Dictionary mapping
                sequence IDs to their pose data
            max_sequences (Optional[int]): Maximum number of sequences to process.
                If None, all sequences will be processed.

        Returns:
            Dict[str, Any]: Batch analysis results including:
                - total_sequences_processed: Number of sequences analyzed
                - sequence_results: Individual results for each sequence
                - summary: Overall statistics and risk distribution

        Example:
            >>> analyzer = GaitSequenceAnalyzer()
            >>> batch_results = analyzer.analyze_multiple_sequences(sequences_data)
            >>> print(f"Processed {batch_results['total_sequences_processed']} sequences")
        """
        if not sequences_data:
            return {
                "total_sequences_processed": 0,
                "sequence_results": {},
                "summary": {"error": "No sequences provided"},
            }

        sequence_results = {}
        processed_count = 0

        for seq_id, pose_data in sequences_data.items():
            if max_sequences and processed_count >= max_sequences:
                break

            try:
                results = self.analyze_sequence(seq_id, pose_data)
                sequence_results[seq_id] = results
                processed_count += 1

            except Exception as e:
                if self.verbose:
                    logger.error(f"Failed to analyze sequence {seq_id}: {str(e)}")
                # Continue with other sequences even if one fails

        # Compile batch summary
        summary = self._compile_batch_summary(sequence_results)

        return {
            "total_sequences_processed": processed_count,
            "sequence_results": sequence_results,
            "summary": summary,
        }

    def _compile_batch_summary(
        self, sequence_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compile summary statistics for batch analysis results.

        This private method calculates overall statistics and risk distribution
        from individual sequence results.

        Args:
            sequence_results (Dict[str, Dict[str, Any]]): Individual sequence results

        Returns:
            Dict[str, Any]: Summary statistics including:
                - average_tinetti_score: Mean Tinetti score across sequences
                - risk_distribution: Count of sequences by risk level
                - total_frames_analyzed: Total frames across all sequences
        """
        if not sequence_results:
            return {"error": "No results to summarize"}

        # Calculate average Tinetti score
        tinetti_scores = [r["tinetti_gait_score"] for r in sequence_results.values()]
        average_tinetti_score = sum(tinetti_scores) / len(tinetti_scores)

        # Calculate risk distribution
        risk_distribution = {
            "low_risk": sum(
                1
                for r in sequence_results.values()
                if r["fall_risk_level"] == "Low Risk"
            ),
            "moderate_risk": sum(
                1
                for r in sequence_results.values()
                if r["fall_risk_level"] == "Moderate Risk"
            ),
            "high_risk": sum(
                1
                for r in sequence_results.values()
                if r["fall_risk_level"] == "High Risk"
            ),
        }

        # Calculate total frames
        total_frames = sum(r["frames_analyzed"] for r in sequence_results.values())

        return {
            "average_tinetti_score": average_tinetti_score,
            "risk_distribution": risk_distribution,
            "total_frames_analyzed": total_frames,
            "sequences_processed": len(sequence_results),
        }


def example_usage():
    """
    Example usage of the GaitDataProcessor for fall risk assessment.

    This function demonstrates how to use the GaitDataProcessor class
    to analyze gait data and assess fall risk.
    """
    # Use the sample gait data from the data directory
    sample_gait_path = get_data_path("sample_gait_data.csv")

    # Initialize processor
    processor = GaitDataProcessor()

    # Process the gait video
    results = processor.process_gait_video(sample_gait_path)

    # Display results
    logger.info("=== Gait Analysis Results ===")
    logger.info(f"Frames analyzed: {results['pose_data_frames']}")
    logger.info(f"Tinetti Gait Score: {results['tinetti_gait_score']}/12")
    logger.info(f"Fall Risk Level: {results['fall_risk_level']}")
    logger.info(f"Fall Risk Score: {results['fall_risk_score']:.2f}")

    logger.info("=== Tinetti Component Scores ===")
    for component, score in results["tinetti_component_scores"].items():
        logger.info(f"  {component}: {score}")

    logger.info("=== Key Gait Metrics ===")
    metrics = results["analysis_summary"]["key_gait_metrics"]
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.2f}")

    return results


if __name__ == "__main__":
    example_usage()
