"""
GAVD dataset service for managing GAVD training data uploads and processing.

Handles GAVD CSV file processing, pose estimation, and training data preparation.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import shutil
from datetime import datetime
from loguru import logger
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ambient.gavd.gavd_processor import GAVDProcessor, create_gavd_processor
from ambient.gavd.pose_estimators import get_pose_estimator
from ambient.core.config import ConfigurationManager


class GAVDService:
    """
    Service for managing GAVD dataset uploads and processing.
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        """
        Initialize GAVD service.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager.config
        self.training_dir = Path(getattr(self.config.storage, 'training_directory', 'data/training'))
        self.gavd_dir = self.training_dir / 'gavd'
        self.metadata_dir = self.gavd_dir / 'metadata'
        self.results_dir = self.gavd_dir / 'results'
        
        # Create directories
        self.gavd_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_dataset_metadata(self, dataset_id: str, metadata: Dict[str, Any]) -> None:
        """
        Save dataset metadata to JSON file.
        
        Args:
            dataset_id: Unique dataset identifier
            metadata: Metadata dictionary
        """
        metadata_file = self.metadata_dir / f"{dataset_id}.json"
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Saved metadata for dataset {dataset_id}")
        except Exception as e:
            logger.error(f"Error saving metadata for dataset {dataset_id}: {str(e)}")
            raise
    
    def get_dataset_metadata(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve dataset metadata.
        
        Args:
            dataset_id: Unique dataset identifier
            
        Returns:
            Metadata dictionary or None if not found
        """
        metadata_file = self.metadata_dir / f"{dataset_id}.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata for dataset {dataset_id}: {str(e)}")
            return None
    
    def update_dataset_metadata(self, dataset_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update dataset metadata.
        
        Args:
            dataset_id: Unique dataset identifier
            updates: Dictionary of updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        metadata = self.get_dataset_metadata(dataset_id)
        if not metadata:
            return False
        
        metadata.update(updates)
        metadata['updated_at'] = datetime.utcnow().isoformat()
        
        self.save_dataset_metadata(dataset_id, metadata)
        return True
    
    async def process_dataset(
        self,
        dataset_id: str,
        max_sequences: Optional[int] = None,
        pose_estimator: str = "mediapipe"
    ) -> None:
        """
        Process GAVD dataset in background.
        
        Args:
            dataset_id: Unique dataset identifier
            max_sequences: Maximum number of sequences to process
            pose_estimator: Pose estimator to use
        """
        try:
            logger.info(f"Starting GAVD dataset processing for {dataset_id}")
            
            # Get metadata
            metadata = self.get_dataset_metadata(dataset_id)
            if not metadata:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            csv_file_path = metadata['file_path']
            
            # Update status
            self.update_dataset_metadata(dataset_id, {
                "status": "processing",
                "processing_started_at": datetime.utcnow().isoformat(),
                "progress": "Initializing..."
            })
            
            # Create pose estimator if specified
            estimator = None
            if pose_estimator and pose_estimator != "none":
                try:
                    self.update_dataset_metadata(dataset_id, {
                        "progress": f"Loading {pose_estimator} pose estimator..."
                    })
                    estimator = get_pose_estimator(pose_estimator)
                    logger.info(f"Using pose estimator: {pose_estimator}")
                except Exception as e:
                    logger.warning(f"Failed to load pose estimator {pose_estimator}: {str(e)}")
                    logger.info("Continuing with placeholder keypoints")
            
            # Create GAVD processor with estimator
            processor = create_gavd_processor()
            if estimator:
                from ambient.gavd.gavd_processor import PoseDataConverter
                processor.data_converter = PoseDataConverter(estimator=estimator)
            
            # Update progress
            self.update_dataset_metadata(dataset_id, {
                "progress": "Loading and validating CSV data..."
            })
            
            # Process dataset
            logger.info(f"Processing GAVD CSV file: {csv_file_path}")
            
            # Update progress before processing
            self.update_dataset_metadata(dataset_id, {
                "progress": "Processing sequences (this may take several minutes)..."
            })
            
            # CRITICAL FIX: Run blocking operation in thread pool to avoid blocking event loop
            import asyncio
            results = await asyncio.to_thread(
                processor.process_gavd_file,
                csv_file_path=csv_file_path,
                max_sequences=max_sequences,
                include_metadata=True,
                verbose=True
            )
            
            # Update progress
            self.update_dataset_metadata(dataset_id, {
                "progress": "Saving results..."
            })
            
            # Save results
            results_file = self.results_dir / f"{dataset_id}_results.json"
            pose_data_file = self.results_dir / f"{dataset_id}_pose_data.json"
            
            with open(results_file, 'w') as f:
                # Convert results to JSON-serializable format
                json_results = {
                    "total_sequences": results["total_sequences"],
                    "summary": results["summary"],
                    "sequences": {}
                }
                
                # Store sequence IDs and basic info (full pose data is too large for JSON)
                for seq_id, seq_data in results["sequences"].items():
                    json_results["sequences"][seq_id] = {
                        "frame_count": seq_data["frame_count"],
                        "has_pose_data": len(seq_data["pose_data"]) > 0
                    }
                
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Saved processing results to {results_file}")
            
            # Save pose data separately for efficient retrieval
            # Include source video dimensions for proper coordinate scaling
            pose_data_dict = {}
            for seq_id, seq_data in results["sequences"].items():
                pose_data_dict[seq_id] = {}
                for frame_data in seq_data["pose_data"]:
                    frame_num = frame_data.get("frame")
                    keypoints = frame_data.get("pose_keypoints_2d", [])
                    if frame_num and keypoints:
                        # Extract source dimensions from first keypoint if available
                        source_width = None
                        source_height = None
                        if keypoints and len(keypoints) > 0:
                            source_width = keypoints[0].get('source_width')
                            source_height = keypoints[0].get('source_height')
                        
                        # Store keypoints with source dimensions metadata
                        pose_data_dict[seq_id][str(frame_num)] = {
                            'keypoints': keypoints,
                            'source_width': source_width,
                            'source_height': source_height
                        }
            
            with open(pose_data_file, 'w') as f:
                json.dump(pose_data_dict, f, indent=2)
            
            logger.info(f"Saved pose data to {pose_data_file}")
            logger.info(f"Total sequences with pose data: {len([s for s in pose_data_dict.values() if s])}")
            
            # Update metadata with completion
            self.update_dataset_metadata(dataset_id, {
                "status": "completed",
                "processing_completed_at": datetime.utcnow().isoformat(),
                "results_file": str(results_file),
                "total_sequences_processed": results["total_sequences"],
                "total_frames_processed": results["summary"]["total_frames"],
                "average_frames_per_sequence": results["summary"]["average_frames_per_sequence"],
                "progress": "Completed"
            })
            
            logger.info(f"GAVD dataset processing completed for {dataset_id}")
            
        except asyncio.TimeoutError:
            logger.error(f"Processing timeout for dataset {dataset_id}")
            self.update_dataset_metadata(dataset_id, {
                "status": "error",
                "error": "Processing timeout - video processing took too long. Try processing fewer sequences or smaller videos.",
                "error_at": datetime.utcnow().isoformat(),
                "progress": "Error: Timeout"
            })
        except MemoryError:
            logger.error(f"Out of memory processing dataset {dataset_id}")
            self.update_dataset_metadata(dataset_id, {
                "status": "error",
                "error": "Out of memory - videos are too large. Try processing fewer sequences.",
                "error_at": datetime.utcnow().isoformat(),
                "progress": "Error: Out of memory"
            })
        except Exception as e:
            logger.error(f"Error processing GAVD dataset {dataset_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            self.update_dataset_metadata(dataset_id, {
                "status": "error",
                "error": str(e),
                "error_at": datetime.utcnow().isoformat(),
                "progress": f"Error: {str(e)}"
            })
    
    def get_dataset_results(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get processed dataset results.
        
        Args:
            dataset_id: Unique dataset identifier
            
        Returns:
            Results dictionary or None if not found
        """
        results_file = self.results_dir / f"{dataset_id}_results.json"
        
        if not results_file.exists():
            return None
        
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading results for dataset {dataset_id}: {str(e)}")
            return None
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """
        Delete a dataset and ALL its associated data completely.
        
        This includes:
        - CSV file (original upload)
        - Metadata file
        - Results file
        - Pose data file
        - Downloaded YouTube videos (if any)
        
        Args:
            dataset_id: Unique dataset identifier
            
        Returns:
            True if successful, False if not found
        """
        metadata = self.get_dataset_metadata(dataset_id)
        if not metadata:
            logger.warning(f"Dataset {dataset_id} not found for deletion")
            return False
        
        deleted_files = []
        errors = []
        
        try:
            # STEP 1: Extract video IDs BEFORE deleting CSV (critical!)
            video_ids_to_delete = set()
            if 'file_path' in metadata:
                file_path = Path(metadata['file_path'])
                if file_path.exists():
                    try:
                        # Load CSV to get video URLs
                        import pandas as pd
                        df = pd.read_csv(file_path)
                        
                        if 'url' in df.columns:
                            # Extract unique video IDs
                            from ambient.utils.youtube_cache import extract_video_id
                            for url in df['url'].dropna().unique():
                                video_id = extract_video_id(str(url))
                                if video_id:
                                    video_ids_to_delete.add(video_id)
                            logger.info(f"Found {len(video_ids_to_delete)} unique videos to delete")
                    except Exception as e:
                        errors.append(f"Failed to extract video IDs: {str(e)}")
                        logger.error(f"Error extracting video IDs: {str(e)}")
            
            # STEP 2: Delete CSV file
            if 'file_path' in metadata:
                file_path = Path(metadata['file_path'])
                if file_path.exists():
                    try:
                        file_path.unlink()
                        deleted_files.append(f"CSV file: {file_path}")
                        logger.info(f"Deleted dataset CSV file: {file_path}")
                    except Exception as e:
                        errors.append(f"Failed to delete CSV file {file_path}: {str(e)}")
                        logger.error(f"Error deleting CSV file: {str(e)}")
            
            # STEP 3: Delete results file
            results_file = self.results_dir / f"{dataset_id}_results.json"
            if results_file.exists():
                try:
                    results_file.unlink()
                    deleted_files.append(f"Results file: {results_file}")
                    logger.info(f"Deleted results file: {results_file}")
                except Exception as e:
                    errors.append(f"Failed to delete results file: {str(e)}")
                    logger.error(f"Error deleting results file: {str(e)}")
            
            # STEP 4: Delete pose data file (CRITICAL - was missing!)
            pose_data_file = self.results_dir / f"{dataset_id}_pose_data.json"
            if pose_data_file.exists():
                try:
                    pose_data_file.unlink()
                    deleted_files.append(f"Pose data file: {pose_data_file}")
                    logger.info(f"Deleted pose data file: {pose_data_file}")
                except Exception as e:
                    errors.append(f"Failed to delete pose data file: {str(e)}")
                    logger.error(f"Error deleting pose data file: {str(e)}")
            
            # STEP 5: Delete downloaded YouTube videos (CRITICAL - was missing!)
            if video_ids_to_delete:
                youtube_dir = Path(getattr(self.config.storage, 'youtube_directory', 'data/youtube'))
                if youtube_dir.exists():
                    for video_id in video_ids_to_delete:
                        # Check for various video formats
                        for ext in ['.mp4', '.webm', '.mkv', '.mov', '.avi']:
                            video_file = youtube_dir / f"{video_id}{ext}"
                            if video_file.exists():
                                try:
                                    video_file.unlink()
                                    deleted_files.append(f"Video file: {video_file}")
                                    logger.info(f"Deleted cached video: {video_file}")
                                except Exception as e:
                                    errors.append(f"Failed to delete video {video_file}: {str(e)}")
                                    logger.error(f"Error deleting video file: {str(e)}")
            
            # STEP 6: Delete metadata file (do this last so we can use it for cleanup)
            metadata_file = self.metadata_dir / f"{dataset_id}.json"
            if metadata_file.exists():
                try:
                    metadata_file.unlink()
                    deleted_files.append(f"Metadata file: {metadata_file}")
                    logger.info(f"Deleted metadata file: {metadata_file}")
                except Exception as e:
                    errors.append(f"Failed to delete metadata file: {str(e)}")
                    logger.error(f"Error deleting metadata file: {str(e)}")
            
            # Log summary
            logger.info(f"Dataset {dataset_id} deletion complete:")
            logger.info(f"  - Deleted {len(deleted_files)} files")
            if errors:
                logger.warning(f"  - {len(errors)} errors occurred during deletion")
                for error in errors:
                    logger.warning(f"    • {error}")
            
            # Return True if at least some files were deleted
            return len(deleted_files) > 0
            
        except Exception as e:
            logger.error(f"Unexpected error during dataset deletion: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def list_datasets(
        self,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all datasets with pagination and filtering.
        
        Args:
            limit: Maximum number of results
            offset: Offset for pagination
            status_filter: Filter by status
            
        Returns:
            List of dataset metadata dictionaries
        """
        logger.debug(f"Listing datasets: limit={limit}, offset={offset}, status_filter={status_filter}")
        logger.debug(f"Metadata directory: {self.metadata_dir}")
        
        # Ensure metadata directory exists
        if not self.metadata_dir.exists():
            logger.warning(f"Metadata directory does not exist: {self.metadata_dir}")
            return []
        
        metadata_files = sorted(
            self.metadata_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        logger.debug(f"Found {len(metadata_files)} metadata files")
        
        datasets = []
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                    # Apply status filter if specified
                    if status_filter and metadata.get("status") != status_filter:
                        continue
                    
                    datasets.append(metadata)
            except Exception as e:
                logger.warning(f"Error loading metadata from {metadata_file}: {str(e)}")
        
        logger.info(f"Loaded {len(datasets)} datasets (before pagination)")
        
        # Apply pagination
        paginated = datasets[offset:offset + limit]
        logger.info(f"Returning {len(paginated)} datasets after pagination")
        
        return paginated
    
    def get_dataset_sequences(
        self,
        dataset_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get sequences from a processed dataset.
        
        Args:
            dataset_id: Unique dataset identifier
            limit: Maximum number of sequences
            offset: Offset for pagination
            
        Returns:
            List of sequence dictionaries or None if not found
        """
        results = self.get_dataset_results(dataset_id)
        if not results:
            return None
        
        sequences = []
        sequence_items = list(results.get("sequences", {}).items())
        
        for seq_id, seq_info in sequence_items[offset:offset + limit]:
            sequences.append({
                "sequence_id": seq_id,
                "frame_count": seq_info["frame_count"],
                "has_pose_data": seq_info["has_pose_data"]
            })
        
        return sequences
    
    def get_sequence_frames(
        self,
        dataset_id: str,
        sequence_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get all frames for a specific sequence with full metadata.
        
        Args:
            dataset_id: Unique dataset identifier
            sequence_id: Sequence identifier
            
        Returns:
            List of frame dictionaries with bbox, vid_info, etc.
        """
        try:
            # Get the original CSV data
            metadata = self.get_dataset_metadata(dataset_id)
            if not metadata:
                logger.error(f"Dataset {dataset_id} not found")
                return None
            
            csv_file_path = metadata.get('file_path')
            if not csv_file_path or not Path(csv_file_path).exists():
                logger.error(f"CSV file not found for dataset {dataset_id}: {csv_file_path}")
                return None
            
            # Load and organize the CSV data
            from ambient.gavd.gavd_processor import GAVDDataLoader
            loader = GAVDDataLoader()
            df = loader.load_gavd_data(csv_file_path, verbose=False)
            sequences = loader.organize_by_sequence(df, verbose=False)
            
            if sequence_id not in sequences:
                logger.warning(f"Sequence {sequence_id} not found in dataset {dataset_id}")
                return None
            
            # Convert sequence data to frame list
            seq_data = sequences[sequence_id]
            frames = []
            
            for _, row in seq_data.iterrows():
                frame = {
                    "frame_num": int(row["frame_num"]),
                    "bbox": row.get("bbox", {}),
                    "vid_info": row.get("vid_info", {}),
                    "url": row.get("url", ""),
                    "gait_event": row.get("gait_event", ""),
                    "cam_view": row.get("cam_view", ""),
                    "gait_pat": row.get("gait_pat", ""),
                    "dataset": row.get("dataset", "")
                }
                frames.append(frame)
            
            logger.info(f"Retrieved {len(frames)} frames for sequence {sequence_id} in dataset {dataset_id}")
            return frames
            
        except Exception as e:
            logger.error(f"Error retrieving sequence frames for {dataset_id}/{sequence_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_frame_pose_data(
        self,
        dataset_id: str,
        sequence_id: str,
        frame_num: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get pose keypoints for a specific frame.
        
        Args:
            dataset_id: Unique dataset identifier
            sequence_id: Sequence identifier
            frame_num: Frame number
            
        Returns:
            Dictionary with keypoints and source dimensions, or None if not found
        """
        try:
            # Check if we have pose data stored
            results_file = self.results_dir / f"{dataset_id}_pose_data.json"
            
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        pose_data = json.load(f)
                    
                    # Navigate to the specific frame
                    if sequence_id in pose_data and str(frame_num) in pose_data[sequence_id]:
                        frame_pose_data = pose_data[sequence_id][str(frame_num)]
                        
                        # Handle both old format (list of keypoints) and new format (dict with metadata)
                        if isinstance(frame_pose_data, dict):
                            logger.debug(f"Retrieved pose data for frame {frame_num} in sequence {sequence_id}")
                            return frame_pose_data
                        elif isinstance(frame_pose_data, list):
                            # Old format - return as keypoints without source dimensions
                            logger.debug(f"Retrieved pose data (old format) for frame {frame_num} in sequence {sequence_id}")
                            return {'keypoints': frame_pose_data, 'source_width': None, 'source_height': None}
                    else:
                        logger.debug(f"No pose data found for frame {frame_num} in sequence {sequence_id}")
                except Exception as e:
                    logger.error(f"Error loading pose data from file: {str(e)}")
            else:
                logger.debug(f"Pose data file not found: {results_file}")
            
            # Fallback: try to extract from processor
            logger.debug(f"Attempting to extract pose data from processor for frame {frame_num}")
            keypoints = self._extract_pose_from_processor(dataset_id, sequence_id, frame_num)
            if keypoints:
                return {'keypoints': keypoints, 'source_width': None, 'source_height': None}
            
            logger.warning(f"No pose data available for frame {frame_num} in sequence {sequence_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving pose data for {dataset_id}/{sequence_id}/frame {frame_num}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_pose_from_processor(
        self,
        dataset_id: str,
        sequence_id: str,
        frame_num: int
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Extract pose data by reprocessing if needed.
        
        This is a fallback when pose data isn't pre-cached.
        """
        try:
            # Get the original CSV data
            metadata = self.get_dataset_metadata(dataset_id)
            if not metadata:
                return None
            
            csv_file_path = metadata['file_path']
            
            # Load and process the specific sequence
            from ambient.gavd.gavd_processor import GAVDDataLoader, PoseDataConverter
            from ambient.gavd.pose_estimators import get_pose_estimator
            
            loader = GAVDDataLoader()
            df = loader.load_gavd_data(csv_file_path, verbose=False)
            sequences = loader.organize_by_sequence(df, verbose=False)
            
            if sequence_id not in sequences:
                return None
            
            seq_data = sequences[sequence_id]
            
            # Find the specific frame
            frame_row = seq_data[seq_data['frame_num'] == frame_num]
            if frame_row.empty:
                return None
            
            # Try to use pose estimator if available
            try:
                estimator = get_pose_estimator("mediapipe")
                converter = PoseDataConverter(estimator=estimator)
            except Exception:
                # Fallback to placeholder keypoints
                converter = PoseDataConverter()
            
            # Convert just this frame
            pose_frames = converter.convert_sequence_to_pose_format(
                frame_row,
                include_metadata=False
            )
            
            if pose_frames and len(pose_frames) > 0:
                return pose_frames[0].get('pose_keypoints_2d', [])
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting pose data: {str(e)}")
            return None

    def get_frame_image(
        self,
        dataset_id: str,
        sequence_id: str,
        frame_num: int,
        show_bbox: bool = True,
        show_pose: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get a frame image with optional overlays.
        
        Args:
            dataset_id: Unique dataset identifier
            sequence_id: Sequence identifier
            frame_num: Frame number
            show_bbox: Whether to draw bounding box
            show_pose: Whether to draw pose keypoints
            
        Returns:
            Dictionary with base64 encoded image and metadata
        """
        import cv2
        import base64
        from pathlib import Path
        
        # Get frame metadata
        frames = self.get_sequence_frames(dataset_id, sequence_id)
        if not frames:
            return None
        
        # Find the specific frame
        frame_data = None
        for frame in frames:
            if frame["frame_num"] == frame_num:
                frame_data = frame
                break
        
        if not frame_data:
            return None
        
        # Extract video ID and find cached video
        from ambient.utils.youtube_cache import extract_video_id
        video_id = extract_video_id(frame_data["url"])
        if not video_id:
            return None
        
        youtube_dir = Path(getattr(self.config.storage, 'youtube_directory', 'data/youtube'))
        video_path = None
        for ext in ['.mp4', '.webm', '.mkv', '.mov']:
            candidate = youtube_dir / f"{video_id}{ext}"
            if candidate.exists():
                video_path = candidate
                break
        
        if not video_path:
            return None
        
        # Extract frame from video
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)  # Convert to 0-based
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw bounding box if requested
        if show_bbox and frame_data["bbox"]:
            bbox = frame_data["bbox"]
            vid_info = frame_data["vid_info"]
            
            # Scale bbox if needed
            actual_height, actual_width = frame_rgb.shape[:2]
            annotation_width = vid_info.get('width', actual_width)
            annotation_height = vid_info.get('height', actual_height)
            
            scale_x = actual_width / annotation_width if annotation_width > 0 else 1.0
            scale_y = actual_height / annotation_height if annotation_height > 0 else 1.0
            
            left = int(bbox.get('left', 0) * scale_x)
            top = int(bbox.get('top', 0) * scale_y)
            width = int(bbox.get('width', 0) * scale_x)
            height = int(bbox.get('height', 0) * scale_y)
            
            cv2.rectangle(frame_rgb, (left, top), (left + width, top + height), (255, 0, 0), 2)
        
        # TODO: Add pose overlay if requested
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "image": image_base64,
            "width": frame_rgb.shape[1],
            "height": frame_rgb.shape[0]
        }
