"""
Pose analysis service for analyzing gait sequences.

This service orchestrates the analysis of pose data from GAVD sequences,
integrating the comprehensive analysis components from the ambient package.

Author: AlexPose Team
Date: January 4, 2026
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
import sys
import json
import time
import hashlib
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ambient.analysis.gait_analyzer import EnhancedGaitAnalyzer
from ambient.storage.sqlite_storage import SQLiteStorage
from server.services.gavd_service import GAVDService


class PoseAnalysisServiceAPI:
    """
    Service for analyzing pose data from GAVD sequences.
    
    This service provides a high-level API for pose analysis, handling:
    - Loading pose data from GAVD service
    - Running comprehensive gait analysis
    - Caching results for performance
    - Error handling and recovery
    """
    
    def __init__(self, config_manager):
        """
        Initialize pose analysis service.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.gavd_service = GAVDService(config_manager)
        
        # Initialize analyzer with appropriate settings
        self.analyzer = EnhancedGaitAnalyzer(
            keypoint_format="COCO_17",  # MediaPipe uses COCO-like format
            fps=30.0  # Default FPS, can be overridden
        )
        
        # Setup cache directory
        self.cache_dir = Path(getattr(
            self.config.config.storage, 
            'cache_directory', 
            'data/cache'
        )) / 'pose_analysis'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database storage for persistence
        db_path = Path(getattr(
            self.config.config.storage,
            'data_directory',
            'data'
        )) / 'storage' / 'alexpose.db'
        self.db_storage = SQLiteStorage(db_path)
        
        logger.info("Pose analysis service initialized")
        logger.debug(f"Cache directory: {self.cache_dir}")
        logger.debug(f"Database path: {db_path}")
    
    def get_sequence_analysis(
        self, 
        dataset_id: str, 
        sequence_id: str,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get complete analysis for a sequence.
        
        This method orchestrates the entire analysis pipeline:
        1. Check cache (if enabled)
        2. Load pose data from GAVD service
        3. Run comprehensive analysis
        4. Cache results
        5. Return formatted results
        
        Args:
            dataset_id: Dataset ID
            sequence_id: Sequence ID
            use_cache: Whether to use cached results
            force_refresh: Force re-analysis even if cached
            
        Returns:
            Analysis results dictionary or None if error
            
        Raises:
            ValueError: If dataset_id or sequence_id is invalid
            RuntimeError: If analysis fails
        """
        # Validate inputs
        if not dataset_id or not sequence_id:
            raise ValueError("dataset_id and sequence_id are required")
        
        logger.info(f"Analyzing sequence {sequence_id} in dataset {dataset_id}")
        
        try:
            # Check cache first (unless force_refresh)
            if use_cache and not force_refresh:
                # First check database for persistent storage
                db_result = self._get_database_analysis(dataset_id, sequence_id)
                if db_result:
                    logger.info(f"Returning database-cached analysis for {sequence_id}")
                    return db_result
                
                # Fallback to file cache
                cached_result = self._get_cached_analysis(dataset_id, sequence_id)
                if cached_result:
                    logger.info(f"Returning file-cached analysis for {sequence_id}")
                    return cached_result
            
            # Load pose data from GAVD service
            logger.debug(f"Loading pose sequence for {sequence_id}")
            pose_sequence = self._load_pose_sequence(dataset_id, sequence_id)
            
            if not pose_sequence:
                logger.warning(f"No pose data found for sequence {sequence_id}")
                return {
                    "error": "no_pose_data",
                    "message": "No pose data available for this sequence. The sequence may not have been processed with pose estimation.",
                    "dataset_id": dataset_id,
                    "sequence_id": sequence_id
                }
            
            logger.info(f"Loaded {len(pose_sequence)} frames with pose data")
            
            # Prepare metadata
            metadata = {
                "dataset_id": dataset_id,
                "sequence_id": sequence_id,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "num_frames": len(pose_sequence)
            }
            
            # Run analysis
            logger.debug("Running gait analysis...")
            start_time = time.time()
            
            results = self.analyzer.analyze_gait_sequence(pose_sequence, metadata)
            
            analysis_time = time.time() - start_time
            logger.info(f"Analysis complete in {analysis_time:.2f}s")
            
            # Add performance metadata
            results["performance"] = {
                "analysis_time_seconds": analysis_time,
                "frames_per_second": len(pose_sequence) / analysis_time if analysis_time > 0 else 0
            }
            
            # Cache results (both file and database)
            if use_cache:
                # Generate hash of pose data for deduplication
                pose_data_hash = self._generate_pose_data_hash(pose_sequence)
                
                # Save to database for persistent storage
                self._save_database_analysis(dataset_id, sequence_id, results, pose_data_hash)
                
                # Also save to file cache for quick access
                self._cache_analysis(dataset_id, sequence_id, results)
            
            return results
            
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error analyzing sequence {sequence_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Analysis failed: {str(e)}") from e
    
    def get_sequence_features(
        self, 
        dataset_id: str, 
        sequence_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get extracted features only (subset of full analysis).
        
        Args:
            dataset_id: Dataset ID
            sequence_id: Sequence ID
            
        Returns:
            Features dictionary or None if error
        """
        try:
            results = self.get_sequence_analysis(dataset_id, sequence_id)
            if results and "features" in results:
                return {
                    "dataset_id": dataset_id,
                    "sequence_id": sequence_id,
                    "features": results["features"]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting features: {str(e)}")
            return None
    
    def get_sequence_cycles(
        self, 
        dataset_id: str, 
        sequence_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get detected gait cycles only (subset of full analysis).
        
        Args:
            dataset_id: Dataset ID
            sequence_id: Sequence ID
            
        Returns:
            Gait cycles dictionary or None if error
        """
        try:
            results = self.get_sequence_analysis(dataset_id, sequence_id)
            if results and "gait_cycles" in results:
                return {
                    "dataset_id": dataset_id,
                    "sequence_id": sequence_id,
                    "gait_cycles": results["gait_cycles"],
                    "timing_analysis": results.get("timing_analysis", {})
                }
            return None
        except Exception as e:
            logger.error(f"Error getting gait cycles: {str(e)}")
            return None
    
    def get_sequence_symmetry(
        self, 
        dataset_id: str, 
        sequence_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get symmetry analysis only (subset of full analysis).
        
        Args:
            dataset_id: Dataset ID
            sequence_id: Sequence ID
            
        Returns:
            Symmetry analysis dictionary or None if error
        """
        try:
            results = self.get_sequence_analysis(dataset_id, sequence_id)
            if results and "symmetry_analysis" in results:
                return {
                    "dataset_id": dataset_id,
                    "sequence_id": sequence_id,
                    "symmetry_analysis": results["symmetry_analysis"]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting symmetry analysis: {str(e)}")
            return None
    
    def _load_pose_sequence(
        self, 
        dataset_id: str, 
        sequence_id: str
    ) -> List[Dict[str, Any]]:
        """
        Load pose data for a sequence from GAVD service.
        
        This method handles:
        - Loading frame metadata
        - Loading pose keypoints for each frame
        - Handling both old and new pose data formats
        - Filtering frames without pose data
        
        Args:
            dataset_id: Dataset ID
            sequence_id: Sequence ID
            
        Returns:
            List of pose data dictionaries, one per frame
        """
        try:
            # Get frames for sequence
            frames = self.gavd_service.get_sequence_frames(dataset_id, sequence_id)
            
            if not frames:
                logger.warning(f"No frames found for sequence {sequence_id}")
                return []
            
            logger.debug(f"Found {len(frames)} frames for sequence {sequence_id}")
            
            # Load pose data for each frame
            pose_sequence = []
            frames_with_pose = 0
            
            for frame in frames:
                frame_num = frame["frame_num"]
                
                try:
                    pose_data = self.gavd_service.get_frame_pose_data(
                        dataset_id, 
                        sequence_id, 
                        frame_num
                    )
                    
                    if pose_data:
                        # Handle both old format (list) and new format (dict with metadata)
                        if isinstance(pose_data, dict):
                            keypoints = pose_data.get('keypoints', [])
                            source_width = pose_data.get('source_width')
                            source_height = pose_data.get('source_height')
                        else:
                            keypoints = pose_data
                            source_width = None
                            source_height = None
                        
                        if keypoints:
                            pose_frame = {
                                "frame_num": frame_num,
                                "keypoints": keypoints
                            }
                            
                            # Add source dimensions if available
                            if source_width and source_height:
                                pose_frame["source_width"] = source_width
                                pose_frame["source_height"] = source_height
                            
                            pose_sequence.append(pose_frame)
                            frames_with_pose += 1
                
                except Exception as e:
                    logger.warning(f"Error loading pose data for frame {frame_num}: {str(e)}")
                    continue
            
            logger.info(f"Loaded pose data for {frames_with_pose}/{len(frames)} frames")
            
            return pose_sequence
            
        except Exception as e:
            logger.error(f"Error loading pose sequence: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _get_cached_analysis(
        self, 
        dataset_id: str, 
        sequence_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached analysis results if available.
        
        Args:
            dataset_id: Dataset ID
            sequence_id: Sequence ID
            
        Returns:
            Cached results or None if not found/expired
        """
        cache_file = self.cache_dir / f"{dataset_id}_{sequence_id}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            # Check cache age (expire after 7 days instead of 1 hour)
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age > 604800:  # 7 days (7 * 24 * 3600)
                logger.debug(f"Cache expired for {sequence_id} (age: {cache_age:.0f}s)")
                return None
            
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            logger.debug(f"Cache hit for {sequence_id} (age: {cache_age:.0f}s)")
            return cached_data
            
        except Exception as e:
            logger.warning(f"Error reading cache: {str(e)}")
            return None
    
    def _cache_analysis(
        self, 
        dataset_id: str, 
        sequence_id: str, 
        results: Dict[str, Any]
    ) -> None:
        """
        Cache analysis results for future use.
        
        Args:
            dataset_id: Dataset ID
            sequence_id: Sequence ID
            results: Analysis results to cache
        """
        cache_file = self.cache_dir / f"{dataset_id}_{sequence_id}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.debug(f"Cached analysis results for {sequence_id}")
            
        except Exception as e:
            logger.warning(f"Error caching results: {str(e)}")
    
    def clear_cache(
        self, 
        dataset_id: Optional[str] = None, 
        sequence_id: Optional[str] = None
    ) -> int:
        """
        Clear cached analysis results.
        
        Args:
            dataset_id: Optional dataset ID to clear (None = all)
            sequence_id: Optional sequence ID to clear (None = all in dataset)
            
        Returns:
            Number of cache files deleted
        """
        deleted_count = 0
        
        try:
            if dataset_id and sequence_id:
                # Clear specific sequence
                cache_file = self.cache_dir / f"{dataset_id}_{sequence_id}.json"
                if cache_file.exists():
                    cache_file.unlink()
                    deleted_count = 1
                    logger.info(f"Cleared cache for {sequence_id}")
            
            elif dataset_id:
                # Clear all sequences in dataset
                pattern = f"{dataset_id}_*.json"
                for cache_file in self.cache_dir.glob(pattern):
                    cache_file.unlink()
                    deleted_count += 1
                logger.info(f"Cleared {deleted_count} cache files for dataset {dataset_id}")
            
            else:
                # Clear all cache
                for cache_file in self.cache_dir.glob("*.json"):
                    cache_file.unlink()
                    deleted_count += 1
                logger.info(f"Cleared all {deleted_count} cache files")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
        
        return deleted_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "cache_directory": str(self.cache_dir),
                "total_files": len(cache_files),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {
                "error": str(e)
            }
    
    def _get_database_analysis(
        self,
        dataset_id: str,
        sequence_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get analysis results from database.
        
        Args:
            dataset_id: Dataset ID
            sequence_id: Sequence ID
            
        Returns:
            Analysis results or None if not found
        """
        try:
            result = self.db_storage.get_pose_analysis_result(dataset_id, sequence_id)
            if result:
                logger.debug(f"Database hit for {dataset_id}/{sequence_id}")
                return result['analysis_data']
            return None
        except Exception as e:
            logger.warning(f"Error reading from database: {str(e)}")
            return None
    
    def _save_database_analysis(
        self,
        dataset_id: str,
        sequence_id: str,
        results: Dict[str, Any],
        pose_data_hash: str
    ) -> None:
        """
        Save analysis results to database for persistent storage.
        
        Args:
            dataset_id: Dataset ID
            sequence_id: Sequence ID
            results: Analysis results
            pose_data_hash: Hash of input pose data
        """
        try:
            self.db_storage.save_pose_analysis_result(
                dataset_id=dataset_id,
                sequence_id=sequence_id,
                analysis_data=results,
                data_hash=pose_data_hash,
                version="1.0"
            )
            logger.debug(f"Saved analysis to database for {dataset_id}/{sequence_id}")
        except Exception as e:
            logger.warning(f"Error saving to database: {str(e)}")
    
    def _generate_pose_data_hash(self, pose_sequence: List[Dict[str, Any]]) -> str:
        """
        Generate hash of pose data for deduplication.
        
        Args:
            pose_sequence: List of pose frames
            
        Returns:
            SHA256 hash of pose data
        """
        try:
            # Create a stable representation of the pose data
            pose_data_str = json.dumps(pose_sequence, sort_keys=True)
            return hashlib.sha256(pose_data_str.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Error generating pose data hash: {str(e)}")
            return f"fallback_{int(time.time())}"
    
    def check_analysis_exists(
        self,
        dataset_id: str,
        sequence_id: str
    ) -> bool:
        """
        Check if analysis exists in database or cache.
        
        Args:
            dataset_id: Dataset ID
            sequence_id: Sequence ID
            
        Returns:
            True if analysis exists, False otherwise
        """
        # Check database first
        if self.db_storage.check_pose_analysis_exists(dataset_id, sequence_id):
            return True
        
        # Check file cache
        cache_file = self.cache_dir / f"{dataset_id}_{sequence_id}.json"
        return cache_file.exists()
    
    def delete_analysis(
        self,
        dataset_id: str,
        sequence_id: str
    ) -> bool:
        """
        Delete analysis from both database and cache.
        
        Args:
            dataset_id: Dataset ID
            sequence_id: Sequence ID
            
        Returns:
            True if deleted, False if not found
        """
        deleted = False
        
        # Delete from database
        try:
            if self.db_storage.delete_pose_analysis_result(dataset_id, sequence_id):
                deleted = True
        except Exception as e:
            logger.warning(f"Error deleting from database: {str(e)}")
        
        # Delete from file cache
        try:
            cache_file = self.cache_dir / f"{dataset_id}_{sequence_id}.json"
            if cache_file.exists():
                cache_file.unlink()
                deleted = True
        except Exception as e:
            logger.warning(f"Error deleting from cache: {str(e)}")
        
        return deleted
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis statistics.
        
        Returns:
            Dictionary with analysis statistics
        """
        try:
            # Get database stats
            db_results = self.db_storage.get_all_pose_analysis_results()
            
            # Get cache stats
            cache_stats = self.get_cache_stats()
            
            return {
                "database": {
                    "total_analyses": len(db_results),
                    "recent_analyses": db_results[:5] if db_results else []
                },
                "cache": cache_stats,
                "service": {
                    "analyzer_type": type(self.analyzer).__name__,
                    "keypoint_format": "COCO_17",
                    "cache_directory": str(self.cache_dir)
                }
            }
        except Exception as e:
            logger.error(f"Error getting analysis stats: {str(e)}")
            return {"error": str(e)}
