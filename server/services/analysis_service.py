"""
Analysis service for managing gait analysis workflows.

Orchestrates video processing, pose estimation, gait analysis, and classification.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import uuid
from datetime import datetime
from loguru import logger

from ambient.core.config import ConfigurationManager
from ambient.core.frame import Frame, FrameSequence
from ambient.video.processor import VideoProcessor
from ambient.pose.factory import PoseEstimatorFactory
from ambient.analysis.gait_analyzer import GaitAnalyzer
from ambient.classification.llm_classifier import LLMClassifier


class AnalysisService:
    """
    Service for managing gait analysis workflows.
    
    Coordinates video processing, pose estimation, gait analysis,
    and classification to produce comprehensive gait analysis results.
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        """
        Initialize analysis service.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager.config
        self.analysis_dir = Path(getattr(self.config.storage, 'analysis_directory', 'data/analysis'))
        self.metadata_dir = self.analysis_dir / 'metadata'
        self.results_dir = self.analysis_dir / 'results'
        
        # Create directories
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.video_processor = VideoProcessor(config_manager)
        self.pose_factory = PoseEstimatorFactory()  # No arguments needed
        # Note: GaitAnalyzer and LLMClassifier require specific initialization
        # They will be created on-demand when needed for actual analysis
        self.gait_analyzer = None
        self.llm_classifier = None
    
    def create_analysis_job(
        self,
        file_id: str,
        pose_estimator: str = "mediapipe",
        frame_rate: float = 30.0,
        use_llm_classification: bool = True,
        llm_model: str = "gpt-4o-mini",
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new analysis job.
        
        Args:
            file_id: Uploaded file ID
            pose_estimator: Pose estimation framework to use
            frame_rate: Frame extraction rate
            use_llm_classification: Whether to use LLM for classification
            llm_model: LLM model to use
            options: Additional analysis options
            
        Returns:
            Analysis job ID
        """
        analysis_id = str(uuid.uuid4())
        
        metadata = {
            "analysis_id": analysis_id,
            "file_id": file_id,
            "pose_estimator": pose_estimator,
            "frame_rate": frame_rate,
            "use_llm_classification": use_llm_classification,
            "llm_model": llm_model,
            "options": options or {},
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "progress": {
                "stage": "initialized",
                "percent": 0
            }
        }
        
        self._save_metadata(analysis_id, metadata)
        logger.info(f"Created analysis job {analysis_id} for file {file_id}")
        
        return analysis_id
    
    def create_batch_analysis(
        self,
        file_ids: List[str],
        pose_estimator: str = "mediapipe",
        frame_rate: float = 30.0,
        use_llm_classification: bool = True,
        llm_model: str = "gpt-4o-mini",
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[str]]:
        """
        Create a batch analysis job.
        
        Args:
            file_ids: List of uploaded file IDs
            pose_estimator: Pose estimation framework to use
            frame_rate: Frame extraction rate
            use_llm_classification: Whether to use LLM for classification
            llm_model: LLM model to use
            options: Additional analysis options
            
        Returns:
            Tuple of (batch_id, list of analysis_ids)
        """
        batch_id = str(uuid.uuid4())
        analysis_ids = []
        
        for file_id in file_ids:
            analysis_id = self.create_analysis_job(
                file_id=file_id,
                pose_estimator=pose_estimator,
                frame_rate=frame_rate,
                use_llm_classification=use_llm_classification,
                llm_model=llm_model,
                options=options
            )
            analysis_ids.append(analysis_id)
        
        # Save batch metadata
        batch_metadata = {
            "batch_id": batch_id,
            "analysis_ids": analysis_ids,
            "created_at": datetime.utcnow().isoformat(),
            "status": "pending"
        }
        
        batch_file = self.metadata_dir / f"batch_{batch_id}.json"
        with open(batch_file, 'w') as f:
            json.dump(batch_metadata, f, indent=2)
        
        logger.info(f"Created batch analysis {batch_id} with {len(analysis_ids)} jobs")
        
        return batch_id, analysis_ids
    
    async def run_analysis(self, analysis_id: str) -> None:
        """
        Run complete gait analysis workflow.
        
        Args:
            analysis_id: Analysis job ID
        """
        try:
            logger.info(f"Starting analysis workflow for {analysis_id}")
            
            # Load metadata
            metadata = self._load_metadata(analysis_id)
            if not metadata:
                raise ValueError(f"Analysis metadata not found: {analysis_id}")
            
            # Update status
            self._update_status(analysis_id, "running", "video_processing", 10)
            
            # Step 1: Load video and extract frames
            file_id = metadata['file_id']
            video_path = self._get_video_path(file_id)
            
            logger.info(f"Extracting frames from {video_path}")
            frame_sequence = self.video_processor.extract_frames(
                video_path=video_path,
                frame_rate=metadata['frame_rate']
            )
            
            self._update_status(analysis_id, "running", "pose_estimation", 30)
            
            # Step 2: Pose estimation
            logger.info(f"Running pose estimation with {metadata['pose_estimator']}")
            pose_estimator = self.pose_factory.create_estimator(
                estimator_type=metadata['pose_estimator']
            )
            
            # Estimate poses for all frames
            for frame in frame_sequence.frames:
                keypoints = pose_estimator.estimate_pose(frame)
                frame.pose_landmarks = keypoints
            
            self._update_status(analysis_id, "running", "gait_analysis", 60)
            
            # Step 3: Gait analysis
            logger.info("Analyzing gait patterns")
            gait_metrics = self.gait_analyzer.analyze_sequence(frame_sequence)
            
            self._update_status(analysis_id, "running", "classification", 80)
            
            # Step 4: Classification
            logger.info("Classifying gait patterns")
            if metadata['use_llm_classification']:
                classification_result = await self.llm_classifier.classify_gait(
                    gait_metrics=gait_metrics,
                    frame_sequence=frame_sequence,
                    model=metadata['llm_model']
                )
            else:
                # Use traditional classification
                classification_result = self._traditional_classification(gait_metrics)
            
            # Step 5: Save results
            results = {
                "analysis_id": analysis_id,
                "file_id": file_id,
                "gait_metrics": self._serialize_gait_metrics(gait_metrics),
                "classification": self._serialize_classification(classification_result),
                "frame_count": len(frame_sequence.frames),
                "duration": frame_sequence.duration,
                "completed_at": datetime.utcnow().isoformat()
            }
            
            self._save_results(analysis_id, results)
            self._update_status(analysis_id, "completed", "finished", 100)
            
            logger.info(f"Analysis completed successfully: {analysis_id}")
            
        except Exception as e:
            logger.error(f"Analysis failed for {analysis_id}: {str(e)}")
            self._update_status(analysis_id, "failed", "error", 0, error=str(e))
            raise
    
    def get_analysis_status(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Get analysis status and progress.
        
        Args:
            analysis_id: Analysis job ID
            
        Returns:
            Status dictionary or None if not found
        """
        return self._load_metadata(analysis_id)
    
    def get_analysis_results(
        self,
        analysis_id: str,
        format: str = "json"
    ) -> Optional[Dict[str, Any]]:
        """
        Get analysis results.
        
        Args:
            analysis_id: Analysis job ID
            format: Result format (json, csv, xml)
            
        Returns:
            Results dictionary or None if not found
        """
        results_file = self.results_dir / f"{analysis_id}.json"
        
        if not results_file.exists():
            return None
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if format == "json":
            return results
        elif format == "csv":
            return self._convert_to_csv(results)
        elif format == "xml":
            return self._convert_to_xml(results)
        else:
            return results
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get batch analysis status.
        
        Args:
            batch_id: Batch job ID
            
        Returns:
            Batch status dictionary or None if not found
        """
        batch_file = self.metadata_dir / f"batch_{batch_id}.json"
        
        if not batch_file.exists():
            return None
        
        with open(batch_file, 'r') as f:
            batch_metadata = json.load(f)
        
        # Get status of each analysis
        analyses_status = []
        for analysis_id in batch_metadata['analysis_ids']:
            status = self.get_analysis_status(analysis_id)
            if status:
                analyses_status.append({
                    "analysis_id": analysis_id,
                    "status": status['status'],
                    "progress": status.get('progress', {})
                })
        
        # Calculate overall progress
        total_progress = sum(a['progress'].get('percent', 0) for a in analyses_status)
        avg_progress = total_progress / len(analyses_status) if analyses_status else 0
        
        # Determine overall status
        statuses = [a['status'] for a in analyses_status]
        if all(s == "completed" for s in statuses):
            overall_status = "completed"
        elif any(s == "failed" for s in statuses):
            overall_status = "partial"
        elif any(s == "running" for s in statuses):
            overall_status = "running"
        else:
            overall_status = "pending"
        
        return {
            "batch_id": batch_id,
            "status": overall_status,
            "progress": avg_progress,
            "total_analyses": len(analyses_status),
            "completed": sum(1 for s in statuses if s == "completed"),
            "failed": sum(1 for s in statuses if s == "failed"),
            "running": sum(1 for s in statuses if s == "running"),
            "pending": sum(1 for s in statuses if s == "pending"),
            "analyses": analyses_status
        }
    
    def delete_analysis(self, analysis_id: str) -> bool:
        """
        Delete an analysis and its results.
        
        Args:
            analysis_id: Analysis job ID
            
        Returns:
            True if successful, False if not found
        """
        metadata_file = self.metadata_dir / f"{analysis_id}.json"
        results_file = self.results_dir / f"{analysis_id}.json"
        
        if not metadata_file.exists():
            return False
        
        # Delete files
        if metadata_file.exists():
            metadata_file.unlink()
        if results_file.exists():
            results_file.unlink()
        
        logger.info(f"Deleted analysis {analysis_id}")
        return True
    
    def list_analyses(
        self,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all analyses with pagination and filtering.
        
        Args:
            limit: Maximum number of results
            offset: Offset for pagination
            status_filter: Filter by status
            
        Returns:
            List of analysis metadata dictionaries
        """
        metadata_files = sorted(
            [f for f in self.metadata_dir.glob("*.json") if not f.name.startswith("batch_")],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        analyses = []
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                    # Apply status filter
                    if status_filter and metadata.get('status') != status_filter:
                        continue
                    
                    analyses.append(metadata)
            except Exception as e:
                logger.warning(f"Error loading metadata from {metadata_file}: {str(e)}")
        
        return analyses[offset:offset + limit]
    
    def get_dashboard_statistics(self) -> Dict[str, Any]:
        """
        Get dashboard statistics for overview.
        
        Returns:
            Dictionary with:
            - total_analyses: Total number of analyses
            - normal_patterns: Count of normal gait patterns
            - abnormal_patterns: Count of abnormal gait patterns
            - avg_confidence: Average confidence score
            - recent_analyses: List of recent analyses with results
            - status_breakdown: Count by status
        """
        try:
            # Get all analyses
            all_analyses = self.list_analyses(limit=1000, offset=0)
            
            # Initialize counters
            total_analyses = len(all_analyses)
            normal_count = 0
            abnormal_count = 0
            confidence_sum = 0.0
            confidence_count = 0
            status_counts = {
                "pending": 0,
                "running": 0,
                "completed": 0,
                "failed": 0
            }
            
            # Get recent analyses with results
            recent_analyses = []
            for metadata in all_analyses[:10]:  # Get top 10 recent
                try:
                    analysis_id = metadata.get('analysis_id')
                    if not analysis_id:
                        continue
                    
                    # Count by status
                    status = metadata.get('status', 'unknown')
                    if status in status_counts:
                        status_counts[status] += 1
                    
                    # Only include completed analyses in statistics
                    if status == 'completed':
                        # Load results
                        results = self.get_analysis_results(analysis_id)
                        
                        if results:
                            classification = results.get('classification', {})
                            is_normal = classification.get('is_normal', None)
                            confidence = classification.get('confidence', 0)
                            
                            # Count normal/abnormal
                            if is_normal is not None:
                                if is_normal:
                                    normal_count += 1
                                else:
                                    abnormal_count += 1
                            
                            # Sum confidence
                            if confidence > 0:
                                confidence_sum += confidence
                                confidence_count += 1
                            
                            # Add to recent analyses
                            recent_analyses.append({
                                "analysis_id": analysis_id,
                                "file_id": metadata.get('file_id', ''),
                                "status": status,
                                "is_normal": is_normal,
                                "confidence": confidence,
                                "explanation": classification.get('explanation', ''),
                                "identified_conditions": classification.get('identified_conditions', []),
                                "created_at": metadata.get('created_at', ''),
                                "completed_at": results.get('completed_at', ''),
                                "frame_count": results.get('frame_count', 0),
                                "duration": results.get('duration', 0)
                            })
                except Exception as e:
                    logger.warning(f"Error processing analysis {analysis_id}: {str(e)}")
                    continue
            
            # Calculate average confidence
            avg_confidence = (confidence_sum / confidence_count) if confidence_count > 0 else 0
            
            # Calculate percentages
            completed_count = status_counts['completed']
            normal_percentage = (normal_count / completed_count * 100) if completed_count > 0 else 0
            abnormal_percentage = (abnormal_count / completed_count * 100) if completed_count > 0 else 0
            
            return {
                "total_analyses": total_analyses,
                "normal_patterns": normal_count,
                "abnormal_patterns": abnormal_count,
                "normal_percentage": round(normal_percentage, 1),
                "abnormal_percentage": round(abnormal_percentage, 1),
                "avg_confidence": round(avg_confidence * 100, 1),  # Convert to percentage
                "recent_analyses": recent_analyses,
                "status_breakdown": status_counts,
                "completed_count": completed_count
            }
        except Exception as e:
            logger.error(f"Error calculating dashboard statistics: {str(e)}")
            # Return empty statistics on error
            return {
                "total_analyses": 0,
                "normal_patterns": 0,
                "abnormal_patterns": 0,
                "normal_percentage": 0.0,
                "abnormal_percentage": 0.0,
                "avg_confidence": 0.0,
                "recent_analyses": [],
                "status_breakdown": {
                    "pending": 0,
                    "running": 0,
                    "completed": 0,
                    "failed": 0
                },
                "completed_count": 0
            }
    
    # Private helper methods
    
    def _save_metadata(self, analysis_id: str, metadata: Dict[str, Any]) -> None:
        """Save analysis metadata."""
        metadata_file = self.metadata_dir / f"{analysis_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Load analysis metadata."""
        metadata_file = self.metadata_dir / f"{analysis_id}.json"
        if not metadata_file.exists():
            return None
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    def _update_status(
        self,
        analysis_id: str,
        status: str,
        stage: str,
        percent: int,
        error: Optional[str] = None
    ) -> None:
        """Update analysis status."""
        metadata = self._load_metadata(analysis_id)
        if metadata:
            metadata['status'] = status
            metadata['progress'] = {
                "stage": stage,
                "percent": percent
            }
            if error:
                metadata['error'] = error
            metadata['updated_at'] = datetime.utcnow().isoformat()
            self._save_metadata(analysis_id, metadata)
    
    def _save_results(self, analysis_id: str, results: Dict[str, Any]) -> None:
        """Save analysis results."""
        results_file = self.results_dir / f"{analysis_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _get_video_path(self, file_id: str) -> Path:
        """Get video file path from file ID."""
        # Load upload metadata to get file path
        from server.services.upload_service import UploadService
        upload_service = UploadService(ConfigurationManager())
        metadata = upload_service.get_upload_metadata(file_id)
        
        if not metadata or 'file_path' not in metadata:
            raise ValueError(f"Video file not found for file_id: {file_id}")
        
        return Path(metadata['file_path'])
    
    def _serialize_gait_metrics(self, gait_metrics: Any) -> Dict[str, Any]:
        """Serialize gait metrics to JSON-compatible format."""
        # Convert gait metrics dataclass to dictionary
        return {
            "stride_length": getattr(gait_metrics, 'stride_length', 0),
            "stride_time": getattr(gait_metrics, 'stride_time', 0),
            "cadence": getattr(gait_metrics, 'cadence', 0),
            "step_width": getattr(gait_metrics, 'step_width', 0),
            "symmetry_index": getattr(gait_metrics, 'symmetry_index', 0),
            # Add more metrics as needed
        }
    
    def _serialize_classification(self, classification_result: Any) -> Dict[str, Any]:
        """Serialize classification result to JSON-compatible format."""
        return {
            "is_normal": getattr(classification_result, 'is_normal', None),
            "confidence": getattr(classification_result, 'confidence', 0),
            "explanation": getattr(classification_result, 'explanation', ''),
            "identified_conditions": getattr(classification_result, 'identified_conditions', [])
        }
    
    def _traditional_classification(self, gait_metrics: Any) -> Any:
        """Fallback traditional classification method."""
        # Implement simple rule-based classification
        # This is a placeholder - implement actual logic
        from dataclasses import dataclass
        
        @dataclass
        class SimpleClassification:
            is_normal: bool
            confidence: float
            explanation: str
            identified_conditions: list
        
        return SimpleClassification(
            is_normal=True,
            confidence=0.5,
            explanation="Traditional classification not fully implemented",
            identified_conditions=[]
        )
    
    def _convert_to_csv(self, results: Dict[str, Any]) -> str:
        """Convert results to CSV format."""
        # Implement CSV conversion
        return "CSV conversion not implemented"
    
    def _convert_to_xml(self, results: Dict[str, Any]) -> str:
        """Convert results to XML format."""
        # Implement XML conversion
        return "XML conversion not implemented"
