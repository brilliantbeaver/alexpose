"""
Realtime gait analysis service.

This service coordinates realtime pose estimation and gait analysis,
managing sessions and providing a high-level interface for the API layer.
"""

import time
import uuid
from typing import Dict, List, Optional, Any
from loguru import logger

from ambient.realtime.stream_processor import StreamProcessor
from ambient.realtime.interfaces import ProcessingMode, IRealtimeService


class RealtimeService(IRealtimeService):
    """
    Service layer for realtime gait analysis.
    
    This service manages processing sessions, coordinates components,
    and provides a clean interface for the API layer.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        default_processing_mode: ProcessingMode = ProcessingMode.BALANCED
    ):
        """
        Initialize realtime service.
        
        Args:
            model_path: Path to pose estimation model
            default_processing_mode: Default processing mode
        """
        self.model_path = model_path
        self.default_processing_mode = default_processing_mode
        
        # Service state
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._current_processor: Optional[StreamProcessor] = None
        self._service_stats = {
            'service_start_time': time.time(),
            'total_sessions': 0,
            'active_sessions': 0,
            'total_frames_processed': 0
        }
        
        # Configuration
        self._current_config = {
            'processing_mode': default_processing_mode.value,
            'buffer_size': 30,
            'enable_tracking': True,
            'confidence_threshold': 0.5,
            'target_fps': 30
        }
        
        logger.info(
            f"RealtimeService initialized: mode={default_processing_mode.value}"
        )
    
    async def handle_frame(self, frame_data: bytes) -> Dict[str, Any]:
        """
        Handle incoming frame data.
        
        Args:
            frame_data: Raw frame data from webcam
            
        Returns:
            Processing result with pose data and metrics
        """
        try:
            # Ensure processor is available
            if self._current_processor is None:
                self._create_processor()
            
            # Process frame
            result = await self._current_processor.process_frame(frame_data)
            
            # Update service statistics
            if result.get('success', False):
                self._service_stats['total_frames_processed'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Frame handling failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current analysis metrics.
        
        Returns:
            Dictionary containing current metrics and statistics
        """
        try:
            metrics = {
                'service_stats': self._service_stats.copy(),
                'active_sessions': len(self._active_sessions),
                'current_config': self._current_config.copy(),
                'timestamp': time.time()
            }
            
            # Add processor statistics if available
            if self._current_processor:
                processor_stats = self._current_processor.get_processing_stats()
                metrics['processor_stats'] = processor_stats
                
                # Add recent poses
                recent_poses = self._current_processor.get_recent_poses(5)
                metrics['recent_poses'] = recent_poses
            
            # Update service uptime
            uptime = time.time() - self._service_stats['service_start_time']
            metrics['service_stats']['uptime_seconds'] = uptime
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            return {
                'error': str(e),
                'timestamp': time.time()
            }
    
    def configure_analysis(self, config: Dict[str, Any]) -> None:
        """
        Configure analysis parameters.
        
        Args:
            config: Configuration parameters to update
        """
        try:
            # Update configuration
            self._current_config.update(config)
            
            # Apply configuration to processor if available
            if self._current_processor:
                self._current_processor.set_processing_parameters(config)
            
            logger.info(f"Configuration updated: {config}")
            
        except Exception as e:
            logger.error(f"Failed to configure analysis: {e}")
            raise
    
    def start_session(self) -> str:
        """
        Start a new analysis session.
        
        Returns:
            Session ID
        """
        try:
            session_id = str(uuid.uuid4())
            
            # Create session record
            session_data = {
                'session_id': session_id,
                'start_time': time.time(),
                'config': self._current_config.copy(),
                'frames_processed': 0,
                'poses_detected': 0,
                'gait_analyses': 0
            }
            
            self._active_sessions[session_id] = session_data
            
            # Create or restart processor
            self._create_processor()
            self._current_processor.start_processing()
            
            # Update service statistics
            self._service_stats['total_sessions'] += 1
            self._service_stats['active_sessions'] = len(self._active_sessions)
            
            logger.info(f"Session started: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            raise
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """
        End an analysis session and return summary.
        
        Args:
            session_id: Session ID to end
            
        Returns:
            Session summary
        """
        try:
            if session_id not in self._active_sessions:
                raise ValueError(f"Session not found: {session_id}")
            
            session_data = self._active_sessions[session_id]
            end_time = time.time()
            
            # Calculate session summary
            duration = end_time - session_data['start_time']
            summary = {
                'session_id': session_id,
                'duration_seconds': duration,
                'start_time': session_data['start_time'],
                'end_time': end_time,
                'config': session_data['config'],
                'frames_processed': session_data['frames_processed'],
                'poses_detected': session_data['poses_detected'],
                'gait_analyses': session_data['gait_analyses']
            }
            
            # Add processor statistics if available
            if self._current_processor:
                processor_stats = self._current_processor.get_processing_stats()
                summary['processor_stats'] = processor_stats
                
                # Stop processor if this is the last session
                if len(self._active_sessions) == 1:
                    self._current_processor.stop_processing()
            
            # Remove session
            del self._active_sessions[session_id]
            self._service_stats['active_sessions'] = len(self._active_sessions)
            
            logger.info(f"Session ended: {session_id}, duration: {duration:.1f}s")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to end session {session_id}: {e}")
            raise
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._current_config.copy()
    
    def is_ready(self) -> bool:
        """Check if service is ready for processing."""
        try:
            # Check if we can create a processor
            if self._current_processor is None:
                self._create_processor()
            
            return (
                self._current_processor is not None and
                self._current_processor.pose_estimator.is_ready()
            )
            
        except Exception as e:
            logger.error(f"Ready check failed: {e}")
            return False
    
    def has_active_session(self) -> bool:
        """Check if there are active sessions."""
        return len(self._active_sessions) > 0
    
    def get_current_timestamp(self) -> float:
        """Get current timestamp."""
        return time.time()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get pose estimation model information."""
        try:
            if self._current_processor is None:
                self._create_processor()
            
            if self._current_processor:
                return self._current_processor.pose_estimator._base_estimator.get_model_info()
            
            return {'error': 'No processor available'}
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {'error': str(e)}
    
    def _create_processor(self) -> None:
        """Create or recreate the stream processor."""
        try:
            # Parse processing mode
            mode_str = self._current_config.get('processing_mode', 'balanced')
            processing_mode = ProcessingMode(mode_str)
            
            # Create processor with current configuration
            self._current_processor = StreamProcessor(
                model_path=self.model_path,
                processing_mode=processing_mode,
                buffer_size=self._current_config.get('buffer_size', 30),
                enable_tracking=self._current_config.get('enable_tracking', True)
            )
            
            logger.debug("Stream processor created")
            
        except Exception as e:
            logger.error(f"Failed to create processor: {e}")
            raise
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific session.
        
        Args:
            session_id: Session ID to query
            
        Returns:
            Session information or None if not found
        """
        return self._active_sessions.get(session_id)
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active sessions.
        
        Returns:
            List of active session information
        """
        return list(self._active_sessions.values())
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service-level statistics."""
        stats = self._service_stats.copy()
        stats['uptime_seconds'] = time.time() - stats['service_start_time']
        return stats
    
    def cleanup_expired_sessions(self, max_age_seconds: float = 3600) -> int:
        """
        Clean up expired sessions.
        
        Args:
            max_age_seconds: Maximum age for sessions in seconds
            
        Returns:
            Number of sessions cleaned up
        """
        try:
            current_time = time.time()
            expired_sessions = []
            
            for session_id, session_data in self._active_sessions.items():
                age = current_time - session_data['start_time']
                if age > max_age_seconds:
                    expired_sessions.append(session_id)
            
            # Clean up expired sessions
            for session_id in expired_sessions:
                try:
                    self.end_session(session_id)
                except Exception as e:
                    logger.error(f"Failed to cleanup session {session_id}: {e}")
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
            return len(expired_sessions)
            
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
            return 0