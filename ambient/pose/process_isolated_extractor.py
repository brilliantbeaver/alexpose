"""
Process-Isolated MediaPipe Extractor for Windows Threading Issues.

This module implements a process-based solution to MediaPipe's threading issues
on Windows. Instead of using threads, it uses separate processes to completely
isolate MediaPipe operations and prevent WinError 1 issues.
"""

import os
import sys
import time
import queue
import pickle
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from ambient.utils.log_config import get_logger
from ambient.pose.keypoint_data import KeypointSet, KeypointFormat

logger = get_logger(__name__)


class ProcessIsolatedMediaPipeWorker:
    """
    Isolated MediaPipe worker that runs in a separate process.
    
    This completely isolates MediaPipe operations from the main process,
    preventing threading conflicts and WinError 1 issues on Windows.
    """
    
    @staticmethod
    def worker_process(
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        model_path: str,
        worker_id: int
    ):
        """
        Worker process function that handles MediaPipe operations.
        
        Args:
            input_queue: Queue for receiving work items
            output_queue: Queue for sending results
            model_path: Path to MediaPipe model
            worker_id: Unique worker identifier
        """
        # Set up signal handling for graceful shutdown
        import signal
        shutdown_requested = False
        
        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            shutdown_requested = True
            logger.info(f"Worker {worker_id}: Shutdown signal received")
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # Import MediaPipe inside the worker process to avoid conflicts
            import mediapipe as mp
            from ambient.pose.model_management import PoseLandmarkerFactory
            from ambient.pose.keypoint_data import MEDIAPIPE_33_NAMES
            from ambient.pose.utils import suppress_stderr_fd
            
            logger.info(f"Worker {worker_id}: Starting MediaPipe worker process (PID: {os.getpid()})")
            
            # Create landmarker factory
            factory = PoseLandmarkerFactory()
            landmarker = None
            
            # Process work items
            while not shutdown_requested:
                try:
                    # Get work item with shorter timeout to check shutdown flag more frequently
                    work_item = input_queue.get(timeout=5.0)
                    
                    if work_item is None:  # Shutdown signal
                        logger.info(f"Worker {worker_id}: Received shutdown signal via queue")
                        break
                    
                    task_type, task_data, task_id = work_item
                    
                    if task_type == "extract_keypoints":
                        # Create landmarker if needed
                        if landmarker is None:
                            try:
                                with suppress_stderr_fd():
                                    landmarker = factory.create_landmarker(model_path)
                                logger.debug(f"Worker {worker_id}: Created landmarker")
                            except Exception as e:
                                error_msg = f"Worker {worker_id}: Failed to create landmarker: {e}"
                                logger.error(error_msg)
                                output_queue.put((task_id, "error", error_msg))
                                continue
                        
                        # Extract keypoints from image
                        image_rgb = task_data
                        
                        try:
                            # Convert to MediaPipe image format
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                            
                            # Detect pose with warning suppression
                            with suppress_stderr_fd():
                                detection_result = landmarker.detect(mp_image)
                            
                            # Extract keypoints
                            height, width = image_rgb.shape[:2]
                            
                            if not detection_result.pose_landmarks:
                                # Return empty result
                                result = KeypointSet(
                                    keypoints=[],
                                    format=KeypointFormat.CUSTOM,
                                    frame_width=width,
                                    frame_height=height
                                )
                            else:
                                pose_landmarks = detection_result.pose_landmarks[0]
                                result = KeypointSet.from_mediapipe(
                                    landmarks=pose_landmarks,
                                    frame_width=width,
                                    frame_height=height,
                                    landmark_names=MEDIAPIPE_33_NAMES
                                )
                            
                            # Send result back
                            output_queue.put((task_id, "success", result))
                            
                        except Exception as e:
                            error_msg = f"Worker {worker_id}: MediaPipe detection failed: {e}"
                            logger.error(error_msg)
                            output_queue.put((task_id, "error", error_msg))
                    
                    else:
                        error_msg = f"Worker {worker_id}: Unknown task type: {task_type}"
                        logger.error(error_msg)
                        output_queue.put((task_id, "error", error_msg))
                
                except queue.Empty:
                    # Timeout waiting for work - check shutdown flag and continue
                    if shutdown_requested:
                        logger.info(f"Worker {worker_id}: Shutdown requested, exiting")
                        break
                    continue
                except Exception as e:
                    logger.error(f"Worker {worker_id}: Unexpected error: {e}")
                    if shutdown_requested:
                        break
            
            logger.info(f"Worker {worker_id}: Shutting down gracefully")
            
        except KeyboardInterrupt:
            logger.info(f"Worker {worker_id}: Interrupted by keyboard")
        except Exception as e:
            logger.error(f"Worker {worker_id}: Fatal error in worker process: {e}")
        finally:
            # Cleanup
            if 'landmarker' in locals() and landmarker is not None:
                try:
                    del landmarker
                    logger.debug(f"Worker {worker_id}: Cleaned up landmarker")
                except Exception:
                    pass
            logger.info(f"Worker {worker_id}: Process terminated")


class ProcessIsolatedExtractor:
    """
    Process-isolated MediaPipe keypoint extractor.
    
    This class manages a pool of worker processes that handle MediaPipe operations
    in complete isolation from the main process, preventing Windows threading issues.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_workers: int = 1,
        worker_timeout: float = 60.0  # Increased from 30s to 60s for batch processing
    ):
        """
        Initialize the process-isolated extractor.
        
        Args:
            model_path: Path to MediaPipe model file
            num_workers: Number of worker processes (1 recommended for Windows)
            worker_timeout: Timeout for worker operations in seconds (default: 60s)
        """
        self.model_path = model_path
        self.num_workers = num_workers
        self.worker_timeout = worker_timeout
        
        # Process management
        self.workers: List[mp.Process] = []
        self.input_queue: Optional[mp.Queue] = None
        self.output_queue: Optional[mp.Queue] = None
        self.task_counter = 0
        self.is_started = False
        self._shutdown_in_progress = False
        
        # Ensure model is available
        if self.model_path is None:
            from ambient.pose.model_management import MediaPipeModelManager
            manager = MediaPipeModelManager()
            self.model_path = manager.ensure_model_available()
            if self.model_path is None:
                raise RuntimeError("Failed to download or locate MediaPipe model")
        
        logger.info(f"Initialized ProcessIsolatedExtractor with {num_workers} workers (timeout: {worker_timeout}s)")
    
    def start(self):
        """Start the worker processes."""
        if self.is_started:
            return
        
        if self._shutdown_in_progress:
            logger.warning("Cannot start workers during shutdown")
            return
        
        logger.info("Starting MediaPipe worker processes...")
        
        # Create queues with larger size for batch processing
        self.input_queue = mp.Queue(maxsize=200)
        self.output_queue = mp.Queue(maxsize=200)
        
        # Start worker processes (non-daemon to ensure proper cleanup)
        for worker_id in range(self.num_workers):
            worker = mp.Process(
                target=ProcessIsolatedMediaPipeWorker.worker_process,
                args=(self.input_queue, self.output_queue, self.model_path, worker_id),
                daemon=False  # Changed from True to ensure proper cleanup
            )
            worker.start()
            self.workers.append(worker)
            logger.info(f"Started worker process {worker_id} (PID: {worker.pid})")
        
        self.is_started = True
        logger.info(f"All {self.num_workers} worker processes started")
    
    def stop(self):
        """Stop all worker processes gracefully."""
        if not self.is_started or self._shutdown_in_progress:
            return
        
        self._shutdown_in_progress = True
        logger.info("Stopping MediaPipe worker processes...")
        
        # Send shutdown signals to all workers
        if self.input_queue:
            for i in range(self.num_workers):
                try:
                    self.input_queue.put(None, timeout=2.0)
                    logger.debug(f"Sent shutdown signal to worker {i}")
                except queue.Full:
                    logger.warning(f"Could not send shutdown signal to worker {i} (queue full)")
        
        # Wait for workers to finish gracefully
        for i, worker in enumerate(self.workers):
            try:
                logger.debug(f"Waiting for worker {i} (PID: {worker.pid}) to finish...")
                worker.join(timeout=10.0)
                
                if worker.is_alive():
                    logger.warning(f"Worker {i} did not shut down gracefully, terminating...")
                    worker.terminate()
                    worker.join(timeout=3.0)
                    
                    if worker.is_alive():
                        logger.error(f"Worker {i} did not terminate, killing...")
                        worker.kill()
                        worker.join(timeout=1.0)
                else:
                    logger.info(f"Worker {i} shut down gracefully")
                    
            except Exception as e:
                logger.error(f"Error stopping worker {i}: {e}")
        
        # Clean up queues
        if self.input_queue:
            try:
                # Drain input queue
                while not self.input_queue.empty():
                    try:
                        self.input_queue.get_nowait()
                    except queue.Empty:
                        break
                self.input_queue.close()
                self.input_queue.join_thread()
            except Exception as e:
                logger.debug(f"Error cleaning up input queue: {e}")
        
        if self.output_queue:
            try:
                # Drain output queue
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        break
                self.output_queue.close()
                self.output_queue.join_thread()
            except Exception as e:
                logger.debug(f"Error cleaning up output queue: {e}")
        
        self.workers.clear()
        self.input_queue = None
        self.output_queue = None
        self.is_started = False
        self._shutdown_in_progress = False
        
        logger.info("All worker processes stopped and cleaned up")
    
    def extract_keypoints(self, image_rgb: np.ndarray) -> Optional[KeypointSet]:
        """
        Extract keypoints from an RGB image using isolated worker process.
        
        Args:
            image_rgb: RGB image array (height, width, 3)
            
        Returns:
            KeypointSet object or None if extraction failed
        """
        if not self.is_started:
            self.start()
        
        if self._shutdown_in_progress:
            logger.warning("Cannot extract keypoints during shutdown")
            return None
        
        # Generate unique task ID
        self.task_counter += 1
        task_id = self.task_counter
        
        try:
            # Submit work to worker process
            work_item = ("extract_keypoints", image_rgb, task_id)
            self.input_queue.put(work_item, timeout=10.0)
            
            # Wait for result with progress logging
            start_time = time.time()
            last_log_time = start_time
            
            while time.time() - start_time < self.worker_timeout:
                try:
                    result_task_id, status, result_data = self.output_queue.get(timeout=2.0)
                    
                    if result_task_id == task_id:
                        if status == "success":
                            elapsed = time.time() - start_time
                            if elapsed > 5.0:  # Log if it took more than 5 seconds
                                logger.debug(f"Task {task_id} completed in {elapsed:.1f}s")
                            return result_data
                        else:
                            logger.error(f"Worker error for task {task_id}: {result_data}")
                            return None
                    else:
                        # Put back result for different task
                        self.output_queue.put((result_task_id, status, result_data))
                        
                except queue.Empty:
                    # Log progress every 10 seconds
                    current_time = time.time()
                    if current_time - last_log_time > 10.0:
                        elapsed = current_time - start_time
                        logger.debug(f"Still waiting for task {task_id} ({elapsed:.0f}s elapsed)...")
                        last_log_time = current_time
                    continue
            
            logger.error(f"Timeout waiting for result from worker (task {task_id}, timeout: {self.worker_timeout}s)")
            return None
            
        except queue.Full:
            logger.error("Input queue is full, cannot submit work to worker")
            return None
        except Exception as e:
            logger.error(f"Error submitting work to worker: {e}")
            return None
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def __del__(self):
        """Destructor - ensure workers are stopped."""
        try:
            if self.is_started and not self._shutdown_in_progress:
                logger.debug("ProcessIsolatedExtractor destructor called, stopping workers")
                self.stop()
        except Exception as e:
            # Suppress errors during cleanup
            pass


class ProcessIsolatedSequenceExtractor:
    """
    High-level sequence extractor using process isolation.
    
    This provides the same interface as SequenceKeypointExtractor but uses
    process isolation to prevent Windows MediaPipe threading issues.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_workers: int = 1,
        worker_timeout: float = 60.0  # Increased from 30s to 60s
    ):
        """
        Initialize the sequence extractor.
        
        Args:
            model_path: Path to MediaPipe model file
            num_workers: Number of worker processes
            worker_timeout: Timeout for worker operations (default: 60s)
        """
        self.extractor = ProcessIsolatedExtractor(
            model_path=model_path,
            num_workers=num_workers,
            worker_timeout=worker_timeout
        )
        logger.info(f"Initialized ProcessIsolatedSequenceExtractor (timeout: {worker_timeout}s)")
    
    def extract_from_image(self, image_rgb: np.ndarray) -> Optional[KeypointSet]:
        """
        Extract keypoints from an RGB image.
        
        Args:
            image_rgb: RGB image array (height, width, 3)
            
        Returns:
            KeypointSet object or None if extraction failed
        """
        return self.extractor.extract_keypoints(image_rgb)
    
    def extract_from_video_frame(
        self,
        video_path: Path,
        frame_number: int
    ) -> Optional[KeypointSet]:
        """
        Extract keypoints from a specific video frame.
        
        Args:
            video_path: Path to video file
            frame_number: Frame number to extract (1-based)
            
        Returns:
            KeypointSet object or None if extraction failed
        """
        try:
            # Use Windows-safe FFmpeg extraction
            from ambient.pose.windows_ffmpeg_handler import WindowsVideoFrameExtractor
            
            frame_extractor = WindowsVideoFrameExtractor(
                prefer_ffmpeg=True,
                verbose=False  # Suppress debug logs for cleaner output
            )
            frame = frame_extractor.extract_frame(video_path, frame_number)
            
            if frame is None:
                logger.warning(f"Failed to extract frame {frame_number} from {video_path}")
                return None
            
            # Convert BGR to RGB
            import cv2
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract keypoints using process isolation
            result = self.extract_from_image(frame_rgb)
            
            if result:
                result.timestamp = float(frame_number)
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting keypoints from frame {frame_number}: {e}")
            return None
    
    def start(self):
        """Start the worker processes."""
        self.extractor.start()
    
    def stop(self):
        """Stop the worker processes."""
        self.extractor.stop()
    
    def __enter__(self):
        """Context manager entry."""
        self.extractor.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.extractor.stop()