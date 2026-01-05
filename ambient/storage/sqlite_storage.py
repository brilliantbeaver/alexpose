"""
SQLite Storage Module

This module provides SQLite database storage for metadata, analysis history,
and structured data in the AlexPose system.

Key Features:
- Analysis result storage and retrieval
- Video processing history tracking
- Metadata indexing and querying
- Transaction support for atomic operations
- Automatic schema migration

Design Principles:
- Single Responsibility: Each table manager handles one entity type
- ACID compliance: All operations are transactional
- Query optimization: Proper indexing for common queries

Author: AlexPose Team
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger


class SQLiteStorage:
    """
    SQLite storage manager for AlexPose system.
    
    This class provides structured storage for analysis results, metadata,
    and processing history using SQLite database.
    """
    
    def __init__(self, db_path: Path = Path("data/storage/alexpose.db")):
        """
        Initialize SQLite storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._initialize_database()
        
        logger.info(f"Initialized SQLiteStorage with database: {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _initialize_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Video analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS video_analysis (
                    id TEXT PRIMARY KEY,
                    video_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata TEXT,
                    error_message TEXT
                )
            """)
            
            # Pose sequence table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pose_sequence (
                    id TEXT PRIMARY KEY,
                    analysis_id TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    pose_data TEXT NOT NULL,
                    gait_metrics TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (analysis_id) REFERENCES video_analysis(id)
                )
            """)
            
            # Classification result table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS classification_result (
                    id TEXT PRIMARY KEY,
                    analysis_id TEXT NOT NULL,
                    is_normal INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    conditions TEXT,
                    explanation TEXT,
                    feature_importance TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (analysis_id) REFERENCES video_analysis(id)
                )
            """)
            
            # Training dataset table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_dataset (
                    id TEXT PRIMARY KEY,
                    dataset_name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    metadata TEXT,
                    imported_at TEXT NOT NULL
                )
            """)
            
            # Training sample table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_sample (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    condition_label TEXT NOT NULL,
                    features TEXT NOT NULL,
                    ground_truth TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (dataset_id) REFERENCES training_dataset(id)
                )
            """)
            
            # Create indices for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_video_analysis_status 
                ON video_analysis(status)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pose_sequence_analysis 
                ON pose_sequence(analysis_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_classification_analysis 
                ON classification_result(analysis_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_sample_dataset 
                ON training_sample(dataset_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_sample_condition 
                ON training_sample(condition_label)
            """)
            
            # Pose analysis results table for persistent storage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pose_analysis_results (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    sequence_id TEXT NOT NULL,
                    analysis_data TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version TEXT DEFAULT '1.0',
                    UNIQUE(dataset_id, sequence_id)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pose_analysis_dataset_sequence 
                ON pose_analysis_results(dataset_id, sequence_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pose_analysis_hash 
                ON pose_analysis_results(data_hash)
            """)
            
            logger.debug("Database schema initialized")
    
    def save_video_analysis(
        self,
        analysis_id: str,
        video_path: str,
        status: str = "pending",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save video analysis record.
        
        Args:
            analysis_id: Unique analysis identifier
            video_path: Path to video file
            status: Analysis status
            metadata: Optional metadata dictionary
            
        Returns:
            Analysis ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO video_analysis 
                (id, video_path, created_at, updated_at, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (analysis_id, video_path, now, now, status, metadata_json))
            
            logger.debug(f"Saved video analysis: {analysis_id}")
            
            return analysis_id
    
    def update_video_analysis_status(
        self,
        analysis_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update video analysis status.
        
        Args:
            analysis_id: Analysis identifier
            status: New status
            error_message: Optional error message
            
        Returns:
            True if updated successfully
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            
            cursor.execute("""
                UPDATE video_analysis 
                SET status = ?, updated_at = ?, error_message = ?
                WHERE id = ?
            """, (status, now, error_message, analysis_id))
            
            success = cursor.rowcount > 0
            
            if success:
                logger.debug(f"Updated analysis status: {analysis_id} -> {status}")
            
            return success
    
    def get_video_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Get video analysis by ID.
        
        Args:
            analysis_id: Analysis identifier
            
        Returns:
            Analysis dictionary or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM video_analysis WHERE id = ?
            """, (analysis_id,))
            
            row = cursor.fetchone()
            
            if row:
                result = dict(row)
                if result['metadata']:
                    result['metadata'] = json.loads(result['metadata'])
                return result
            
            return None
    
    def list_video_analyses(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List video analyses with optional filtering.
        
        Args:
            status: Optional status filter
            limit: Maximum number of results
            offset: Result offset for pagination
            
        Returns:
            List of analysis dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if status:
                cursor.execute("""
                    SELECT * FROM video_analysis 
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (status, limit, offset))
            else:
                cursor.execute("""
                    SELECT * FROM video_analysis 
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                result = dict(row)
                if result['metadata']:
                    result['metadata'] = json.loads(result['metadata'])
                results.append(result)
            
            return results
    
    def save_classification_result(
        self,
        result_id: str,
        analysis_id: str,
        is_normal: bool,
        confidence: float,
        conditions: Optional[List[Dict[str, Any]]] = None,
        explanation: str = "",
        feature_importance: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Save classification result.
        
        Args:
            result_id: Unique result identifier
            analysis_id: Associated analysis ID
            is_normal: Whether gait is classified as normal
            confidence: Classification confidence
            conditions: List of identified conditions
            explanation: Classification explanation
            feature_importance: Feature importance scores
            
        Returns:
            Result ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            conditions_json = json.dumps(conditions) if conditions else None
            feature_importance_json = json.dumps(feature_importance) if feature_importance else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO classification_result 
                (id, analysis_id, is_normal, confidence, conditions, explanation, 
                 feature_importance, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (result_id, analysis_id, int(is_normal), confidence, conditions_json,
                  explanation, feature_importance_json, now))
            
            logger.debug(f"Saved classification result: {result_id}")
            
            return result_id
    
    def get_classification_result(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Get classification result for an analysis.
        
        Args:
            analysis_id: Analysis identifier
            
        Returns:
            Classification result dictionary or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM classification_result WHERE analysis_id = ?
            """, (analysis_id,))
            
            row = cursor.fetchone()
            
            if row:
                result = dict(row)
                result['is_normal'] = bool(result['is_normal'])
                if result['conditions']:
                    result['conditions'] = json.loads(result['conditions'])
                if result['feature_importance']:
                    result['feature_importance'] = json.loads(result['feature_importance'])
                return result
            
            return None
    
    def save_training_dataset(
        self,
        dataset_id: str,
        dataset_name: str,
        source_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save training dataset record.
        
        Args:
            dataset_id: Unique dataset identifier
            dataset_name: Dataset name
            source_path: Path to dataset source
            metadata: Optional metadata
            
        Returns:
            Dataset ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO training_dataset 
                (id, dataset_name, source_path, metadata, imported_at)
                VALUES (?, ?, ?, ?, ?)
            """, (dataset_id, dataset_name, source_path, metadata_json, now))
            
            logger.debug(f"Saved training dataset: {dataset_id}")
            
            return dataset_id
    
    def save_training_sample(
        self,
        sample_id: str,
        dataset_id: str,
        condition_label: str,
        features: Dict[str, Any],
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save training sample.
        
        Args:
            sample_id: Unique sample identifier
            dataset_id: Associated dataset ID
            condition_label: Condition label
            features: Feature dictionary
            ground_truth: Optional ground truth data
            
        Returns:
            Sample ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            features_json = json.dumps(features)
            ground_truth_json = json.dumps(ground_truth) if ground_truth else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO training_sample 
                (id, dataset_id, condition_label, features, ground_truth, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (sample_id, dataset_id, condition_label, features_json, 
                  ground_truth_json, now))
            
            logger.debug(f"Saved training sample: {sample_id}")
            
            return sample_id
    
    def get_training_samples_by_condition(
        self,
        condition_label: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get training samples by condition label.
        
        Args:
            condition_label: Condition to filter by
            limit: Maximum number of results
            
        Returns:
            List of sample dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM training_sample 
                WHERE condition_label = ?
                LIMIT ?
            """, (condition_label, limit))
            
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                result = dict(row)
                result['features'] = json.loads(result['features'])
                if result['ground_truth']:
                    result['ground_truth'] = json.loads(result['ground_truth'])
                results.append(result)
            
            return results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Count records in each table
            tables = ['video_analysis', 'pose_sequence', 'classification_result',
                     'training_dataset', 'training_sample']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats[f"{table}_count"] = count
            
            # Get database file size
            if self.db_path.exists():
                stats['database_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
            else:
                stats['database_size_mb'] = 0
            
            return stats
    
    def vacuum(self) -> None:
        """Vacuum database to reclaim space and optimize."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")
            logger.info("Database vacuumed successfully")
    
    def backup(self, backup_path: Union[str, Path]) -> Path:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path for backup file
            
        Returns:
            Path to backup file
        """
        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_connection() as conn:
            backup_conn = sqlite3.connect(str(backup_path))
            conn.backup(backup_conn)
            backup_conn.close()
        
        logger.info(f"Database backed up to {backup_path}")
        return backup_path
    
    def save_pose_analysis_result(
        self,
        dataset_id: str,
        sequence_id: str,
        analysis_data: Dict[str, Any],
        data_hash: str,
        version: str = "1.0"
    ) -> str:
        """
        Save pose analysis result to database for persistent storage.
        
        Args:
            dataset_id: Dataset identifier
            sequence_id: Sequence identifier
            analysis_data: Complete analysis results dictionary
            data_hash: Hash of input pose data for deduplication
            version: Analysis version for compatibility
            
        Returns:
            Analysis result ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            now = datetime.now().isoformat()
            analysis_json = json.dumps(analysis_data)
            result_id = f"{dataset_id}_{sequence_id}_{int(time.time())}"
            
            cursor.execute("""
                INSERT OR REPLACE INTO pose_analysis_results 
                (id, dataset_id, sequence_id, analysis_data, data_hash, 
                 created_at, updated_at, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (result_id, dataset_id, sequence_id, analysis_json, data_hash,
                  now, now, version))
            
            logger.info(f"Saved pose analysis result: {dataset_id}/{sequence_id}")
            
            return result_id
    
    def get_pose_analysis_result(
        self,
        dataset_id: str,
        sequence_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get pose analysis result from database.
        
        Args:
            dataset_id: Dataset identifier
            sequence_id: Sequence identifier
            
        Returns:
            Analysis result dictionary or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM pose_analysis_results 
                WHERE dataset_id = ? AND sequence_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
            """, (dataset_id, sequence_id))
            
            row = cursor.fetchone()
            
            if row:
                result = dict(row)
                result['analysis_data'] = json.loads(result['analysis_data'])
                logger.debug(f"Retrieved pose analysis result: {dataset_id}/{sequence_id}")
                return result
            
            logger.debug(f"No pose analysis result found: {dataset_id}/{sequence_id}")
            return None
    
    def check_pose_analysis_exists(
        self,
        dataset_id: str,
        sequence_id: str,
        data_hash: Optional[str] = None
    ) -> bool:
        """
        Check if pose analysis result exists in database.
        
        Args:
            dataset_id: Dataset identifier
            sequence_id: Sequence identifier
            data_hash: Optional data hash for exact match
            
        Returns:
            True if analysis exists, False otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if data_hash:
                cursor.execute("""
                    SELECT COUNT(*) FROM pose_analysis_results 
                    WHERE dataset_id = ? AND sequence_id = ? AND data_hash = ?
                """, (dataset_id, sequence_id, data_hash))
            else:
                cursor.execute("""
                    SELECT COUNT(*) FROM pose_analysis_results 
                    WHERE dataset_id = ? AND sequence_id = ?
                """, (dataset_id, sequence_id))
            
            count = cursor.fetchone()[0]
            return count > 0
    
    def delete_pose_analysis_result(
        self,
        dataset_id: str,
        sequence_id: str
    ) -> bool:
        """
        Delete pose analysis result from database.
        
        Args:
            dataset_id: Dataset identifier
            sequence_id: Sequence identifier
            
        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM pose_analysis_results 
                WHERE dataset_id = ? AND sequence_id = ?
            """, (dataset_id, sequence_id))
            
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted pose analysis result: {dataset_id}/{sequence_id}")
            
            return deleted
    
    def get_all_pose_analysis_results(
        self,
        dataset_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all pose analysis results, optionally filtered by dataset.
        
        Args:
            dataset_id: Optional dataset filter
            limit: Maximum number of results
            
        Returns:
            List of analysis result dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if dataset_id:
                cursor.execute("""
                    SELECT id, dataset_id, sequence_id, data_hash, created_at, updated_at, version
                    FROM pose_analysis_results 
                    WHERE dataset_id = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (dataset_id, limit))
            else:
                cursor.execute("""
                    SELECT id, dataset_id, sequence_id, data_hash, created_at, updated_at, version
                    FROM pose_analysis_results 
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
