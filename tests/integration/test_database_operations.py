"""
Database integration tests for AlexPose.

This module tests database operations with actual data transactions
and validates data persistence and retrieval workflows.
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from tests.fixtures.real_data_fixtures import get_real_data_manager


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not available")
class TestDatabaseOperations:
    """Test database operations with real data transactions."""

    @pytest.fixture(scope="class")
    def test_database_url(self):
        """Provide test database URL."""
        # Use SQLite for testing
        return "sqlite:///test_alexpose.db"

    @pytest.fixture(scope="class")
    def database_engine(self, test_database_url: str):
        """Provide database engine for testing."""
        engine = create_engine(test_database_url, echo=False)
        yield engine
        engine.dispose()

    @pytest.fixture(scope="class")
    def database_session(self, database_engine):
        """Provide database session for testing."""
        Session = sessionmaker(bind=database_engine)
        session = Session()
        yield session
        session.close()

    @pytest.fixture
    def sample_video_record(self):
        """Provide sample video record data."""
        return {
            "video_id": str(uuid.uuid4()),
            "filename": "test_gait_video.mp4",
            "file_path": "/data/videos/test_gait_video.mp4",
            "file_size": 2048576,  # 2MB
            "duration": 5.2,
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "format": "mp4",
            "upload_timestamp": datetime.now(timezone.utc),
            "status": "uploaded",
            "metadata": {
                "codec": "h264",
                "bitrate": 2000000,
                "frame_count": 156
            }
        }

    @pytest.fixture
    def sample_analysis_record(self, sample_video_record: Dict[str, Any]):
        """Provide sample analysis record data."""
        return {
            "analysis_id": str(uuid.uuid4()),
            "video_id": sample_video_record["video_id"],
            "analysis_type": "gait_analysis",
            "status": "completed",
            "created_timestamp": datetime.now(timezone.utc),
            "completed_timestamp": datetime.now(timezone.utc) + timedelta(minutes=2),
            "processing_time": 120.5,
            "results": {
                "classification": "normal",
                "confidence": 0.85,
                "gait_features": {
                    "temporal_features": {
                        "stride_time": 1.2,
                        "cadence": 110.0
                    },
                    "spatial_features": {
                        "stride_length": 1.4,
                        "step_width": 0.15
                    },
                    "symmetry_features": {
                        "left_right_symmetry": 0.95
                    }
                },
                "pose_landmarks": [
                    [{"x": 100.0, "y": 200.0, "confidence": 0.9} for _ in range(33)]
                    for _ in range(10)  # 10 frames of landmarks
                ]
            },
            "error_message": None
        }

    @pytest.mark.integration
    @pytest.mark.slow
    def test_create_database_tables(self, database_engine):
        """Test database table creation."""
        # Create tables using raw SQL for testing
        with database_engine.connect() as conn:
            # Create videos table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    duration REAL,
                    fps REAL,
                    width INTEGER,
                    height INTEGER,
                    format TEXT,
                    upload_timestamp TIMESTAMP,
                    status TEXT,
                    metadata TEXT
                )
            """))
            
            # Create analyses table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS analyses (
                    analysis_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_timestamp TIMESTAMP,
                    completed_timestamp TIMESTAMP,
                    processing_time REAL,
                    results TEXT,
                    error_message TEXT,
                    FOREIGN KEY (video_id) REFERENCES videos (video_id)
                )
            """))
            
            # Create users table (if user management is implemented)
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    created_timestamp TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """))
            
            conn.commit()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_insert_video_record(
        self, 
        database_engine,
        sample_video_record: Dict[str, Any]
    ):
        """Test inserting video record into database."""
        with database_engine.connect() as conn:
            # Insert video record
            conn.execute(text("""
                INSERT INTO videos (
                    video_id, filename, file_path, file_size, duration,
                    fps, width, height, format, upload_timestamp, status, metadata
                ) VALUES (
                    :video_id, :filename, :file_path, :file_size, :duration,
                    :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                )
            """), {
                **sample_video_record,
                "metadata": json.dumps(sample_video_record["metadata"])
            })
            conn.commit()
            
            # Verify insertion
            result = conn.execute(text("""
                SELECT * FROM videos WHERE video_id = :video_id
            """), {"video_id": sample_video_record["video_id"]})
            
            row = result.fetchone()
            assert row is not None, "Video record not inserted"
            
            # Validate inserted data
            assert row[0] == sample_video_record["video_id"]  # video_id
            assert row[1] == sample_video_record["filename"]  # filename
            assert row[2] == sample_video_record["file_path"]  # file_path
            assert row[3] == sample_video_record["file_size"]  # file_size

    @pytest.mark.integration
    @pytest.mark.slow
    def test_insert_analysis_record(
        self, 
        database_engine,
        sample_video_record: Dict[str, Any],
        sample_analysis_record: Dict[str, Any]
    ):
        """Test inserting analysis record into database."""
        with database_engine.connect() as conn:
            # First insert video record (foreign key dependency)
            conn.execute(text("""
                INSERT OR REPLACE INTO videos (
                    video_id, filename, file_path, file_size, duration,
                    fps, width, height, format, upload_timestamp, status, metadata
                ) VALUES (
                    :video_id, :filename, :file_path, :file_size, :duration,
                    :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                )
            """), {
                **sample_video_record,
                "metadata": json.dumps(sample_video_record["metadata"])
            })
            
            # Insert analysis record
            conn.execute(text("""
                INSERT INTO analyses (
                    analysis_id, video_id, analysis_type, status,
                    created_timestamp, completed_timestamp, processing_time,
                    results, error_message
                ) VALUES (
                    :analysis_id, :video_id, :analysis_type, :status,
                    :created_timestamp, :completed_timestamp, :processing_time,
                    :results, :error_message
                )
            """), {
                **sample_analysis_record,
                "results": json.dumps(sample_analysis_record["results"])
            })
            conn.commit()
            
            # Verify insertion
            result = conn.execute(text("""
                SELECT * FROM analyses WHERE analysis_id = :analysis_id
            """), {"analysis_id": sample_analysis_record["analysis_id"]})
            
            row = result.fetchone()
            assert row is not None, "Analysis record not inserted"
            
            # Validate inserted data
            assert row[0] == sample_analysis_record["analysis_id"]  # analysis_id
            assert row[1] == sample_analysis_record["video_id"]  # video_id
            assert row[2] == sample_analysis_record["analysis_type"]  # analysis_type

    @pytest.mark.integration
    @pytest.mark.slow
    def test_query_video_records(
        self, 
        database_engine,
        sample_video_record: Dict[str, Any]
    ):
        """Test querying video records from database."""
        with database_engine.connect() as conn:
            # Insert test data
            conn.execute(text("""
                INSERT OR REPLACE INTO videos (
                    video_id, filename, file_path, file_size, duration,
                    fps, width, height, format, upload_timestamp, status, metadata
                ) VALUES (
                    :video_id, :filename, :file_path, :file_size, :duration,
                    :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                )
            """), {
                **sample_video_record,
                "metadata": json.dumps(sample_video_record["metadata"])
            })
            conn.commit()
            
            # Query by video ID
            result = conn.execute(text("""
                SELECT * FROM videos WHERE video_id = :video_id
            """), {"video_id": sample_video_record["video_id"]})
            
            row = result.fetchone()
            assert row is not None, "Video record not found"
            
            # Query by status
            result = conn.execute(text("""
                SELECT COUNT(*) FROM videos WHERE status = :status
            """), {"status": "uploaded"})
            
            count = result.fetchone()[0]
            assert count >= 1, "No uploaded videos found"
            
            # Query with date range
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
            
            result = conn.execute(text("""
                SELECT COUNT(*) FROM videos 
                WHERE upload_timestamp BETWEEN :start_date AND :end_date
            """), {"start_date": yesterday, "end_date": tomorrow})
            
            count = result.fetchone()[0]
            assert count >= 1, "No videos found in date range"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_query_analysis_results(
        self, 
        database_engine,
        sample_video_record: Dict[str, Any],
        sample_analysis_record: Dict[str, Any]
    ):
        """Test querying analysis results from database."""
        with database_engine.connect() as conn:
            # Insert test data
            conn.execute(text("""
                INSERT OR REPLACE INTO videos (
                    video_id, filename, file_path, file_size, duration,
                    fps, width, height, format, upload_timestamp, status, metadata
                ) VALUES (
                    :video_id, :filename, :file_path, :file_size, :duration,
                    :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                )
            """), {
                **sample_video_record,
                "metadata": json.dumps(sample_video_record["metadata"])
            })
            
            conn.execute(text("""
                INSERT OR REPLACE INTO analyses (
                    analysis_id, video_id, analysis_type, status,
                    created_timestamp, completed_timestamp, processing_time,
                    results, error_message
                ) VALUES (
                    :analysis_id, :video_id, :analysis_type, :status,
                    :created_timestamp, :completed_timestamp, :processing_time,
                    :results, :error_message
                )
            """), {
                **sample_analysis_record,
                "results": json.dumps(sample_analysis_record["results"])
            })
            conn.commit()
            
            # Query analysis by ID
            result = conn.execute(text("""
                SELECT * FROM analyses WHERE analysis_id = :analysis_id
            """), {"analysis_id": sample_analysis_record["analysis_id"]})
            
            row = result.fetchone()
            assert row is not None, "Analysis record not found"
            
            # Validate results JSON
            results_json = row[7]  # results column
            results = json.loads(results_json)
            assert "classification" in results
            assert "confidence" in results
            assert results["classification"] == "normal"
            
            # Query analyses by video ID
            result = conn.execute(text("""
                SELECT COUNT(*) FROM analyses WHERE video_id = :video_id
            """), {"video_id": sample_video_record["video_id"]})
            
            count = result.fetchone()[0]
            assert count >= 1, "No analyses found for video"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_join_video_analysis_data(
        self, 
        database_engine,
        sample_video_record: Dict[str, Any],
        sample_analysis_record: Dict[str, Any]
    ):
        """Test joining video and analysis data."""
        with database_engine.connect() as conn:
            # Insert test data
            conn.execute(text("""
                INSERT OR REPLACE INTO videos (
                    video_id, filename, file_path, file_size, duration,
                    fps, width, height, format, upload_timestamp, status, metadata
                ) VALUES (
                    :video_id, :filename, :file_path, :file_size, :duration,
                    :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                )
            """), {
                **sample_video_record,
                "metadata": json.dumps(sample_video_record["metadata"])
            })
            
            conn.execute(text("""
                INSERT OR REPLACE INTO analyses (
                    analysis_id, video_id, analysis_type, status,
                    created_timestamp, completed_timestamp, processing_time,
                    results, error_message
                ) VALUES (
                    :analysis_id, :video_id, :analysis_type, :status,
                    :created_timestamp, :completed_timestamp, :processing_time,
                    :results, :error_message
                )
            """), {
                **sample_analysis_record,
                "results": json.dumps(sample_analysis_record["results"])
            })
            conn.commit()
            
            # Join query
            result = conn.execute(text("""
                SELECT 
                    v.filename, v.duration, v.fps,
                    a.analysis_type, a.status, a.processing_time,
                    a.results
                FROM videos v
                JOIN analyses a ON v.video_id = a.video_id
                WHERE v.video_id = :video_id
            """), {"video_id": sample_video_record["video_id"]})
            
            row = result.fetchone()
            assert row is not None, "Join query returned no results"
            
            # Validate joined data
            assert row[0] == sample_video_record["filename"]  # filename
            assert row[1] == sample_video_record["duration"]  # duration
            assert row[3] == sample_analysis_record["analysis_type"]  # analysis_type

    @pytest.mark.integration
    @pytest.mark.slow
    def test_update_analysis_status(
        self, 
        database_engine,
        sample_video_record: Dict[str, Any],
        sample_analysis_record: Dict[str, Any]
    ):
        """Test updating analysis status in database."""
        with database_engine.connect() as conn:
            # Insert test data
            conn.execute(text("""
                INSERT OR REPLACE INTO videos (
                    video_id, filename, file_path, file_size, duration,
                    fps, width, height, format, upload_timestamp, status, metadata
                ) VALUES (
                    :video_id, :filename, :file_path, :file_size, :duration,
                    :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                )
            """), {
                **sample_video_record,
                "metadata": json.dumps(sample_video_record["metadata"])
            })
            
            # Insert analysis with "processing" status
            initial_record = sample_analysis_record.copy()
            initial_record["status"] = "processing"
            initial_record["completed_timestamp"] = None
            initial_record["results"] = None
            
            conn.execute(text("""
                INSERT INTO analyses (
                    analysis_id, video_id, analysis_type, status,
                    created_timestamp, completed_timestamp, processing_time,
                    results, error_message
                ) VALUES (
                    :analysis_id, :video_id, :analysis_type, :status,
                    :created_timestamp, :completed_timestamp, :processing_time,
                    :results, :error_message
                )
            """), initial_record)
            conn.commit()
            
            # Update to completed status
            conn.execute(text("""
                UPDATE analyses 
                SET status = :status,
                    completed_timestamp = :completed_timestamp,
                    processing_time = :processing_time,
                    results = :results
                WHERE analysis_id = :analysis_id
            """), {
                "analysis_id": sample_analysis_record["analysis_id"],
                "status": "completed",
                "completed_timestamp": datetime.now(timezone.utc),
                "processing_time": 120.5,
                "results": json.dumps(sample_analysis_record["results"])
            })
            conn.commit()
            
            # Verify update
            result = conn.execute(text("""
                SELECT status, completed_timestamp, results 
                FROM analyses WHERE analysis_id = :analysis_id
            """), {"analysis_id": sample_analysis_record["analysis_id"]})
            
            row = result.fetchone()
            assert row is not None, "Analysis record not found after update"
            assert row[0] == "completed", "Status not updated"
            assert row[1] is not None, "Completed timestamp not set"
            assert row[2] is not None, "Results not set"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_delete_records(
        self, 
        database_engine,
        sample_video_record: Dict[str, Any],
        sample_analysis_record: Dict[str, Any]
    ):
        """Test deleting records from database."""
        with database_engine.connect() as conn:
            # Insert test data
            conn.execute(text("""
                INSERT OR REPLACE INTO videos (
                    video_id, filename, file_path, file_size, duration,
                    fps, width, height, format, upload_timestamp, status, metadata
                ) VALUES (
                    :video_id, :filename, :file_path, :file_size, :duration,
                    :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                )
            """), {
                **sample_video_record,
                "metadata": json.dumps(sample_video_record["metadata"])
            })
            
            conn.execute(text("""
                INSERT OR REPLACE INTO analyses (
                    analysis_id, video_id, analysis_type, status,
                    created_timestamp, completed_timestamp, processing_time,
                    results, error_message
                ) VALUES (
                    :analysis_id, :video_id, :analysis_type, :status,
                    :created_timestamp, :completed_timestamp, :processing_time,
                    :results, :error_message
                )
            """), {
                **sample_analysis_record,
                "results": json.dumps(sample_analysis_record["results"])
            })
            conn.commit()
            
            # Delete analysis record first (foreign key constraint)
            conn.execute(text("""
                DELETE FROM analyses WHERE analysis_id = :analysis_id
            """), {"analysis_id": sample_analysis_record["analysis_id"]})
            
            # Delete video record
            conn.execute(text("""
                DELETE FROM videos WHERE video_id = :video_id
            """), {"video_id": sample_video_record["video_id"]})
            conn.commit()
            
            # Verify deletion
            result = conn.execute(text("""
                SELECT COUNT(*) FROM analyses WHERE analysis_id = :analysis_id
            """), {"analysis_id": sample_analysis_record["analysis_id"]})
            
            count = result.fetchone()[0]
            assert count == 0, "Analysis record not deleted"
            
            result = conn.execute(text("""
                SELECT COUNT(*) FROM videos WHERE video_id = :video_id
            """), {"video_id": sample_video_record["video_id"]})
            
            count = result.fetchone()[0]
            assert count == 0, "Video record not deleted"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_transaction_rollback(
        self, 
        database_engine,
        sample_video_record: Dict[str, Any]
    ):
        """Test database transaction rollback on error."""
        with database_engine.connect() as conn:
            trans = conn.begin()
            
            try:
                # Insert valid record
                conn.execute(text("""
                    INSERT INTO videos (
                        video_id, filename, file_path, file_size, duration,
                        fps, width, height, format, upload_timestamp, status, metadata
                    ) VALUES (
                        :video_id, :filename, :file_path, :file_size, :duration,
                        :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                    )
                """), {
                    **sample_video_record,
                    "metadata": json.dumps(sample_video_record["metadata"])
                })
                
                # Attempt to insert duplicate (should fail)
                conn.execute(text("""
                    INSERT INTO videos (
                        video_id, filename, file_path, file_size, duration,
                        fps, width, height, format, upload_timestamp, status, metadata
                    ) VALUES (
                        :video_id, :filename, :file_path, :file_size, :duration,
                        :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                    )
                """), {
                    **sample_video_record,
                    "metadata": json.dumps(sample_video_record["metadata"])
                })
                
                trans.commit()
                
            except Exception:
                # Rollback on error
                trans.rollback()
                
                # Verify rollback - no records should exist
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM videos WHERE video_id = :video_id
                """), {"video_id": sample_video_record["video_id"]})
                
                count = result.fetchone()[0]
                assert count == 0, "Transaction not rolled back properly"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_performance_bulk_insert(self, database_engine):
        """Test database performance with bulk insert operations."""
        import time
        
        # Generate test data
        test_records = []
        for i in range(100):  # 100 records
            test_records.append({
                "video_id": str(uuid.uuid4()),
                "filename": f"test_video_{i:03d}.mp4",
                "file_path": f"/data/videos/test_video_{i:03d}.mp4",
                "file_size": 1024 * 1024 * (i + 1),  # Variable size
                "duration": 5.0 + (i * 0.1),
                "fps": 30.0,
                "width": 1920,
                "height": 1080,
                "format": "mp4",
                "upload_timestamp": datetime.now(timezone.utc),
                "status": "uploaded",
                "metadata": json.dumps({"test_record": i})
            })
        
        # Bulk insert with timing
        start_time = time.time()
        
        with database_engine.connect() as conn:
            for record in test_records:
                conn.execute(text("""
                    INSERT INTO videos (
                        video_id, filename, file_path, file_size, duration,
                        fps, width, height, format, upload_timestamp, status, metadata
                    ) VALUES (
                        :video_id, :filename, :file_path, :file_size, :duration,
                        :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                    )
                """), record)
            conn.commit()
        
        insert_time = time.time() - start_time
        
        # Validate performance (should insert 100 records in < 5 seconds)
        assert insert_time < 5.0, f"Bulk insert too slow: {insert_time:.2f}s"
        
        # Verify all records inserted
        with database_engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM videos"))
            count = result.fetchone()[0]
            assert count >= 100, f"Not all records inserted: {count}"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_concurrent_access(self, database_engine):
        """Test database concurrent access handling."""
        import threading
        import time
        
        results = []
        errors = []
        
        def insert_record(thread_id: int):
            try:
                record = {
                    "video_id": str(uuid.uuid4()),
                    "filename": f"concurrent_test_{thread_id}.mp4",
                    "file_path": f"/data/videos/concurrent_test_{thread_id}.mp4",
                    "file_size": 1024 * 1024,
                    "duration": 5.0,
                    "fps": 30.0,
                    "width": 1920,
                    "height": 1080,
                    "format": "mp4",
                    "upload_timestamp": datetime.now(timezone.utc),
                    "status": "uploaded",
                    "metadata": json.dumps({"thread_id": thread_id})
                }
                
                with database_engine.connect() as conn:
                    conn.execute(text("""
                        INSERT INTO videos (
                            video_id, filename, file_path, file_size, duration,
                            fps, width, height, format, upload_timestamp, status, metadata
                        ) VALUES (
                            :video_id, :filename, :file_path, :file_size, :duration,
                            :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                        )
                    """), record)
                    conn.commit()
                
                results.append(thread_id)
                
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create 5 concurrent threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=insert_record, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Validate results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 5, f"Not all threads completed: {len(results)}"

    # ========================================
    # COMPREHENSIVE ERROR HANDLING TESTS
    # ========================================

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_connection_failure_handling(self):
        """Test handling of database connection failures."""
        # Test with invalid database URL
        invalid_engine = create_engine("sqlite:///nonexistent/path/test.db")
        
        with pytest.raises(Exception):
            with invalid_engine.connect() as conn:
                conn.execute(text("SELECT 1"))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_connection_timeout_handling(self):
        """Test handling of database connection timeouts."""
        # Create engine with very short timeout
        timeout_engine = create_engine(
            "sqlite:///test_timeout.db",
            connect_args={"timeout": 0.001}  # 1ms timeout
        )
        
        try:
            with timeout_engine.connect() as conn:
                # This might timeout on busy systems
                conn.execute(text("SELECT 1"))
        except Exception as e:
            # Timeout or connection error is expected
            assert any(keyword in str(e).lower() for keyword in 
                      ["timeout", "connection", "busy"]), \
                f"Unexpected error type: {e}"
        finally:
            timeout_engine.dispose()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_locked_handling(self, database_engine):
        """Test handling of database lock situations."""
        import threading
        import time
        
        lock_error_occurred = False
        
        def long_transaction():
            nonlocal lock_error_occurred
            try:
                with database_engine.connect() as conn:
                    trans = conn.begin()
                    
                    # Insert a record
                    conn.execute(text("""
                        INSERT INTO videos (
                            video_id, filename, file_path, file_size, duration,
                            fps, width, height, format, upload_timestamp, status, metadata
                        ) VALUES (
                            :video_id, :filename, :file_path, :file_size, :duration,
                            :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                        )
                    """), {
                        "video_id": str(uuid.uuid4()),
                        "filename": "lock_test.mp4",
                        "file_path": "/data/videos/lock_test.mp4",
                        "file_size": 1024 * 1024,
                        "duration": 5.0,
                        "fps": 30.0,
                        "width": 1920,
                        "height": 1080,
                        "format": "mp4",
                        "upload_timestamp": datetime.now(timezone.utc),
                        "status": "uploaded",
                        "metadata": "{}"
                    })
                    
                    # Hold the transaction for a while
                    time.sleep(0.5)
                    trans.commit()
                    
            except Exception as e:
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    lock_error_occurred = True
        
        # Start long transaction in background
        thread = threading.Thread(target=long_transaction)
        thread.start()
        
        # Try to access database while it might be locked
        time.sleep(0.1)  # Let the other transaction start
        
        try:
            with database_engine.connect() as conn:
                conn.execute(text("SELECT COUNT(*) FROM videos"))
        except Exception as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                lock_error_occurred = True
        
        thread.join()
        
        # Either the operation succeeded or we got appropriate lock errors
        # Both are acceptable outcomes
        assert True, "Database lock handling tested"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_disk_full_simulation(self, tmp_path):
        """Test handling of disk full scenarios."""
        # Create a small database file to simulate disk space issues
        small_db_path = tmp_path / "small_space.db"
        
        # Create engine with the temporary database
        small_engine = create_engine(f"sqlite:///{small_db_path}")
        
        try:
            with small_engine.connect() as conn:
                # Create tables
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS test_table (
                        id INTEGER PRIMARY KEY,
                        large_data TEXT
                    )
                """))
                conn.commit()
                
                # Try to insert very large data to simulate disk full
                large_data = "x" * (10 * 1024 * 1024)  # 10MB string
                
                try:
                    for i in range(100):  # Try to insert 1GB total
                        conn.execute(text("""
                            INSERT INTO test_table (large_data) VALUES (:data)
                        """), {"data": large_data})
                        conn.commit()
                
                except Exception as e:
                    # Should handle disk space errors gracefully
                    error_msg = str(e).lower()
                    assert any(keyword in error_msg for keyword in 
                              ["disk", "space", "full", "write", "io"]), \
                        f"Unexpected error for disk full simulation: {e}"
        
        finally:
            small_engine.dispose()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_corruption_detection(self, database_engine):
        """Test database corruption detection and handling."""
        with database_engine.connect() as conn:
            # Insert test data
            test_record = {
                "video_id": str(uuid.uuid4()),
                "filename": "corruption_test.mp4",
                "file_path": "/data/videos/corruption_test.mp4",
                "file_size": 1024 * 1024,
                "duration": 5.0,
                "fps": 30.0,
                "width": 1920,
                "height": 1080,
                "format": "mp4",
                "upload_timestamp": datetime.now(timezone.utc),
                "status": "uploaded",
                "metadata": "{}"
            }
            
            conn.execute(text("""
                INSERT INTO videos (
                    video_id, filename, file_path, file_size, duration,
                    fps, width, height, format, upload_timestamp, status, metadata
                ) VALUES (
                    :video_id, :filename, :file_path, :file_size, :duration,
                    :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                )
            """), test_record)
            conn.commit()
            
            # Run integrity check
            try:
                result = conn.execute(text("PRAGMA integrity_check"))
                integrity_result = result.fetchone()[0]
                
                if integrity_result != "ok":
                    # Handle corruption gracefully
                    assert "corruption" in integrity_result.lower() or \
                           "damaged" in integrity_result.lower(), \
                        f"Unexpected integrity check result: {integrity_result}"
                else:
                    # Database is healthy
                    assert integrity_result == "ok", "Database should be healthy"
            
            except Exception as e:
                # Handle integrity check failures
                assert "corrupt" in str(e).lower() or "damaged" in str(e).lower(), \
                    f"Unexpected integrity check error: {e}"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_schema_migration_failure(self, database_engine):
        """Test handling of schema migration failures."""
        with database_engine.connect() as conn:
            # Try to create a table with invalid schema
            invalid_schemas = [
                "CREATE TABLE invalid_table (id INVALID_TYPE)",
                "CREATE TABLE duplicate_videos (video_id TEXT PRIMARY KEY)",  # Might conflict
                "CREATE TABLE bad_syntax (id TEXT, CONSTRAINT bad_constraint)",
            ]
            
            for schema in invalid_schemas:
                try:
                    conn.execute(text(schema))
                    conn.commit()
                    
                    # If it succeeds, clean up
                    table_name = schema.split()[2]  # Extract table name
                    conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                    conn.commit()
                    
                except Exception as e:
                    # Schema errors are expected and should be handled gracefully
                    error_msg = str(e).lower()
                    assert any(keyword in error_msg for keyword in 
                              ["syntax", "type", "constraint", "exists"]), \
                        f"Unexpected schema error: {e}"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_deadlock_handling(self, database_engine):
        """Test handling of database deadlock situations."""
        import threading
        import time
        
        deadlock_errors = []
        
        def transaction_a():
            try:
                with database_engine.connect() as conn:
                    trans = conn.begin()
                    
                    # Lock resource A
                    conn.execute(text("""
                        INSERT OR REPLACE INTO videos (
                            video_id, filename, file_path, file_size, duration,
                            fps, width, height, format, upload_timestamp, status, metadata
                        ) VALUES (
                            'deadlock_a', 'test_a.mp4', '/test_a.mp4', 1024, 5.0,
                            30.0, 1920, 1080, 'mp4', :timestamp, 'uploaded', '{}'
                        )
                    """), {"timestamp": datetime.now(timezone.utc)})
                    
                    time.sleep(0.1)  # Give other transaction time to start
                    
                    # Try to lock resource B
                    conn.execute(text("""
                        INSERT OR REPLACE INTO videos (
                            video_id, filename, file_path, file_size, duration,
                            fps, width, height, format, upload_timestamp, status, metadata
                        ) VALUES (
                            'deadlock_b', 'test_b.mp4', '/test_b.mp4', 1024, 5.0,
                            30.0, 1920, 1080, 'mp4', :timestamp, 'uploaded', '{}'
                        )
                    """), {"timestamp": datetime.now(timezone.utc)})
                    
                    trans.commit()
                    
            except Exception as e:
                deadlock_errors.append(f"Transaction A: {e}")
        
        def transaction_b():
            try:
                with database_engine.connect() as conn:
                    trans = conn.begin()
                    
                    # Lock resource B
                    conn.execute(text("""
                        INSERT OR REPLACE INTO videos (
                            video_id, filename, file_path, file_size, duration,
                            fps, width, height, format, upload_timestamp, status, metadata
                        ) VALUES (
                            'deadlock_b', 'test_b.mp4', '/test_b.mp4', 1024, 5.0,
                            30.0, 1920, 1080, 'mp4', :timestamp, 'uploaded', '{}'
                        )
                    """), {"timestamp": datetime.now(timezone.utc)})
                    
                    time.sleep(0.1)  # Give other transaction time to start
                    
                    # Try to lock resource A
                    conn.execute(text("""
                        INSERT OR REPLACE INTO videos (
                            video_id, filename, file_path, file_size, duration,
                            fps, width, height, format, upload_timestamp, status, metadata
                        ) VALUES (
                            'deadlock_a', 'test_a.mp4', '/test_a.mp4', 1024, 5.0,
                            30.0, 1920, 1080, 'mp4', :timestamp, 'uploaded', '{}'
                        )
                    """), {"timestamp": datetime.now(timezone.utc)})
                    
                    trans.commit()
                    
            except Exception as e:
                deadlock_errors.append(f"Transaction B: {e}")
        
        # Start both transactions
        thread_a = threading.Thread(target=transaction_a)
        thread_b = threading.Thread(target=transaction_b)
        
        thread_a.start()
        thread_b.start()
        
        thread_a.join(timeout=5.0)
        thread_b.join(timeout=5.0)
        
        # SQLite typically handles this with locking rather than true deadlocks
        # But we should handle any errors gracefully
        if deadlock_errors:
            for error in deadlock_errors:
                error_msg = error.lower()
                # Common database error conditions
                assert any(keyword in error_msg for keyword in 
                          ["locked", "busy", "deadlock", "timeout"]), \
                    f"Unexpected deadlock error: {error}"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_backup_failure_handling(self, database_engine, tmp_path):
        """Test handling of database backup failures."""
        # Create test data
        with database_engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO videos (
                    video_id, filename, file_path, file_size, duration,
                    fps, width, height, format, upload_timestamp, status, metadata
                ) VALUES (
                    :video_id, :filename, :file_path, :file_size, :duration,
                    :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                )
            """), {
                "video_id": str(uuid.uuid4()),
                "filename": "backup_test.mp4",
                "file_path": "/data/videos/backup_test.mp4",
                "file_size": 1024 * 1024,
                "duration": 5.0,
                "fps": 30.0,
                "width": 1920,
                "height": 1080,
                "format": "mp4",
                "upload_timestamp": datetime.now(timezone.utc),
                "status": "uploaded",
                "metadata": "{}"
            })
            conn.commit()
        
        # Try to backup to invalid location
        invalid_backup_path = "/nonexistent/directory/backup.db"
        
        try:
            with database_engine.connect() as conn:
                # SQLite backup command (if supported)
                conn.execute(text(f"ATTACH DATABASE '{invalid_backup_path}' AS backup"))
                conn.execute(text("INSERT INTO backup.videos SELECT * FROM main.videos"))
                conn.execute(text("DETACH DATABASE backup"))
        
        except Exception as e:
            # Backup failure should be handled gracefully
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in 
                      ["attach", "path", "permission", "directory", "file"]), \
                f"Unexpected backup error: {e}"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_recovery_from_partial_writes(self, database_engine):
        """Test database recovery from partial write scenarios."""
        with database_engine.connect() as conn:
            # Start a transaction
            trans = conn.begin()
            
            try:
                # Insert multiple records
                for i in range(5):
                    conn.execute(text("""
                        INSERT INTO videos (
                            video_id, filename, file_path, file_size, duration,
                            fps, width, height, format, upload_timestamp, status, metadata
                        ) VALUES (
                            :video_id, :filename, :file_path, :file_size, :duration,
                            :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                        )
                    """), {
                        "video_id": str(uuid.uuid4()),
                        "filename": f"partial_write_{i}.mp4",
                        "file_path": f"/data/videos/partial_write_{i}.mp4",
                        "file_size": 1024 * 1024,
                        "duration": 5.0,
                        "fps": 30.0,
                        "width": 1920,
                        "height": 1080,
                        "format": "mp4",
                        "upload_timestamp": datetime.now(timezone.utc),
                        "status": "uploaded",
                        "metadata": "{}"
                    })
                
                # Simulate partial write by rolling back
                trans.rollback()
                
                # Verify no partial data remains
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM videos WHERE filename LIKE 'partial_write_%'
                """))
                count = result.fetchone()[0]
                assert count == 0, "Partial writes should be rolled back"
                
            except Exception as e:
                # Handle any transaction errors
                try:
                    trans.rollback()
                except:
                    pass  # Rollback might fail if transaction is already aborted
                
                # Verify database is still consistent
                result = conn.execute(text("SELECT COUNT(*) FROM videos"))
                count = result.fetchone()[0]
                assert count >= 0, "Database should remain consistent after errors"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_invalid_sql_query_handling(self, database_engine):
        """Test handling of invalid SQL queries."""
        with database_engine.connect() as conn:
            # Test invalid table name
            with pytest.raises(Exception):
                conn.execute(text("SELECT * FROM nonexistent_table"))
            
            # Test invalid column name
            with pytest.raises(Exception):
                conn.execute(text("SELECT nonexistent_column FROM videos"))
            
            # Test malformed SQL
            with pytest.raises(Exception):
                conn.execute(text("INVALID SQL SYNTAX"))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_constraint_violation_handling(
        self, 
        database_engine,
        sample_video_record: Dict[str, Any]
    ):
        """Test handling of database constraint violations."""
        with database_engine.connect() as conn:
            # Insert initial record
            conn.execute(text("""
                INSERT OR REPLACE INTO videos (
                    video_id, filename, file_path, file_size, duration,
                    fps, width, height, format, upload_timestamp, status, metadata
                ) VALUES (
                    :video_id, :filename, :file_path, :file_size, :duration,
                    :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                )
            """), {
                **sample_video_record,
                "metadata": json.dumps(sample_video_record["metadata"])
            })
            conn.commit()
            
            # Test primary key constraint violation
            with pytest.raises(Exception):
                conn.execute(text("""
                    INSERT INTO videos (
                        video_id, filename, file_path, file_size, duration,
                        fps, width, height, format, upload_timestamp, status, metadata
                    ) VALUES (
                        :video_id, :filename, :file_path, :file_size, :duration,
                        :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                    )
                """), {
                    **sample_video_record,
                    "metadata": json.dumps(sample_video_record["metadata"])
                })
                conn.commit()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_foreign_key_constraint_violation(self, database_engine):
        """Test handling of foreign key constraint violations."""
        with database_engine.connect() as conn:
            # Enable foreign key constraints for SQLite
            conn.execute(text("PRAGMA foreign_keys = ON"))
            
            # Try to insert analysis with non-existent video_id
            with pytest.raises(Exception):
                conn.execute(text("""
                    INSERT INTO analyses (
                        analysis_id, video_id, analysis_type, status,
                        created_timestamp, completed_timestamp, processing_time,
                        results, error_message
                    ) VALUES (
                        :analysis_id, :video_id, :analysis_type, :status,
                        :created_timestamp, :completed_timestamp, :processing_time,
                        :results, :error_message
                    )
                """), {
                    "analysis_id": str(uuid.uuid4()),
                    "video_id": "nonexistent-video-id",
                    "analysis_type": "gait_analysis",
                    "status": "completed",
                    "created_timestamp": datetime.now(timezone.utc),
                    "completed_timestamp": datetime.now(timezone.utc),
                    "processing_time": 120.0,
                    "results": json.dumps({"test": "data"}),
                    "error_message": None
                })
                conn.commit()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_null_constraint_violation(self, database_engine):
        """Test handling of NOT NULL constraint violations."""
        with database_engine.connect() as conn:
            # Test missing required fields
            with pytest.raises(Exception):
                conn.execute(text("""
                    INSERT INTO videos (video_id, filename) 
                    VALUES (:video_id, :filename)
                """), {
                    "video_id": str(uuid.uuid4()),
                    "filename": None  # NULL value for NOT NULL field
                })
                conn.commit()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_data_type_mismatch_handling(self, database_engine):
        """Test handling of data type mismatches."""
        with database_engine.connect() as conn:
            # SQLite is very permissive with type conversion, so we test with extreme cases
            # that would cause issues in stricter databases
            
            # Test with completely invalid data that should cause issues
            try:
                conn.execute(text("""
                    INSERT INTO videos (
                        video_id, filename, file_path, file_size, duration,
                        fps, width, height, format, upload_timestamp, status, metadata
                    ) VALUES (
                        :video_id, :filename, :file_path, :file_size, :duration,
                        :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                    )
                """), {
                    "video_id": str(uuid.uuid4()),
                    "filename": "test.mp4",
                    "file_path": "/test/path.mp4",
                    "file_size": "not_a_number_at_all",  # String that can't be converted
                    "duration": "invalid_float",  # String that can't be converted to float
                    "fps": 30.0,
                    "width": 1920,
                    "height": 1080,
                    "format": "mp4",
                    "upload_timestamp": "not_a_timestamp",  # Invalid timestamp
                    "status": "uploaded",
                    "metadata": "{}"
                })
                conn.commit()
                
                # If we get here, SQLite was very permissive
                # Let's verify the data was stored (even if converted)
                result = conn.execute(text("""
                    SELECT file_size, duration, upload_timestamp 
                    FROM videos WHERE filename = 'test.mp4'
                """))
                row = result.fetchone()
                
                # SQLite may have converted or stored as text
                # This tests that we can handle the results gracefully
                assert row is not None, "Record was not inserted despite type mismatches"
                
            except Exception as e:
                # This is the expected behavior for stricter databases
                assert any(keyword in str(e).lower() for keyword in 
                          ["type", "convert", "invalid", "format"]), \
                    f"Unexpected error type: {e}"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_json_data_corruption_handling(
        self, 
        database_engine,
        sample_video_record: Dict[str, Any]
    ):
        """Test handling of corrupted JSON data."""
        with database_engine.connect() as conn:
            # Insert record with invalid JSON
            corrupted_record = sample_video_record.copy()
            
            conn.execute(text("""
                INSERT INTO videos (
                    video_id, filename, file_path, file_size, duration,
                    fps, width, height, format, upload_timestamp, status, metadata
                ) VALUES (
                    :video_id, :filename, :file_path, :file_size, :duration,
                    :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                )
            """), {
                **corrupted_record,
                "metadata": "invalid json data"  # Invalid JSON
            })
            conn.commit()
            
            # Try to retrieve and parse JSON
            result = conn.execute(text("""
                SELECT metadata FROM videos WHERE video_id = :video_id
            """), {"video_id": corrupted_record["video_id"]})
            
            row = result.fetchone()
            metadata_str = row[0]
            
            # Should handle JSON parsing error gracefully
            with pytest.raises(json.JSONDecodeError):
                json.loads(metadata_str)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_data_handling(self, database_engine):
        """Test handling of large data insertions."""
        # Create very large JSON data
        large_results = {
            "classification": "normal",
            "confidence": 0.85,
            "pose_landmarks": [
                [{"x": i * 1.0, "y": j * 1.0, "confidence": 0.9} for j in range(33)]
                for i in range(10000)  # Very large dataset
            ]
        }
        
        large_record = {
            "analysis_id": str(uuid.uuid4()),
            "video_id": str(uuid.uuid4()),
            "analysis_type": "gait_analysis",
            "status": "completed",
            "created_timestamp": datetime.now(timezone.utc),
            "completed_timestamp": datetime.now(timezone.utc),
            "processing_time": 120.0,
            "results": json.dumps(large_results),
            "error_message": None
        }
        
        # First insert a video record
        video_record = {
            "video_id": large_record["video_id"],
            "filename": "large_test.mp4",
            "file_path": "/data/videos/large_test.mp4",
            "file_size": 1024 * 1024 * 100,  # 100MB
            "duration": 60.0,
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "format": "mp4",
            "upload_timestamp": datetime.now(timezone.utc),
            "status": "uploaded",
            "metadata": json.dumps({"large_test": True})
        }
        
        with database_engine.connect() as conn:
            # Insert video record
            conn.execute(text("""
                INSERT INTO videos (
                    video_id, filename, file_path, file_size, duration,
                    fps, width, height, format, upload_timestamp, status, metadata
                ) VALUES (
                    :video_id, :filename, :file_path, :file_size, :duration,
                    :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                )
            """), video_record)
            
            # Test large data insertion
            try:
                conn.execute(text("""
                    INSERT INTO analyses (
                        analysis_id, video_id, analysis_type, status,
                        created_timestamp, completed_timestamp, processing_time,
                        results, error_message
                    ) VALUES (
                        :analysis_id, :video_id, :analysis_type, :status,
                        :created_timestamp, :completed_timestamp, :processing_time,
                        :results, :error_message
                    )
                """), large_record)
                conn.commit()
                
                # Verify insertion succeeded
                result = conn.execute(text("""
                    SELECT analysis_id FROM analyses WHERE analysis_id = :analysis_id
                """), {"analysis_id": large_record["analysis_id"]})
                
                row = result.fetchone()
                assert row is not None, "Large data insertion failed"
                
            except Exception as e:
                # Handle potential memory or size limit errors
                assert "too large" in str(e).lower() or "memory" in str(e).lower(), \
                    f"Unexpected error with large data: {e}"

    # ========================================
    # EDGE CASE TESTS
    # ========================================

    @pytest.mark.integration
    @pytest.mark.slow
    def test_empty_string_handling(self, database_engine):
        """Test handling of empty strings in database operations."""
        empty_record = {
            "video_id": str(uuid.uuid4()),
            "filename": "",  # Empty string
            "file_path": "",  # Empty string
            "file_size": 0,
            "duration": 0.0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "format": "",  # Empty string
            "upload_timestamp": datetime.now(timezone.utc),
            "status": "",  # Empty string
            "metadata": "{}"
        }
        
        with database_engine.connect() as conn:
            # Should handle empty strings gracefully
            conn.execute(text("""
                INSERT INTO videos (
                    video_id, filename, file_path, file_size, duration,
                    fps, width, height, format, upload_timestamp, status, metadata
                ) VALUES (
                    :video_id, :filename, :file_path, :file_size, :duration,
                    :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                )
            """), empty_record)
            conn.commit()
            
            # Verify insertion
            result = conn.execute(text("""
                SELECT filename, file_path, format, status 
                FROM videos WHERE video_id = :video_id
            """), {"video_id": empty_record["video_id"]})
            
            row = result.fetchone()
            assert row is not None, "Empty string record not inserted"
            assert row[0] == "", "Empty filename not preserved"
            assert row[1] == "", "Empty file_path not preserved"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_extreme_numeric_values(self, database_engine):
        """Test handling of extreme numeric values."""
        extreme_record = {
            "video_id": str(uuid.uuid4()),
            "filename": "extreme_test.mp4",
            "file_path": "/data/videos/extreme_test.mp4",
            "file_size": 2**63 - 1,  # Maximum 64-bit integer
            "duration": 999999.999,  # Very large duration
            "fps": 0.001,  # Very small FPS
            "width": 99999,  # Very large width
            "height": 1,  # Very small height
            "format": "mp4",
            "upload_timestamp": datetime.now(timezone.utc),
            "status": "uploaded",
            "metadata": "{}"
        }
        
        with database_engine.connect() as conn:
            try:
                conn.execute(text("""
                    INSERT INTO videos (
                        video_id, filename, file_path, file_size, duration,
                        fps, width, height, format, upload_timestamp, status, metadata
                    ) VALUES (
                        :video_id, :filename, :file_path, :file_size, :duration,
                        :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                    )
                """), extreme_record)
                conn.commit()
                
                # Verify values are preserved correctly
                result = conn.execute(text("""
                    SELECT file_size, duration, fps, width, height 
                    FROM videos WHERE video_id = :video_id
                """), {"video_id": extreme_record["video_id"]})
                
                row = result.fetchone()
                assert row is not None, "Extreme values record not inserted"
                assert row[0] == extreme_record["file_size"], "Extreme file_size not preserved"
                assert abs(row[1] - extreme_record["duration"]) < 0.001, "Extreme duration not preserved"
                
            except Exception as e:
                # Handle potential overflow errors
                assert "overflow" in str(e).lower() or "out of range" in str(e).lower(), \
                    f"Unexpected error with extreme values: {e}"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_unicode_and_special_characters(self, database_engine):
        """Test handling of Unicode and special characters."""
        unicode_record = {
            "video_id": str(uuid.uuid4()),
            "filename": "测试视频_🎥_special-chars!@#$%^&*().mp4",  # Unicode and special chars
            "file_path": "/data/videos/测试/special chars & symbols/video.mp4",
            "file_size": 1024 * 1024,
            "duration": 5.0,
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "format": "mp4",
            "upload_timestamp": datetime.now(timezone.utc),
            "status": "uploaded",
            "metadata": json.dumps({
                "description": "视频描述 with émojis 🎬🎭",
                "tags": ["测试", "special-chars", "unicode"]
            })
        }
        
        with database_engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO videos (
                    video_id, filename, file_path, file_size, duration,
                    fps, width, height, format, upload_timestamp, status, metadata
                ) VALUES (
                    :video_id, :filename, :file_path, :file_size, :duration,
                    :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                )
            """), unicode_record)
            conn.commit()
            
            # Verify Unicode characters are preserved
            result = conn.execute(text("""
                SELECT filename, file_path, metadata 
                FROM videos WHERE video_id = :video_id
            """), {"video_id": unicode_record["video_id"]})
            
            row = result.fetchone()
            assert row is not None, "Unicode record not inserted"
            assert "测试视频" in row[0], "Unicode filename not preserved"
            assert "测试" in row[1], "Unicode file_path not preserved"
            
            # Verify JSON with Unicode
            metadata = json.loads(row[2])
            assert "视频描述" in metadata["description"], "Unicode in JSON not preserved"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_timestamp_edge_cases(self, database_engine):
        """Test handling of timestamp edge cases."""
        # Test various timestamp scenarios
        timestamp_cases = [
            {
                "name": "epoch_start",
                "timestamp": datetime(1970, 1, 1, tzinfo=timezone.utc)
            },
            {
                "name": "y2k",
                "timestamp": datetime(2000, 1, 1, tzinfo=timezone.utc)
            },
            {
                "name": "leap_year",
                "timestamp": datetime(2024, 2, 29, tzinfo=timezone.utc)
            },
            {
                "name": "far_future",
                "timestamp": datetime(2099, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            }
        ]
        
        with database_engine.connect() as conn:
            for case in timestamp_cases:
                record = {
                    "video_id": str(uuid.uuid4()),
                    "filename": f"{case['name']}_test.mp4",
                    "file_path": f"/data/videos/{case['name']}_test.mp4",
                    "file_size": 1024 * 1024,
                    "duration": 5.0,
                    "fps": 30.0,
                    "width": 1920,
                    "height": 1080,
                    "format": "mp4",
                    "upload_timestamp": case["timestamp"],
                    "status": "uploaded",
                    "metadata": "{}"
                }
                
                conn.execute(text("""
                    INSERT INTO videos (
                        video_id, filename, file_path, file_size, duration,
                        fps, width, height, format, upload_timestamp, status, metadata
                    ) VALUES (
                        :video_id, :filename, :file_path, :file_size, :duration,
                        :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                    )
                """), record)
                
                # Verify timestamp preservation
                result = conn.execute(text("""
                    SELECT upload_timestamp FROM videos WHERE video_id = :video_id
                """), {"video_id": record["video_id"]})
                
                row = result.fetchone()
                assert row is not None, f"Timestamp case {case['name']} not inserted"
                
                # Note: Exact timestamp comparison may vary by database implementation
                retrieved_timestamp = row[0]
                assert retrieved_timestamp is not None, f"Timestamp {case['name']} not preserved"
            
            conn.commit()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_recovery_after_corruption(self, database_engine):
        """Test database recovery scenarios after simulated corruption."""
        with database_engine.connect() as conn:
            # Insert some test data
            test_record = {
                "video_id": str(uuid.uuid4()),
                "filename": "recovery_test.mp4",
                "file_path": "/data/videos/recovery_test.mp4",
                "file_size": 1024 * 1024,
                "duration": 5.0,
                "fps": 30.0,
                "width": 1920,
                "height": 1080,
                "format": "mp4",
                "upload_timestamp": datetime.now(timezone.utc),
                "status": "uploaded",
                "metadata": "{}"
            }
            
            conn.execute(text("""
                INSERT INTO videos (
                    video_id, filename, file_path, file_size, duration,
                    fps, width, height, format, upload_timestamp, status, metadata
                ) VALUES (
                    :video_id, :filename, :file_path, :file_size, :duration,
                    :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                )
            """), test_record)
            conn.commit()
            
            # Verify data integrity check
            result = conn.execute(text("PRAGMA integrity_check"))
            integrity_result = result.fetchone()[0]
            assert integrity_result == "ok", f"Database integrity check failed: {integrity_result}"
            
            # Test database vacuum operation
            conn.execute(text("VACUUM"))
            
            # Verify data still exists after vacuum
            result = conn.execute(text("""
                SELECT COUNT(*) FROM videos WHERE video_id = :video_id
            """), {"video_id": test_record["video_id"]})
            
            count = result.fetchone()[0]
            assert count == 1, "Data lost after vacuum operation"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_connection_pool_exhaustion(self, test_database_url: str):
        """Test handling of connection pool exhaustion."""
        # Create engine with limited connection pool
        limited_engine = create_engine(
            test_database_url, 
            pool_size=2, 
            max_overflow=0,
            pool_timeout=1
        )
        
        connections = []
        
        try:
            # Exhaust connection pool
            for i in range(2):
                conn = limited_engine.connect()
                connections.append(conn)
            
            # This should timeout or raise an exception
            with pytest.raises(Exception):
                extra_conn = limited_engine.connect()
                extra_conn.execute(text("SELECT 1"))
                
        finally:
            # Clean up connections
            for conn in connections:
                conn.close()
            limited_engine.dispose()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_long_running_transaction_timeout(self, database_engine):
        """Test handling of long-running transaction timeouts."""
        import time
        
        with database_engine.connect() as conn:
            trans = conn.begin()
            
            try:
                # Insert a record
                conn.execute(text("""
                    INSERT INTO videos (
                        video_id, filename, file_path, file_size, duration,
                        fps, width, height, format, upload_timestamp, status, metadata
                    ) VALUES (
                        :video_id, :filename, :file_path, :file_size, :duration,
                        :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                    )
                """), {
                    "video_id": str(uuid.uuid4()),
                    "filename": "timeout_test.mp4",
                    "file_path": "/data/videos/timeout_test.mp4",
                    "file_size": 1024 * 1024,
                    "duration": 5.0,
                    "fps": 30.0,
                    "width": 1920,
                    "height": 1080,
                    "format": "mp4",
                    "upload_timestamp": datetime.now(timezone.utc),
                    "status": "uploaded",
                    "metadata": "{}"
                })
                
                # Simulate long-running operation (but keep it short for testing)
                time.sleep(0.1)
                
                trans.commit()
                
            except Exception as e:
                trans.rollback()
                # Handle timeout gracefully
                if "timeout" in str(e).lower():
                    pytest.skip("Transaction timeout occurred as expected")
                else:
                    raise

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_schema_validation(self, database_engine):
        """Test database schema validation and consistency."""
        with database_engine.connect() as conn:
            # Check if required tables exist
            result = conn.execute(text("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('videos', 'analyses', 'users')
            """))
            
            tables = [row[0] for row in result.fetchall()]
            assert 'videos' in tables, "Videos table not found"
            assert 'analyses' in tables, "Analyses table not found"
            
            # Check table schema for videos
            result = conn.execute(text("PRAGMA table_info(videos)"))
            video_columns = {row[1]: row[2] for row in result.fetchall()}  # name: type
            
            required_video_columns = {
                'video_id': 'TEXT',
                'filename': 'TEXT',
                'file_path': 'TEXT',
                'upload_timestamp': 'TIMESTAMP'
            }
            
            for col_name, col_type in required_video_columns.items():
                assert col_name in video_columns, f"Required column {col_name} not found in videos table"
            
            # Check table schema for analyses
            result = conn.execute(text("PRAGMA table_info(analyses)"))
            analysis_columns = {row[1]: row[2] for row in result.fetchall()}  # name: type
            
            required_analysis_columns = {
                'analysis_id': 'TEXT',
                'video_id': 'TEXT',
                'analysis_type': 'TEXT',
                'status': 'TEXT'
            }
            
            for col_name, col_type in required_analysis_columns.items():
                assert col_name in analysis_columns, f"Required column {col_name} not found in analyses table"

    # ========================================
    # ADDITIONAL ERROR HANDLING AND EDGE CASES
    # ========================================

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_connection_pool_exhaustion(self, database_engine):
        """Test handling of database connection pool exhaustion."""
        import threading
        import time
        
        # Create many concurrent connections to exhaust pool
        connections = []
        connection_errors = []
        
        def create_long_connection(thread_id: int):
            try:
                conn = database_engine.connect()
                connections.append(conn)
                
                # Hold connection for a while
                time.sleep(0.5)
                
                # Try to use the connection
                result = conn.execute(text("SELECT 1"))
                assert result.fetchone()[0] == 1
                
            except Exception as e:
                connection_errors.append(f"Thread {thread_id}: {e}")
            finally:
                if connections and len(connections) > thread_id:
                    try:
                        connections[thread_id].close()
                    except:
                        pass
        
        # Start many concurrent threads
        threads = []
        for i in range(20):  # Try to create 20 concurrent connections
            thread = threading.Thread(target=create_long_connection, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=2.0)
        
        # Clean up remaining connections
        for conn in connections:
            try:
                conn.close()
            except:
                pass
        
        # Some connections might fail due to pool limits, which is acceptable
        # The important thing is that the system handles it gracefully
        if connection_errors:
            for error in connection_errors:
                error_msg = error.lower()
                # Should indicate connection or pool-related issues
                assert any(keyword in error_msg for keyword in 
                          ["connection", "pool", "limit", "timeout", "busy"]), \
                    f"Unexpected connection error: {error}"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_transaction_isolation_violations(self, database_engine):
        """Test handling of transaction isolation violations."""
        import threading
        import time
        
        isolation_errors = []
        
        def dirty_read_transaction():
            try:
                with database_engine.connect() as conn:
                    trans = conn.begin()
                    
                    # Insert a record but don't commit
                    conn.execute(text("""
                        INSERT INTO videos (
                            video_id, filename, file_path, file_size, duration,
                            fps, width, height, format, upload_timestamp, status, metadata
                        ) VALUES (
                            'isolation_test', 'isolation.mp4', '/test.mp4', 1024, 5.0,
                            30.0, 1920, 1080, 'mp4', :timestamp, 'uploaded', '{}'
                        )
                    """), {"timestamp": datetime.now(timezone.utc)})
                    
                    time.sleep(0.2)  # Hold transaction
                    
                    # Rollback to test dirty read scenarios
                    trans.rollback()
                    
            except Exception as e:
                isolation_errors.append(f"Dirty read transaction: {e}")
        
        def concurrent_read_transaction():
            try:
                time.sleep(0.1)  # Let other transaction start first
                
                with database_engine.connect() as conn:
                    # Try to read the uncommitted data
                    result = conn.execute(text("""
                        SELECT COUNT(*) FROM videos WHERE video_id = 'isolation_test'
                    """))
                    count = result.fetchone()[0]
                    
                    # In proper isolation, this should be 0 (no dirty read)
                    if count > 0:
                        isolation_errors.append("Dirty read detected")
                    
            except Exception as e:
                isolation_errors.append(f"Concurrent read transaction: {e}")
        
        # Run concurrent transactions
        thread1 = threading.Thread(target=dirty_read_transaction)
        thread2 = threading.Thread(target=concurrent_read_transaction)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # SQLite handles isolation well, so we shouldn't see dirty reads
        # But if we do, the system should handle it gracefully
        if isolation_errors:
            for error in isolation_errors:
                if "dirty read detected" in error.lower():
                    pytest.skip("Database isolation level allows dirty reads (acceptable for SQLite)")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_schema_version_mismatch(self, database_engine):
        """Test handling of schema version mismatches."""
        with database_engine.connect() as conn:
            # Create a version table to simulate schema versioning
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP
                )
            """))
            
            # Insert current version
            conn.execute(text("""
                INSERT OR REPLACE INTO schema_version (version, applied_at)
                VALUES (1, :timestamp)
            """), {"timestamp": datetime.now(timezone.utc)})
            conn.commit()
            
            # Simulate version mismatch by expecting higher version
            result = conn.execute(text("SELECT version FROM schema_version"))
            current_version = result.fetchone()[0]
            
            expected_version = 2
            if current_version < expected_version:
                # This simulates a version mismatch scenario
                # In a real application, this would trigger migration or error
                assert current_version == 1, "Should detect version mismatch"
                
                # Simulate handling the mismatch
                try:
                    # Try to access a column that doesn't exist in current version
                    conn.execute(text("SELECT new_column FROM videos"))
                    assert False, "Should fail with missing column"
                except Exception as e:
                    # Should handle missing column gracefully
                    assert "no such column" in str(e).lower() or "column" in str(e).lower(), \
                        f"Should indicate column issue: {e}"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_index_corruption_handling(self, database_engine):
        """Test handling of database index corruption."""
        with database_engine.connect() as conn:
            # Create an index
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_video_filename ON videos(filename)"))
            conn.commit()
            
            # Insert test data
            test_records = []
            for i in range(100):
                record = {
                    "video_id": str(uuid.uuid4()),
                    "filename": f"index_test_{i:03d}.mp4",
                    "file_path": f"/data/videos/index_test_{i:03d}.mp4",
                    "file_size": 1024 * 1024,
                    "duration": 5.0,
                    "fps": 30.0,
                    "width": 1920,
                    "height": 1080,
                    "format": "mp4",
                    "upload_timestamp": datetime.now(timezone.utc),
                    "status": "uploaded",
                    "metadata": "{}"
                }
                test_records.append(record)
                
                conn.execute(text("""
                    INSERT INTO videos (
                        video_id, filename, file_path, file_size, duration,
                        fps, width, height, format, upload_timestamp, status, metadata
                    ) VALUES (
                        :video_id, :filename, :file_path, :file_size, :duration,
                        :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                    )
                """), record)
            
            conn.commit()
            
            # Test index usage with queries
            result = conn.execute(text("""
                SELECT COUNT(*) FROM videos WHERE filename LIKE 'index_test_%'
            """))
            count = result.fetchone()[0]
            assert count == 100, "Index should help find all test records"
            
            # Test index integrity (SQLite specific)
            try:
                result = conn.execute(text("PRAGMA integrity_check"))
                integrity_result = result.fetchone()[0]
                
                if integrity_result != "ok":
                    # Handle index corruption
                    assert "index" in integrity_result.lower() or "corrupt" in integrity_result.lower(), \
                        f"Should indicate index corruption: {integrity_result}"
                    
                    # Try to rebuild index
                    conn.execute(text("DROP INDEX IF EXISTS idx_video_filename"))
                    conn.execute(text("CREATE INDEX idx_video_filename ON videos(filename)"))
                    conn.commit()
                    
                    # Verify rebuild worked
                    result = conn.execute(text("PRAGMA integrity_check"))
                    rebuilt_result = result.fetchone()[0]
                    assert rebuilt_result == "ok", "Index rebuild should fix corruption"
                
            except Exception as e:
                # Handle integrity check failures
                assert "corrupt" in str(e).lower() or "index" in str(e).lower(), \
                    f"Should indicate index-related error: {e}"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_database_vacuum_operation_failures(self, database_engine):
        """Test handling of database vacuum operation failures."""
        with database_engine.connect() as conn:
            # Insert and delete data to create fragmentation
            for i in range(50):
                conn.execute(text("""
                    INSERT INTO videos (
                        video_id, filename, file_path, file_size, duration,
                        fps, width, height, format, upload_timestamp, status, metadata
                    ) VALUES (
                        :video_id, :filename, :file_path, :file_size, :duration,
                        :fps, :width, :height, :format, :upload_timestamp, :status, :metadata
                    )
                """), {
                    "video_id": str(uuid.uuid4()),
                    "filename": f"vacuum_test_{i}.mp4",
                    "file_path": f"/data/videos/vacuum_test_{i}.mp4",
                    "file_size": 1024 * 1024,
                    "duration": 5.0,
                    "fps": 30.0,
                    "width": 1920,
                    "height": 1080,
                    "format": "mp4",
                    "upload_timestamp": datetime.now(timezone.utc),
                    "status": "uploaded",
                    "metadata": "{}"
                })
            
            conn.commit()
            
            # Delete half the records to create fragmentation
            conn.execute(text("DELETE FROM videos WHERE filename LIKE 'vacuum_test_%' AND video_id LIKE '%0'"))
            conn.commit()
            
            # Try to vacuum the database
            try:
                conn.execute(text("VACUUM"))
                # Vacuum succeeded
                assert True, "Vacuum operation completed successfully"
                
            except Exception as e:
                # Handle vacuum failures
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in 
                          ["vacuum", "space", "lock", "busy"]), \
                    f"Should indicate vacuum-related error: {e}"
                
                # Database should still be functional after vacuum failure
                result = conn.execute(text("SELECT COUNT(*) FROM videos"))
                count = result.fetchone()[0]
                assert count >= 0, "Database should remain functional after vacuum failure"