"""
Tests for Storage Manager

This module provides comprehensive tests for the storage management system,
including unit tests, property-based tests, and integration tests.

Feature: gavd-gait-analysis
"""

import json
import pickle
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from ambient.storage.storage_manager import (
    JSONStorageBackend,
    PickleStorageBackend,
    StorageManager,
)


class TestJSONStorageBackend:
    """Tests for JSON storage backend."""
    
    def test_initialization(self):
        """Test backend initialization."""
        backend = JSONStorageBackend(indent=2, compress=False)
        assert backend.indent == 2
        assert backend.compress is False
    
    def test_save_and_load_dict(self):
        """Test saving and loading dictionary."""
        backend = JSONStorageBackend()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"key": "value", "number": 42}
            
            saved_path = backend.save(data, path)
            assert saved_path.exists()
            
            loaded_data = backend.load(saved_path)
            assert loaded_data == data
    
    def test_save_and_load_dataframe(self):
        """Test saving and loading DataFrame."""
        backend = JSONStorageBackend()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            
            saved_path = backend.save(df, path)
            assert saved_path.exists()
            
            loaded_data = backend.load(saved_path)
            assert isinstance(loaded_data, list)
    
    def test_save_with_compression(self):
        """Test saving with compression."""
        backend = JSONStorageBackend(compress=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"key": "value"}
            
            saved_path = backend.save(data, path)
            assert saved_path.suffix == '.gz'
            assert saved_path.exists()
            
            loaded_data = backend.load(saved_path)
            assert loaded_data == data
    
    def test_exists(self):
        """Test checking file existence."""
        backend = JSONStorageBackend()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            assert not backend.exists(path)
            
            backend.save({"key": "value"}, path)
            assert backend.exists(path)
    
    def test_delete(self):
        """Test deleting file."""
        backend = JSONStorageBackend()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            backend.save({"key": "value"}, path)
            
            assert backend.exists(path)
            assert backend.delete(path)
            assert not backend.exists(path)


class TestPickleStorageBackend:
    """Tests for pickle storage backend."""
    
    def test_initialization(self):
        """Test backend initialization."""
        backend = PickleStorageBackend(protocol=pickle.HIGHEST_PROTOCOL, compress=False)
        assert backend.protocol == pickle.HIGHEST_PROTOCOL
        assert backend.compress is False
    
    def test_save_and_load_object(self):
        """Test saving and loading Python object."""
        backend = PickleStorageBackend()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pkl"
            data = {"key": "value", "list": [1, 2, 3], "nested": {"a": 1}}
            
            saved_path = backend.save(data, path)
            assert saved_path.exists()
            
            loaded_data = backend.load(saved_path)
            assert loaded_data == data
    
    def test_save_and_load_dataframe(self):
        """Test saving and loading DataFrame."""
        backend = PickleStorageBackend()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pkl"
            df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            
            saved_path = backend.save(df, path)
            assert saved_path.exists()
            
            loaded_df = backend.load(saved_path)
            pd.testing.assert_frame_equal(loaded_df, df)
    
    def test_save_with_compression(self):
        """Test saving with compression."""
        backend = PickleStorageBackend(compress=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pkl"
            data = {"key": "value"}
            
            saved_path = backend.save(data, path)
            assert saved_path.suffix == '.gz'
            assert saved_path.exists()
            
            loaded_data = backend.load(saved_path)
            assert loaded_data == data


class TestStorageManager:
    """Tests for storage manager."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def manager(self, temp_storage_dir):
        """Create storage manager."""
        return StorageManager(base_dir=temp_storage_dir)
    
    def test_initialization(self, manager, temp_storage_dir):
        """Test manager initialization."""
        assert manager is not None
        assert manager.base_dir == temp_storage_dir
        assert temp_storage_dir.exists()
        assert 'json' in manager.backends
        assert 'pickle' in manager.backends
    
    def test_save_and_load_with_pickle(self, manager):
        """Test saving and loading with pickle backend."""
        data = {"key": "value", "number": 42}
        
        saved_path = manager.save(data, "test_data", backend="pickle")
        assert saved_path.exists()
        
        loaded_data = manager.load("test_data", backend="pickle")
        assert loaded_data == data
    
    def test_save_and_load_with_json(self, manager):
        """Test saving and loading with JSON backend."""
        data = {"key": "value", "number": 42}
        
        saved_path = manager.save(data, "test_data", backend="json")
        assert saved_path.exists()
        
        loaded_data = manager.load("test_data", backend="json")
        assert loaded_data == data
    
    def test_save_with_metadata(self, manager):
        """Test saving with metadata."""
        data = {"key": "value"}
        metadata = {"description": "Test data", "version": "1.0"}
        
        saved_path = manager.save(data, "test_data", backend="pickle", metadata=metadata)
        assert saved_path.exists()
        
        loaded_metadata = manager.get_metadata("test_data")
        assert loaded_metadata is not None
        assert loaded_metadata["description"] == "Test data"
        assert "saved_at" in loaded_metadata
    
    def test_exists(self, manager):
        """Test checking data existence."""
        assert not manager.exists("nonexistent")
        
        manager.save({"key": "value"}, "test_data", backend="pickle")
        assert manager.exists("test_data")
    
    def test_delete(self, manager):
        """Test deleting data."""
        manager.save({"key": "value"}, "test_data", backend="pickle")
        assert manager.exists("test_data")
        
        assert manager.delete("test_data")
        assert not manager.exists("test_data")
    
    def test_list_keys(self, manager):
        """Test listing storage keys."""
        manager.save({"key": "value1"}, "data1", backend="pickle")
        manager.save({"key": "value2"}, "data2", backend="json")
        
        keys = manager.list_keys()
        assert "data1" in keys
        assert "data2" in keys
    
    def test_get_storage_stats(self, manager):
        """Test getting storage statistics."""
        manager.save({"key": "value1"}, "data1", backend="pickle")
        manager.save({"key": "value2"}, "data2", backend="json")
        
        stats = manager.get_storage_stats()
        
        assert stats["total_files"] >= 2
        assert stats["total_size_mb"] > 0
        assert "backend_counts" in stats


# Property-Based Tests

@given(
    data_dict=st.dictionaries(
        keys=st.text(min_size=1, max_size=10),
        values=st.one_of(st.integers(), st.floats(allow_nan=False), st.text()),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=30, deadline=None)
def test_storage_roundtrip_property(data_dict):
    """
    Feature: gavd-gait-analysis, Property 11: Data Management Integrity
    
    For any data stored and retrieved, the system should maintain data integrity
    with exact roundtrip preservation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = StorageManager(base_dir=Path(tmpdir))
        
        # Save data
        manager.save(data_dict, "test_data", backend="pickle")
        
        # Load data
        loaded_data = manager.load("test_data", backend="pickle")
        
        # Verify exact match
        assert loaded_data == data_dict


@given(
    key_count=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=20, deadline=None)
def test_multiple_storage_operations_property(key_count):
    """
    Feature: gavd-gait-analysis, Property 11: Data Management Integrity
    
    For any number of storage operations, the system should maintain consistency
    and allow retrieval of all stored items.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = StorageManager(base_dir=Path(tmpdir))
        
        # Store multiple items
        stored_keys = []
        for i in range(key_count):
            key = f"data_{i}"
            data = {"index": i, "value": f"value_{i}"}
            manager.save(data, key, backend="pickle")
            stored_keys.append(key)
        
        # Verify all items exist
        for key in stored_keys:
            assert manager.exists(key)
        
        # Verify list_keys returns all keys
        listed_keys = manager.list_keys()
        for key in stored_keys:
            assert key in listed_keys
        
        # Verify all items can be loaded
        for i, key in enumerate(stored_keys):
            loaded_data = manager.load(key, backend="pickle")
            assert loaded_data["index"] == i
            assert loaded_data["value"] == f"value_{i}"


@given(
    backend_choice=st.sampled_from(['json', 'pickle'])
)
@settings(max_examples=10, deadline=None)
def test_backend_consistency_property(backend_choice):
    """
    Feature: gavd-gait-analysis, Property 11: Data Management Integrity
    
    For any storage backend, data should be stored and retrieved consistently
    with proper format handling.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = StorageManager(base_dir=Path(tmpdir))
        
        # Create test data
        data = {
            "string": "test",
            "number": 42,
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": 2}
        }
        
        # Save with chosen backend
        manager.save(data, "test_data", backend=backend_choice)
        
        # Load with same backend
        loaded_data = manager.load("test_data", backend=backend_choice)
        
        # Verify data integrity
        assert loaded_data["string"] == data["string"]
        assert loaded_data["number"] == data["number"]
        assert loaded_data["list"] == data["list"]
        assert loaded_data["nested"] == data["nested"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
