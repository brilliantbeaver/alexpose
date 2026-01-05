"""
Tests for Dataset Versioning

This module provides comprehensive tests for dataset versioning and provenance tracking.

Feature: gavd-gait-analysis
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from ambient.data.versioning import (
    DatasetVersion,
    DatasetVersionManager,
    ProvenanceRecord,
)


class TestDatasetVersionManager:
    """Tests for dataset version manager."""
    
    @pytest.fixture
    def temp_versions_dir(self):
        """Create temporary versions directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def manager(self, temp_versions_dir):
        """Create version manager."""
        return DatasetVersionManager(versions_dir=temp_versions_dir)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': ['x', 'y', 'z']
        })
    
    def test_initialization(self, manager, temp_versions_dir):
        """Test manager initialization."""
        assert manager is not None
        assert manager.versions_dir == temp_versions_dir
        assert temp_versions_dir.exists()
    
    def test_create_version(self, manager, sample_dataframe):
        """Test creating a dataset version."""
        version = manager.create_version(
            dataset_name="test_dataset",
            data=sample_dataframe,
            version_number="1.0.0",
            description="Initial version"
        )
        
        assert version is not None
        assert version.version_number == "1.0.0"
        assert version.sample_count == 3
        assert version.checksum != ""
    
    def test_load_version(self, manager, sample_dataframe):
        """Test loading a dataset version."""
        manager.create_version(
            dataset_name="test_dataset",
            data=sample_dataframe,
            version_number="1.0.0"
        )
        
        loaded_df, version = manager.load_version("test_dataset", "1.0.0")
        
        pd.testing.assert_frame_equal(loaded_df, sample_dataframe)
        assert version.version_number == "1.0.0"
    
    def test_list_versions(self, manager, sample_dataframe):
        """Test listing dataset versions."""
        manager.create_version("test_dataset", sample_dataframe, "1.0.0")
        manager.create_version("test_dataset", sample_dataframe, "1.1.0")
        manager.create_version("test_dataset", sample_dataframe, "2.0.0")
        
        versions = manager.list_versions("test_dataset")
        
        assert len(versions) == 3
        version_numbers = [v.version_number for v in versions]
        assert "1.0.0" in version_numbers
        assert "1.1.0" in version_numbers
        assert "2.0.0" in version_numbers
    
    def test_get_latest_version(self, manager, sample_dataframe):
        """Test getting latest version."""
        manager.create_version("test_dataset", sample_dataframe, "1.0.0")
        manager.create_version("test_dataset", sample_dataframe, "1.1.0")
        
        latest = manager.get_latest_version("test_dataset")
        
        assert latest is not None
        # Latest should be most recently created
        assert latest.version_number in ["1.0.0", "1.1.0"]
    
    def test_compare_versions(self, manager, sample_dataframe):
        """Test comparing two versions."""
        manager.create_version("test_dataset", sample_dataframe, "1.0.0")
        
        # Create modified version
        modified_df = sample_dataframe.copy()
        modified_df['d'] = [7, 8, 9]
        manager.create_version("test_dataset", modified_df, "1.1.0")
        
        comparison = manager.compare_versions("test_dataset", "1.0.0", "1.1.0")
        
        assert "version1" in comparison
        assert "version2" in comparison
        assert "checksum_match" in comparison
        assert comparison["checksum_match"] is False  # Different data
    
    def test_delete_version(self, manager, sample_dataframe):
        """Test deleting a version."""
        manager.create_version("test_dataset", sample_dataframe, "1.0.0")
        
        versions_before = manager.list_versions("test_dataset")
        assert len(versions_before) == 1
        
        success = manager.delete_version("test_dataset", "1.0.0", force=True)
        assert success is True
        
        versions_after = manager.list_versions("test_dataset")
        assert len(versions_after) == 0
    
    def test_get_version_lineage(self, manager, sample_dataframe):
        """Test getting version lineage."""
        manager.create_version("test_dataset", sample_dataframe, "1.0.0")
        manager.create_version(
            "test_dataset",
            sample_dataframe,
            "1.1.0",
            parent_version="1.0.0"
        )
        manager.create_version(
            "test_dataset",
            sample_dataframe,
            "1.2.0",
            parent_version="1.1.0"
        )
        
        lineage = manager.get_version_lineage("test_dataset", "1.2.0")
        
        assert len(lineage) == 3
        assert lineage[0].version_number == "1.0.0"
        assert lineage[1].version_number == "1.1.0"
        assert lineage[2].version_number == "1.2.0"
    
    def test_validate_version_integrity(self, manager, sample_dataframe):
        """Test validating version integrity."""
        manager.create_version("test_dataset", sample_dataframe, "1.0.0")
        
        validation = manager.validate_version_integrity("test_dataset", "1.0.0")
        
        assert validation["valid"] is True
        assert validation["checksum_valid"] is True
        assert validation["sample_count_valid"] is True
    
    def test_export_version_history(self, manager, sample_dataframe, temp_versions_dir):
        """Test exporting version history."""
        manager.create_version("test_dataset", sample_dataframe, "1.0.0")
        manager.create_version("test_dataset", sample_dataframe, "1.1.0")
        
        output_path = temp_versions_dir / "history.json"
        result_path = manager.export_version_history("test_dataset", output_path)
        
        assert result_path.exists()
        assert result_path == output_path


# Property-Based Tests

@given(
    version_count=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=10, deadline=None)
def test_multiple_versions_property(version_count):
    """
    Feature: gavd-gait-analysis, Property 18: Training Data Management
    
    For any number of dataset versions, the system should maintain versioning
    with provenance tracking.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = DatasetVersionManager(versions_dir=Path(tmpdir))
        
        df = pd.DataFrame({'a': [1, 2, 3]})
        
        # Create multiple versions
        for i in range(version_count):
            manager.create_version(
                dataset_name="test_dataset",
                data=df,
                version_number=f"1.{i}.0"
            )
        
        # Verify all versions exist
        versions = manager.list_versions("test_dataset")
        assert len(versions) == version_count
        
        # Verify each version can be loaded
        for i in range(version_count):
            loaded_df, version = manager.load_version("test_dataset", f"1.{i}.0")
            assert len(loaded_df) == 3
            assert version.version_number == f"1.{i}.0"


@given(
    sample_count=st.integers(min_value=1, max_value=50)
)
@settings(max_examples=20, deadline=None)
def test_version_integrity_property(sample_count):
    """
    Feature: gavd-gait-analysis, Property 18: Training Data Management
    
    For any dataset version, the system should maintain data integrity
    with accurate checksums and sample counts.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = DatasetVersionManager(versions_dir=Path(tmpdir))
        
        # Create dataset with variable size
        df = pd.DataFrame({
            'index': list(range(sample_count)),
            'value': [f'value_{i}' for i in range(sample_count)]
        })
        
        # Create version
        version = manager.create_version(
            dataset_name="test_dataset",
            data=df,
            version_number="1.0.0"
        )
        
        # Verify sample count
        assert version.sample_count == sample_count
        
        # Validate integrity
        validation = manager.validate_version_integrity("test_dataset", "1.0.0")
        assert validation["valid"] is True
        assert validation["sample_count_valid"] is True
        assert validation["expected_samples"] == sample_count
        assert validation["actual_samples"] == sample_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
