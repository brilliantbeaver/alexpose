"""
Tests for Training Data Manager

This module provides comprehensive tests for the training data management system,
including unit tests, property-based tests, and integration tests.

Feature: gavd-gait-analysis
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from ambient.data.training_manager import (
    GAVDDatasetLoader,
    TrainingDataManager,
)


class TestGAVDDatasetLoader:
    """Tests for GAVD dataset loader."""
    
    def test_initialization(self):
        """Test loader initialization."""
        loader = GAVDDatasetLoader()
        assert loader is not None
        assert loader.data_loader is not None
    
    def test_load_dataset_file_not_found(self):
        """Test loading non-existent dataset."""
        loader = GAVDDatasetLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_dataset(Path("nonexistent.csv"))
    
    def test_get_dataset_info_before_load(self):
        """Test getting info before loading dataset."""
        loader = GAVDDatasetLoader()
        
        with pytest.raises(ValueError, match="Dataset not loaded"):
            loader.get_dataset_info()


class TestTrainingDataManager:
    """Tests for training data manager."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def manager(self, temp_data_dir):
        """Create training data manager."""
        return TrainingDataManager(data_dir=temp_data_dir)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'seq': ['seq1', 'seq1', 'seq2', 'seq2', 'seq3'],
            'frame_num': [1, 2, 1, 2, 1],
            'gait_pat': ['Normal', 'Normal', 'Abnormal', 'Abnormal', 'Normal'],
            'bbox': [
                {'left': 100, 'top': 100, 'width': 50, 'height': 100},
                {'left': 105, 'top': 100, 'width': 50, 'height': 100},
                {'left': 200, 'top': 150, 'width': 60, 'height': 110},
                {'left': 205, 'top': 150, 'width': 60, 'height': 110},
                {'left': 300, 'top': 200, 'width': 55, 'height': 105},
            ]
        })
    
    def test_initialization(self, manager, temp_data_dir):
        """Test manager initialization."""
        assert manager is not None
        assert manager.data_dir == temp_data_dir
        assert temp_data_dir.exists()
        assert len(manager.datasets) == 0
    
    def test_register_dataset_loader(self, manager):
        """Test registering custom dataset loader."""
        loader = GAVDDatasetLoader()
        manager.register_dataset_loader("CustomGAVD", loader)
        
        assert "CustomGAVD" in manager.loaders
    
    def test_list_datasets_empty(self, manager):
        """Test listing datasets when none loaded."""
        datasets = manager.list_datasets()
        assert datasets == []
    
    def test_get_dataset_not_loaded(self, manager):
        """Test getting non-existent dataset."""
        with pytest.raises(ValueError, match="not loaded"):
            manager.get_dataset("nonexistent")
    
    def test_create_balanced_dataset(self, manager, sample_dataframe):
        """Test creating balanced dataset."""
        # Add dataset to manager
        manager.datasets["test_dataset"] = sample_dataframe
        manager.dataset_info["test_dataset"] = type('obj', (object,), {
            'dataset_name': 'test_dataset',
            'version': '1.0',
            'source_path': Path('test.csv'),
            'sample_count': len(sample_dataframe),
            'condition_distribution': sample_dataframe['gait_pat'].value_counts().to_dict(),
            'metadata': {}
        })()
        
        balanced_df, distribution = manager.create_balanced_dataset(
            "test_dataset",
            label_column="gait_pat",
            normal_label="Normal",
            balance_ratio=1.0
        )
        
        assert len(balanced_df) > 0
        assert distribution["Normal"] == distribution["abnormal"]
    
    def test_organize_by_condition(self, manager, sample_dataframe):
        """Test organizing dataset by condition."""
        manager.datasets["test_dataset"] = sample_dataframe
        manager.dataset_info["test_dataset"] = type('obj', (object,), {
            'dataset_name': 'test_dataset',
            'version': '1.0',
            'source_path': Path('test.csv'),
            'sample_count': len(sample_dataframe),
            'condition_distribution': {},
            'metadata': {}
        })()
        
        organized = manager.organize_by_condition("test_dataset", condition_column="gait_pat")
        
        assert "Normal" in organized
        assert "Abnormal" in organized
        assert len(organized["Normal"]) == 3
        assert len(organized["Abnormal"]) == 2
    
    def test_save_and_load_dataset_version(self, manager, sample_dataframe):
        """Test saving and loading dataset versions."""
        manager.datasets["test_dataset"] = sample_dataframe
        manager.dataset_info["test_dataset"] = type('obj', (object,), {
            'dataset_name': 'test_dataset',
            'version': '1.0',
            'source_path': Path('test.csv'),
            'sample_count': len(sample_dataframe),
            'condition_distribution': {},
            'metadata': {}
        })()
        
        # Save version
        version_dir = manager.save_dataset_version(
            "test_dataset",
            "v1.0",
            metadata={"description": "Test version"}
        )
        
        assert version_dir.exists()
        assert (version_dir / "data.pkl").exists()
        assert (version_dir / "metadata.json").exists()
        
        # Load version
        loaded_df, metadata = manager.load_dataset_version("test_dataset", "v1.0")
        
        assert len(loaded_df) == len(sample_dataframe)
        assert metadata["version"] == "v1.0"
        assert metadata["description"] == "Test version"
    
    def test_export_for_training(self, manager, sample_dataframe, temp_data_dir):
        """Test exporting dataset for training."""
        manager.datasets["test_dataset"] = sample_dataframe
        manager.dataset_info["test_dataset"] = type('obj', (object,), {
            'dataset_name': 'test_dataset',
            'version': '1.0',
            'source_path': Path('test.csv'),
            'sample_count': len(sample_dataframe),
            'condition_distribution': {},
            'metadata': {},
            'created_at': '2025-01-01T00:00:00'
        })()
        
        # Export as pickle
        output_path = temp_data_dir / "export.pkl"
        exported_path = manager.export_for_training(
            "test_dataset",
            output_path,
            format="pickle",
            include_metadata=True
        )
        
        assert exported_path.exists()
        assert (temp_data_dir / "export_metadata.json").exists()
    
    def test_get_statistics(self, manager, sample_dataframe):
        """Test getting dataset statistics."""
        manager.datasets["test_dataset"] = sample_dataframe
        manager.dataset_info["test_dataset"] = type('obj', (object,), {
            'dataset_name': 'test_dataset',
            'version': '1.0',
            'source_path': Path('test.csv'),
            'sample_count': len(sample_dataframe),
            'condition_distribution': {'Normal': 3, 'Abnormal': 2},
            'metadata': {}
        })()
        
        stats = manager.get_statistics("test_dataset")
        
        assert stats["dataset_name"] == "test_dataset"
        assert stats["total_samples"] == 5
        assert "columns" in stats
        assert "condition_distribution" in stats
        assert stats["unique_sequences"] == 3


# Property-Based Tests

@given(
    sample_count=st.integers(min_value=10, max_value=100),
    normal_ratio=st.floats(min_value=0.3, max_value=0.7)
)
@settings(max_examples=50, deadline=None)
def test_balanced_dataset_property(sample_count, normal_ratio):
    """
    Feature: gavd-gait-analysis, Property 13: Dataset Balance Verification
    
    For any training dataset preparation, the normal/abnormal class distribution
    should be within acceptable balance thresholds.
    """
    # Create sample dataset
    normal_count = int(sample_count * normal_ratio)
    abnormal_count = sample_count - normal_count
    
    df = pd.DataFrame({
        'seq': [f'seq{i}' for i in range(sample_count)],
        'frame_num': list(range(sample_count)),
        'gait_pat': ['Normal'] * normal_count + ['Abnormal'] * abnormal_count,
        'bbox': [{'left': 100, 'top': 100, 'width': 50, 'height': 100}] * sample_count
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TrainingDataManager(data_dir=Path(tmpdir))
        manager.datasets["test"] = df
        manager.dataset_info["test"] = type('obj', (object,), {
            'dataset_name': 'test',
            'version': '1.0',
            'source_path': Path('test.csv'),
            'sample_count': len(df),
            'condition_distribution': df['gait_pat'].value_counts().to_dict(),
            'metadata': {}
        })()
        
        balanced_df, distribution = manager.create_balanced_dataset(
            "test",
            label_column="gait_pat",
            normal_label="Normal",
            balance_ratio=1.0
        )
        
        # Verify balance
        assert distribution["Normal"] > 0
        assert distribution["abnormal"] > 0
        
        # Check balance ratio (should be 1.0 or close to it)
        balance_ratio = distribution["Normal"] / distribution["abnormal"]
        assert 0.9 <= balance_ratio <= 1.1, f"Balance ratio {balance_ratio} outside acceptable range"


@given(
    condition_count=st.integers(min_value=2, max_value=5),
    samples_per_condition=st.integers(min_value=5, max_value=20)
)
@settings(max_examples=30, deadline=None)
def test_organize_by_condition_property(condition_count, samples_per_condition):
    """
    Feature: gavd-gait-analysis, Property 12: Training Data Organization by Condition
    
    For any GAVD dataset with condition labels, sequences should be correctly
    grouped by their associated health conditions.
    """
    # Create dataset with multiple conditions
    conditions = [f'Condition{i}' for i in range(condition_count)]
    
    data = []
    for condition in conditions:
        for i in range(samples_per_condition):
            data.append({
                'seq': f'{condition}_seq{i}',
                'frame_num': i,
                'gait_pat': condition,
                'bbox': {'left': 100, 'top': 100, 'width': 50, 'height': 100}
            })
    
    df = pd.DataFrame(data)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TrainingDataManager(data_dir=Path(tmpdir))
        manager.datasets["test"] = df
        manager.dataset_info["test"] = type('obj', (object,), {
            'dataset_name': 'test',
            'version': '1.0',
            'source_path': Path('test.csv'),
            'sample_count': len(df),
            'condition_distribution': {},
            'metadata': {}
        })()
        
        organized = manager.organize_by_condition("test", condition_column="gait_pat")
        
        # Verify all conditions are present
        assert len(organized) == condition_count
        
        # Verify each condition has correct number of samples
        for condition in conditions:
            assert condition in organized
            assert len(organized[condition]) == samples_per_condition
            
            # Verify all samples in group have correct condition
            assert all(organized[condition]['gait_pat'] == condition)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
