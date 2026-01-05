"""
Tests for Data Augmentation

This module provides comprehensive tests for data augmentation capabilities,
including unit tests, property-based tests, and integration tests.

Feature: gavd-gait-analysis
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from ambient.core.frame import Frame, FrameSequence
from ambient.data.augmentation import (
    AugmentationPipeline,
    DataAugmenter,
    HorizontalFlipAugmentation,
    NoiseInjectionAugmentation,
    SpatialScalingAugmentation,
    TemporalSpeedAugmentation,
)


class TestTemporalSpeedAugmentation:
    """Tests for temporal speed augmentation."""
    
    def test_initialization(self):
        """Test augmentation initialization."""
        aug = TemporalSpeedAugmentation(speed_range=(0.8, 1.2))
        assert aug.speed_range == (0.8, 1.2)
    
    def test_augment_sequence(self):
        """Test augmenting a frame sequence."""
        # Create sample sequence
        frames = [Frame.empty(shape=(480, 640, 3)) for _ in range(30)]
        sequence = FrameSequence(frames=frames, sequence_id="test_seq")
        
        aug = TemporalSpeedAugmentation(speed_range=(0.8, 1.2), random_seed=42)
        augmented = aug.augment(sequence)
        
        assert augmented is not None
        assert len(augmented.frames) != len(sequence.frames)  # Length should change
        assert 'augmentation' in augmented.metadata
    
    def test_get_parameters(self):
        """Test getting augmentation parameters."""
        aug = TemporalSpeedAugmentation(speed_range=(0.9, 1.1))
        params = aug.get_parameters()
        
        assert params['type'] == 'temporal_speed'
        assert params['speed_range'] == (0.9, 1.1)


class TestSpatialScalingAugmentation:
    """Tests for spatial scaling augmentation."""
    
    def test_initialization(self):
        """Test augmentation initialization."""
        aug = SpatialScalingAugmentation(scale_range=(0.9, 1.1))
        assert aug.scale_range == (0.9, 1.1)
    
    def test_augment_dataframe(self):
        """Test augmenting DataFrame with bounding boxes."""
        df = pd.DataFrame({
            'seq': ['seq1', 'seq1'],
            'frame_num': [1, 2],
            'bbox': [
                {'left': 100, 'top': 100, 'width': 50, 'height': 100},
                {'left': 105, 'top': 100, 'width': 50, 'height': 100}
            ]
        })
        
        aug = SpatialScalingAugmentation(scale_range=(0.9, 1.1), random_seed=42)
        augmented = aug.augment(df)
        
        assert len(augmented) == len(df)
        assert 'bbox' in augmented.columns
        
        # Check that bboxes were scaled
        original_bbox = df.iloc[0]['bbox']
        augmented_bbox = augmented.iloc[0]['bbox']
        assert augmented_bbox['width'] != original_bbox['width']


class TestNoiseInjectionAugmentation:
    """Tests for noise injection augmentation."""
    
    def test_initialization(self):
        """Test augmentation initialization."""
        aug = NoiseInjectionAugmentation(noise_level=0.01)
        assert aug.noise_level == 0.01
    
    def test_augment_dataframe(self):
        """Test adding noise to DataFrame."""
        df = pd.DataFrame({
            'seq': ['seq1', 'seq1'],
            'frame_num': [1, 2],
            'bbox': [
                {'left': 100, 'top': 100, 'width': 50, 'height': 100},
                {'left': 105, 'top': 100, 'width': 50, 'height': 100}
            ]
        })
        
        aug = NoiseInjectionAugmentation(noise_level=0.01, random_seed=42)
        augmented = aug.augment(df)
        
        assert len(augmented) == len(df)
        
        # Check that noise was added
        original_bbox = df.iloc[0]['bbox']
        augmented_bbox = augmented.iloc[0]['bbox']
        assert augmented_bbox['left'] != original_bbox['left']


class TestHorizontalFlipAugmentation:
    """Tests for horizontal flip augmentation."""
    
    def test_initialization(self):
        """Test augmentation initialization."""
        aug = HorizontalFlipAugmentation(flip_probability=0.5)
        assert aug.flip_probability == 0.5
    
    def test_augment_dataframe(self):
        """Test flipping DataFrame."""
        df = pd.DataFrame({
            'seq': ['seq1', 'seq1'],
            'frame_num': [1, 2],
            'bbox': [
                {'left': 100, 'top': 100, 'width': 50, 'height': 100},
                {'left': 105, 'top': 100, 'width': 50, 'height': 100}
            ],
            'vid_info': [
                {'width': 1920, 'height': 1080},
                {'width': 1920, 'height': 1080}
            ]
        })
        
        aug = HorizontalFlipAugmentation(flip_probability=1.0, random_seed=42)
        augmented = aug.augment(df)
        
        assert len(augmented) == len(df)


class TestAugmentationPipeline:
    """Tests for augmentation pipeline."""
    
    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = AugmentationPipeline()
        assert len(pipeline.augmentations) == 0
    
    def test_add_augmentation(self):
        """Test adding augmentations to pipeline."""
        pipeline = AugmentationPipeline()
        aug = SpatialScalingAugmentation()
        
        pipeline.add_augmentation(aug)
        assert len(pipeline.augmentations) == 1
    
    def test_augment_with_pipeline(self):
        """Test augmenting data with pipeline."""
        df = pd.DataFrame({
            'seq': ['seq1', 'seq1'],
            'frame_num': [1, 2],
            'bbox': [
                {'left': 100, 'top': 100, 'width': 50, 'height': 100},
                {'left': 105, 'top': 100, 'width': 50, 'height': 100}
            ]
        })
        
        pipeline = AugmentationPipeline()
        pipeline.add_augmentation(SpatialScalingAugmentation(random_seed=42))
        pipeline.add_augmentation(NoiseInjectionAugmentation(random_seed=42))
        
        augmented = pipeline.augment(df)
        
        assert len(augmented) == len(df)
    
    def test_get_pipeline_config(self):
        """Test getting pipeline configuration."""
        pipeline = AugmentationPipeline()
        pipeline.add_augmentation(SpatialScalingAugmentation())
        pipeline.add_augmentation(NoiseInjectionAugmentation())
        
        config = pipeline.get_pipeline_config()
        
        assert len(config) == 2
        assert config[0]['type'] == 'spatial_scaling'
        assert config[1]['type'] == 'noise_injection'


class TestDataAugmenter:
    """Tests for data augmenter."""
    
    def test_initialization(self):
        """Test augmenter initialization."""
        augmenter = DataAugmenter(random_seed=42)
        assert augmenter.random_seed == 42
    
    def test_create_default_pipeline(self):
        """Test creating default pipeline."""
        augmenter = DataAugmenter()
        pipeline = augmenter.create_default_pipeline()
        
        assert len(pipeline.augmentations) > 0
    
    def test_augment_dataset(self):
        """Test augmenting a dataset."""
        df = pd.DataFrame({
            'seq': ['seq1', 'seq1', 'seq2'],
            'frame_num': [1, 2, 1],
            'bbox': [
                {'left': 100, 'top': 100, 'width': 50, 'height': 100},
                {'left': 105, 'top': 100, 'width': 50, 'height': 100},
                {'left': 200, 'top': 150, 'width': 60, 'height': 110}
            ]
        })
        
        augmenter = DataAugmenter(random_seed=42)
        augmented = augmenter.augment_dataset(df, augmentation_factor=2)
        
        assert len(augmented) >= len(df)
    
    def test_validate_augmentation(self):
        """Test validating augmentation."""
        original = pd.DataFrame({
            'seq': ['seq1', 'seq1'],
            'frame_num': [1, 2],
            'bbox': [
                {'left': 100, 'top': 100, 'width': 50, 'height': 100},
                {'left': 105, 'top': 100, 'width': 50, 'height': 100}
            ]
        })
        
        augmented = pd.concat([original, original], ignore_index=True)
        
        augmenter = DataAugmenter()
        validation = augmenter.validate_augmentation(original, augmented)
        
        assert 'valid' in validation
        assert 'checks' in validation
        assert validation['checks']['augmentation_factor'] == 2.0


# Property-Based Tests

@given(
    sequence_length=st.integers(min_value=10, max_value=50),
    speed_factor=st.floats(min_value=0.5, max_value=2.0)
)
@settings(max_examples=30, deadline=None)
def test_temporal_augmentation_preserves_characteristics(sequence_length, speed_factor):
    """
    Feature: gavd-gait-analysis, Property 14: Data Augmentation Preservation
    
    For any original gait sequence that undergoes augmentation, the augmented versions
    should preserve essential gait characteristics while introducing controlled variations.
    """
    # Create sample sequence
    frames = [Frame.empty(shape=(480, 640, 3)) for _ in range(sequence_length)]
    sequence = FrameSequence(frames=frames, sequence_id="test_seq")
    
    # Apply temporal speed augmentation
    aug = TemporalSpeedAugmentation(speed_range=(speed_factor, speed_factor), random_seed=42)
    augmented = aug.augment(sequence)
    
    # Verify augmentation preserves structure
    assert augmented is not None
    assert len(augmented.frames) > 0
    
    # Verify length change is proportional to speed factor
    expected_length = int(sequence_length / speed_factor)
    actual_length = len(augmented.frames)
    
    # Allow some tolerance for rounding
    assert abs(actual_length - expected_length) <= 2, \
        f"Expected length ~{expected_length}, got {actual_length}"
    
    # Verify metadata is preserved
    assert augmented.sequence_id is not None
    assert 'augmentation' in augmented.metadata


@given(
    sample_count=st.integers(min_value=5, max_value=20),
    scale_factor=st.floats(min_value=0.8, max_value=1.2)
)
@settings(max_examples=30, deadline=None)
def test_spatial_augmentation_preserves_structure(sample_count, scale_factor):
    """
    Feature: gavd-gait-analysis, Property 14: Data Augmentation Preservation
    
    Spatial augmentation should preserve the relative structure of bounding boxes
    while scaling them proportionally.
    """
    # Create sample DataFrame
    df = pd.DataFrame({
        'seq': [f'seq{i}' for i in range(sample_count)],
        'frame_num': list(range(sample_count)),
        'bbox': [
            {'left': 100 + i*10, 'top': 100, 'width': 50, 'height': 100}
            for i in range(sample_count)
        ]
    })
    
    # Apply spatial scaling
    aug = SpatialScalingAugmentation(
        scale_range=(scale_factor, scale_factor),
        random_seed=42
    )
    augmented = aug.augment(df)
    
    # Verify structure is preserved
    assert len(augmented) == len(df)
    assert list(augmented.columns) == list(df.columns)
    
    # Verify scaling is applied correctly
    for i in range(sample_count):
        original_bbox = df.iloc[i]['bbox']
        augmented_bbox = augmented.iloc[i]['bbox']
        
        # Check that scaling is proportional
        width_ratio = augmented_bbox['width'] / original_bbox['width']
        height_ratio = augmented_bbox['height'] / original_bbox['height']
        
        # Ratios should be close to scale_factor
        assert abs(width_ratio - scale_factor) < 0.01
        assert abs(height_ratio - scale_factor) < 0.01


@given(
    augmentation_factor=st.integers(min_value=2, max_value=5)
)
@settings(max_examples=20, deadline=None)
def test_augmentation_increases_dataset_size(augmentation_factor):
    """
    Feature: gavd-gait-analysis, Property 14: Data Augmentation Preservation
    
    Augmentation should increase dataset size by the specified factor while
    preserving data structure.
    """
    # Create sample DataFrame
    df = pd.DataFrame({
        'seq': ['seq1', 'seq2', 'seq3'],
        'frame_num': [1, 1, 1],
        'bbox': [
            {'left': 100, 'top': 100, 'width': 50, 'height': 100},
            {'left': 200, 'top': 150, 'width': 60, 'height': 110},
            {'left': 300, 'top': 200, 'width': 55, 'height': 105}
        ]
    })
    
    augmenter = DataAugmenter(random_seed=42)
    augmented = augmenter.augment_dataset(df, augmentation_factor=augmentation_factor)
    
    # Verify size increase
    expected_size = len(df) * augmentation_factor
    assert len(augmented) == expected_size
    
    # Verify structure is preserved (augmentation adds version column)
    expected_columns = set(df.columns) | {'augmentation_version'}
    assert set(augmented.columns) == expected_columns
    
    # Verify original data is included
    original_seqs = set(df['seq'])
    augmented_seqs = set(augmented['seq'])
    assert original_seqs.issubset(augmented_seqs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
