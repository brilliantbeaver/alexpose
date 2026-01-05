"""
Training Data Management Module

This module provides comprehensive training data management for the AlexPose system,
supporting GAVD and other datasets with versioning, augmentation, and balanced dataset creation.

Key Features:
- Multi-dataset support (GAVD and additional sources)
- Dataset versioning and provenance tracking
- Balanced dataset creation for normal/abnormal classification
- Data augmentation for model training
- Export capabilities for model training pipelines

Design Principles:
- Single Responsibility: Each class handles one aspect of data management
- Open/Closed: Extensible for new dataset types
- Dependency Inversion: Depends on abstractions
- Interface Segregation: Small, focused interfaces

Author: AlexPose Team
"""

import json
import pickle
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from ambient.core.data_models import (
    DatasetInfo,
    GaitFeatures,
    TrainingDataSample,
)
from ambient.core.frame import Frame, FrameSequence
from ambient.gavd.gavd_processor import GAVDDataLoader


class IDatasetLoader(ABC):
    """Interface for dataset loaders."""
    
    @abstractmethod
    def load_dataset(self, source_path: Path) -> pd.DataFrame:
        """Load dataset from source path."""
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> DatasetInfo:
        """Get dataset information."""
        pass


class GAVDDatasetLoader(IDatasetLoader):
    """Loader for GAVD dataset."""
    
    def __init__(self, data_loader: Optional[GAVDDataLoader] = None):
        """
        Initialize GAVD dataset loader.
        
        Args:
            data_loader: Optional GAVDDataLoader instance
        """
        self.data_loader = data_loader or GAVDDataLoader()
        self.dataset_info: Optional[DatasetInfo] = None
    
    def load_dataset(self, source_path: Path) -> pd.DataFrame:
        """
        Load GAVD dataset from CSV file.
        
        Args:
            source_path: Path to GAVD CSV file
            
        Returns:
            DataFrame with GAVD data
        """
        logger.info(f"Loading GAVD dataset from {source_path}")
        
        df = self.data_loader.load_gavd_data(
            str(source_path),
            convert_frame_num=True,
            verbose=False
        )
        
        # Create dataset info
        condition_distribution = {}
        if 'gait_pat' in df.columns:
            condition_distribution = df['gait_pat'].value_counts().to_dict()
        
        self.dataset_info = DatasetInfo(
            dataset_name="GAVD",
            version="1.0",
            source_path=source_path,
            sample_count=len(df),
            condition_distribution=condition_distribution,
            created_at=datetime.now().isoformat(),
            metadata={
                "columns": list(df.columns),
                "unique_sequences": df['seq'].nunique() if 'seq' in df.columns else 0
            }
        )
        
        logger.info(f"Loaded {len(df)} samples from GAVD dataset")
        return df
    
    def get_dataset_info(self) -> DatasetInfo:
        """Get dataset information."""
        if self.dataset_info is None:
            raise ValueError("Dataset not loaded yet")
        return self.dataset_info


class TrainingDataManager:
    """
    Comprehensive training data management system.
    
    This class manages multiple datasets, provides versioning, and supports
    balanced dataset creation for training machine learning models.
    """
    
    def __init__(self, data_dir: Path = Path("data/training")):
        """
        Initialize training data manager.
        
        Args:
            data_dir: Directory for training data storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.dataset_info: Dict[str, DatasetInfo] = {}
        self.loaders: Dict[str, IDatasetLoader] = {
            "GAVD": GAVDDatasetLoader()
        }
        
        logger.info(f"Initialized TrainingDataManager with data_dir: {self.data_dir}")
    
    def register_dataset_loader(self, dataset_type: str, loader: IDatasetLoader) -> None:
        """
        Register a new dataset loader.
        
        Args:
            dataset_type: Type identifier for the dataset
            loader: Dataset loader instance
        """
        self.loaders[dataset_type] = loader
        logger.info(f"Registered dataset loader for type: {dataset_type}")
    
    def load_dataset(
        self,
        dataset_name: str,
        source_path: Union[str, Path],
        dataset_type: str = "GAVD"
    ) -> pd.DataFrame:
        """
        Load a dataset using the appropriate loader.
        
        Args:
            dataset_name: Name to identify this dataset
            source_path: Path to dataset file
            dataset_type: Type of dataset (must have registered loader)
            
        Returns:
            Loaded DataFrame
        """
        if dataset_type not in self.loaders:
            raise ValueError(f"No loader registered for dataset type: {dataset_type}")
        
        loader = self.loaders[dataset_type]
        df = loader.load_dataset(Path(source_path))
        
        self.datasets[dataset_name] = df
        self.dataset_info[dataset_name] = loader.get_dataset_info()
        
        logger.info(f"Loaded dataset '{dataset_name}' with {len(df)} samples")
        return df
    
    def get_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Get a loaded dataset by name.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame with dataset data
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not loaded")
        return self.datasets[dataset_name]
    
    def get_dataset_info(self, dataset_name: str) -> DatasetInfo:
        """
        Get information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            DatasetInfo object
        """
        if dataset_name not in self.dataset_info:
            raise ValueError(f"Dataset '{dataset_name}' not loaded")
        return self.dataset_info[dataset_name]
    
    def list_datasets(self) -> List[str]:
        """Get list of loaded dataset names."""
        return list(self.datasets.keys())
    
    def create_balanced_dataset(
        self,
        dataset_name: str,
        label_column: str = "gait_pat",
        normal_label: str = "Normal",
        balance_ratio: float = 1.0,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Create a balanced dataset for normal/abnormal classification.
        
        Args:
            dataset_name: Name of the source dataset
            label_column: Column containing labels
            normal_label: Label value for normal samples
            balance_ratio: Ratio of abnormal to normal samples (1.0 = equal)
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (balanced DataFrame, class distribution dict)
        """
        df = self.get_dataset(dataset_name)
        
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        
        # Separate normal and abnormal samples
        normal_samples = df[df[label_column] == normal_label]
        abnormal_samples = df[df[label_column] != normal_label]
        
        logger.info(f"Original distribution - Normal: {len(normal_samples)}, Abnormal: {len(abnormal_samples)}")
        
        # Calculate target counts
        if len(normal_samples) == 0 or len(abnormal_samples) == 0:
            logger.warning("Dataset has only one class, returning original dataset")
            return df, {normal_label: len(normal_samples), "abnormal": len(abnormal_samples)}
        
        # Balance the dataset
        np.random.seed(random_seed)
        
        if balance_ratio == 1.0:
            # Equal balance
            target_count = min(len(normal_samples), len(abnormal_samples))
            normal_balanced = normal_samples.sample(n=target_count, random_state=random_seed)
            abnormal_balanced = abnormal_samples.sample(n=target_count, random_state=random_seed)
        else:
            # Custom ratio
            if len(abnormal_samples) < len(normal_samples):
                abnormal_balanced = abnormal_samples
                target_normal = int(len(abnormal_samples) / balance_ratio)
                normal_balanced = normal_samples.sample(n=min(target_normal, len(normal_samples)), random_state=random_seed)
            else:
                normal_balanced = normal_samples
                target_abnormal = int(len(normal_samples) * balance_ratio)
                abnormal_balanced = abnormal_samples.sample(n=min(target_abnormal, len(abnormal_samples)), random_state=random_seed)
        
        # Combine and shuffle
        balanced_df = pd.concat([normal_balanced, abnormal_balanced], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        distribution = {
            normal_label: len(normal_balanced),
            "abnormal": len(abnormal_balanced)
        }
        
        logger.info(f"Created balanced dataset - Normal: {distribution[normal_label]}, Abnormal: {distribution['abnormal']}")
        
        return balanced_df, distribution
    
    def organize_by_condition(
        self,
        dataset_name: str,
        condition_column: str = "gait_pat"
    ) -> Dict[str, pd.DataFrame]:
        """
        Organize dataset by health conditions.
        
        Args:
            dataset_name: Name of the dataset
            condition_column: Column containing condition labels
            
        Returns:
            Dictionary mapping condition names to DataFrames
        """
        df = self.get_dataset(dataset_name)
        
        if condition_column not in df.columns:
            raise ValueError(f"Condition column '{condition_column}' not found in dataset")
        
        organized = {}
        for condition in df[condition_column].unique():
            condition_data = df[df[condition_column] == condition]
            organized[str(condition)] = condition_data
            logger.debug(f"Condition '{condition}': {len(condition_data)} samples")
        
        logger.info(f"Organized dataset into {len(organized)} conditions")
        return organized
    
    def save_dataset_version(
        self,
        dataset_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save a versioned copy of a dataset.
        
        Args:
            dataset_name: Name of the dataset
            version: Version identifier
            metadata: Optional metadata to save with the dataset
            
        Returns:
            Path to saved dataset
        """
        df = self.get_dataset(dataset_name)
        
        # Create version directory
        version_dir = self.data_dir / dataset_name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data_path = version_dir / "data.pkl"
        df.to_pickle(data_path)
        
        # Save metadata
        version_metadata = {
            "dataset_name": dataset_name,
            "version": version,
            "sample_count": len(df),
            "created_at": datetime.now().isoformat(),
            "columns": list(df.columns)
        }
        
        if metadata:
            version_metadata.update(metadata)
        
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(version_metadata, f, indent=2)
        
        logger.info(f"Saved dataset version '{version}' to {version_dir}")
        return version_dir
    
    def load_dataset_version(
        self,
        dataset_name: str,
        version: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load a versioned dataset.
        
        Args:
            dataset_name: Name of the dataset
            version: Version identifier
            
        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        version_dir = self.data_dir / dataset_name / version
        
        if not version_dir.exists():
            raise FileNotFoundError(f"Dataset version not found: {version_dir}")
        
        # Load data
        data_path = version_dir / "data.pkl"
        df = pd.read_pickle(data_path)
        
        # Load metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded dataset version '{version}' from {version_dir}")
        return df, metadata
    
    def export_for_training(
        self,
        dataset_name: str,
        output_path: Union[str, Path],
        format: str = "pickle",
        include_metadata: bool = True
    ) -> Path:
        """
        Export dataset for model training.
        
        Args:
            dataset_name: Name of the dataset
            output_path: Path for exported data
            format: Export format ("pickle", "csv", "json")
            include_metadata: Whether to include metadata
            
        Returns:
            Path to exported file
        """
        df = self.get_dataset(dataset_name)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "pickle":
            df.to_pickle(output_path)
        elif format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        if include_metadata:
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
            info = self.get_dataset_info(dataset_name)
            with open(metadata_path, 'w') as f:
                json.dump({
                    "dataset_name": info.dataset_name,
                    "version": info.version,
                    "sample_count": info.sample_count,
                    "condition_distribution": info.condition_distribution,
                    "created_at": info.created_at,
                    "metadata": info.metadata
                }, f, indent=2)
        
        logger.info(f"Exported dataset to {output_path} in {format} format")
        return output_path
    
    def get_statistics(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with statistics
        """
        df = self.get_dataset(dataset_name)
        info = self.get_dataset_info(dataset_name)
        
        stats = {
            "dataset_name": dataset_name,
            "total_samples": len(df),
            "columns": list(df.columns),
            "condition_distribution": info.condition_distribution,
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "null_counts": df.isnull().sum().to_dict()
        }
        
        # Add sequence statistics if available
        if 'seq' in df.columns:
            stats["unique_sequences"] = df['seq'].nunique()
            stats["avg_frames_per_sequence"] = len(df) / df['seq'].nunique()
        
        return stats
