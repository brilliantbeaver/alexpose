"""
Dataset Versioning and Provenance Tracking Module

This module provides comprehensive versioning and provenance tracking for training datasets,
ensuring reproducibility and traceability of model training data.

Key Features:
- Dataset versioning with semantic versioning support
- Provenance tracking (data lineage, transformations, sources)
- Version comparison and diff capabilities
- Rollback and restore functionality
- Audit trail for all dataset modifications

Design Principles:
- Single Responsibility: Each class handles one aspect of versioning
- Open/Closed: Extensible for new versioning strategies
- Immutability: Versions are immutable once created

Author: AlexPose Team
"""

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from loguru import logger


@dataclass
class DatasetVersion:
    """Represents a specific version of a dataset."""
    
    version_id: str
    dataset_name: str
    version_number: str
    created_at: str
    created_by: str = "system"
    parent_version: Optional[str] = None
    description: str = ""
    sample_count: int = 0
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceRecord:
    """Records the provenance (origin and history) of a dataset."""
    
    source_datasets: List[str] = field(default_factory=list)
    source_files: List[str] = field(default_factory=list)
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    augmentations: List[Dict[str, Any]] = field(default_factory=list)
    filters: List[Dict[str, Any]] = field(default_factory=list)
    merge_operations: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = "system"
    notes: str = ""


class DatasetVersionManager:
    """
    Manages dataset versions with provenance tracking.
    
    This class provides comprehensive version control for training datasets,
    including creation, comparison, and rollback capabilities.
    """
    
    def __init__(self, versions_dir: Path = Path("data/training/versions")):
        """
        Initialize dataset version manager.
        
        Args:
            versions_dir: Directory for storing dataset versions
        """
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        self.versions: Dict[str, Dict[str, DatasetVersion]] = {}
        self._load_existing_versions()
        
        logger.info(f"Initialized DatasetVersionManager with versions_dir: {self.versions_dir}")
    
    def _load_existing_versions(self) -> None:
        """Load existing versions from disk."""
        for dataset_dir in self.versions_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            self.versions[dataset_name] = {}
            
            for version_dir in dataset_dir.iterdir():
                if not version_dir.is_dir():
                    continue
                
                metadata_file = version_dir / "version_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            version_data = json.load(f)
                        
                        version = DatasetVersion(**version_data)
                        self.versions[dataset_name][version.version_number] = version
                        
                    except Exception as e:
                        logger.warning(f"Failed to load version from {version_dir}: {e}")
        
        logger.info(f"Loaded {sum(len(v) for v in self.versions.values())} existing versions")
    
    def create_version(
        self,
        dataset_name: str,
        data: pd.DataFrame,
        version_number: str,
        description: str = "",
        parent_version: Optional[str] = None,
        provenance: Optional[ProvenanceRecord] = None,
        created_by: str = "system"
    ) -> DatasetVersion:
        """
        Create a new dataset version.
        
        Args:
            dataset_name: Name of the dataset
            data: DataFrame to version
            version_number: Version identifier (e.g., "1.0.0", "1.1.0")
            description: Description of this version
            parent_version: Parent version number if this is derived
            provenance: Provenance record for this version
            created_by: User or system creating the version
            
        Returns:
            Created DatasetVersion object
        """
        # Create version directory
        version_dir = self.versions_dir / dataset_name / version_number
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data_file = version_dir / "data.pkl"
        data.to_pickle(data_file)
        
        # Calculate checksum
        checksum = self._calculate_checksum(data)
        
        # Create version object
        version = DatasetVersion(
            version_id=f"{dataset_name}_{version_number}",
            dataset_name=dataset_name,
            version_number=version_number,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            parent_version=parent_version,
            description=description,
            sample_count=len(data),
            checksum=checksum,
            metadata={
                "columns": list(data.columns),
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            provenance=asdict(provenance) if provenance else {}
        )
        
        # Save version metadata
        metadata_file = version_dir / "version_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(version), f, indent=2)
        
        # Update in-memory registry
        if dataset_name not in self.versions:
            self.versions[dataset_name] = {}
        self.versions[dataset_name][version_number] = version
        
        logger.info(f"Created version {version_number} for dataset {dataset_name}")
        
        return version
    
    def _calculate_checksum(self, data: pd.DataFrame) -> str:
        """
        Calculate checksum for dataset.
        
        Args:
            data: DataFrame to checksum
            
        Returns:
            SHA256 checksum string
        """
        # Convert DataFrame to bytes for hashing
        data_bytes = pd.util.hash_pandas_object(data, index=True).values.tobytes()
        checksum = hashlib.sha256(data_bytes).hexdigest()
        return checksum
    
    def load_version(
        self,
        dataset_name: str,
        version_number: str
    ) -> Tuple[pd.DataFrame, DatasetVersion]:
        """
        Load a specific dataset version.
        
        Args:
            dataset_name: Name of the dataset
            version_number: Version to load
            
        Returns:
            Tuple of (DataFrame, DatasetVersion)
        """
        if dataset_name not in self.versions:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        if version_number not in self.versions[dataset_name]:
            raise ValueError(f"Version '{version_number}' not found for dataset '{dataset_name}'")
        
        version = self.versions[dataset_name][version_number]
        version_dir = self.versions_dir / dataset_name / version_number
        data_file = version_dir / "data.pkl"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        data = pd.read_pickle(data_file)
        
        logger.info(f"Loaded version {version_number} of dataset {dataset_name}")
        
        return data, version
    
    def list_versions(self, dataset_name: str) -> List[DatasetVersion]:
        """
        List all versions of a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of DatasetVersion objects
        """
        if dataset_name not in self.versions:
            return []
        
        versions = list(self.versions[dataset_name].values())
        versions.sort(key=lambda v: v.created_at, reverse=True)
        
        return versions
    
    def get_latest_version(self, dataset_name: str) -> Optional[DatasetVersion]:
        """
        Get the latest version of a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Latest DatasetVersion or None
        """
        versions = self.list_versions(dataset_name)
        return versions[0] if versions else None
    
    def compare_versions(
        self,
        dataset_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions of a dataset.
        
        Args:
            dataset_name: Name of the dataset
            version1: First version number
            version2: Second version number
            
        Returns:
            Dictionary with comparison results
        """
        if dataset_name not in self.versions:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        if version1 not in self.versions[dataset_name]:
            raise ValueError(f"Version '{version1}' not found")
        
        if version2 not in self.versions[dataset_name]:
            raise ValueError(f"Version '{version2}' not found")
        
        v1 = self.versions[dataset_name][version1]
        v2 = self.versions[dataset_name][version2]
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "sample_count_diff": v2.sample_count - v1.sample_count,
            "checksum_match": v1.checksum == v2.checksum,
            "created_at_diff": v2.created_at,
            "metadata_diff": self._compare_metadata(v1.metadata, v2.metadata),
            "transformations_diff": {
                "v1_count": len(v1.transformations),
                "v2_count": len(v2.transformations)
            }
        }
        
        logger.info(f"Compared versions {version1} and {version2} of dataset {dataset_name}")
        
        return comparison
    
    def _compare_metadata(self, meta1: Dict[str, Any], meta2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two metadata dictionaries."""
        diff = {
            "added_keys": list(set(meta2.keys()) - set(meta1.keys())),
            "removed_keys": list(set(meta1.keys()) - set(meta2.keys())),
            "changed_values": {}
        }
        
        for key in set(meta1.keys()) & set(meta2.keys()):
            if meta1[key] != meta2[key]:
                diff["changed_values"][key] = {
                    "old": meta1[key],
                    "new": meta2[key]
                }
        
        return diff
    
    def delete_version(
        self,
        dataset_name: str,
        version_number: str,
        force: bool = False
    ) -> bool:
        """
        Delete a dataset version.
        
        Args:
            dataset_name: Name of the dataset
            version_number: Version to delete
            force: Force deletion even if it has children
            
        Returns:
            True if deleted successfully
        """
        if dataset_name not in self.versions:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        if version_number not in self.versions[dataset_name]:
            raise ValueError(f"Version '{version_number}' not found")
        
        # Check if this version has children
        if not force:
            children = [
                v for v in self.versions[dataset_name].values()
                if v.parent_version == version_number
            ]
            if children:
                raise ValueError(
                    f"Version {version_number} has {len(children)} child versions. "
                    "Use force=True to delete anyway."
                )
        
        # Delete from disk
        version_dir = self.versions_dir / dataset_name / version_number
        if version_dir.exists():
            shutil.rmtree(version_dir)
        
        # Remove from registry
        del self.versions[dataset_name][version_number]
        
        logger.info(f"Deleted version {version_number} of dataset {dataset_name}")
        
        return True
    
    def get_version_lineage(
        self,
        dataset_name: str,
        version_number: str
    ) -> List[DatasetVersion]:
        """
        Get the lineage (ancestry) of a version.
        
        Args:
            dataset_name: Name of the dataset
            version_number: Version to trace
            
        Returns:
            List of DatasetVersion objects from root to current
        """
        if dataset_name not in self.versions:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        if version_number not in self.versions[dataset_name]:
            raise ValueError(f"Version '{version_number}' not found")
        
        lineage = []
        current_version = self.versions[dataset_name][version_number]
        
        while current_version:
            lineage.insert(0, current_version)
            
            if current_version.parent_version:
                current_version = self.versions[dataset_name].get(current_version.parent_version)
            else:
                break
        
        logger.info(f"Retrieved lineage for version {version_number}: {len(lineage)} versions")
        
        return lineage
    
    def export_version_history(
        self,
        dataset_name: str,
        output_path: Union[str, Path]
    ) -> Path:
        """
        Export version history to JSON file.
        
        Args:
            dataset_name: Name of the dataset
            output_path: Path for output file
            
        Returns:
            Path to exported file
        """
        if dataset_name not in self.versions:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        history = {
            "dataset_name": dataset_name,
            "total_versions": len(self.versions[dataset_name]),
            "exported_at": datetime.now().isoformat(),
            "versions": [
                asdict(version) for version in self.versions[dataset_name].values()
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Exported version history to {output_path}")
        
        return output_path
    
    def validate_version_integrity(
        self,
        dataset_name: str,
        version_number: str
    ) -> Dict[str, Any]:
        """
        Validate the integrity of a dataset version.
        
        Args:
            dataset_name: Name of the dataset
            version_number: Version to validate
            
        Returns:
            Validation results dictionary
        """
        data, version = self.load_version(dataset_name, version_number)
        
        # Recalculate checksum
        current_checksum = self._calculate_checksum(data)
        checksum_valid = current_checksum == version.checksum
        
        # Validate sample count
        sample_count_valid = len(data) == version.sample_count
        
        # Validate columns
        columns_valid = set(data.columns) == set(version.metadata.get("columns", []))
        
        validation = {
            "valid": checksum_valid and sample_count_valid and columns_valid,
            "checksum_valid": checksum_valid,
            "sample_count_valid": sample_count_valid,
            "columns_valid": columns_valid,
            "expected_checksum": version.checksum,
            "actual_checksum": current_checksum,
            "expected_samples": version.sample_count,
            "actual_samples": len(data)
        }
        
        if not validation["valid"]:
            logger.warning(f"Version {version_number} failed integrity validation")
        else:
            logger.info(f"Version {version_number} passed integrity validation")
        
        return validation
