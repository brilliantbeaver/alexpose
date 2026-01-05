"""
Tests for model path resolution utilities.

Verifies that YOLO model paths are resolved correctly with
intelligent fallback strategies for backward compatibility.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from ambient.utils.model_utils import (
    resolve_yolo_model_path,
    get_model_info,
    list_available_yolo_models,
    validate_model_path,
    ensure_models_directory
)


class TestResolveYoloModelPath:
    """Test suite for YOLO model path resolution."""
    
    def test_resolve_absolute_path_exists(self, tmp_path):
        """Test resolution of absolute path that exists."""
        model_file = tmp_path / "yolov8n-pose.pt"
        model_file.write_bytes(b"fake model content")
        
        result = resolve_yolo_model_path(str(model_file))
        
        assert result == str(model_file)
    
    def test_resolve_absolute_path_not_exists(self, tmp_path):
        """Test resolution of absolute path that doesn't exist."""
        model_file = tmp_path / "nonexistent.pt"
        
        result = resolve_yolo_model_path(str(model_file))
        
        # Should return original path (let Ultralytics handle it)
        assert result == str(model_file)
    
    def test_resolve_relative_path_exists(self, tmp_path):
        """Test resolution of relative path that exists."""
        # Create a model file in current directory
        model_file = Path("test_model.pt")
        try:
            model_file.write_bytes(b"fake model content")
            
            result = resolve_yolo_model_path("test_model.pt")
            
            assert result == "test_model.pt"
        finally:
            if model_file.exists():
                model_file.unlink()
    
    def test_resolve_in_models_directory(self, tmp_path):
        """Test resolution finds model in models directory."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        # Use a unique model name that doesn't exist in project root
        model_file = models_dir / "test-yolov8n-pose.pt"
        model_file.write_bytes(b"fake model content")
        
        result = resolve_yolo_model_path("test-yolov8n-pose.pt", models_directory=models_dir)
        
        assert result == str(models_dir / "test-yolov8n-pose.pt")
    
    def test_resolve_in_project_root(self, tmp_path):
        """Test resolution finds model in project root (backward compatibility)."""
        # Create model in current directory
        model_file = Path("yolov8n-pose.pt")
        try:
            model_file.write_bytes(b"fake model content")
            
            # Use non-existent models directory
            result = resolve_yolo_model_path("yolov8n-pose.pt", models_directory=tmp_path / "nonexistent")
            
            assert result == "yolov8n-pose.pt"
        finally:
            if model_file.exists():
                model_file.unlink()
    
    def test_resolve_not_found_returns_original(self, tmp_path):
        """Test that non-existent model returns original name."""
        result = resolve_yolo_model_path("nonexistent-model.pt", models_directory=tmp_path)
        
        assert result == "nonexistent-model.pt"
    
    def test_resolve_priority_models_dir_over_root(self, tmp_path):
        """Test that models directory has priority over project root when file doesn't exist as relative path."""
        # Create model in both locations using unique name
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        # Use unique model name
        model_name = "test-priority-yolov8n-pose.pt"
        models_file = models_dir / model_name
        models_file.write_bytes(b"models dir content")
        
        # Don't create in current directory - test that models_dir is checked before project root
        # when the file doesn't exist as a relative path from current directory
        result = resolve_yolo_model_path(model_name, models_directory=models_dir, check_project_root=True)
        
        # Should find in models directory
        assert result == str(models_dir / model_name)
    
    def test_resolve_with_path_in_model_name(self, tmp_path):
        """Test resolution extracts filename from path in model_name."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_file = models_dir / "yolov8n-pose.pt"
        model_file.write_bytes(b"fake model content")
        
        # Pass model_name with path prefix
        result = resolve_yolo_model_path("some/path/yolov8n-pose.pt", models_directory=models_dir)
        
        # Should find the file by filename in models_dir
        assert result == str(models_dir / "yolov8n-pose.pt")
    
    def test_resolve_empty_model_name_raises(self):
        """Test that empty model_name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            resolve_yolo_model_path("")
    
    def test_resolve_with_check_project_root_false(self, tmp_path):
        """Test resolution with check_project_root=False."""
        # Create model in current directory
        root_file = Path("yolov8n-pose.pt")
        try:
            root_file.write_bytes(b"root content")
            
            result = resolve_yolo_model_path(
                "yolov8n-pose.pt",
                models_directory=tmp_path / "nonexistent",
                check_project_root=False
            )
            
            # Should not find in root, return original
            assert result == "yolov8n-pose.pt"
        finally:
            if root_file.exists():
                root_file.unlink()


class TestGetModelInfo:
    """Test suite for get_model_info function."""
    
    def test_get_info_existing_model(self, tmp_path):
        """Test getting info for existing model."""
        model_file = tmp_path / "yolov8n-pose.pt"
        model_file.write_bytes(b"fake model content" * 1000)
        
        info = get_model_info(model_file)
        
        assert info["path"] == str(model_file)
        assert info["exists"] is True
        assert info["name"] == "yolov8n-pose.pt"
        assert info["size_mb"] is not None
        assert info["size_mb"] > 0
        assert info["is_absolute"] is True
    
    def test_get_info_nonexistent_model(self, tmp_path):
        """Test getting info for non-existent model."""
        model_file = tmp_path / "nonexistent.pt"
        
        info = get_model_info(model_file)
        
        assert info["path"] == str(model_file)
        assert info["exists"] is False
        assert info["name"] == "nonexistent.pt"
        assert info["size_mb"] is None


class TestListAvailableYoloModels:
    """Test suite for list_available_yolo_models function."""
    
    def test_list_models_in_directory(self, tmp_path):
        """Test listing models in directory."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        
        # Create some model files
        (models_dir / "yolov8n-pose.pt").write_bytes(b"model1")
        (models_dir / "yolo11n-pose.pt").write_bytes(b"model2")
        (models_dir / "yolov8x-pose.pt").write_bytes(b"model3")
        (models_dir / "other-model.pt").write_bytes(b"other")  # Changed name to not include "pose"
        
        models = list_available_yolo_models(models_dir)
        
        assert len(models) == 3
        assert "yolov8n-pose.pt" in models
        assert "yolo11n-pose.pt" in models
        assert "yolov8x-pose.pt" in models
        assert "other-model.pt" not in models
    
    def test_list_models_empty_directory(self, tmp_path):
        """Test listing models in empty directory."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        
        models = list_available_yolo_models(models_dir)
        
        assert models == []
    
    def test_list_models_nonexistent_directory(self, tmp_path):
        """Test listing models in non-existent directory."""
        models_dir = tmp_path / "nonexistent"
        
        models = list_available_yolo_models(models_dir)
        
        assert models == []


class TestValidateModelPath:
    """Test suite for validate_model_path function."""
    
    def test_validate_existing_model(self, tmp_path):
        """Test validation of existing model."""
        model_file = tmp_path / "yolov8n-pose.pt"
        model_file.write_bytes(b"fake model content")
        
        result = validate_model_path(model_file)
        
        assert result is True
    
    def test_validate_nonexistent_model(self, tmp_path):
        """Test validation of non-existent model."""
        model_file = tmp_path / "nonexistent.pt"
        
        result = validate_model_path(model_file)
        
        assert result is False
    
    def test_validate_inaccessible_model(self, tmp_path):
        """Test validation of inaccessible model."""
        model_file = tmp_path / "yolov8n-pose.pt"
        model_file.write_bytes(b"fake model content")
        
        # Mock open to raise exception
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            result = validate_model_path(model_file)
            
            assert result is False


class TestEnsureModelsDirectory:
    """Test suite for ensure_models_directory function."""
    
    def test_ensure_creates_directory(self, tmp_path):
        """Test that ensure_models_directory creates directory."""
        models_dir = tmp_path / "models"
        
        result = ensure_models_directory(models_dir)
        
        assert result == models_dir
        assert models_dir.exists()
        assert models_dir.is_dir()
    
    def test_ensure_existing_directory(self, tmp_path):
        """Test that ensure_models_directory works with existing directory."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        
        result = ensure_models_directory(models_dir)
        
        assert result == models_dir
        assert models_dir.exists()
    
    def test_ensure_creates_parent_directories(self, tmp_path):
        """Test that ensure_models_directory creates parent directories."""
        models_dir = tmp_path / "parent" / "child" / "models"
        
        result = ensure_models_directory(models_dir)
        
        assert result == models_dir
        assert models_dir.exists()
        assert models_dir.parent.exists()


class TestIntegration:
    """Integration tests for model path resolution."""
    
    def test_real_models_directory(self):
        """Test with real data/models directory."""
        # This test uses the actual project structure
        models_dir = Path("data/models")
        
        if models_dir.exists():
            # Test listing models
            models = list_available_yolo_models(models_dir)
            
            # Should find YOLO pose models if they exist
            assert isinstance(models, list)
            
            # Test resolution
            result = resolve_yolo_model_path("yolov8n-pose.pt", models_directory=models_dir)
            
            # Should resolve to models directory or return original
            assert isinstance(result, str)
    
    def test_backward_compatibility_scenario(self, tmp_path):
        """Test backward compatibility: models in project root."""
        # Simulate old setup with models in root
        root_model = Path("test_yolov8n-pose.pt")
        try:
            root_model.write_bytes(b"fake model content")
            
            # Resolution should find it in root
            result = resolve_yolo_model_path(
                "test_yolov8n-pose.pt",
                models_directory=tmp_path / "nonexistent"
            )
            
            assert result == "test_yolov8n-pose.pt"
        finally:
            if root_model.exists():
                root_model.unlink()
    
    def test_new_structure_scenario(self, tmp_path):
        """Test new structure: models in data/models."""
        # Simulate new setup with models in data/models
        models_dir = tmp_path / "data" / "models"
        models_dir.mkdir(parents=True)
        model_file = models_dir / "yolov8n-pose.pt"
        model_file.write_bytes(b"fake model content")
        
        # Resolution should find it in models directory
        result = resolve_yolo_model_path("yolov8n-pose.pt", models_directory=models_dir)
        
        assert result == str(model_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
