"""
Tests for YOLO model display names.

Verifies that YOLO model display names match official Ultralytics naming:
- YOLOv8 (with capital V)
- YOLO11 (no v)
"""

import pytest
from unittest.mock import Mock, MagicMock

from server.services.models_service import PoseEstimatorProvider, ModelsService
from ambient.core.config import ConfigurationManager


class TestYOLODisplayNames:
    """Test suite for YOLO model display name formatting."""
    
    def test_yolov8_display_name_correct(self):
        """Verify YOLOv8 uses correct display name with capital V."""
        provider = PoseEstimatorProvider()
        
        display_name = provider._format_display_name("yolov8-pose")
        
        assert display_name == "YOLOv8 Pose", f"Expected 'YOLOv8 Pose', got '{display_name}'"
        assert "YOLOv8" in display_name, "Display name must contain 'YOLOv8' with capital V"
        assert "Yolov8" not in display_name, "Display name must not contain 'Yolov8' with lowercase v"
    
    def test_yolo11_display_name_correct(self):
        """Verify YOLO11 uses correct display name without v."""
        provider = PoseEstimatorProvider()
        
        display_name = provider._format_display_name("yolov11-pose")
        
        assert display_name == "YOLO11 Pose", f"Expected 'YOLO11 Pose', got '{display_name}'"
        assert "YOLO11" in display_name, "Display name must contain 'YOLO11' without v"
        assert "YOLOv11" not in display_name, "Display name must not contain 'YOLOv11' with v"
        assert "Yolov11" not in display_name, "Display name must not contain 'Yolov11'"
    
    def test_yolov8_description_correct(self):
        """Verify YOLOv8 description uses correct naming."""
        provider = PoseEstimatorProvider()
        
        description = provider._get_description("yolov8-pose")
        
        assert "YOLOv8" in description, "Description must contain 'YOLOv8' with capital V"
        assert "Yolov8" not in description, "Description must not contain 'Yolov8'"
    
    def test_yolo11_description_correct(self):
        """Verify YOLO11 description uses correct naming."""
        provider = PoseEstimatorProvider()
        
        description = provider._get_description("yolov11-pose")
        
        assert "YOLO11" in description, "Description must contain 'YOLO11' without v"
        assert "YOLOv11" not in description, "Description must not contain 'YOLOv11'"
        assert "Yolov11" not in description, "Description must not contain 'Yolov11'"
    
    def test_other_models_title_case(self):
        """Verify other models use standard title case formatting."""
        provider = PoseEstimatorProvider()
        
        # Test MediaPipe
        assert provider._format_display_name("mediapipe") == "Mediapipe"
        
        # Test OpenPose
        assert provider._format_display_name("openpose") == "Openpose"
        
        # Test AlphaPose
        assert provider._format_display_name("alphapose") == "Alphapose"
    
    def test_models_service_integration(self):
        """Test that ModelsService returns correct display names."""
        # Create mock config
        mock_config = Mock(spec=ConfigurationManager)
        mock_config.llm_models = {}
        
        # Create service
        service = ModelsService(mock_config)
        
        # Get all models
        all_models = service.get_all_models()
        
        # Find YOLO models
        yolo_models = [m for m in all_models["models"] if "yolo" in m["name"].lower()]
        
        for model in yolo_models:
            if model["name"] == "yolov8-pose":
                assert model["display_name"] == "YOLOv8 Pose", \
                    f"YOLOv8 display name incorrect: {model['display_name']}"
                assert "description" in model, "Model should include description"
                assert "YOLOv8" in model["description"], \
                    f"YOLOv8 description should contain 'YOLOv8': {model['description']}"
            elif model["name"] == "yolov11-pose":
                assert model["display_name"] == "YOLO11 Pose", \
                    f"YOLO11 display name incorrect: {model['display_name']}"
                assert "description" in model, "Model should include description"
                assert "YOLO11" in model["description"], \
                    f"YOLO11 description should contain 'YOLO11': {model['description']}"
    
    def test_search_with_correct_names(self):
        """Test that search works with correct YOLO names."""
        mock_config = Mock(spec=ConfigurationManager)
        mock_config.llm_models = {}
        
        service = ModelsService(mock_config)
        
        # Search for YOLOv8 (with capital V)
        results = service.search_models("YOLOv8")
        yolov8_found = any(m["name"] == "yolov8-pose" for m in results)
        assert yolov8_found, "Should find yolov8-pose when searching for 'YOLOv8'"
        
        # Search for YOLO11 (no v)
        results = service.search_models("YOLO11")
        yolo11_found = any(m["name"] == "yolov11-pose" for m in results)
        assert yolo11_found, "Should find yolov11-pose when searching for 'YOLO11'"
    
    def test_display_names_match_official_ultralytics(self):
        """Verify display names match official Ultralytics documentation."""
        provider = PoseEstimatorProvider()
        
        # Official Ultralytics naming from documentation
        official_names = {
            "yolov8-pose": "YOLOv8 Pose",  # With 'v' and capital V
            "yolov11-pose": "YOLO11 Pose"  # No 'v'
        }
        
        for model_name, expected_display in official_names.items():
            actual_display = provider._format_display_name(model_name)
            assert actual_display == expected_display, \
                f"Model {model_name}: expected '{expected_display}', got '{actual_display}'"


class TestYOLONamingConsistency:
    """Test suite for YOLO naming consistency across the system."""
    
    def test_model_file_names_correct(self):
        """Verify model file names match Ultralytics official names."""
        from pathlib import Path
        
        models_dir = Path("data/models")
        if models_dir.exists():
            # Check YOLOv8 (with v)
            yolov8_files = list(models_dir.glob("yolov8*-pose.pt"))
            if yolov8_files:
                assert any("yolov8" in f.name for f in yolov8_files), \
                    "YOLOv8 model files should contain 'yolov8' (with v)"
            
            # Check YOLO11 (no v)
            yolo11_files = list(models_dir.glob("yolo11*-pose.pt"))
            if yolo11_files:
                assert any("yolo11" in f.name for f in yolo11_files), \
                    "YOLO11 model files should contain 'yolo11' (no v)"
                assert not any("yolov11" in f.name for f in yolo11_files), \
                    "YOLO11 model files should NOT contain 'yolov11' (with v)"
    
    def test_estimator_names_consistent(self):
        """Verify estimator names are consistent."""
        from ambient.pose.factory import PoseEstimatorFactory
        
        factory = PoseEstimatorFactory()
        estimators = factory.list_available_estimators()
        
        # Check that estimator names use lowercase with hyphens
        yolo_estimators = [e for e in estimators if "yolo" in e.lower()]
        
        for estimator in yolo_estimators:
            assert "-" in estimator, f"Estimator name should use hyphens: {estimator}"
            assert estimator.islower(), f"Estimator name should be lowercase: {estimator}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
