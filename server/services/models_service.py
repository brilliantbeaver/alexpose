"""
Models service for managing and discovering available models.

This service provides a unified interface for discovering and managing
different types of models (pose estimators, LLM models, etc.) with
extensibility and dependency injection in mind.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Protocol
from pathlib import Path
from loguru import logger

from ambient.core.config import ConfigurationManager
from ambient.pose.factory import PoseEstimatorFactory


class ModelProvider(Protocol):
    """Protocol for model providers."""
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        ...
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        ...
    
    def get_provider_type(self) -> str:
        """Get the type of models this provider handles."""
        ...


class PoseEstimatorProvider:
    """Provider for pose estimation models."""
    
    def __init__(self, factory: Optional[PoseEstimatorFactory] = None):
        """
        Initialize pose estimator provider.
        
        Args:
            factory: Optional PoseEstimatorFactory instance for dependency injection
        """
        self.factory = factory or PoseEstimatorFactory()
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available pose estimators."""
        estimators = self.factory.list_available_estimators()
        
        models = []
        for estimator_name in estimators:
            try:
                info = self.factory.get_estimator_info(estimator_name)
                models.append({
                    "name": estimator_name,
                    "display_name": self._format_display_name(estimator_name),
                    "class_name": info.get("class", "Unknown"),
                    "module": info.get("module", "Unknown"),
                    "available": info.get("available", False),
                    "error": info.get("error"),
                    "type": "pose_estimator",
                    "category": "pose_estimation",
                    "provider": "local",
                    "description": self._get_description(estimator_name)  # Add description
                })
            except Exception as e:
                logger.warning(f"Error getting info for {estimator_name}: {e}")
                models.append({
                    "name": estimator_name,
                    "display_name": self._format_display_name(estimator_name),
                    "class_name": "Unknown",
                    "module": "Unknown",
                    "available": False,
                    "error": str(e),
                    "type": "pose_estimator",
                    "category": "pose_estimation",
                    "provider": "local",
                    "description": self._get_description(estimator_name)  # Add description
                })
        
        return models
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific pose estimator."""
        info = self.factory.get_estimator_info(model_name)
        
        return {
            "name": model_name,
            "display_name": self._format_display_name(model_name),
            "class_name": info.get("class", "Unknown"),
            "module": info.get("module", "Unknown"),
            "available": info.get("available", False),
            "error": info.get("error"),
            "type": "pose_estimator",
            "category": "pose_estimation",
            "provider": "local",
            "description": self._get_description(model_name)
        }
    
    def get_provider_type(self) -> str:
        """Get the type of models this provider handles."""
        return "pose_estimator"
    
    def _format_display_name(self, name: str) -> str:
        """Format model name for display with proper capitalization."""
        # Special handling for YOLO models to match official naming
        if name == "yolov8-pose":
            return "YOLOv8 Pose"
        elif name == "yolov11-pose":
            return "YOLO11 Pose"
        
        # Convert snake_case or kebab-case to Title Case for other models
        return name.replace("_", " ").replace("-", " ").title()
    
    def _get_description(self, name: str) -> str:
        """Get description for a pose estimator."""
        descriptions = {
            "mediapipe": "Google MediaPipe Pose - Fast and accurate pose estimation with 33 keypoints",
            "openpose": "OpenPose - Multi-person pose estimation with BODY_25 format",
            "yolov8-pose": "YOLOv8 Pose - Real-time pose estimation using YOLOv8 architecture (2023)",
            "yolov11-pose": "YOLO11 Pose - Advanced pose estimation using YOLO11 architecture (2024)",
            "alphapose": "AlphaPose - Accurate multi-person pose estimation"
        }
        return descriptions.get(name.lower(), f"{name} pose estimation model")


class LLMModelProvider:
    """Provider for LLM classification models."""
    
    def __init__(self, config_manager: ConfigurationManager):
        """
        Initialize LLM model provider.
        
        Args:
            config_manager: Configuration manager instance for dependency injection
        """
        self.config_manager = config_manager
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available LLM models."""
        models = []
        
        if hasattr(self.config_manager, 'llm_models') and self.config_manager.llm_models:
            for model_name, model_config in self.config_manager.llm_models.items():
                models.append(self._format_model_info(model_name, model_config))
        
        return models
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific LLM model."""
        if not hasattr(self.config_manager, 'llm_models') or not self.config_manager.llm_models:
            raise ValueError("No LLM models configured")
        
        if model_name not in self.config_manager.llm_models:
            raise ValueError(f"Model {model_name} not found")
        
        model_config = self.config_manager.llm_models[model_name]
        return self._format_model_info(model_name, model_config, detailed=True)
    
    def get_provider_type(self) -> str:
        """Get the type of models this provider handles."""
        return "llm_model"
    
    def _format_model_info(self, model_name: str, model_config: Dict[str, Any], detailed: bool = False) -> Dict[str, Any]:
        """Format model configuration into standardized info structure."""
        info = {
            "name": model_name,
            "display_name": model_name.upper().replace("-", " ").replace("_", " "),
            "provider": model_config.get("provider", "unknown"),
            "capability": model_config.get("capability", "text_only"),
            "description": model_config.get("description", ""),
            "type": "llm_model",
            "category": "classification",
            "available": True  # Assume available if configured
        }
        
        if detailed:
            info.update({
                "max_tokens": model_config.get("max_tokens", 0),
                "context_window": model_config.get("context_window", 0),
                "cost_per_1k_tokens": model_config.get("cost_per_1k_tokens", {}),
                "supports_json_mode": model_config.get("supports_json_mode", False),
                "supports_video": model_config.get("supports_video", False),
                "multimodal": model_config.get("multimodal", False),
                "reasoning": model_config.get("reasoning", "standard"),
                "cost_tier": model_config.get("cost_tier", "medium"),
                "text": model_config.get("text", True),
                "images": model_config.get("images", False),
                "video": model_config.get("video", False),
                "temperature": model_config.get("temperature", 0.1)
            })
        else:
            # Include summary info for list view
            info.update({
                "cost_tier": model_config.get("cost_tier", "medium"),
                "multimodal": model_config.get("multimodal", False),
                "reasoning": model_config.get("reasoning", "standard")
            })
        
        return info


class ModelsService:
    """
    Unified service for managing all types of models.
    
    This service uses dependency injection and the strategy pattern to
    support multiple model providers in an extensible way.
    """
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        pose_provider: Optional[PoseEstimatorProvider] = None,
        llm_provider: Optional[LLMModelProvider] = None
    ):
        """
        Initialize models service with dependency injection.
        
        Args:
            config_manager: Configuration manager instance
            pose_provider: Optional pose estimator provider (created if not provided)
            llm_provider: Optional LLM model provider (created if not provided)
        """
        self.config_manager = config_manager
        
        # Initialize providers with dependency injection
        self.providers: Dict[str, ModelProvider] = {}
        
        # Register pose estimator provider
        self.pose_provider = pose_provider or PoseEstimatorProvider()
        self.register_provider(self.pose_provider)
        
        # Register LLM model provider
        self.llm_provider = llm_provider or LLMModelProvider(config_manager)
        self.register_provider(self.llm_provider)
        
        logger.info(f"Models service initialized with {len(self.providers)} providers")
    
    def register_provider(self, provider: ModelProvider) -> None:
        """
        Register a new model provider.
        
        This allows for extensibility - new model types can be added
        by implementing the ModelProvider protocol and registering them.
        
        Args:
            provider: Model provider instance
        """
        provider_type = provider.get_provider_type()
        self.providers[provider_type] = provider
        logger.info(f"Registered model provider: {provider_type}")
    
    def get_all_models(self) -> Dict[str, Any]:
        """
        Get all available models from all providers.
        
        Returns:
            Dictionary containing models grouped by type
        """
        result = {
            "models": [],
            "by_type": {},
            "by_category": {},
            "summary": {
                "total_models": 0,
                "by_type": {},
                "by_category": {}
            }
        }
        
        for provider_type, provider in self.providers.items():
            try:
                models = provider.get_models()
                result["models"].extend(models)
                result["by_type"][provider_type] = models
                
                # Group by category
                for model in models:
                    category = model.get("category", "other")
                    if category not in result["by_category"]:
                        result["by_category"][category] = []
                    result["by_category"][category].append(model)
                
                # Update summary
                result["summary"]["by_type"][provider_type] = len(models)
                
            except Exception as e:
                logger.error(f"Error getting models from provider {provider_type}: {e}")
                result["by_type"][provider_type] = []
                result["summary"]["by_type"][provider_type] = 0
        
        # Calculate category counts
        for category, models in result["by_category"].items():
            result["summary"]["by_category"][category] = len(models)
        
        result["summary"]["total_models"] = len(result["models"])
        
        return result
    
    def get_models_by_type(self, model_type: str) -> List[Dict[str, Any]]:
        """
        Get models of a specific type.
        
        Args:
            model_type: Type of models to retrieve (e.g., "pose_estimator", "llm_model")
            
        Returns:
            List of models of the specified type
        """
        if model_type not in self.providers:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return self.providers[model_type].get_models()
    
    def get_model_info(self, model_type: str, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_type: Type of model (e.g., "pose_estimator", "llm_model")
            model_name: Name of the model
            
        Returns:
            Detailed model information
        """
        if model_type not in self.providers:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return self.providers[model_type].get_model_info(model_name)
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of available model provider types.
        
        Returns:
            List of provider type names
        """
        return list(self.providers.keys())
    
    def search_models(self, query: str, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for models by name or description.
        
        Args:
            query: Search query string
            model_type: Optional model type to filter by
            
        Returns:
            List of matching models
        """
        all_models = self.get_all_models()["models"]
        
        # Filter by type if specified
        if model_type:
            all_models = [m for m in all_models if m.get("type") == model_type]
        
        # Search by name or description
        query_lower = query.lower()
        results = []
        
        for model in all_models:
            if (query_lower in model.get("name", "").lower() or
                query_lower in model.get("display_name", "").lower() or
                query_lower in model.get("description", "").lower()):
                results.append(model)
        
        return results
    
    def get_models_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get models by category.
        
        Args:
            category: Category name (e.g., "pose_estimation", "classification")
            
        Returns:
            List of models in the specified category
        """
        all_models = self.get_all_models()
        return all_models["by_category"].get(category, [])
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about available models.
        
        Returns:
            Dictionary with model statistics
        """
        all_models = self.get_all_models()
        
        stats = {
            "total_models": all_models["summary"]["total_models"],
            "by_type": all_models["summary"]["by_type"],
            "by_category": all_models["summary"]["by_category"],
            "available_count": sum(1 for m in all_models["models"] if m.get("available", False)),
            "unavailable_count": sum(1 for m in all_models["models"] if not m.get("available", False)),
            "providers": self.get_available_providers()
        }
        
        return stats
