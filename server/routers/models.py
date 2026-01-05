"""
Models management endpoints for AlexPose FastAPI application.

Provides endpoints for browsing available models (pose estimators and LLM models)
using a service-oriented architecture with dependency injection.
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Dict, Any, List, Optional
from loguru import logger

from server.services.models_service import ModelsService
from ambient.core.config import ConfigurationManager

router = APIRouter(prefix="/api/v1/models", tags=["models"])


def get_models_service(request: Request) -> ModelsService:
    """
    Dependency injection for ModelsService.
    
    Args:
        request: FastAPI request object
        
    Returns:
        ModelsService instance
    """
    config_manager = request.app.state.config
    return ModelsService(config_manager)


@router.get("/list")
async def list_all_models(
    models_service: ModelsService = Depends(get_models_service)
) -> Dict[str, Any]:
    """
    List all available models from all providers.
    
    Returns:
        Dictionary containing:
        - models: List of all models
        - by_type: Models grouped by type
        - by_category: Models grouped by category
        - summary: Statistics summary
    """
    try:
        result = models_service.get_all_models()
        
        return {
            "success": True,
            **result
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/statistics")
async def get_models_statistics(
    models_service: ModelsService = Depends(get_models_service)
) -> Dict[str, Any]:
    """
    Get statistics about available models.
    
    Returns:
        Statistics including counts by type, category, and availability
    """
    try:
        stats = models_service.get_statistics()
        
        return {
            "success": True,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting model statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.get("/search")
async def search_models(
    query: str,
    model_type: Optional[str] = None,
    models_service: ModelsService = Depends(get_models_service)
) -> Dict[str, Any]:
    """
    Search for models by name or description.
    
    Args:
        query: Search query string
        model_type: Optional model type filter
        
    Returns:
        List of matching models
    """
    try:
        results = models_service.search_models(query, model_type)
        
        return {
            "success": True,
            "query": query,
            "model_type": model_type,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to search models: {str(e)}")


@router.get("/category/{category}")
async def get_models_by_category(
    category: str,
    models_service: ModelsService = Depends(get_models_service)
) -> Dict[str, Any]:
    """
    Get models by category.
    
    Args:
        category: Category name (e.g., "pose_estimation", "classification")
        
    Returns:
        List of models in the specified category
    """
    try:
        models = models_service.get_models_by_category(category)
        
        return {
            "success": True,
            "category": category,
            "models": models,
            "count": len(models)
        }
        
    except Exception as e:
        logger.error(f"Error getting models by category: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


@router.get("/type/{model_type}")
async def get_models_by_type(
    model_type: str,
    models_service: ModelsService = Depends(get_models_service)
) -> Dict[str, Any]:
    """
    Get models of a specific type.
    
    Args:
        model_type: Type of models (e.g., "pose_estimator", "llm_model")
        
    Returns:
        List of models of the specified type
    """
    try:
        models = models_service.get_models_by_type(model_type)
        
        return {
            "success": True,
            "model_type": model_type,
            "models": models,
            "count": len(models)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting models by type: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


@router.get("/{model_type}/{model_name}")
async def get_model_info(
    model_type: str,
    model_name: str,
    models_service: ModelsService = Depends(get_models_service)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific model.
    
    Args:
        model_type: Type of model (e.g., "pose_estimator", "llm_model")
        model_name: Name of the model
        
    Returns:
        Detailed model information
    """
    try:
        model_info = models_service.get_model_info(model_type, model_name)
        
        return {
            "success": True,
            "model": model_info
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.get("/providers")
async def list_providers(
    models_service: ModelsService = Depends(get_models_service)
) -> Dict[str, Any]:
    """
    List available model providers.
    
    Returns:
        List of provider type names
    """
    try:
        providers = models_service.get_available_providers()
        
        return {
            "success": True,
            "providers": providers,
            "count": len(providers)
        }
        
    except Exception as e:
        logger.error(f"Error listing providers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list providers: {str(e)}")
