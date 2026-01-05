"""
Integration tests for Models Browse Page and API endpoints.

Tests the complete flow from frontend to backend for models browsing.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from server.main import app
from ambient.core.config import ConfigurationManager


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_config():
    """Create mock configuration with LLM models."""
    config = Mock(spec=ConfigurationManager)
    config.llm_models = {
        "gpt-5.2": {
            "provider": "openai",
            "capability": "multimodal",
            "description": "Latest GPT-5.2 flagship model",
            "max_tokens": 16384,
            "context_window": 200000,
            "cost_tier": "premium",
            "multimodal": True,
            "reasoning": "advanced"
        },
        "gemini-3-flash-preview": {
            "provider": "gemini",
            "capability": "multimodal",
            "description": "Gemini 3 Flash Preview - fast model",
            "max_tokens": 8192,
            "context_window": 1000000,
            "cost_tier": "medium",
            "multimodal": True,
            "reasoning": "advanced"
        }
    }
    return config


class TestModelsListEndpoint:
    """Test /api/v1/models/list endpoint."""
    
    def test_list_all_models_success(self, client, mock_config):
        """Test listing all models returns success."""
        with patch.object(app.state, 'config', mock_config):
            response = client.get("/api/v1/models/list")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert "models" in data
            assert "by_type" in data
            assert "by_category" in data
            assert "summary" in data
            
            # Verify structure
            assert isinstance(data["models"], list)
            assert len(data["models"]) > 0
            
            # Verify types are present
            assert "pose_estimator" in data["by_type"]
            assert "llm_model" in data["by_type"]
            
            # Verify summary
            assert data["summary"]["total_models"] > 0
            assert "pose_estimator" in data["summary"]["by_type"]
            assert "llm_model" in data["summary"]["by_type"]
    
    def test_list_models_includes_pose_estimators(self, client, mock_config):
        """Test that pose estimators are included in list."""
        with patch.object(app.state, 'config', mock_config):
            response = client.get("/api/v1/models/list")
            data = response.json()
            
            pose_models = data["by_type"]["pose_estimator"]
            assert len(pose_models) > 0
            
            # Verify structure of pose estimator
            for model in pose_models:
                assert model["type"] == "pose_estimator"
                assert model["category"] == "pose_estimation"
                assert "name" in model
                assert "display_name" in model
                assert "available" in model
    
    def test_list_models_includes_llm_models(self, client, mock_config):
        """Test that LLM models are included in list."""
        with patch.object(app.state, 'config', mock_config):
            response = client.get("/api/v1/models/list")
            data = response.json()
            
            llm_models = data["by_type"]["llm_model"]
            assert len(llm_models) == 2  # gpt-5.2 and gemini-3-flash-preview
            
            # Verify structure of LLM model
            for model in llm_models:
                assert model["type"] == "llm_model"
                assert model["category"] == "classification"
                assert "name" in model
                assert "display_name" in model
                assert "provider" in model
                assert "cost_tier" in model


class TestModelsStatisticsEndpoint:
    """Test /api/v1/models/statistics endpoint."""
    
    def test_statistics_success(self, client, mock_config):
        """Test getting model statistics."""
        with patch.object(app.state, 'config', mock_config):
            response = client.get("/api/v1/models/statistics")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert "statistics" in data
            
            stats = data["statistics"]
            assert "total_models" in stats
            assert "by_type" in stats
            assert "by_category" in stats
            assert "available_count" in stats
            assert "unavailable_count" in stats
            assert "providers" in stats
            
            # Verify counts
            assert stats["total_models"] > 0
            assert stats["available_count"] >= 0
            assert stats["unavailable_count"] >= 0
            assert stats["total_models"] == stats["available_count"] + stats["unavailable_count"]
    
    def test_statistics_includes_provider_types(self, client, mock_config):
        """Test that statistics include provider types."""
        with patch.object(app.state, 'config', mock_config):
            response = client.get("/api/v1/models/statistics")
            data = response.json()
            
            providers = data["statistics"]["providers"]
            assert "pose_estimator" in providers
            assert "llm_model" in providers


class TestModelsSearchEndpoint:
    """Test /api/v1/models/search endpoint."""
    
    def test_search_by_name(self, client, mock_config):
        """Test searching models by name."""
        with patch.object(app.state, 'config', mock_config):
            response = client.get("/api/v1/models/search?query=gpt")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert "results" in data
            assert data["query"] == "gpt"
            
            # Should find gpt-5.2
            assert len(data["results"]) > 0
            assert any("gpt" in r["name"].lower() for r in data["results"])
    
    def test_search_with_type_filter(self, client, mock_config):
        """Test searching with type filter."""
        with patch.object(app.state, 'config', mock_config):
            response = client.get("/api/v1/models/search?query=model&model_type=llm_model")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["model_type"] == "llm_model"
            
            # All results should be LLM models
            for result in data["results"]:
                assert result["type"] == "llm_model"
    
    def test_search_no_results(self, client, mock_config):
        """Test search with no matching results."""
        with patch.object(app.state, 'config', mock_config):
            response = client.get("/api/v1/models/search?query=nonexistent_model_xyz")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert len(data["results"]) == 0
            assert data["count"] == 0


class TestModelsTypeEndpoint:
    """Test /api/v1/models/type/{model_type} endpoint."""
    
    def test_get_pose_estimators(self, client, mock_config):
        """Test getting pose estimators by type."""
        with patch.object(app.state, 'config', mock_config):
            response = client.get("/api/v1/models/type/pose_estimator")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["model_type"] == "pose_estimator"
            assert "models" in data
            
            # All should be pose estimators
            for model in data["models"]:
                assert model["type"] == "pose_estimator"
    
    def test_get_llm_models(self, client, mock_config):
        """Test getting LLM models by type."""
        with patch.object(app.state, 'config', mock_config):
            response = client.get("/api/v1/models/type/llm_model")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["model_type"] == "llm_model"
            
            # All should be LLM models
            for model in data["models"]:
                assert model["type"] == "llm_model"
    
    def test_get_invalid_type(self, client, mock_config):
        """Test getting models with invalid type."""
        with patch.object(app.state, 'config', mock_config):
            response = client.get("/api/v1/models/type/invalid_type")
            
            assert response.status_code == 404
            data = response.json()
            assert "Unknown model type" in data["detail"]


class TestModelsCategoryEndpoint:
    """Test /api/v1/models/category/{category} endpoint."""
    
    def test_get_pose_estimation_category(self, client, mock_config):
        """Test getting models by pose_estimation category."""
        with patch.object(app.state, 'config', mock_config):
            response = client.get("/api/v1/models/category/pose_estimation")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["category"] == "pose_estimation"
            
            # All should be in pose_estimation category
            for model in data["models"]:
                assert model["category"] == "pose_estimation"
    
    def test_get_classification_category(self, client, mock_config):
        """Test getting models by classification category."""
        with patch.object(app.state, 'config', mock_config):
            response = client.get("/api/v1/models/category/classification")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["category"] == "classification"
            
            # All should be in classification category
            for model in data["models"]:
                assert model["category"] == "classification"


class TestModelsDetailEndpoint:
    """Test /api/v1/models/{model_type}/{model_name} endpoint."""
    
    def test_get_pose_estimator_detail(self, client, mock_config):
        """Test getting detailed info for pose estimator."""
        with patch.object(app.state, 'config', mock_config):
            response = client.get("/api/v1/models/pose_estimator/mediapipe")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert "model" in data
            
            model = data["model"]
            assert model["name"] == "mediapipe"
            assert model["type"] == "pose_estimator"
            assert "description" in model
    
    def test_get_llm_model_detail(self, client, mock_config):
        """Test getting detailed info for LLM model."""
        with patch.object(app.state, 'config', mock_config):
            response = client.get("/api/v1/models/llm_model/gpt-5.2")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            model = data["model"]
            
            assert model["name"] == "gpt-5.2"
            assert model["type"] == "llm_model"
            assert model["provider"] == "openai"
            assert "max_tokens" in model
            assert "context_window" in model
    
    def test_get_nonexistent_model(self, client, mock_config):
        """Test getting info for nonexistent model."""
        with patch.object(app.state, 'config', mock_config):
            response = client.get("/api/v1/models/llm_model/nonexistent")
            
            assert response.status_code == 404
            data = response.json()
            assert "not found" in data["detail"].lower()


class TestModelsProvidersEndpoint:
    """Test /api/v1/models/providers endpoint."""
    
    def test_list_providers(self, client, mock_config):
        """Test listing available providers."""
        with patch.object(app.state, 'config', mock_config):
            response = client.get("/api/v1/models/providers")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert "providers" in data
            assert "count" in data
            
            providers = data["providers"]
            assert "pose_estimator" in providers
            assert "llm_model" in providers
            assert data["count"] == len(providers)


class TestModelsIntegrationFlow:
    """Test complete integration flow for models browsing."""
    
    def test_complete_browse_flow(self, client, mock_config):
        """Test complete flow: list -> filter -> detail."""
        with patch.object(app.state, 'config', mock_config):
            # 1. List all models
            list_response = client.get("/api/v1/models/list")
            assert list_response.status_code == 200
            list_data = list_response.json()
            assert list_data["success"] is True
            
            # 2. Get statistics
            stats_response = client.get("/api/v1/models/statistics")
            assert stats_response.status_code == 200
            stats_data = stats_response.json()
            assert stats_data["success"] is True
            
            # Verify counts match
            assert stats_data["statistics"]["total_models"] == len(list_data["models"])
            
            # 3. Filter by type
            type_response = client.get("/api/v1/models/type/llm_model")
            assert type_response.status_code == 200
            type_data = type_response.json()
            assert type_data["success"] is True
            
            # 4. Get detail for first LLM model
            if type_data["models"]:
                first_model = type_data["models"][0]
                detail_response = client.get(f"/api/v1/models/llm_model/{first_model['name']}")
                assert detail_response.status_code == 200
                detail_data = detail_response.json()
                assert detail_data["success"] is True
                assert detail_data["model"]["name"] == first_model["name"]
    
    def test_search_and_filter_flow(self, client, mock_config):
        """Test search and filter flow."""
        with patch.object(app.state, 'config', mock_config):
            # 1. Search for all models
            search_all = client.get("/api/v1/models/search?query=model")
            assert search_all.status_code == 200
            all_results = search_all.json()["results"]
            
            # 2. Search with type filter
            search_filtered = client.get("/api/v1/models/search?query=model&model_type=pose_estimator")
            assert search_filtered.status_code == 200
            filtered_results = search_filtered.json()["results"]
            
            # Filtered should be subset of all
            assert len(filtered_results) <= len(all_results)
            
            # All filtered results should be pose estimators
            for result in filtered_results:
                assert result["type"] == "pose_estimator"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
