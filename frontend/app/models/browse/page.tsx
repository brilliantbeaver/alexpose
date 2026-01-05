/**
 * Models Browse Page
 * Browse available pose estimation models and LLM classification models
 */

'use client';

import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';

interface Model {
  name: string;
  display_name: string;
  type: string;
  category: string;
  provider: string;
  available: boolean;
  description?: string;
  error?: string;
  // Pose estimator fields
  class_name?: string;
  module?: string;
  // LLM model fields
  capability?: string;
  cost_tier?: string;
  multimodal?: boolean;
  reasoning?: string;
}

interface ModelsData {
  models: Model[];
  by_type: {
    pose_estimator: Model[];
    llm_model: Model[];
  };
  by_category: {
    [key: string]: Model[];
  };
  summary: {
    total_models: number;
    by_type: {
      [key: string]: number;
    };
    by_category: {
      [key: string]: number;
    };
  };
}

interface Statistics {
  total_models: number;
  by_type: {
    [key: string]: number;
  };
  by_category: {
    [key: string]: number;
  };
  available_count: number;
  unavailable_count: number;
  providers: string[];
}

export default function ModelsBrowsePage() {
  const [modelsData, setModelsData] = useState<ModelsData | null>(null);
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filterType, setFilterType] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      setLoading(true);
      setError(null);

      console.log('[Models Browse] Fetching models from API...');
      
      // Load models list
      const modelsResponse = await fetch('http://localhost:8000/api/v1/models/list');
      console.log('[Models Browse] Models response status:', modelsResponse.status);

      if (!modelsResponse.ok) {
        throw new Error(`Failed to load models: ${modelsResponse.statusText}`);
      }

      const modelsResult = await modelsResponse.json();
      console.log('[Models Browse] Models result:', modelsResult);

      if (modelsResult.success) {
        setModelsData(modelsResult);
      }

      // Load statistics
      const statsResponse = await fetch('http://localhost:8000/api/v1/models/statistics');
      console.log('[Models Browse] Statistics response status:', statsResponse.status);

      if (statsResponse.ok) {
        const statsResult = await statsResponse.json();
        console.log('[Models Browse] Statistics result:', statsResult);

        if (statsResult.success) {
          setStatistics(statsResult.statistics);
        }
      }

    } catch (err) {
      console.error('[Models Browse] Error loading models:', err);
      const errorMsg = err instanceof Error ? err.message : 'Failed to load models';
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const getAvailabilityBadge = (model: Model) => {
    if (model.available) {
      return <Badge variant="default" className="bg-green-600">Available</Badge>;
    } else {
      return <Badge variant="destructive">Unavailable</Badge>;
    }
  };

  const getCostTierBadge = (costTier?: string) => {
    if (!costTier) return null;
    
    const colors = {
      low: 'bg-green-600',
      medium: 'bg-blue-600',
      high: 'bg-orange-600',
      premium: 'bg-purple-600'
    };

    return (
      <Badge variant="default" className={colors[costTier as keyof typeof colors] || 'bg-gray-600'}>
        {costTier}
      </Badge>
    );
  };

  const getReasoningBadge = (reasoning?: string) => {
    if (!reasoning || reasoning === 'standard') return null;
    
    return (
      <Badge variant="secondary">
        {reasoning} reasoning
      </Badge>
    );
  };

  const getFilteredModels = () => {
    if (!modelsData) return [];

    let filtered = modelsData.models;

    // Filter by type
    if (filterType) {
      filtered = filtered.filter(m => m.type === filterType);
    }

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(m => 
        m.name.toLowerCase().includes(query) ||
        m.display_name.toLowerCase().includes(query) ||
        (m.description && m.description.toLowerCase().includes(query))
      );
    }

    return filtered;
  };

  const renderModelCard = (model: Model) => {
    const isPoseEstimator = model.type === 'pose_estimator';
    const isLLM = model.type === 'llm_model';

    return (
      <Card key={model.name} className="hover:shadow-md transition-shadow">
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <CardTitle className="text-lg">{model.display_name}</CardTitle>
              <CardDescription className="mt-1">
                {model.description || `${model.name} model`}
              </CardDescription>
            </div>
            {getAvailabilityBadge(model)}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Model Type and Provider */}
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline">
              {model.category.replace('_', ' ')}
            </Badge>
            <Badge variant="secondary">
              {model.provider}
            </Badge>
            {isLLM && getCostTierBadge(model.cost_tier)}
            {isLLM && getReasoningBadge(model.reasoning)}
            {isLLM && model.multimodal && (
              <Badge variant="default" className="bg-indigo-600">
                Multimodal
              </Badge>
            )}
          </div>

          {/* Pose Estimator Details */}
          {isPoseEstimator && (
            <div className="text-sm space-y-1">
              {model.class_name && (
                <div>
                  <span className="text-muted-foreground">Class:</span>{' '}
                  <span className="font-mono text-xs">{model.class_name}</span>
                </div>
              )}
              {model.module && (
                <div>
                  <span className="text-muted-foreground">Module:</span>{' '}
                  <span className="font-mono text-xs">{model.module}</span>
                </div>
              )}
            </div>
          )}

          {/* LLM Details */}
          {isLLM && (
            <div className="text-sm space-y-1">
              {model.capability && (
                <div>
                  <span className="text-muted-foreground">Capability:</span>{' '}
                  <span>{model.capability.replace('_', ' ')}</span>
                </div>
              )}
            </div>
          )}

          {/* Error Message */}
          {model.error && (
            <Alert variant="destructive" className="py-2">
              <AlertDescription className="text-xs">
                {model.error}
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    );
  };

  if (loading) {
    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Browse Models</h1>
            <p className="text-muted-foreground">Loading available models...</p>
          </div>
        </div>

        {/* Loading Spinner */}
        <div className="flex flex-col items-center justify-center py-12">
          <div className="relative w-16 h-16">
            <div className="absolute top-0 left-0 w-full h-full border-4 border-blue-200 rounded-full"></div>
            <div className="absolute top-0 left-0 w-full h-full border-4 border-blue-600 rounded-full animate-spin border-t-transparent"></div>
          </div>
          <p className="mt-4 text-sm text-muted-foreground">Loading models...</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-6 w-32 mb-2" />
                <Skeleton className="h-4 w-full" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-20 w-full" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (error || !modelsData) {
    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Browse Models</h1>
            <p className="text-muted-foreground">Explore available models</p>
          </div>
        </div>

        <Alert variant="destructive">
          <AlertTitle>Error Loading Models</AlertTitle>
          <AlertDescription>
            {error || 'Failed to load models. Please try again.'}
            <Button 
              variant="outline" 
              size="sm" 
              className="ml-4"
              onClick={loadModels}
            >
              Retry
            </Button>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const filteredModels = getFilteredModels();
  const poseEstimators = filteredModels.filter(m => m.type === 'pose_estimator');
  const llmModels = filteredModels.filter(m => m.type === 'llm_model');

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Browse Models</h1>
          <p className="text-muted-foreground">
            Explore available pose estimation and LLM classification models
          </p>
        </div>
      </div>

      {/* Statistics Cards */}
      {statistics && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Total Models</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{statistics.total_models}</div>
              <p className="text-xs text-muted-foreground">
                {statistics.available_count} available
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Pose Estimators</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">
                {statistics.by_type.pose_estimator || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                For pose detection
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">LLM Models</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-purple-600">
                {statistics.by_type.llm_model || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                For classification
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Providers</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">
                {statistics.providers.length}
              </div>
              <p className="text-xs text-muted-foreground">
                Model providers
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Filters and Search */}
      <Card>
        <CardHeader>
          <CardTitle>Filter Models</CardTitle>
          <CardDescription>Search and filter by model type</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Search */}
          <div>
            <input
              type="text"
              placeholder="Search models by name or description..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Type Filter */}
          <div className="flex gap-2">
            <Button
              variant={filterType === null ? 'default' : 'outline'}
              size="sm"
              onClick={() => setFilterType(null)}
            >
              All Models ({modelsData.models.length})
            </Button>
            <Button
              variant={filterType === 'pose_estimator' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setFilterType('pose_estimator')}
            >
              Pose Estimators ({modelsData.by_type.pose_estimator?.length || 0})
            </Button>
            <Button
              variant={filterType === 'llm_model' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setFilterType('llm_model')}
            >
              LLM Models ({modelsData.by_type.llm_model?.length || 0})
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Pose Estimators Section */}
      {(!filterType || filterType === 'pose_estimator') && poseEstimators.length > 0 && (
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-bold">Pose Estimation Models</h2>
            <p className="text-muted-foreground">
              Models for detecting and tracking human pose keypoints
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {poseEstimators.map(renderModelCard)}
          </div>
        </div>
      )}

      {/* LLM Models Section */}
      {(!filterType || filterType === 'llm_model') && llmModels.length > 0 && (
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-bold">LLM Classification Models</h2>
            <p className="text-muted-foreground">
              Large language models for gait pattern classification and analysis
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {llmModels.map(renderModelCard)}
          </div>
        </div>
      )}

      {/* No Results */}
      {filteredModels.length === 0 && (
        <Alert>
          <AlertTitle>No Models Found</AlertTitle>
          <AlertDescription>
            No models match your current filters. Try adjusting your search or filter criteria.
            <div className="mt-4">
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => {
                  setSearchQuery('');
                  setFilterType(null);
                }}
              >
                Clear Filters
              </Button>
            </div>
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
