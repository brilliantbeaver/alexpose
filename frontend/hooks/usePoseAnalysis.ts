/**
 * React hook for managing pose analysis data
 */

'use client';

import { useState, useEffect, useCallback } from 'react';
import { apiClient, APIError } from '@/lib/api-client';
import { PoseAnalysisResult } from '@/lib/pose-analysis-types';

interface UsePoseAnalysisOptions {
  useCache?: boolean;
  forceRefresh?: boolean;
  autoFetch?: boolean;
}

interface UsePoseAnalysisReturn {
  analysis: PoseAnalysisResult | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  clearError: () => void;
  status: {
    exists: boolean;
    cached: boolean;
    lastUpdated?: string;
  } | null;
}

export function usePoseAnalysis(
  datasetId: string | null,
  sequenceId: string | null,
  options: UsePoseAnalysisOptions = {}
): UsePoseAnalysisReturn {
  const {
    useCache = true,
    forceRefresh = false,
    autoFetch = true,
  } = options;

  const [analysis, setAnalysis] = useState<PoseAnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<{
    exists: boolean;
    cached: boolean;
    lastUpdated?: string;
  } | null>(null);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const fetchAnalysis = useCallback(async () => {
    if (!datasetId || !sequenceId) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // First check status
      const statusResponse = await apiClient.getPoseAnalysisStatus(datasetId, sequenceId);
      setStatus(statusResponse);

      // If analysis doesn't exist and we're not forcing refresh, show appropriate message
      if (!statusResponse.exists && !forceRefresh) {
        setError('No analysis found for this sequence. Analysis may need to be generated first.');
        setAnalysis(null);
        return;
      }

      // Fetch the full analysis
      const response = await apiClient.getPoseAnalysis(datasetId, sequenceId, {
        useCache,
        forceRefresh,
      });

      if (response.success && response.analysis) {
        setAnalysis(response.analysis);
        setError(null);
      } else {
        setError('Failed to load analysis data');
        setAnalysis(null);
      }
    } catch (err) {
      console.error('Error fetching pose analysis:', err);
      
      if (err instanceof APIError) {
        switch (err.status) {
          case 404:
            setError('Analysis not found. The sequence may not have been analyzed yet.');
            break;
          case 400:
            setError('Invalid request. Please check the dataset and sequence IDs.');
            break;
          case 500:
            setError('Server error occurred while processing the analysis.');
            break;
          case 0:
            setError('Network error. Please check your connection and try again.');
            break;
          default:
            setError(err.message || 'An unexpected error occurred');
        }
      } else {
        setError('An unexpected error occurred while loading the analysis');
      }
      
      setAnalysis(null);
    } finally {
      setLoading(false);
    }
  }, [datasetId, sequenceId, useCache, forceRefresh]);

  const refetch = useCallback(async () => {
    await fetchAnalysis();
  }, [fetchAnalysis]);

  // Auto-fetch on mount and when dependencies change
  useEffect(() => {
    if (autoFetch && datasetId && sequenceId) {
      fetchAnalysis();
    }
  }, [autoFetch, fetchAnalysis, datasetId, sequenceId]);

  return {
    analysis,
    loading,
    error,
    refetch,
    clearError,
    status,
  };
}

// Hook for fetching analysis features only
export function usePoseAnalysisFeatures(
  datasetId: string | null,
  sequenceId: string | null
) {
  const [features, setFeatures] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchFeatures = useCallback(async () => {
    if (!datasetId || !sequenceId) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await apiClient.getPoseAnalysisFeatures(datasetId, sequenceId);
      setFeatures(response);
    } catch (err) {
      console.error('Error fetching pose analysis features:', err);
      setError(err instanceof APIError ? err.message : 'Failed to load features');
    } finally {
      setLoading(false);
    }
  }, [datasetId, sequenceId]);

  useEffect(() => {
    fetchFeatures();
  }, [fetchFeatures]);

  return {
    features,
    loading,
    error,
    refetch: fetchFeatures,
  };
}

// Hook for fetching gait cycles only
export function usePoseAnalysisCycles(
  datasetId: string | null,
  sequenceId: string | null
) {
  const [cycles, setCycles] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchCycles = useCallback(async () => {
    if (!datasetId || !sequenceId) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await apiClient.getPoseAnalysisCycles(datasetId, sequenceId);
      setCycles(response);
    } catch (err) {
      console.error('Error fetching pose analysis cycles:', err);
      setError(err instanceof APIError ? err.message : 'Failed to load gait cycles');
    } finally {
      setLoading(false);
    }
  }, [datasetId, sequenceId]);

  useEffect(() => {
    fetchCycles();
  }, [fetchCycles]);

  return {
    cycles,
    loading,
    error,
    refetch: fetchCycles,
  };
}

// Hook for fetching symmetry analysis only
export function usePoseAnalysisSymmetry(
  datasetId: string | null,
  sequenceId: string | null
) {
  const [symmetry, setSymmetry] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSymmetry = useCallback(async () => {
    if (!datasetId || !sequenceId) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await apiClient.getPoseAnalysisSymmetry(datasetId, sequenceId);
      setSymmetry(response);
    } catch (err) {
      console.error('Error fetching pose analysis symmetry:', err);
      setError(err instanceof APIError ? err.message : 'Failed to load symmetry analysis');
    } finally {
      setLoading(false);
    }
  }, [datasetId, sequenceId]);

  useEffect(() => {
    fetchSymmetry();
  }, [fetchSymmetry]);

  return {
    symmetry,
    loading,
    error,
    refetch: fetchSymmetry,
  };
}