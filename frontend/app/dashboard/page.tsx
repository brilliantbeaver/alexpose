/**
 * Dashboard Page
 * Overview of recent analyses and quick actions
 */

'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';
import { AnalysesTable } from '@/components/ui/analyses-table';

interface DashboardStatistics {
  total_analyses: number;
  total_gait_analyses: number;
  total_gavd_datasets: number;
  normal_patterns: number;
  abnormal_patterns: number;
  normal_percentage: number;
  abnormal_percentage: number;
  avg_confidence: number;
  gavd_completed: number;
  gavd_processing: number;
  gavd_uploaded: number;
  gavd_error: number;
  total_gavd_sequences: number;
  total_gavd_frames: number;
  recent_analyses: RecentAnalysis[];
  status_breakdown: {
    gait_analysis: {
      pending: number;
      running: number;
      completed: number;
      failed: number;
    };
    gavd_datasets: {
      uploaded: number;
      processing: number;
      completed: number;
      error: number;
    };
  };
  completed_count: number;
}

interface RecentAnalysis {
  type: 'gait_analysis' | 'gavd_dataset';
  // Gait analysis fields
  analysis_id?: string;
  file_id?: string;
  is_normal?: boolean | null;
  confidence?: number;
  explanation?: string;
  identified_conditions?: string[];
  frame_count?: number;
  duration?: number;
  // GAVD dataset fields
  dataset_id?: string;
  filename?: string;
  sequence_count?: number;
  row_count?: number;
  total_sequences_processed?: number;
  total_frames_processed?: number;
  progress?: string;
  // GAVD sequence-specific fields
  seq?: string;
  gait_pat?: string;
  // Common fields
  status: string;
  created_at?: string;
  uploaded_at?: string;
  completed_at?: string;
}

export default function DashboardPage() {
  const [statistics, setStatistics] = useState<DashboardStatistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deletingAnalysis, setDeletingAnalysis] = useState<string | null>(null);

  useEffect(() => {
    loadDashboardStatistics();
  }, []);

  const loadDashboardStatistics = async () => {
    try {
      setLoading(true);
      setError(null);
      
      console.log('[Dashboard] Fetching statistics from API...');
      const response = await fetch('http://localhost:8000/api/v1/analysis/statistics');
      console.log('[Dashboard] Response status:', response.status);
      
      const result = await response.json();
      console.log('[Dashboard] Response data:', result);
      
      if (response.ok && result.success) {
        setStatistics(result.statistics);
        console.log('[Dashboard] Statistics loaded successfully');
      } else {
        const errorMsg = result.message || result.detail || 'Failed to load dashboard statistics';
        console.error('[Dashboard] API error:', errorMsg);
        setError(errorMsg);
      }
    } catch (err) {
      console.error('[Dashboard] Error loading dashboard statistics:', err);
      const errorMsg = err instanceof Error ? err.message : 'Failed to connect to server. Please ensure the backend is running.';
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteAnalysis = async (analysis: RecentAnalysis) => {
    const title = analysis.type === 'gavd_dataset' 
      ? (analysis.filename || 'GAVD Dataset')
      : `Analysis ${(analysis.analysis_id || '').substring(0, 8)}`;
    const id = analysis.dataset_id || analysis.analysis_id;
    
    if (!id) return;
    
    const confirmMessage = analysis.type === 'gavd_dataset'
      ? `Are you sure you want to delete "${title}"?\n\nThis will permanently delete:\n• Original CSV file\n• All processing results\n• Pose data\n• Downloaded videos\n\nThis action cannot be undone.`
      : `Are you sure you want to delete analysis "${title}"?\n\nThis will permanently delete all analysis results.\n\nThis action cannot be undone.`;
    
    if (!confirm(confirmMessage)) {
      return;
    }
    
    setDeletingAnalysis(id);
    
    try {
      const endpoint = analysis.type === 'gavd_dataset'
        ? `http://localhost:8000/api/v1/gavd/${id}`
        : `http://localhost:8000/api/v1/analysis/${id}`;
      
      const response = await fetch(endpoint, {
        method: 'DELETE',
      });
      
      const result = await response.json();
      
      if (response.ok && result.success) {
        // Reload dashboard statistics
        loadDashboardStatistics();
      } else {
        alert(`Failed to delete: ${result.detail || result.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Delete error:', error);
      alert(`Error deleting: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setDeletingAnalysis(null);
    }
  };

  if (loading) {
    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Dashboard</h1>
            <p className="text-muted-foreground">Loading your overview...</p>
          </div>
        </div>

        {/* Loading Spinner */}
        <div className="flex flex-col items-center justify-center py-12">
          <div className="relative w-16 h-16">
            <div className="absolute top-0 left-0 w-full h-full border-4 border-blue-200 rounded-full"></div>
            <div className="absolute top-0 left-0 w-full h-full border-4 border-blue-600 rounded-full animate-spin border-t-transparent"></div>
          </div>
          <p className="mt-4 text-sm text-muted-foreground">Loading dashboard statistics...</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i}>
              <CardHeader className="pb-3">
                <Skeleton className="h-4 w-24" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-8 w-16 mb-2" />
                <Skeleton className="h-3 w-32" />
              </CardContent>
            </Card>
          ))}
        </div>

        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-32" />
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-20 w-full" />
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error || !statistics) {
    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Dashboard</h1>
            <p className="text-muted-foreground">Welcome back! Here's your overview.</p>
          </div>
          <Button asChild>
            <Link href="/analyze/upload">
              📤 New Analysis
            </Link>
          </Button>
        </div>

        <Alert variant="destructive">
          <AlertTitle>Error Loading Dashboard</AlertTitle>
          <AlertDescription>
            {error || 'Failed to load dashboard data. Please try again.'}
            <Button 
              variant="outline" 
              size="sm" 
              className="ml-4"
              onClick={loadDashboardStatistics}
            >
              Retry
            </Button>
          </AlertDescription>
        </Alert>

        <Alert>
          <AlertTitle>No Analyses Yet</AlertTitle>
          <AlertDescription>
            Get started by uploading a video or analyzing a YouTube URL.
            <div className="mt-4 flex gap-2">
              <Button asChild size="sm">
                <Link href="/analyze/upload">📤 Upload Video</Link>
              </Button>
              <Button asChild size="sm" variant="outline">
                <Link href="/analyze/youtube">🔗 YouTube URL</Link>
              </Button>
            </div>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const { 
    total_analyses, 
    total_gait_analyses,
    total_gavd_datasets,
    normal_patterns, 
    abnormal_patterns, 
    normal_percentage, 
    abnormal_percentage, 
    avg_confidence, 
    gavd_completed,
    gavd_processing,
    total_gavd_sequences,
    total_gavd_frames,
    recent_analyses, 
    status_breakdown 
  } = statistics;

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <p className="text-muted-foreground">Welcome back! Here's your overview.</p>
        </div>
        <Button asChild>
          <Link href="/analyze/upload">
            📤 New Analysis
          </Link>
        </Button>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="hover:shadow-md transition-shadow cursor-pointer">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Total Analyses</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{total_analyses}</div>
            <p className="text-xs text-muted-foreground">
              {total_gait_analyses} gait, {total_gavd_datasets} GAVD
            </p>
          </CardContent>
        </Card>

        <Card className="hover:shadow-md transition-shadow cursor-pointer">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">GAVD Datasets</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">{gavd_completed}</div>
            <p className="text-xs text-muted-foreground">
              {gavd_processing} processing, {total_gavd_sequences} sequences
            </p>
          </CardContent>
        </Card>

        <Card className="hover:shadow-md transition-shadow cursor-pointer">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Normal Patterns</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">{normal_patterns}</div>
            <p className="text-xs text-muted-foreground">{normal_percentage.toFixed(0)}% of gait analyses</p>
          </CardContent>
        </Card>

        <Card className="hover:shadow-md transition-shadow cursor-pointer">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Avg. Confidence</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-purple-600">{avg_confidence.toFixed(0)}%</div>
            <p className="text-xs text-muted-foreground">
              {avg_confidence >= 90 ? 'High accuracy' : avg_confidence >= 70 ? 'Good accuracy' : 'Moderate accuracy'}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Recent Analyses */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Recent Analyses</CardTitle>
              <CardDescription>Your latest gait analyses and GAVD datasets</CardDescription>
            </div>
            <Button asChild variant="outline" size="sm">
              <Link href="/analyses">View All →</Link>
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {recent_analyses.length === 0 ? (
            <Alert>
              <AlertTitle>No Completed Analyses</AlertTitle>
              <AlertDescription>
                Start analyzing videos or upload GAVD datasets to see results here.
                <div className="mt-4 flex gap-2">
                  <Button asChild size="sm">
                    <Link href="/analyze/upload">📤 Upload Video</Link>
                  </Button>
                  <Button asChild size="sm" variant="outline">
                    <Link href="/analyze/youtube">🔗 YouTube URL</Link>
                  </Button>
                  <Button asChild size="sm" variant="outline">
                    <Link href="/gavd">📊 Upload GAVD</Link>
                  </Button>
                </div>
              </AlertDescription>
            </Alert>
          ) : (
            <AnalysesTable 
              analyses={recent_analyses}
              onDelete={handleDeleteAnalysis}
              deletingAnalysis={deletingAnalysis}
              maxRows={5}
              showFilters={true}
            />
          )}
        </CardContent>
      </Card>

      {/* System Status */}
      <Alert>
        <AlertTitle>✅ System Status: Operational</AlertTitle>
        <AlertDescription>
          All services are running normally. Ready to process new analyses.
          {status_breakdown.gait_analysis.failed > 0 && (
            <span className="text-orange-600 ml-2">
              ({status_breakdown.gait_analysis.failed} failed gait analysis{status_breakdown.gait_analysis.failed > 1 ? 'es' : ''})
            </span>
          )}
          {status_breakdown.gavd_datasets.error > 0 && (
            <span className="text-orange-600 ml-2">
              ({status_breakdown.gavd_datasets.error} GAVD dataset error{status_breakdown.gavd_datasets.error > 1 ? 's' : ''})
            </span>
          )}
        </AlertDescription>
      </Alert>
    </div>
  );
}
