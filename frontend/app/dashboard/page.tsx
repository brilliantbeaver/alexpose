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

  const formatTimeAgo = (dateString: string): string => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
  };

  const getStatusBadge = (analysis: RecentAnalysis) => {
    if (analysis.type === 'gavd_dataset') {
      const status = analysis.status;
      if (status === 'completed') return <Badge variant="default" className="bg-green-600">Completed</Badge>;
      if (status === 'processing') return <Badge variant="default" className="bg-blue-600">Processing</Badge>;
      if (status === 'uploaded') return <Badge variant="secondary">Uploaded</Badge>;
      if (status === 'error') return <Badge variant="destructive">Error</Badge>;
      return <Badge variant="secondary">{status}</Badge>;
    } else {
      // Gait analysis
      const isNormal = analysis.is_normal;
      if (isNormal === null) return <Badge variant="secondary">Unknown</Badge>;
      return isNormal ? (
        <Badge variant="default" className="bg-green-600">Normal</Badge>
      ) : (
        <Badge variant="destructive">Abnormal</Badge>
      );
    }
  };

  const getAnalysisTitle = (analysis: RecentAnalysis) => {
    if (analysis.type === 'gavd_dataset') {
      return analysis.filename || 'GAVD Dataset';
    } else {
      return `Analysis ${(analysis.analysis_id || '').substring(0, 8)}...`;
    }
  };

  const getAnalysisSubtitle = (analysis: RecentAnalysis) => {
    const date = analysis.completed_at || analysis.uploaded_at || analysis.created_at || '';
    const timeAgo = date ? formatTimeAgo(date) : 'Unknown time';
    
    if (analysis.type === 'gavd_dataset') {
      const details = [];
      if (analysis.total_sequences_processed) {
        details.push(`${analysis.total_sequences_processed} sequences`);
      }
      if (analysis.total_frames_processed) {
        details.push(`${analysis.total_frames_processed} frames`);
      }
      return `${timeAgo}${details.length > 0 ? ' • ' + details.join(', ') : ''}`;
    } else {
      return `${timeAgo}${analysis.frame_count ? ` • ${analysis.frame_count} frames` : ''}`;
    }
  };

  const getAnalysisLink = (analysis: RecentAnalysis) => {
    if (analysis.type === 'gavd_dataset') {
      // Link to full dataset analysis page
      return `/gavd/${analysis.dataset_id}`;
    } else {
      return `/results/${analysis.analysis_id}`;
    }
  };

  const getAnalysisIcon = (analysis: RecentAnalysis) => {
    if (analysis.type === 'gavd_dataset') {
      return '📊'; // Dataset icon
    } else {
      return '🎥'; // Video analysis icon
    }
  };

  const handleDeleteAnalysis = async (analysis: RecentAnalysis, event: React.MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();
    
    const title = getAnalysisTitle(analysis);
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
          <CardTitle>Recent Analyses</CardTitle>
          <CardDescription>Your latest gait analyses and GAVD datasets</CardDescription>
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
                    <Link href="/gavd/upload">📊 Upload GAVD</Link>
                  </Button>
                </div>
              </AlertDescription>
            </Alert>
          ) : (
            <div className="space-y-4">
              {recent_analyses.map((analysis, index) => {
                const analysisId = analysis.dataset_id || analysis.analysis_id || `${index}`;
                return (
                  <div 
                    key={analysisId} 
                    className="flex items-center justify-between p-4 border rounded-lg hover:bg-accent/50 transition-colors"
                  >
                    <div className="flex items-center space-x-4 flex-1">
                      <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center text-white text-2xl">
                        {getAnalysisIcon(analysis)}
                      </div>
                      <div className="flex-1">
                        <div className="font-medium">
                          {getAnalysisTitle(analysis)}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {getAnalysisSubtitle(analysis)}
                        </div>
                        {analysis.type === 'gait_analysis' && analysis.identified_conditions && analysis.identified_conditions.length > 0 && (
                          <div className="text-xs text-muted-foreground mt-1">
                            Conditions: {analysis.identified_conditions.join(', ')}
                          </div>
                        )}
                        {analysis.type === 'gavd_dataset' && analysis.progress && (
                          <div className="text-xs text-muted-foreground mt-1">
                            {analysis.progress}
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      {getStatusBadge(analysis)}
                      {analysis.type === 'gait_analysis' && analysis.confidence !== undefined && (
                        <div className="text-sm text-muted-foreground">
                          {(analysis.confidence * 100).toFixed(0)}%
                        </div>
                      )}
                      <Button variant="ghost" size="sm" asChild>
                        <Link href={getAnalysisLink(analysis)}>View →</Link>
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => handleDeleteAnalysis(analysis, e)}
                        disabled={deletingAnalysis === analysisId}
                        className="text-red-600 hover:text-red-700 hover:bg-red-50"
                      >
                        {deletingAnalysis === analysisId ? (
                          <span className="animate-spin">⏳</span>
                        ) : (
                          '🗑️'
                        )}
                      </Button>
                    </div>
                  </div>
                );
              })}
            </div>
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
