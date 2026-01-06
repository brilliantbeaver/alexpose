'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';
import { AnalysesTable } from '@/components/ui/analyses-table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

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

interface DashboardStatistics {
  total_analyses: number;
  total_gait_analyses: number;
  total_gavd_datasets: number;
  recent_analyses: RecentAnalysis[];
}

export default function AllAnalysesPage() {
  const [statistics, setStatistics] = useState<DashboardStatistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deletingAnalysis, setDeletingAnalysis] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('all');

  useEffect(() => {
    loadAllAnalyses();
  }, []);

  const loadAllAnalyses = async () => {
    try {
      setLoading(true);
      setError(null);
      
      console.log('[AllAnalyses] Fetching all analyses from API...');
      const response = await fetch('http://localhost:8000/api/v1/analysis/statistics');
      console.log('[AllAnalyses] Response status:', response.status);
      
      const result = await response.json();
      console.log('[AllAnalyses] Response data:', result);
      
      if (response.ok && result.success) {
        setStatistics(result.statistics);
        console.log('[AllAnalyses] Statistics loaded successfully');
      } else {
        const errorMsg = result.message || result.detail || 'Failed to load analyses';
        console.error('[AllAnalyses] API error:', errorMsg);
        setError(errorMsg);
      }
    } catch (err) {
      console.error('[AllAnalyses] Error loading analyses:', err);
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
        // Reload analyses
        loadAllAnalyses();
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
            <h1 className="text-3xl font-bold">All Analyses</h1>
            <p className="text-muted-foreground">Loading your complete analysis history...</p>
          </div>
        </div>

        <div className="flex flex-col items-center justify-center py-12">
          <div className="relative w-16 h-16">
            <div className="absolute top-0 left-0 w-full h-full border-4 border-blue-200 rounded-full"></div>
            <div className="absolute top-0 left-0 w-full h-full border-4 border-blue-600 rounded-full animate-spin border-t-transparent"></div>
          </div>
          <p className="mt-4 text-sm text-muted-foreground">Loading all analyses...</p>
        </div>

        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-32" />
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[1, 2, 3, 4, 5].map((i) => (
                <Skeleton key={i} className="h-16 w-full" />
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
            <h1 className="text-3xl font-bold">All Analyses</h1>
            <p className="text-muted-foreground">Your complete analysis history</p>
          </div>
          <Button asChild>
            <Link href="/dashboard">← Back to Dashboard</Link>
          </Button>
        </div>

        <Alert variant="destructive">
          <AlertTitle>Error Loading Analyses</AlertTitle>
          <AlertDescription>
            {error || 'Failed to load analysis data. Please try again.'}
            <Button 
              variant="outline" 
              size="sm" 
              className="ml-4"
              onClick={loadAllAnalyses}
            >
              Retry
            </Button>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const { 
    total_analyses, 
    total_gait_analyses,
    total_gavd_datasets,
    recent_analyses
  } = statistics;

  // Filter analyses by type
  const gaitAnalyses = recent_analyses.filter(a => a.type === 'gait_analysis');
  const gavdDatasets = recent_analyses.filter(a => a.type === 'gavd_dataset');

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">All Analyses</h1>
          <p className="text-muted-foreground">
            Complete history of your {total_analyses} analyses ({total_gait_analyses} gait, {total_gavd_datasets} GAVD)
          </p>
        </div>
        <div className="flex gap-2">
          <Button asChild variant="outline">
            <Link href="/dashboard">← Dashboard</Link>
          </Button>
          <Button asChild>
            <Link href="/analyze/upload">📤 New Analysis</Link>
          </Button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Total Analyses</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{total_analyses}</div>
            <p className="text-xs text-muted-foreground">All time</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Gait Analyses</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">{total_gait_analyses}</div>
            <p className="text-xs text-muted-foreground">Video analyses</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">GAVD Datasets</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">{total_gavd_datasets}</div>
            <p className="text-xs text-muted-foreground">Dataset analyses</p>
          </CardContent>
        </Card>
      </div>

      {/* Analyses Table with Tabs */}
      <Card>
        <CardHeader>
          <CardTitle>Analysis History</CardTitle>
          <CardDescription>Browse and manage all your analyses</CardDescription>
        </CardHeader>
        <CardContent>
          {recent_analyses.length === 0 ? (
            <Alert>
              <AlertTitle>No Analyses Found</AlertTitle>
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
                    <Link href="/gavd">📊 GAVD Datasets</Link>
                  </Button>
                </div>
              </AlertDescription>
            </Alert>
          ) : (
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="all">All ({recent_analyses.length})</TabsTrigger>
                <TabsTrigger value="gait">Gait ({gaitAnalyses.length})</TabsTrigger>
                <TabsTrigger value="gavd">GAVD ({gavdDatasets.length})</TabsTrigger>
              </TabsList>
              
              <TabsContent value="all" className="mt-6">
                <AnalysesTable 
                  analyses={recent_analyses}
                  onDelete={handleDeleteAnalysis}
                  deletingAnalysis={deletingAnalysis}
                  showFilters={true}
                />
              </TabsContent>
              
              <TabsContent value="gait" className="mt-6">
                <AnalysesTable 
                  analyses={gaitAnalyses}
                  onDelete={handleDeleteAnalysis}
                  deletingAnalysis={deletingAnalysis}
                  showFilters={true}
                />
              </TabsContent>
              
              <TabsContent value="gavd" className="mt-6">
                <AnalysesTable 
                  analyses={gavdDatasets}
                  onDelete={handleDeleteAnalysis}
                  deletingAnalysis={deletingAnalysis}
                  showFilters={true}
                />
              </TabsContent>
            </Tabs>
          )}
        </CardContent>
      </Card>
    </div>
  );
}