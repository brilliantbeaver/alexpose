/**
 * Individual Result Detail Page
 * Display detailed gait analysis results from backend API
 */

'use client';

import { use, useState, useEffect } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { VideoPlayer } from '@/components/video/VideoPlayer';
import PoseAnalysisOverview from '@/components/pose-analysis/PoseAnalysisOverview';
import { usePoseAnalysis } from '@/hooks/usePoseAnalysis';
import { PerformanceMetrics } from '@/applib/pose-analysis-types';
import { AlertCircle, RefreshCw, Download, Eye, BarChart3 } from 'lucide-react';

interface ResultDetailPageProps {
  params: Promise<{ id: string }>;
}

// Helper function to parse the ID into dataset and sequence
function parseAnalysisId(id: string): { datasetId: string; sequenceId: string } | null {
  // Expected format: datasetId_sequenceId or datasetId-sequenceId
  const parts = id.includes('_') ? id.split('_') : id.split('-');
  
  if (parts.length >= 2) {
    return {
      datasetId: parts[0],
      sequenceId: parts.slice(1).join('_'), // Rejoin in case sequence has underscores
    };
  }
  
  // Fallback: assume it's a sequence ID in a default dataset
  return {
    datasetId: 'default',
    sequenceId: id,
  };
}

export default function ResultDetailPage({ params }: ResultDetailPageProps) {
  const { id } = use(params);
  const [forceRefresh, setForceRefresh] = useState(false);
  
  // Parse the ID to get dataset and sequence
  const parsedId = parseAnalysisId(id);
  
  const {
    analysis,
    loading,
    error,
    refetch,
    clearError,
    status,
  } = usePoseAnalysis(
    parsedId?.datasetId || null,
    parsedId?.sequenceId || null,
    {
      useCache: !forceRefresh,
      forceRefresh,
      autoFetch: true,
    }
  );

  // Reset force refresh after use
  useEffect(() => {
    if (forceRefresh) {
      setForceRefresh(false);
    }
  }, [forceRefresh]);

  const handleRefresh = async () => {
    clearError();
    setForceRefresh(true);
    await refetch();
  };

  const handleExportReport = () => {
    if (analysis) {
      const reportData = {
        analysis_id: analysis.analysis_id,
        sequence_info: analysis.sequence_info,
        summary: analysis.summary,
        gait_cycles: analysis.gait_cycles,
        performance: analysis.performance,
        timestamp: analysis.timestamp,
        exported_at: new Date().toISOString(),
      };
      
      const blob = new Blob([JSON.stringify(reportData, null, 2)], {
        type: 'application/json',
      });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `pose-analysis-${analysis.analysis_id}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  // Loading state
  if (loading) {
    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <Button variant="ghost" size="sm" asChild>
                <Link href="/results">← Back</Link>
              </Button>
            </div>
            <div className="h-8 w-64 bg-muted animate-pulse rounded" />
            <div className="h-4 w-48 bg-muted animate-pulse rounded" />
          </div>
        </div>
        
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-center space-x-2 py-8">
              <RefreshCw className="h-6 w-6 animate-spin" />
              <span className="text-lg">Loading analysis...</span>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <Button variant="ghost" size="sm" asChild>
                <Link href="/results">← Back</Link>
              </Button>
            </div>
            <h1 className="text-3xl font-bold">Analysis Error</h1>
            <p className="text-muted-foreground">
              Analysis ID: {id}
            </p>
          </div>
        </div>

        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Failed to Load Analysis</AlertTitle>
          <AlertDescription className="mt-2">
            {error}
          </AlertDescription>
        </Alert>

        <Card>
          <CardContent className="pt-6">
            <div className="text-center space-y-4">
              <div className="text-6xl">⚠️</div>
              <h2 className="text-2xl font-bold">Analysis Not Available</h2>
              <p className="text-muted-foreground max-w-md mx-auto">
                The analysis could not be loaded. This might be because:
              </p>
              <ul className="text-sm text-muted-foreground space-y-1 max-w-md mx-auto text-left">
                <li>• The analysis hasn't been generated yet</li>
                <li>• The sequence ID is incorrect</li>
                <li>• The backend server is not running</li>
                <li>• There was an error during analysis</li>
              </ul>
              <div className="flex justify-center space-x-2 pt-4">
                <Button onClick={handleRefresh}>
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Retry
                </Button>
                <Button variant="outline" asChild>
                  <Link href="/results">← Back to Results</Link>
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // No analysis found
  if (!analysis) {
    return (
      <div className="space-y-8">
        <Card>
          <CardContent className="pt-6">
            <div className="text-center space-y-4">
              <div className="text-6xl">📊</div>
              <h2 className="text-2xl font-bold">No Analysis Data</h2>
              <p className="text-muted-foreground">
                No analysis data was returned for ID {id}.
              </p>
              <Button asChild>
                <Link href="/results">← Back to Results</Link>
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Extract data from analysis
  const sequenceInfo = analysis.sequence_info || analysis.metadata || {};
  const summary = analysis.summary || {};
  const overallAssessment = summary.overall_assessment || {};
  const gaitCycles = analysis.gait_cycles || [];
  const performance: PerformanceMetrics = analysis.performance || { analysis_time_seconds: 0 };

  // Format confidence score
  const confidenceScore = performance.confidence_score || (typeof overallAssessment.confidence === 'number' ? overallAssessment.confidence : 0);
  const qualityScore = performance.quality_score || 0;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <Button variant="ghost" size="sm" asChild>
              <Link href="/results">← Back</Link>
            </Button>
          </div>
          <h1 className="text-3xl font-bold">
            Sequence {sequenceInfo.sequence_id || parsedId?.sequenceId}
          </h1>
          <p className="text-muted-foreground">
            Analysis ID: {analysis.analysis_id} • Generated: {analysis.timestamp ? new Date(analysis.timestamp).toLocaleString() : 'N/A'}
          </p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline" onClick={handleExportReport}>
            <Download className="h-4 w-4 mr-2" />
            Export Report
          </Button>
          <Button variant="outline" onClick={handleRefresh}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Re-analyze
          </Button>
        </div>
      </div>

      {/* Status Overview */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Analysis Summary</CardTitle>
              <CardDescription>
                Comprehensive gait analysis results • Version {analysis.version}
              </CardDescription>
            </div>
            <Badge
              variant={overallAssessment.overall_level === 'good' ? 'default' : 'destructive'}
              className="text-lg px-4 py-2"
            >
              {overallAssessment.overall_level || 'Unknown'}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <div className="text-sm text-muted-foreground mb-1">Confidence Score</div>
              <div className="text-2xl font-bold">{Math.round(confidenceScore * 100)}%</div>
              <Progress value={confidenceScore * 100} className="mt-2" />
            </div>
            <div>
              <div className="text-sm text-muted-foreground mb-1">Video Duration</div>
              <div className="text-2xl font-bold">
                {sequenceInfo.duration ? `${sequenceInfo.duration.toFixed(1)}s` : 'N/A'}
              </div>
              <div className="text-sm text-muted-foreground mt-1">
                {sequenceInfo.frame_count || 0} frames • {sequenceInfo.fps || 30} FPS
              </div>
            </div>
            <div>
              <div className="text-sm text-muted-foreground mb-1">Gait Cycles</div>
              <div className="text-2xl font-bold">{gaitCycles.length}</div>
              <div className="text-sm text-muted-foreground mt-1">
                Detected cycles
              </div>
            </div>
          </div>

          {/* Processing Performance */}
          {performance.processing_time && (
            <>
              <Separator />
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Processing Time:</span>
                <span className="font-medium">{performance.processing_time.toFixed(2)}s</span>
              </div>
            </>
          )}

          {/* Cache Status */}
          {status && (
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Cache Status:</span>
              <span className="font-medium">
                {status.cached ? '✓ Cached' : '○ Fresh'} 
                {status.lastUpdated && ` • Updated ${new Date(status.lastUpdated).toLocaleString()}`}
              </span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Detailed Analysis Tabs */}
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">
            <BarChart3 className="h-4 w-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="cycles">Gait Cycles</TabsTrigger>
          <TabsTrigger value="features">Features</TabsTrigger>
          <TabsTrigger value="raw">Raw Data</TabsTrigger>
        </TabsList>

        {/* Overview Tab - Use the existing PoseAnalysisOverview component */}
        <TabsContent value="overview" className="space-y-4">
          <PoseAnalysisOverview
            analysis={analysis}
            loading={false}
            error={null}
          />
        </TabsContent>

        {/* Gait Cycles Tab */}
        <TabsContent value="cycles" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Gait Cycles Analysis</CardTitle>
              <CardDescription>
                Detected gait cycles with timing and characteristics
              </CardDescription>
            </CardHeader>
            <CardContent>
              {gaitCycles.length > 0 ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {gaitCycles.map((cycle, index) => (
                      <Card key={cycle.cycle_id || index} className="p-4">
                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="font-medium">Cycle {cycle.cycle_id || index + 1}</span>
                            <Badge variant="outline">
                              {cycle.duration ? `${cycle.duration.toFixed(2)}s` : 'N/A'}
                            </Badge>
                          </div>
                          <div className="text-sm space-y-1">
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Start Frame:</span>
                              <span>{cycle.start_frame}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">End Frame:</span>
                              <span>{cycle.end_frame}</span>
                            </div>
                            {cycle.step_length && (
                              <div className="flex justify-between">
                                <span className="text-muted-foreground">Step Length:</span>
                                <span>{cycle.step_length.toFixed(3)}m</span>
                              </div>
                            )}
                            {cycle.stride_length && (
                              <div className="flex justify-between">
                                <span className="text-muted-foreground">Stride Length:</span>
                                <span>{cycle.stride_length.toFixed(3)}m</span>
                              </div>
                            )}
                          </div>
                        </div>
                      </Card>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <div className="text-4xl mb-4">🚶</div>
                  <p className="text-muted-foreground">No gait cycles detected</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Features Tab */}
        <TabsContent value="features" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Extracted Features</CardTitle>
              <CardDescription>
                Detailed feature analysis from pose estimation
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Alert>
                <Eye className="h-4 w-4" />
                <AlertTitle>Feature Analysis</AlertTitle>
                <AlertDescription>
                  Detailed feature extraction data would be displayed here. 
                  This includes kinematic features, joint angles, temporal features, 
                  stride characteristics, and symmetry measures.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Raw Data Tab */}
        <TabsContent value="raw" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Raw Analysis Data</CardTitle>
              <CardDescription>
                Complete analysis response from the backend API
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="bg-muted p-4 rounded-lg overflow-auto text-sm max-h-96">
                {JSON.stringify(analysis, null, 2)}
              </pre>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
