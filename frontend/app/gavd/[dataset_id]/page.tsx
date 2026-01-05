/**
 * GAVD Dataset Detail Page
 * Displays detailed information about a GAVD dataset including sequences and frames
 */

'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';

interface DatasetMetadata {
  dataset_id: string;
  original_filename: string;
  file_path: string;
  file_size: number;
  description?: string;
  uploaded_at: string;
  status: string;
  validation: {
    valid: boolean;
    row_count: number;
    sequence_count: number;
    headers: string[];
  };
  row_count: number;
  sequence_count: number;
  processing_started_at?: string;
  processing_completed_at?: string;
  total_sequences_processed?: number;
  total_frames_processed?: number;
  average_frames_per_sequence?: number;
  progress?: string;
  error?: string;
}

interface Sequence {
  sequence_id: string;
  frame_count: number;
  has_pose_data: boolean;
}

export default function GAVDDatasetPage() {
  const params = useParams();
  const router = useRouter();
  const dataset_id = params.dataset_id as string;

  const [metadata, setMetadata] = useState<DatasetMetadata | null>(null);
  const [sequences, setSequences] = useState<Sequence[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    loadDatasetDetails();
  }, [dataset_id]);

  const loadDatasetDetails = async () => {
    try {
      setLoading(true);
      setError(null);

      console.log('[GAVD Detail] Loading dataset:', dataset_id);

      // Load metadata
      const metadataResponse = await fetch(`http://localhost:8000/api/v1/gavd/status/${dataset_id}`);
      console.log('[GAVD Detail] Metadata response status:', metadataResponse.status);

      if (!metadataResponse.ok) {
        throw new Error(`Failed to load dataset metadata: ${metadataResponse.statusText}`);
      }

      const metadataResult = await metadataResponse.json();
      console.log('[GAVD Detail] Metadata result:', metadataResult);

      if (metadataResult.success && metadataResult.metadata) {
        setMetadata(metadataResult.metadata);

        // Load sequences if dataset is completed
        if (metadataResult.metadata.status === 'completed') {
          const sequencesResponse = await fetch(`http://localhost:8000/api/v1/gavd/sequences/${dataset_id}?limit=100`);
          console.log('[GAVD Detail] Sequences response status:', sequencesResponse.status);

          if (sequencesResponse.ok) {
            const sequencesResult = await sequencesResponse.json();
            console.log('[GAVD Detail] Sequences result:', sequencesResult);

            if (sequencesResult.success && sequencesResult.sequences) {
              setSequences(sequencesResult.sequences);
            }
          }
        }
      } else {
        throw new Error('Invalid response format');
      }
    } catch (err) {
      console.error('[GAVD Detail] Error loading dataset:', err);
      const errorMsg = err instanceof Error ? err.message : 'Failed to load dataset details';
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge variant="default" className="bg-green-600">Completed</Badge>;
      case 'processing':
        return <Badge variant="default" className="bg-blue-600">Processing</Badge>;
      case 'uploaded':
        return <Badge variant="secondary">Uploaded</Badge>;
      case 'error':
        return <Badge variant="destructive">Error</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const handleDelete = async () => {
    if (!metadata) return;
    
    if (!confirm(
      `Are you sure you want to delete "${metadata.original_filename}"?\n\n` +
      `This will permanently delete:\n` +
      `• Original CSV file\n` +
      `• All processing results (${metadata.total_sequences_processed || metadata.sequence_count} sequences)\n` +
      `• Pose data (${metadata.total_frames_processed || metadata.row_count} frames)\n` +
      `• Downloaded videos\n\n` +
      `This action cannot be undone.`
    )) {
      return;
    }
    
    setDeleting(true);
    
    try {
      const response = await fetch(`http://localhost:8000/api/v1/gavd/${dataset_id}`, {
        method: 'DELETE',
      });
      
      const result = await response.json();
      
      if (response.ok && result.success) {
        alert('Dataset deleted successfully');
        router.push('/dashboard');
      } else {
        alert(`Failed to delete dataset: ${result.detail || result.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Delete error:', error);
      alert(`Error deleting dataset: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setDeleting(false);
    }
  };

  if (loading) {
    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <Skeleton className="h-8 w-64 mb-2" />
            <Skeleton className="h-4 w-96" />
          </div>
          <Skeleton className="h-10 w-32" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-4 w-24" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-8 w-16" />
              </CardContent>
            </Card>
          ))}
        </div>

        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-32" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-40 w-full" />
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error || !metadata) {
    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">GAVD Dataset</h1>
            <p className="text-muted-foreground">Dataset not found</p>
          </div>
          <Button variant="outline" onClick={() => router.push('/dashboard')}>
            ← Back to Dashboard
          </Button>
        </div>

        <Alert variant="destructive">
          <AlertTitle>Error Loading Dataset</AlertTitle>
          <AlertDescription>
            {error || 'Dataset not found. It may have been deleted or the ID is incorrect.'}
            <div className="mt-4">
              <Button variant="outline" size="sm" onClick={loadDatasetDetails}>
                Retry
              </Button>
            </div>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">{metadata.original_filename}</h1>
          <p className="text-muted-foreground">GAVD Dataset Details</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => router.push('/dashboard')}>
            ← Back to Dashboard
          </Button>
          <Button 
            variant="destructive" 
            onClick={handleDelete}
            disabled={deleting}
          >
            {deleting ? (
              <>
                <span className="animate-spin mr-2">⏳</span>
                Deleting...
              </>
            ) : (
              <>
                🗑️ Delete Dataset
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Status Badge */}
      <div className="flex items-center gap-4">
        {getStatusBadge(metadata.status)}
        {metadata.progress && (
          <span className="text-sm text-muted-foreground">{metadata.progress}</span>
        )}
      </div>

      {/* Error Alert */}
      {metadata.error && (
        <Alert variant="destructive">
          <AlertTitle>Processing Error</AlertTitle>
          <AlertDescription>{metadata.error}</AlertDescription>
        </Alert>
      )}

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Total Sequences</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metadata.total_sequences_processed || metadata.sequence_count || 0}</div>
            <p className="text-xs text-muted-foreground">
              {metadata.sequence_count} in CSV
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Total Frames</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metadata.total_frames_processed || metadata.row_count || 0}</div>
            <p className="text-xs text-muted-foreground">
              {metadata.row_count} rows in CSV
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Avg Frames/Seq</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metadata.average_frames_per_sequence?.toFixed(1) || 'N/A'}
            </div>
            <p className="text-xs text-muted-foreground">Per sequence</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">File Size</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatFileSize(metadata.file_size)}</div>
            <p className="text-xs text-muted-foreground">CSV file</p>
          </CardContent>
        </Card>
      </div>

      {/* Dataset Information */}
      <Card>
        <CardHeader>
          <CardTitle>Dataset Information</CardTitle>
          <CardDescription>Metadata and processing details</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-sm font-medium text-muted-foreground">Dataset ID</div>
              <div className="text-sm font-mono">{metadata.dataset_id}</div>
            </div>
            <div>
              <div className="text-sm font-medium text-muted-foreground">Filename</div>
              <div className="text-sm">{metadata.original_filename}</div>
            </div>
            <div>
              <div className="text-sm font-medium text-muted-foreground">Uploaded At</div>
              <div className="text-sm">{formatDate(metadata.uploaded_at)}</div>
            </div>
            {metadata.processing_completed_at && (
              <div>
                <div className="text-sm font-medium text-muted-foreground">Completed At</div>
                <div className="text-sm">{formatDate(metadata.processing_completed_at)}</div>
              </div>
            )}
            {metadata.description && (
              <div className="col-span-2">
                <div className="text-sm font-medium text-muted-foreground">Description</div>
                <div className="text-sm">{metadata.description}</div>
              </div>
            )}
          </div>

          {metadata.validation && (
            <div className="pt-4 border-t">
              <div className="text-sm font-medium mb-2">CSV Validation</div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-muted-foreground">Valid:</span>{' '}
                  {metadata.validation.valid ? '✅ Yes' : '❌ No'}
                </div>
                <div>
                  <span className="text-muted-foreground">Rows:</span> {metadata.validation.row_count}
                </div>
                <div>
                  <span className="text-muted-foreground">Sequences:</span> {metadata.validation.sequence_count}
                </div>
                <div>
                  <span className="text-muted-foreground">Columns:</span> {metadata.validation.headers.length}
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Sequences List */}
      {sequences.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Sequences</CardTitle>
            <CardDescription>Available sequences in this dataset</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {sequences.map((sequence) => (
                <div
                  key={sequence.sequence_id}
                  className="flex items-center justify-between p-4 border rounded-lg hover:bg-accent/50 transition-colors"
                >
                  <div>
                    <div className="font-medium">Sequence {sequence.sequence_id}</div>
                    <div className="text-sm text-muted-foreground">
                      {sequence.frame_count} frames
                      {sequence.has_pose_data && ' • Pose data available'}
                    </div>
                  </div>
                  <Button variant="ghost" size="sm" asChild>
                    <Link href={`/gavd/${dataset_id}/sequence/${sequence.sequence_id}`}>
                      View →
                    </Link>
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Processing Status */}
      {metadata.status === 'processing' && (
        <Alert>
          <AlertTitle>Processing in Progress</AlertTitle>
          <AlertDescription>
            This dataset is currently being processed. Sequences will appear here once processing is complete.
            <div className="mt-4">
              <Button variant="outline" size="sm" onClick={loadDatasetDetails}>
                Refresh Status
              </Button>
            </div>
          </AlertDescription>
        </Alert>
      )}

      {/* No Sequences */}
      {metadata.status === 'completed' && sequences.length === 0 && (
        <Alert>
          <AlertTitle>No Sequences Found</AlertTitle>
          <AlertDescription>
            This dataset has been processed but no sequences were found. This may indicate an issue with the processing.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
