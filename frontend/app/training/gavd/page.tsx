/**
 * GAVD Dataset Upload and Management Page
 * Modern, guided workflow for GAVD training dataset processing
 */

'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

interface UploadResult {
  success: boolean;
  dataset_id?: string;
  filename?: string;
  file_size?: number;
  row_count?: number;
  sequence_count?: number;
  status?: string;
  message?: string;
  error?: string;
}

interface DatasetStatus {
  dataset_id: string;
  status: string;
  original_filename: string;
  row_count: number;
  sequence_count: number;
  uploaded_at: string;
  processing_started_at?: string;
  processing_completed_at?: string;
  total_sequences_processed?: number;
  total_frames_processed?: number;
  progress?: string;
  error?: string;
}

export default function GAVDUploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [description, setDescription] = useState('');
  const [processImmediately, setProcessImmediately] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus | null>(null);
  const [statusPolling, setStatusPolling] = useState(false);
  const [recentDatasets, setRecentDatasets] = useState<DatasetStatus[]>([]);
  const [loadingRecent, setLoadingRecent] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [deletingDataset, setDeletingDataset] = useState<string | null>(null);

  // Load recent datasets on mount
  useEffect(() => {
    loadRecentDatasets();
  }, []);

  const loadRecentDatasets = async () => {
    setLoadError(null);
    try {
      console.log('Loading recent datasets from:', 'http://localhost:8000/api/v1/gavd/list?limit=5');
      const response = await fetch('http://localhost:8000/api/v1/gavd/list?limit=5');
      console.log('Response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Failed to load datasets:', response.status, response.statusText, errorText);
        setLoadError(`Server error: ${response.status} ${response.statusText}`);
        setLoadingRecent(false);
        return;
      }
      
      const result = await response.json();
      console.log('Datasets loaded:', result);
      
      if (result.success && result.datasets) {
        setRecentDatasets(result.datasets);
      } else {
        console.warn('Unexpected response format:', result);
        setRecentDatasets([]);
      }
    } catch (error) {
      console.error('Failed to load recent datasets:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      
      // Check if it's a network error
      if (errorMessage.includes('fetch') || errorMessage.includes('NetworkError') || errorMessage.includes('Failed to fetch')) {
        setLoadError('Cannot connect to server. Please ensure the backend server is running on http://localhost:8000');
      } else {
        setLoadError(`Error loading datasets: ${errorMessage}`);
      }
      
      setRecentDatasets([]);
    } finally {
      setLoadingRecent(false);
    }
  };

  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.name.endsWith('.csv')) {
        setSelectedFile(file);
        setUploadResult(null);
        setDatasetStatus(null);
      } else {
        alert('Please drop a CSV file');
      }
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.name.endsWith('.csv')) {
        alert('Please select a CSV file');
        return;
      }
      setSelectedFile(file);
      setUploadResult(null);
      setDatasetStatus(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select a file first');
      return;
    }

    setUploading(true);
    setUploadResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('description', description);
      formData.append('process_immediately', processImmediately.toString());

      const response = await fetch('http://localhost:8000/api/v1/gavd/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        setUploadResult(result);
        
        // Reload recent datasets
        loadRecentDatasets();
        
        // Start polling for status if processing immediately
        if (processImmediately && result.dataset_id) {
          startStatusPolling(result.dataset_id);
        }
      } else {
        setUploadResult({
          success: false,
          error: result.detail || 'Upload failed'
        });
      }
    } catch (error) {
      console.error('Upload error:', error);
      setUploadResult({
        success: false,
        error: error instanceof Error ? error.message : 'Upload failed'
      });
    } finally {
      setUploading(false);
    }
  };

  const startStatusPolling = (datasetId: string) => {
    setStatusPolling(true);
    let pollCount = 0;
    const maxPolls = 300; // 10 minutes max (300 * 2 seconds)
    
    const pollStatus = async () => {
      try {
        pollCount++;
        
        const response = await fetch(`http://localhost:8000/api/v1/gavd/status/${datasetId}`);
        const result = await response.json();
        
        if (response.ok && result.metadata) {
          setDatasetStatus(result.metadata);
          
          // Stop polling if completed or error
          if (result.metadata.status === 'completed' || result.metadata.status === 'error') {
            setStatusPolling(false);
            loadRecentDatasets(); // Refresh the list
            return;
          }
          
          // Stop polling after max attempts (but keep showing processing status)
          if (pollCount >= maxPolls) {
            console.log('Max polling attempts reached. Processing continues in background.');
            setStatusPolling(false);
            return;
          }
          
          // Continue polling
          setTimeout(() => pollStatus(), 2000);
        }
      } catch (error) {
        console.error('Status polling error:', error);
        setStatusPolling(false);
      }
    };
    
    pollStatus();
  };

  const getStatusBadge = (status: string) => {
    const statusConfig: Record<string, { variant: 'default' | 'secondary' | 'destructive' | 'outline', label: string, color: string }> = {
      uploaded: { variant: 'secondary', label: 'Uploaded', color: 'bg-gray-100 text-gray-800' },
      processing: { variant: 'default', label: 'Processing', color: 'bg-blue-100 text-blue-800' },
      completed: { variant: 'default', label: 'Completed', color: 'bg-green-100 text-green-800' },
      error: { variant: 'destructive', label: 'Error', color: 'bg-red-100 text-red-800' }
    };
    
    const config = statusConfig[status] || { variant: 'outline', label: status, color: 'bg-gray-100' };
    return (
      <Badge variant={config.variant} className={config.color}>
        {config.label}
      </Badge>
    );
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} min ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    return date.toLocaleDateString();
  };

  const handleDeleteDataset = async (datasetId: string, filename: string, event: React.MouseEvent) => {
    event.preventDefault();
    event.stopPropagation();
    
    if (!confirm(`Are you sure you want to delete "${filename}"?\n\nThis will permanently delete:\n• Original CSV file\n• All processing results\n• Pose data\n• Downloaded videos\n\nThis action cannot be undone.`)) {
      return;
    }
    
    setDeletingDataset(datasetId);
    
    try {
      const response = await fetch(`http://localhost:8000/api/v1/gavd/${datasetId}`, {
        method: 'DELETE',
      });
      
      const result = await response.json();
      
      if (response.ok && result.success) {
        // Remove from list
        setRecentDatasets(prev => prev.filter(d => d.dataset_id !== datasetId));
        alert('Dataset deleted successfully');
      } else {
        alert(`Failed to delete dataset: ${result.detail || result.message || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Delete error:', error);
      alert(`Error deleting dataset: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setDeletingDataset(null);
    }
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Hero Section */}
      <div className="text-center space-y-4">
        <div className="inline-block">
          <div className="text-6xl mb-4">📊</div>
        </div>
        <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
          GAVD Dataset Analysis
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Upload and process GAVD training datasets for gait abnormality detection and model training
        </p>
      </div>

      <Tabs defaultValue="upload" className="space-y-6">
        <TabsList className="grid w-full grid-cols-2 max-w-md mx-auto">
          <TabsTrigger value="upload" className="text-base">
            📤 Upload Dataset
          </TabsTrigger>
          <TabsTrigger value="recent" className="text-base">
            📋 Recent Datasets
          </TabsTrigger>
        </TabsList>

        {/* Upload Tab */}
        <TabsContent value="upload" className="space-y-6">
          {/* Upload Card */}
          <Card className="border-2 border-dashed hover:border-purple-500 transition-colors">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <span>📁</span>
                <span>Upload GAVD CSV File</span>
              </CardTitle>
              <CardDescription>
                Drag and drop your GAVD dataset CSV file or click to browse
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Drag & Drop Zone */}
              <div
                className={`border-2 border-dashed rounded-lg p-12 text-center transition-all ${
                  dragActive 
                    ? 'border-purple-500 bg-purple-50' 
                    : 'border-gray-300 hover:border-purple-400 hover:bg-gray-50'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                {selectedFile ? (
                  <div className="space-y-4">
                    <div className="text-5xl">✅</div>
                    <div>
                      <p className="text-lg font-medium">{selectedFile.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {formatFileSize(selectedFile.size)}
                      </p>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setSelectedFile(null)}
                    >
                      Remove File
                    </Button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="text-5xl">📂</div>
                    <div>
                      <p className="text-lg font-medium">Drop your CSV file here</p>
                      <p className="text-sm text-muted-foreground">or click to browse</p>
                    </div>
                    <Input
                      id="file"
                      type="file"
                      accept=".csv"
                      onChange={handleFileChange}
                      disabled={uploading}
                      className="hidden"
                    />
                    <Button
                      variant="outline"
                      onClick={() => document.getElementById('file')?.click()}
                      disabled={uploading}
                    >
                      Browse Files
                    </Button>
                  </div>
                )}
              </div>

              {/* Description */}
              <div className="space-y-2">
                <label htmlFor="description" className="text-sm font-medium">Description (Optional)</label>
                <textarea
                  id="description"
                  placeholder="e.g., Parkinsons gait dataset from clinical study..."
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  disabled={uploading}
                  rows={3}
                  className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 resize-none"
                />
              </div>

              {/* Processing Options */}
              <div className="flex items-center space-x-2 p-4 bg-blue-50 rounded-lg border border-blue-200">
                <input
                  type="checkbox"
                  id="process"
                  checked={processImmediately}
                  onChange={(e) => setProcessImmediately(e.target.checked)}
                  disabled={uploading}
                  className="rounded"
                />
                <label htmlFor="process" className="cursor-pointer flex-1">
                  <div>
                    <p className="font-medium">Process immediately after upload</p>
                    <p className="text-sm text-muted-foreground">
                      Automatically download videos, extract frames, and run pose estimation
                    </p>
                  </div>
                </label>
              </div>

              {/* Upload Button */}
              <Button
                onClick={handleUpload}
                disabled={!selectedFile || uploading}
                className="w-full h-12 text-lg bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700"
              >
                {uploading ? (
                  <>
                    <span className="animate-spin mr-2">⏳</span>
                    Uploading...
                  </>
                ) : (
                  <>
                    🚀 Upload and Process Dataset
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Upload Result */}
          {uploadResult && (
            <Alert variant={uploadResult.success ? 'default' : 'destructive'} className="border-2">
              <AlertTitle className="text-lg flex items-center space-x-2">
                <span>{uploadResult.success ? '✅' : '❌'}</span>
                <span>{uploadResult.success ? 'Upload Successful!' : 'Upload Failed'}</span>
              </AlertTitle>
              <AlertDescription>
                {uploadResult.success ? (
                  <div className="space-y-3 mt-3">
                    <p className="text-base">{uploadResult.message}</p>
                    <div className="grid grid-cols-2 gap-3 text-sm bg-white p-4 rounded-lg">
                      <div>
                        <p className="text-muted-foreground">Filename</p>
                        <p className="font-medium">{uploadResult.filename}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Status</p>
                        <div className="mt-1">{getStatusBadge(uploadResult.status || 'uploaded')}</div>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Total Rows</p>
                        <p className="font-medium">{uploadResult.row_count?.toLocaleString()}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Sequences</p>
                        <p className="font-medium">{uploadResult.sequence_count}</p>
                      </div>
                    </div>
                    {uploadResult.dataset_id && (
                      <Button asChild className="w-full">
                        <Link href={`/training/gavd/${uploadResult.dataset_id}`}>
                          View Dataset Analysis →
                        </Link>
                      </Button>
                    )}
                  </div>
                ) : (
                  <p className="text-base mt-2">{uploadResult.error}</p>
                )}
              </AlertDescription>
            </Alert>
          )}

          {/* Processing Status */}
          {datasetStatus && (
            <Card className="border-2 border-blue-500">
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Processing Status</span>
                  {getStatusBadge(datasetStatus.status)}
                </CardTitle>
                <CardDescription>
                  {datasetStatus.status === 'processing' 
                    ? 'Processing in background - this may take several minutes for large datasets'
                    : 'Real-time processing updates'}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {datasetStatus.status === 'processing' && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-sm">
                      <span>{datasetStatus.progress || 'Processing dataset...'}</span>
                      <span className="animate-pulse">⚡</span>
                    </div>
                    <Progress value={undefined} className="w-full h-2" />
                    <Alert className="bg-blue-50 border-blue-200">
                      <AlertDescription className="text-sm">
                        <p className="font-medium mb-2">⏱️ Processing Steps:</p>
                        <ul className="space-y-1 text-xs">
                          <li>• Downloading YouTube videos (if not cached)</li>
                          <li>• Extracting frames from videos</li>
                          <li>• Running pose estimation on {datasetStatus.row_count.toLocaleString()} frames</li>
                          <li>• Saving results and pose data</li>
                        </ul>
                        <p className="mt-3 text-xs text-muted-foreground">
                          💡 You can safely navigate away - processing continues in the background.
                          Check the "Recent Datasets" tab to see when it completes.
                        </p>
                      </AlertDescription>
                    </Alert>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-muted-foreground">Filename</p>
                    <p className="font-medium truncate">{datasetStatus.original_filename}</p>
                  </div>
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-muted-foreground">Uploaded</p>
                    <p className="font-medium">{formatDate(datasetStatus.uploaded_at)}</p>
                  </div>
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-muted-foreground">Total Rows</p>
                    <p className="font-medium">{datasetStatus.row_count.toLocaleString()}</p>
                  </div>
                  <div className="p-3 bg-gray-50 rounded-lg">
                    <p className="text-muted-foreground">Sequences</p>
                    <p className="font-medium">{datasetStatus.sequence_count}</p>
                  </div>
                </div>

                {datasetStatus.status === 'completed' && (
                  <div className="space-y-3 pt-4 border-t">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div className="p-3 bg-green-50 rounded-lg border border-green-200">
                        <p className="text-muted-foreground">Sequences Processed</p>
                        <p className="text-lg font-bold text-green-600">
                          {datasetStatus.total_sequences_processed}
                        </p>
                      </div>
                      <div className="p-3 bg-green-50 rounded-lg border border-green-200">
                        <p className="text-muted-foreground">Frames Processed</p>
                        <p className="text-lg font-bold text-green-600">
                          {datasetStatus.total_frames_processed?.toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <Button asChild className="w-full" size="lg">
                      <Link href={`/training/gavd/${datasetStatus.dataset_id}`}>
                        🔍 Analyze Dataset →
                      </Link>
                    </Button>
                  </div>
                )}

                {datasetStatus.status === 'error' && datasetStatus.error && (
                  <Alert variant="destructive">
                    <AlertTitle>Processing Error</AlertTitle>
                    <AlertDescription>{datasetStatus.error}</AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          )}

          {/* Information Card */}
          <Card className="bg-gradient-to-br from-purple-50 to-blue-50 border-purple-200">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <span>💡</span>
                <span>About GAVD Datasets</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h3 className="font-semibold mb-2 flex items-center space-x-2">
                  <span>📋</span>
                  <span>Required CSV Columns:</span>
                </h3>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="flex items-center space-x-2">
                    <span className="text-green-600">✓</span>
                    <span><strong>seq:</strong> Sequence ID</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-green-600">✓</span>
                    <span><strong>frame_num:</strong> Frame number</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-green-600">✓</span>
                    <span><strong>bbox:</strong> Bounding box</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-green-600">✓</span>
                    <span><strong>url:</strong> YouTube URL</span>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="font-semibold mb-2 flex items-center space-x-2">
                  <span>⚙️</span>
                  <span>Processing Pipeline:</span>
                </h3>
                <ol className="space-y-1 text-sm">
                  <li className="flex items-center space-x-2">
                    <span className="text-blue-600">1.</span>
                    <span>Validate CSV structure</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <span className="text-blue-600">2.</span>
                    <span>Download YouTube videos</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <span className="text-blue-600">3.</span>
                    <span>Extract frames at specified positions</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <span className="text-blue-600">4.</span>
                    <span>Run pose estimation (MediaPipe)</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <span className="text-blue-600">5.</span>
                    <span>Organize sequences and save results</span>
                  </li>
                </ol>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Recent Datasets Tab */}
        <TabsContent value="recent" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Recent Datasets</CardTitle>
              <CardDescription>Your recently uploaded GAVD datasets</CardDescription>
            </CardHeader>
            <CardContent>
              {loadingRecent ? (
                <div className="text-center py-8">
                  <Progress value={undefined} className="w-64 mx-auto" />
                  <p className="text-sm text-muted-foreground mt-4">Loading datasets...</p>
                </div>
              ) : loadError ? (
                <div className="text-center py-12">
                  <Alert variant="destructive" className="max-w-2xl mx-auto">
                    <AlertTitle className="flex items-center space-x-2">
                      <span>⚠️</span>
                      <span>Connection Error</span>
                    </AlertTitle>
                    <AlertDescription className="mt-2">
                      <p className="mb-3">{loadError}</p>
                      <div className="text-sm text-left bg-red-50 p-3 rounded border border-red-200">
                        <p className="font-medium mb-2">Troubleshooting:</p>
                        <ul className="space-y-1 text-xs">
                          <li>• Check if the backend server is running</li>
                          <li>• Run: <code className="bg-red-100 px-1 py-0.5 rounded">python -m uvicorn server.main:app --reload</code></li>
                          <li>• Or use the startup script: <code className="bg-red-100 px-1 py-0.5 rounded">./scripts/start-dev.ps1</code></li>
                          <li>• Verify the server is accessible at http://localhost:8000</li>
                        </ul>
                      </div>
                      <Button 
                        onClick={loadRecentDatasets} 
                        className="mt-4"
                        variant="outline"
                      >
                        🔄 Retry
                      </Button>
                    </AlertDescription>
                  </Alert>
                </div>
              ) : recentDatasets.length === 0 ? (
                <div className="text-center py-12">
                  <div className="text-5xl mb-4">📂</div>
                  <p className="text-lg font-medium">No datasets yet</p>
                  <p className="text-sm text-muted-foreground mb-4">
                    Upload your first GAVD dataset to get started
                  </p>
                  <Button onClick={() => document.querySelector('[value="upload"]')?.dispatchEvent(new Event('click'))}>
                    Upload Dataset
                  </Button>
                </div>
              ) : (
                <div className="space-y-3">
                  {recentDatasets.map((dataset) => (
                    <div
                      key={dataset.dataset_id}
                      className="flex items-center justify-between p-4 border rounded-lg hover:bg-accent/50 hover:border-purple-500 transition-all"
                    >
                      <Link
                        href={`/training/gavd/${dataset.dataset_id}`}
                        className="flex-1 flex items-center space-x-3"
                      >
                        <div className="flex-1">
                          <div className="flex items-center space-x-3 mb-2">
                            <p className="font-medium">{dataset.original_filename}</p>
                            {getStatusBadge(dataset.status)}
                          </div>
                          <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                            <span>📊 {dataset.sequence_count} sequences</span>
                            <span>📝 {dataset.row_count.toLocaleString()} rows</span>
                            <span>🕒 {formatDate(dataset.uploaded_at)}</span>
                          </div>
                        </div>
                      </Link>
                      <div className="flex items-center space-x-2">
                        <Button 
                          variant="ghost" 
                          size="sm"
                          asChild
                        >
                          <Link href={`/training/gavd/${dataset.dataset_id}`}>
                            View →
                          </Link>
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => handleDeleteDataset(dataset.dataset_id, dataset.original_filename, e)}
                          disabled={deletingDataset === dataset.dataset_id}
                          className="text-red-600 hover:text-red-700 hover:bg-red-50"
                        >
                          {deletingDataset === dataset.dataset_id ? (
                            <span className="animate-spin">⏳</span>
                          ) : (
                            '🗑️'
                          )}
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
