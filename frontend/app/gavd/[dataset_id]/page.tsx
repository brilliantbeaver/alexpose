/**
 * GAVD Dataset Analysis Page
 * Interactive visualization and analysis of processed GAVD datasets
 * Mirrors the explore2.ipynb notebook workflow
 */

'use client';

import { useState, useEffect, useCallback } from 'react';
import { useParams, useSearchParams } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Progress } from '@/components/ui/progress';
import GAVDVideoPlayer from '@/components/GAVDVideoPlayer';
import PoseAnalysisOverview from '@/components/pose-analysis/PoseAnalysisOverview';
import { 
  DatasetLoadingState, 
  SequenceLoadingState, 
  PoseAnalysisLoadingState,
  FrameProcessingState
} from '@/components/ui/loading-spinner';
import { 
  DatasetStatsSkeleton,
  SequenceListSkeleton,
  VideoPlayerSkeleton,
  FrameInfoSkeleton
} from '@/components/ui/skeleton-loader';

interface DatasetMetadata {
  dataset_id: string;
  original_filename: string;
  status: string;
  row_count: number;
  sequence_count: number;
  uploaded_at: string;
  total_sequences_processed?: number;
  total_frames_processed?: number;
  error?: string;
}

interface Sequence {
  sequence_id: string;
  frame_count: number;
  has_pose_data: boolean;
  gait_pattern?: string;
  dataset_type?: string;
}

interface PoseKeypoint {
  x: number;
  y: number;
  confidence: number;
  keypoint_id: number;
}

interface FrameData {
  frame_num: number;
  bbox: {
    left: number;
    top: number;
    width: number;
    height: number;
  };
  vid_info: {
    width: number;
    height: number;
  };
  url: string;
  gait_event?: string;
  cam_view?: string;
  pose_keypoints?: PoseKeypoint[];
  // Source video dimensions for pose keypoint scaling
  // These are the actual dimensions of the video used for pose estimation
  pose_source_width?: number;
  pose_source_height?: number;
}

export default function GAVDAnalysisPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const datasetId = params.dataset_id as string;
  
  // Get query parameters for deep linking
  const sequenceParam = searchParams.get('sequence');
  const tabParam = searchParams.get('tab');

  const [metadata, setMetadata] = useState<DatasetMetadata | null>(null);
  const [sequences, setSequences] = useState<Sequence[]>([]);
  const [selectedSequence, setSelectedSequence] = useState<string | null>(null);
  const [sequenceFrames, setSequenceFrames] = useState<FrameData[]>([]);
  const [selectedFrameIndex, setSelectedFrameIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingFrames, setLoadingFrames] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [framesError, setFramesError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [showBbox, setShowBbox] = useState(true);
  const [showPose, setShowPose] = useState(false);
  const [poseAnalysis, setPoseAnalysis] = useState<any>(null);
  const [poseAnalysisCache, setPoseAnalysisCache] = useState<Map<string, any>>(new Map());
  const [loadingPoseAnalysis, setLoadingPoseAnalysis] = useState(false);
  const [poseAnalysisError, setPoseAnalysisError] = useState<string | null>(null);

  // Define callback functions before useEffect hooks
  const loadDatasetMetadata = useCallback(async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/gavd/status/${datasetId}`);
      const result = await response.json();
      
      if (response.ok) {
        setMetadata(result.metadata);
      } else {
        setError('Failed to load dataset metadata');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dataset');
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  const loadSequences = useCallback(async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/gavd/sequences/${datasetId}?limit=100`);
      const result = await response.json();
      
      if (response.ok) {
        setSequences(result.sequences);
        if (result.sequences.length > 0) {
          setSelectedSequence(result.sequences[0].sequence_id);
        }
      }
    } catch (err) {
      console.error('Failed to load sequences:', err);
    }
  }, [datasetId]);

  const loadPoseAnalysis = useCallback(async (sequenceId: string) => {
    if (!sequenceId) {
      console.warn('[loadPoseAnalysis] No sequence ID provided');
      return;
    }

    // Check if analysis is already cached
    if (poseAnalysisCache.has(sequenceId)) {
      console.log(`[loadPoseAnalysis] Using cached analysis for sequence: ${sequenceId}`);
      setPoseAnalysis(poseAnalysisCache.get(sequenceId));
      setPoseAnalysisError(null);
      return;
    }

    setLoadingPoseAnalysis(true);
    setPoseAnalysisError(null);
    
    try {
      console.log(`[loadPoseAnalysis] Starting to load analysis for sequence: ${sequenceId}`);
      const response = await fetch(
        `http://localhost:8000/api/v1/pose-analysis/sequence/${datasetId}/${sequenceId}`
      );
      
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error('No pose data available for this sequence');
        }
        throw new Error(`Failed to load pose analysis: ${response.status} ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (!result.success || !result.analysis) {
        throw new Error('Invalid response format from server');
      }
      
      console.log(`[loadPoseAnalysis] Analysis loaded successfully for sequence ${sequenceId}`);
      
      // Cache the analysis result
      const newCache = new Map(poseAnalysisCache);
      newCache.set(sequenceId, result.analysis);
      setPoseAnalysisCache(newCache);
      
      setPoseAnalysis(result.analysis);
      setPoseAnalysisError(null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load pose analysis';
      console.error('[loadPoseAnalysis] Error:', errorMessage, err);
      setPoseAnalysisError(errorMessage);
      setPoseAnalysis(null);
    } finally {
      setLoadingPoseAnalysis(false);
    }
  }, [datasetId, poseAnalysisCache]);

  const loadSequenceFrames = useCallback(async (sequenceId: string) => {
    if (!sequenceId) {
      console.warn('[loadSequenceFrames] No sequence ID provided');
      return;
    }

    setLoadingFrames(true);
    setFramesError(null);
    
    try {
      console.log(`[loadSequenceFrames] Starting to load frames for sequence: ${sequenceId}`);
      const response = await fetch(
        `http://localhost:8000/api/v1/gavd/sequence/${datasetId}/${sequenceId}/frames`
      );
      
      if (!response.ok) {
        throw new Error(`Failed to load frames: ${response.status} ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (!result.success || !result.frames) {
        throw new Error('Invalid response format from server');
      }
      
      console.log(`[loadSequenceFrames] Loaded ${result.frames.length} frames for sequence ${sequenceId}`);
      
      if (result.frames.length === 0) {
        setFramesError('No frames found for this sequence');
        setSequenceFrames([]);
        setLoadingFrames(false);
        return;
      }
      
      // Load pose data for each frame if available
      const framesWithPose = await Promise.all(
        result.frames.map(async (frame: FrameData) => {
          try {
            const poseResponse = await fetch(
              `http://localhost:8000/api/v1/gavd/sequence/${datasetId}/${sequenceId}/frame/${frame.frame_num}/pose`
            );
            if (poseResponse.ok) {
              const poseData = await poseResponse.json();
              console.log(`[loadSequenceFrames] Loaded pose data for frame ${frame.frame_num}:`, poseData.pose_keypoints?.length || 0, 'keypoints');
              
              // Include source video dimensions for proper scaling
              const sourceVideoWidth = poseData.source_video_width;
              const sourceVideoHeight = poseData.source_video_height;
              
              if (sourceVideoWidth && sourceVideoHeight) {
                console.log(`[loadSequenceFrames] Source video dimensions for frame ${frame.frame_num}: ${sourceVideoWidth}x${sourceVideoHeight}`);
              }
              
              return { 
                ...frame, 
                pose_keypoints: poseData.pose_keypoints,
                pose_source_width: sourceVideoWidth,
                pose_source_height: sourceVideoHeight
              };
            } else {
              console.warn(`[loadSequenceFrames] No pose data for frame ${frame.frame_num}: ${poseResponse.status}`);
            }
          } catch (err) {
            console.warn(`[loadSequenceFrames] Failed to load pose data for frame ${frame.frame_num}:`, err);
          }
          return frame;
        })
      );
      
      console.log(`[loadSequenceFrames] Frames with pose data loaded: ${framesWithPose.filter(f => f.pose_keypoints).length} of ${framesWithPose.length}`);
      setSequenceFrames(framesWithPose);
      setSelectedFrameIndex(0);
      setFramesError(null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load sequence frames';
      console.error('[loadSequenceFrames] Error:', errorMessage, err);
      setFramesError(errorMessage);
      setSequenceFrames([]);
    } finally {
      setLoadingFrames(false);
    }
  }, [datasetId]);

  // Load dataset metadata
  useEffect(() => {
    loadDatasetMetadata();
  }, [loadDatasetMetadata]);

  // Load sequences when metadata is available
  useEffect(() => {
    if (metadata && metadata.status === 'completed') {
      loadSequences();
    }
  }, [metadata, loadSequences]);

  // Load sequence frames when a sequence is selected
  useEffect(() => {
    if (selectedSequence) {
      console.log(`Loading frames for sequence: ${selectedSequence}`);
      loadSequenceFrames(selectedSequence);
      
      // Load cached pose analysis if available, but don't clear it
      if (poseAnalysisCache.has(selectedSequence)) {
        setPoseAnalysis(poseAnalysisCache.get(selectedSequence));
        setPoseAnalysisError(null);
      } else {
        // Only clear if we don't have cached data
        setPoseAnalysis(null);
        setPoseAnalysisError(null);
      }
    } else {
      // Clear frames when no sequence is selected
      setSequenceFrames([]);
      setSelectedFrameIndex(0);
      setFramesError(null);
      setPoseAnalysis(null);
      setPoseAnalysisError(null);
    }
  }, [selectedSequence, loadSequenceFrames, poseAnalysisCache]);

  // Reload frames when switching to visualization tab if sequence is selected but frames not loaded
  useEffect(() => {
    if (activeTab === 'visualization' && selectedSequence && sequenceFrames.length === 0 && !loadingFrames) {
      console.log(`Visualization tab activated, loading frames for sequence: ${selectedSequence}`);
      loadSequenceFrames(selectedSequence);
    }
  }, [activeTab, selectedSequence, sequenceFrames.length, loadingFrames, loadSequenceFrames]);

  // Load pose analysis when switching to pose tab or when sequence changes
  useEffect(() => {
    if (activeTab === 'pose' && selectedSequence && !loadingPoseAnalysis) {
      // Check if we already have analysis for this sequence
      if (poseAnalysisCache.has(selectedSequence)) {
        console.log(`Pose tab activated, using cached analysis for sequence: ${selectedSequence}`);
        setPoseAnalysis(poseAnalysisCache.get(selectedSequence));
        setPoseAnalysisError(null);
      } else if (!poseAnalysis) {
        console.log(`Pose tab activated, loading analysis for sequence: ${selectedSequence}`);
        loadPoseAnalysis(selectedSequence);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, selectedSequence, poseAnalysisCache]);

  // Handle URL query parameters for deep linking (sequence and tab)
  useEffect(() => {
    // Auto-select sequence from query parameter
    if (sequenceParam && sequences.length > 0) {
      const sequenceExists = sequences.some(seq => seq.sequence_id === sequenceParam);
      if (sequenceExists && selectedSequence !== sequenceParam) {
        console.log(`[Query Param] Auto-selecting sequence: ${sequenceParam}`);
        setSelectedSequence(sequenceParam);
      }
    }
    
    // Auto-switch to tab from query parameter
    if (tabParam && tabParam !== activeTab) {
      const validTabs = ['overview', 'sequences', 'visualization', 'pose'];
      if (validTabs.includes(tabParam)) {
        console.log(`[Query Param] Auto-switching to tab: ${tabParam}`);
        setActiveTab(tabParam);
      }
    }
  }, [sequenceParam, tabParam, sequences, selectedSequence, activeTab]);

  const getStatusBadge = (status: string) => {
    const config: Record<string, { variant: 'default' | 'secondary' | 'destructive', label: string }> = {
      uploaded: { variant: 'secondary', label: 'Uploaded' },
      processing: { variant: 'default', label: 'Processing' },
      completed: { variant: 'default', label: 'Completed' },
      error: { variant: 'destructive', label: 'Error' }
    };
    
    const { variant, label } = config[status] || { variant: 'secondary', label: status };
    return <Badge variant={variant}>{label}</Badge>;
  };

  if (loading) {
    return <DatasetLoadingState />;
  }

  if (error || !metadata) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error || 'Dataset not found'}</AlertDescription>
      </Alert>
    );
  }

  const currentFrame = sequenceFrames[selectedFrameIndex];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">GAVD Dataset Analysis</h1>
          <p className="text-muted-foreground">{metadata.original_filename}</p>
        </div>
        {getStatusBadge(metadata.status)}
      </div>

      {/* Dataset Statistics */}
      {loading ? (
        <DatasetStatsSkeleton />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Total Sequences</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{metadata.total_sequences_processed || metadata.sequence_count}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Total Frames</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{metadata.total_frames_processed || metadata.row_count}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Avg Frames/Seq</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {metadata.total_sequences_processed 
                  ? Math.round(metadata.total_frames_processed! / metadata.total_sequences_processed)
                  : Math.round(metadata.row_count / metadata.sequence_count)}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Processing Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-sm font-medium text-green-600">
                {metadata.status === 'completed' ? '✓ Complete' : metadata.status}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Main Analysis Interface */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="sequences">Sequences</TabsTrigger>
          <TabsTrigger value="visualization">Visualization</TabsTrigger>
          <TabsTrigger value="pose">Pose Analysis</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Dataset Information</CardTitle>
              <CardDescription>Summary of the GAVD dataset</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Dataset ID</p>
                  <p className="font-mono text-xs">{metadata.dataset_id}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Uploaded</p>
                  <p>{new Date(metadata.uploaded_at).toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Original Filename</p>
                  <p>{metadata.original_filename}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Status</p>
                  <p>{metadata.status}</p>
                </div>
              </div>

              {metadata.status === 'completed' && (
                <Alert>
                  <AlertTitle>✓ Processing Complete</AlertTitle>
                  <AlertDescription>
                    Dataset has been successfully processed. You can now explore sequences and visualize frames.
                  </AlertDescription>
                </Alert>
              )}

              {metadata.status === 'error' && metadata.error && (
                <Alert variant="destructive">
                  <AlertTitle>Processing Error</AlertTitle>
                  <AlertDescription>{metadata.error}</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Sequence List Preview */}
          {sequences.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Sequences ({sequences.length})</CardTitle>
                <CardDescription>Available gait sequences in this dataset</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {sequences.slice(0, 5).map((seq) => (
                    <div
                      key={seq.sequence_id}
                      className="flex items-center justify-between p-3 border rounded-lg hover:bg-accent/50 cursor-pointer"
                      onClick={() => {
                        setSelectedSequence(seq.sequence_id);
                        setActiveTab('sequences');
                      }}
                    >
                      <div>
                        <p className="font-medium">{seq.sequence_id}</p>
                        <p className="text-sm text-muted-foreground">
                          {seq.frame_count} frames
                          {seq.gait_pattern && ` • ${seq.gait_pattern}`}
                        </p>
                      </div>
                      <Button variant="ghost" size="sm">
                        View →
                      </Button>
                    </div>
                  ))}
                  {sequences.length > 5 && (
                    <p className="text-sm text-muted-foreground text-center pt-2">
                      + {sequences.length - 5} more sequences
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Sequences Tab */}
        <TabsContent value="sequences" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Sequence Explorer</CardTitle>
              <CardDescription>Browse and analyze individual gait sequences</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center space-x-4">
                <Label className="text-sm font-medium">Select Sequence:</Label>
                <Select value={selectedSequence || ''} onValueChange={setSelectedSequence}>
                  <SelectTrigger className="w-[400px]">
                    <SelectValue placeholder="Choose a sequence" />
                  </SelectTrigger>
                  <SelectContent>
                    {sequences.map((seq) => (
                      <SelectItem key={seq.sequence_id} value={seq.sequence_id}>
                        {seq.sequence_id} ({seq.frame_count} frames)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {selectedSequence && sequenceFrames.length > 0 && (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="text-muted-foreground">Sequence ID</p>
                      <p className="font-mono text-xs">{selectedSequence}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Total Frames</p>
                      <p className="font-medium">{sequenceFrames.length}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Frame Range</p>
                      <p className="font-medium">
                        {sequenceFrames[0]?.frame_num} - {sequenceFrames[sequenceFrames.length - 1]?.frame_num}
                      </p>
                    </div>
                  </div>

                  <div className="border rounded-lg p-4 space-y-2">
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-medium">Frame Timeline</p>
                      <p className="text-sm text-muted-foreground">
                        Frame {selectedFrameIndex + 1} of {sequenceFrames.length}
                      </p>
                    </div>
                    <Slider
                      value={[selectedFrameIndex]}
                      onValueChange={(value) => setSelectedFrameIndex(value[0])}
                      max={sequenceFrames.length - 1}
                      step={1}
                      className="w-full"
                    />
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>Start</span>
                      <span>Frame #{currentFrame?.frame_num}</span>
                      <span>End</span>
                    </div>
                  </div>

                  {currentFrame && (
                    <div className="grid grid-cols-2 gap-4 text-sm border rounded-lg p-4">
                      <div>
                        <p className="text-muted-foreground">Frame Number</p>
                        <p className="font-medium">{currentFrame.frame_num}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Camera View</p>
                        <p className="font-medium">{currentFrame.cam_view || 'N/A'}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Gait Event</p>
                        <p className="font-medium">{currentFrame.gait_event || 'None'}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Video Resolution</p>
                        <p className="font-medium">
                          {currentFrame.vid_info?.width}x{currentFrame.vid_info?.height}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Visualization Tab */}
        <TabsContent value="visualization" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Frame Visualization</CardTitle>
              <CardDescription>
                View frames with bounding boxes and pose overlays
                {selectedSequence && ` • Sequence: ${selectedSequence}`}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Loading State */}
              {loadingFrames && (
                <SequenceLoadingState sequenceId={selectedSequence || undefined} />
              )}

              {/* Error State */}
              {!loadingFrames && framesError && (
                <Alert variant="destructive">
                  <AlertTitle>Error Loading Frames</AlertTitle>
                  <AlertDescription>
                    {framesError}
                    {selectedSequence && (
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="mt-2"
                        onClick={() => loadSequenceFrames(selectedSequence)}
                      >
                        Retry
                      </Button>
                    )}
                  </AlertDescription>
                </Alert>
              )}

              {/* No Sequence Selected State */}
              {!loadingFrames && !framesError && !selectedSequence && (
                <Alert>
                  <AlertTitle>No Sequence Selected</AlertTitle>
                  <AlertDescription>
                    Select a sequence from the Sequences tab to visualize frames
                  </AlertDescription>
                </Alert>
              )}

              {/* Frames Loaded Successfully */}
              {!loadingFrames && !framesError && selectedSequence && sequenceFrames.length > 0 && (
                <>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-4">
                      <div className="flex items-center space-x-2">
                        <Label className="text-sm font-medium">Sequence:</Label>
                        <Select value={selectedSequence || ''} onValueChange={setSelectedSequence}>
                          <SelectTrigger className="w-[300px]">
                            <SelectValue placeholder="Choose a sequence" />
                          </SelectTrigger>
                          <SelectContent>
                            {sequences.map((seq) => (
                              <SelectItem key={seq.sequence_id} value={seq.sequence_id}>
                                {seq.sequence_id} ({seq.frame_count} frames)
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          id="showBbox"
                          checked={showBbox}
                          onChange={(e) => setShowBbox(e.target.checked)}
                          className="rounded"
                        />
                        <Label htmlFor="showBbox">Show Bounding Box</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          id="showPose"
                          checked={showPose}
                          onChange={(e) => setShowPose(e.target.checked)}
                          className="rounded"
                        />
                        <Label htmlFor="showPose">Show Pose Overlay</Label>
                      </div>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {sequenceFrames.filter(f => f.pose_keypoints).length} frames with pose data
                    </div>
                  </div>

                  <GAVDVideoPlayer
                    frames={sequenceFrames}
                    currentFrameIndex={selectedFrameIndex}
                    onFrameChange={setSelectedFrameIndex}
                    showBbox={showBbox}
                    showPose={showPose}
                  />

                  <div className="grid grid-cols-2 gap-4 text-sm mt-4">
                    <div className="border rounded-lg p-3">
                      <p className="text-muted-foreground mb-2">Bounding Box</p>
                      <div className="space-y-1 font-mono text-xs">
                        <p>Left: {currentFrame.bbox.left.toFixed(1)}</p>
                        <p>Top: {currentFrame.bbox.top.toFixed(1)}</p>
                        <p>Width: {currentFrame.bbox.width.toFixed(1)}</p>
                        <p>Height: {currentFrame.bbox.height.toFixed(1)}</p>
                      </div>
                    </div>
                    <div className="border rounded-lg p-3">
                      <p className="text-muted-foreground mb-2">Video Info</p>
                      <div className="space-y-1 font-mono text-xs">
                        <p>Width: {currentFrame.vid_info.width}px</p>
                        <p>Height: {currentFrame.vid_info.height}px</p>
                        <p>Keypoints: {currentFrame.pose_keypoints?.length || 0}</p>
                        <p className="truncate">URL: {currentFrame.url.substring(0, 30)}...</p>
                      </div>
                    </div>
                  </div>
                </>
              )}

              {/* Sequence Selected but No Frames (edge case) */}
              {!loadingFrames && !framesError && selectedSequence && sequenceFrames.length === 0 && (
                <Alert>
                  <AlertTitle>No Frames Available</AlertTitle>
                  <AlertDescription>
                    The selected sequence does not have any frames available.
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="mt-2"
                      onClick={() => loadSequenceFrames(selectedSequence)}
                    >
                      Reload Frames
                    </Button>
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Pose Analysis Tab */}
        <TabsContent value="pose" className="space-y-4">
          {/* Sequence Selection */}
          {!selectedSequence && (
            <Alert>
              <AlertTitle>No Sequence Selected</AlertTitle>
              <AlertDescription>
                Select a sequence from the Sequences tab to view pose analysis results.
              </AlertDescription>
            </Alert>
          )}

          {selectedSequence && (
            <>
              {/* Sequence Selector */}
              <Card>
                <CardHeader>
                  <CardTitle>Pose Analysis</CardTitle>
                  <CardDescription>
                    Comprehensive gait analysis with feature extraction, cycle detection, and symmetry assessment
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center space-x-4">
                    <Label className="text-sm font-medium">Sequence:</Label>
                    <Select value={selectedSequence || ''} onValueChange={setSelectedSequence}>
                      <SelectTrigger className="w-[400px]">
                        <SelectValue placeholder="Choose a sequence" />
                      </SelectTrigger>
                      <SelectContent>
                        {sequences.map((seq) => (
                          <SelectItem key={seq.sequence_id} value={seq.sequence_id}>
                            {seq.sequence_id} ({seq.frame_count} frames)
                            {seq.has_pose_data && ' • Has Pose Data'}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    {!loadingPoseAnalysis && selectedSequence && (
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={() => loadPoseAnalysis(selectedSequence)}
                      >
                        Refresh Analysis
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Pose Analysis Results */}
              <PoseAnalysisOverview 
                analysis={poseAnalysis}
                loading={loadingPoseAnalysis}
                error={poseAnalysisError}
              />
            </>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}

function Label({ children, htmlFor, className }: { children: React.ReactNode; htmlFor?: string; className?: string }) {
  return (
    <label htmlFor={htmlFor} className={className}>
      {children}
    </label>
  );
}
