/**
 * GAVD Video Player Component
 * 
 * A robust video player for GAVD dataset analysis with:
 * - Frame-accurate seeking using ABSOLUTE frame numbers from GAVD annotations
 * - Bounding box overlay (only for annotated frames)
 * - Pose keypoint visualization
 * - Playback controls constrained to annotated frame range
 * - Frame-by-frame navigation
 * 
 * IMPORTANT: GAVD frame numbers are ABSOLUTE frame numbers in the original YouTube video.
 * For example, if the first annotated frame is 1757, we seek to time 1757/fps in the video.
 */

'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface BoundingBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

interface VideoInfo {
  width: number;
  height: number;
}

interface PoseKeypoint {
  x: number;
  y: number;
  confidence: number;
  keypoint_id: number;
}

interface FrameData {
  frame_num: number;
  bbox: BoundingBox;
  vid_info: VideoInfo;
  url: string;
  gait_event?: string;
  cam_view?: string;
  pose_keypoints?: PoseKeypoint[];
  // Source video dimensions for pose keypoint scaling
  // These are the actual dimensions of the video used for pose estimation
  pose_source_width?: number;
  pose_source_height?: number;
}

interface GAVDVideoPlayerProps {
  frames: FrameData[];
  currentFrameIndex: number;
  onFrameChange: (index: number) => void;
  showBbox?: boolean;
  showPose?: boolean;
  autoPlay?: boolean;
}

export default function GAVDVideoPlayer({
  frames,
  currentFrameIndex,
  onFrameChange,
  showBbox = true,
  showPose = false,
}: GAVDVideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [streamUrl, setStreamUrl] = useState<string | null>(null);
  const [fps, setFps] = useState(30); // Default FPS
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  
  // Use refs to track mutable state that needs to be accessed in animation loops
  // This avoids stale closure issues
  const animationFrameRef = useRef<number | null>(null);
  const isPlayingRef = useRef(false);
  const currentFrameIndexRef = useRef(currentFrameIndex);
  const framesRef = useRef(frames);
  const showBboxRef = useRef(showBbox);
  const showPoseRef = useRef(showPose);
  const fpsRef = useRef(fps);

  // Keep refs in sync with props/state
  useEffect(() => {
    currentFrameIndexRef.current = currentFrameIndex;
  }, [currentFrameIndex]);

  useEffect(() => {
    framesRef.current = frames;
  }, [frames]);

  useEffect(() => {
    showBboxRef.current = showBbox;
  }, [showBbox]);

  useEffect(() => {
    showPoseRef.current = showPose;
  }, [showPose]);

  useEffect(() => {
    fpsRef.current = fps;
  }, [fps]);

  const currentFrame = frames[currentFrameIndex];

  // Get the range of annotated frames
  const firstAnnotatedFrame = frames.length > 0 ? frames[0].frame_num : 0;
  const lastAnnotatedFrame = frames.length > 0 ? frames[frames.length - 1].frame_num : 0;

  // Extract video ID from URL
  useEffect(() => {
    if (currentFrame?.url) {
      extractVideoId(currentFrame.url);
    }
  }, [currentFrame?.url]);

  const extractVideoId = async (url: string) => {
    try {
      const response = await fetch(
        `http://localhost:8000/api/v1/video/url-to-id?url=${encodeURIComponent(url)}`
      );
      const result = await response.json();
      
      if (result.success) {
        setVideoId(result.video_id);
        setStreamUrl(`http://localhost:8000${result.stream_url}`);
        setVideoError(null);
      } else {
        setVideoError('Failed to extract video ID');
      }
    } catch (error) {
      console.error('Error extracting video ID:', error);
      setVideoError('Failed to load video information');
    }
  };

  // Load video metadata
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !streamUrl) return;

    const handleLoadedMetadata = () => {
      setDuration(video.duration);
      setVideoReady(true);
      
      // Estimate FPS (most videos are 24, 30, or 60 fps)
      // TODO: Could potentially get this from video metadata or server
      const estimatedFps = 30;
      setFps(estimatedFps);
      
      // Seek to the first annotated frame (ABSOLUTE frame number)
      console.log(`Video loaded. Duration: ${video.duration}s. Seeking to first annotated frame: ${firstAnnotatedFrame}`);
      seekToFrame(0); // Seek to first frame in our annotation list
    };

    const handleError = (e: Event) => {
      console.error('Video error:', e);
      setVideoError('Failed to load video. Video may not be cached.');
      setVideoReady(false);
    };

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
    };

    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('error', handleError);
    video.addEventListener('timeupdate', handleTimeUpdate);

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('error', handleError);
      video.removeEventListener('timeupdate', handleTimeUpdate);
    };
  }, [streamUrl, firstAnnotatedFrame]);

  /**
   * Seek to a specific frame by index in the frames array.
   * Uses ABSOLUTE frame numbers from GAVD annotations.
   * 
   * GAVD frame_num values are absolute frame numbers in the original YouTube video.
   * To seek to frame 1757, we calculate: time = 1757 / fps
   */
  const seekToFrame = useCallback((frameIndex: number) => {
    const video = videoRef.current;
    const currentFrames = framesRef.current;
    const currentFps = fpsRef.current;
    
    if (!video || !videoReady || !currentFrames[frameIndex]) {
      console.log(`Cannot seek: video=${!!video}, videoReady=${videoReady}, frame exists=${!!currentFrames[frameIndex]}`);
      return;
    }

    const frame = currentFrames[frameIndex];
    
    // CRITICAL: Use ABSOLUTE frame number to calculate seek time
    // GAVD frame numbers are absolute positions in the original YouTube video
    // Frame 1757 means we need to seek to time 1757/fps in the video
    const targetTime = frame.frame_num / currentFps;
    
    console.log(`Seeking to GAVD frame ${frame.frame_num} (index ${frameIndex}), target time: ${targetTime.toFixed(3)}s`);
    
    video.currentTime = targetTime;
    
    // Draw overlays after seeking
    setTimeout(() => drawOverlaysForFrame(frameIndex), 50);
  }, [videoReady]);

  // Update video position when frame changes (only when not playing)
  useEffect(() => {
    if (videoReady && !isPlayingRef.current) {
      seekToFrame(currentFrameIndex);
    }
  }, [currentFrameIndex, videoReady, seekToFrame]);

  /**
   * Check if a video time corresponds to an annotated frame.
   * Returns the frame index if found, -1 otherwise.
   */
  const findFrameIndexForTime = useCallback((videoTime: number): number => {
    const currentFrames = framesRef.current;
    const currentFps = fpsRef.current;
    
    if (currentFrames.length === 0) return -1;
    
    // Calculate the absolute frame number from video time
    const absoluteFrameNum = Math.round(videoTime * currentFps);
    
    // Check if this frame is within our annotated range
    const firstFrame = currentFrames[0].frame_num;
    const lastFrame = currentFrames[currentFrames.length - 1].frame_num;
    
    if (absoluteFrameNum < firstFrame || absoluteFrameNum > lastFrame) {
      return -1; // Outside annotated range
    }
    
    // Find the closest annotated frame
    let closestIndex = 0;
    let minDiff = Infinity;
    
    for (let i = 0; i < currentFrames.length; i++) {
      const diff = Math.abs(currentFrames[i].frame_num - absoluteFrameNum);
      if (diff < minDiff) {
        minDiff = diff;
        closestIndex = i;
      }
    }
    
    return closestIndex;
  }, []);

  /**
   * Draw bounding box and pose overlays for a specific frame index.
   * Only draws overlays for frames that are in the GAVD annotation.
   */
  const drawOverlaysForFrame = useCallback((frameIndex: number) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const currentFrames = framesRef.current;
    
    if (!canvas || !video) {
      console.log('Cannot draw overlays: missing canvas or video');
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.log('Cannot get canvas context');
      return;
    }

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Check if frameIndex is valid
    if (frameIndex < 0 || frameIndex >= currentFrames.length) {
      console.log(`Frame index ${frameIndex} is outside annotated range, not drawing overlays`);
      return;
    }

    const frameData = currentFrames[frameIndex];
    if (!frameData) {
      console.log('No frame data for index', frameIndex);
      return;
    }

    const shouldShowBbox = showBboxRef.current;
    const shouldShowPose = showPoseRef.current;

    console.log(`Drawing overlays for GAVD frame ${frameData.frame_num} (index ${frameIndex}), showBbox=${shouldShowBbox}, showPose=${shouldShowPose}`);

    // Draw bounding box if enabled
    if (shouldShowBbox && frameData.bbox) {
      const bbox = frameData.bbox;
      const vidInfo = frameData.vid_info;

      // Scale bbox coordinates if video resolution differs from annotation
      const scaleX = video.videoWidth / (vidInfo?.width || video.videoWidth);
      const scaleY = video.videoHeight / (vidInfo?.height || video.videoHeight);

      const scaledLeft = bbox.left * scaleX;
      const scaledTop = bbox.top * scaleY;
      const scaledWidth = bbox.width * scaleX;
      const scaledHeight = bbox.height * scaleY;

      // Draw bounding box
      ctx.strokeStyle = '#FF0000';
      ctx.lineWidth = 3;
      ctx.strokeRect(scaledLeft, scaledTop, scaledWidth, scaledHeight);

      // Draw label with correct frame number
      ctx.fillStyle = '#FF0000';
      ctx.fillRect(scaledLeft, scaledTop - 25, 150, 25);
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '14px Arial';
      ctx.fillText(`Frame ${frameData.frame_num}`, scaledLeft + 5, scaledTop - 7);
    }

    // Draw pose keypoints if enabled and available
    if (shouldShowPose && frameData.pose_keypoints && frameData.pose_keypoints.length > 0) {
      drawPoseKeypoints(
        ctx, 
        frameData.pose_keypoints, 
        video, 
        frameData.vid_info,
        frameData.pose_source_width,
        frameData.pose_source_height
      );
    }
  }, []);

  // Draw pose keypoints with skeleton connections
  const drawPoseKeypoints = (
    ctx: CanvasRenderingContext2D,
    keypoints: PoseKeypoint[] | any[],
    video: HTMLVideoElement,
    vidInfo: VideoInfo,
    poseSourceWidth?: number,
    poseSourceHeight?: number
  ) => {
    if (!keypoints || keypoints.length === 0) return;

    // Normalize keypoint format
    const normalizedKeypoints: PoseKeypoint[] = keypoints.map((kp: any) => ({
      x: kp.x || 0,
      y: kp.y || 0,
      confidence: kp.confidence || 0,
      keypoint_id: kp.keypoint_id !== undefined ? kp.keypoint_id : (kp.id !== undefined ? kp.id : 0)
    }));

    // CRITICAL: Determine the correct source dimensions for pose keypoint scaling
    // Priority:
    // 1. Use pose_source_width/height if available (NEW format with metadata)
    // 2. Fall back to actual video dimensions (for OLD format data)
    // 3. Last resort: use vid_info (but this is often wrong for downloaded videos)
    let sourceWidth: number;
    let sourceHeight: number;
    
    if (poseSourceWidth && poseSourceHeight) {
      // NEW format: Use the stored source dimensions
      sourceWidth = poseSourceWidth;
      sourceHeight = poseSourceHeight;
      console.log(`Using stored source dimensions: ${sourceWidth}x${sourceHeight}`);
    } else {
      // OLD format: Assume keypoints are in the actual video's coordinate space
      // This is a safe assumption because MediaPipe generates keypoints based on the video it processes
      sourceWidth = video.videoWidth;
      sourceHeight = video.videoHeight;
      console.log(`No source dimensions found, using actual video dimensions: ${sourceWidth}x${sourceHeight}`);
      console.log(`Note: vid_info says ${vidInfo?.width}x${vidInfo?.height}, but using actual video dimensions instead`);
    }
    
    const scaleX = video.videoWidth / sourceWidth;
    const scaleY = video.videoHeight / sourceHeight;
    
    console.log(`Pose scaling: source=${sourceWidth}x${sourceHeight}, display=${video.videoWidth}x${video.videoHeight}, scale=${scaleX.toFixed(2)}x${scaleY.toFixed(2)}`);

    // MediaPipe Pose skeleton connections (33 landmarks)
    // Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
    // Based on official MediaPipe POSE_CONNECTIONS
    const connections = [
      // Face contour
      [0, 1], [1, 2], [2, 3], [3, 7],  // Nose to left eye to left ear
      [0, 4], [4, 5], [5, 6], [6, 8],  // Nose to right eye to right ear
      [9, 10],  // Mouth left to mouth right
      
      // Shoulders and arms
      [11, 12],  // Left shoulder to right shoulder
      [11, 13], [13, 15],  // Left shoulder to left elbow to left wrist
      [12, 14], [14, 16],  // Right shoulder to right elbow to right wrist
      
      // Left hand
      [15, 17], [15, 19], [15, 21],  // Left wrist to pinky, index, thumb
      [17, 19],  // Left pinky to left index
      
      // Right hand
      [16, 18], [16, 20], [16, 22],  // Right wrist to pinky, index, thumb
      [18, 20],  // Right pinky to right index
      
      // Torso
      [11, 23], [12, 24],  // Shoulders to hips
      [23, 24],  // Left hip to right hip
      
      // Left leg
      [23, 25], [25, 27],  // Left hip to left knee to left ankle
      [27, 29], [27, 31],  // Left ankle to left heel and left foot index
      [29, 31],  // Left heel to left foot index
      
      // Right leg
      [24, 26], [26, 28],  // Right hip to right knee to right ankle
      [28, 30], [28, 32],  // Right ankle to right heel and right foot index
      [30, 32]  // Right heel to right foot index
    ];

    // Draw skeleton connections
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 2;
    
    connections.forEach(([startId, endId]) => {
      const start = normalizedKeypoints.find(kp => kp.keypoint_id === startId);
      const end = normalizedKeypoints.find(kp => kp.keypoint_id === endId);
      
      if (start && end && start.confidence > 0.3 && end.confidence > 0.3) {
        ctx.beginPath();
        ctx.moveTo(start.x * scaleX, start.y * scaleY);
        ctx.lineTo(end.x * scaleX, end.y * scaleY);
        ctx.stroke();
      }
    });

    // Draw keypoints
    normalizedKeypoints.forEach((kp) => {
      if (kp.confidence > 0.3) {
        const x = kp.x * scaleX;
        const y = kp.y * scaleY;
        
        ctx.fillStyle = `rgba(255, 0, 0, ${Math.min(1, kp.confidence)})`;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fill();
        
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    });
  };

  // Redraw overlays when video seeks (for manual navigation)
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleSeeked = () => {
      if (!isPlayingRef.current) {
        drawOverlaysForFrame(currentFrameIndexRef.current);
      }
    };

    video.addEventListener('seeked', handleSeeked);
    return () => video.removeEventListener('seeked', handleSeeked);
  }, [drawOverlaysForFrame]);

  // Redraw overlays when showBbox or showPose changes
  useEffect(() => {
    if (videoReady) {
      drawOverlaysForFrame(currentFrameIndex);
    }
  }, [showBbox, showPose, videoReady, currentFrameIndex, drawOverlaysForFrame]);

  /**
   * Playback control with proper frame synchronization.
   * Only plays through annotated frames, stopping at the last annotated frame.
   */
  const togglePlayPause = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlayingRef.current) {
      // Stop playback
      video.pause();
      isPlayingRef.current = false;
      setIsPlaying(false);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      drawOverlaysForFrame(currentFrameIndexRef.current);
    } else {
      // Start playback
      video.play().then(() => {
        isPlayingRef.current = true;
        setIsPlaying(true);
        
        const updateFrame = () => {
          if (!isPlayingRef.current || video.paused) {
            return;
          }

          const currentFrames = framesRef.current;
          const currentFps = fpsRef.current;
          
          if (currentFrames.length === 0) {
            animationFrameRef.current = requestAnimationFrame(updateFrame);
            return;
          }

          // Calculate absolute frame number from current video time
          const absoluteFrameNum = Math.round(video.currentTime * currentFps);
          
          // Get annotated frame range
          const firstFrame = currentFrames[0].frame_num;
          const lastFrame = currentFrames[currentFrames.length - 1].frame_num;
          
          // Check if we've gone past the last annotated frame
          if (absoluteFrameNum > lastFrame) {
            console.log(`Reached end of annotated frames (frame ${absoluteFrameNum} > last annotated ${lastFrame})`);
            video.pause();
            isPlayingRef.current = false;
            setIsPlaying(false);
            
            // Seek back to last annotated frame
            const lastIndex = currentFrames.length - 1;
            currentFrameIndexRef.current = lastIndex;
            onFrameChange(lastIndex);
            video.currentTime = lastFrame / currentFps;
            drawOverlaysForFrame(lastIndex);
            return;
          }
          
          // Find the closest annotated frame
          let closestIndex = currentFrameIndexRef.current;
          let minDiff = Infinity;
          
          for (let i = 0; i < currentFrames.length; i++) {
            const diff = Math.abs(currentFrames[i].frame_num - absoluteFrameNum);
            if (diff < minDiff) {
              minDiff = diff;
              closestIndex = i;
            }
          }
          
          // Update frame index if changed
          if (closestIndex !== currentFrameIndexRef.current) {
            console.log(`Playback: frame ${absoluteFrameNum} -> index ${closestIndex} (GAVD frame ${currentFrames[closestIndex].frame_num})`);
            currentFrameIndexRef.current = closestIndex;
            onFrameChange(closestIndex);
          }
          
          // Draw overlays for current frame
          drawOverlaysForFrame(closestIndex);
          
          animationFrameRef.current = requestAnimationFrame(updateFrame);
        };
        
        animationFrameRef.current = requestAnimationFrame(updateFrame);
      }).catch((error) => {
        console.error('Failed to start video playback:', error);
        isPlayingRef.current = false;
        setIsPlaying(false);
      });
    }
  }, [onFrameChange, drawOverlaysForFrame]);

  // Cleanup animation frame on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  // Frame navigation
  const previousFrame = useCallback(() => {
    if (currentFrameIndexRef.current > 0) {
      const newIndex = currentFrameIndexRef.current - 1;
      currentFrameIndexRef.current = newIndex;
      onFrameChange(newIndex);
    }
  }, [onFrameChange]);

  const nextFrame = useCallback(() => {
    if (currentFrameIndexRef.current < framesRef.current.length - 1) {
      const newIndex = currentFrameIndexRef.current + 1;
      currentFrameIndexRef.current = newIndex;
      onFrameChange(newIndex);
    }
  }, [onFrameChange]);

  // Keyboard controls
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      switch (e.key) {
        case ' ':
          e.preventDefault();
          togglePlayPause();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          previousFrame();
          break;
        case 'ArrowRight':
          e.preventDefault();
          nextFrame();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [togglePlayPause, previousFrame, nextFrame]);

  if (!currentFrame) {
    return (
      <Alert>
        <AlertDescription>No frame data available</AlertDescription>
      </Alert>
    );
  }

  if (videoError) {
    return (
      <Alert variant="destructive">
        <AlertDescription>
          {videoError}
          <br />
          <span className="text-xs">
            Video ID: {videoId || 'Unknown'} • URL: {currentFrame.url}
          </span>
        </AlertDescription>
      </Alert>
    );
  }

  // Calculate the time range for annotated frames
  const annotatedStartTime = firstAnnotatedFrame / fps;
  const annotatedEndTime = lastAnnotatedFrame / fps;

  return (
    <div className="space-y-4">
      {/* Video Container */}
      <div className="relative bg-black rounded-lg overflow-hidden">
        <div className="aspect-video relative">
          <video
            ref={videoRef}
            src={streamUrl || undefined}
            className="w-full h-full object-contain"
            preload="metadata"
            crossOrigin="anonymous"
          />
          
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none"
            style={{ objectFit: 'contain' }}
          />

          {!videoReady && streamUrl && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50">
              <div className="text-white text-center space-y-2">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto"></div>
                <p>Loading video...</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="space-y-3">
        <div className="flex items-center justify-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={previousFrame}
            disabled={currentFrameIndex === 0}
          >
            ⏮ Previous
          </Button>
          
          <Button
            variant="default"
            size="sm"
            onClick={togglePlayPause}
            disabled={!videoReady}
            className="w-24"
          >
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            onClick={nextFrame}
            disabled={currentFrameIndex === frames.length - 1}
          >
            Next ⏭
          </Button>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">
              Frame {currentFrameIndex + 1} of {frames.length} (GAVD Frame #{currentFrame.frame_num})
            </span>
            <span className="text-muted-foreground">
              {currentFrame.gait_event && `Event: ${currentFrame.gait_event}`}
            </span>
          </div>
          <Slider
            value={[currentFrameIndex]}
            onValueChange={(value) => {
              const newIndex = value[0];
              currentFrameIndexRef.current = newIndex;
              onFrameChange(newIndex);
            }}
            max={frames.length - 1}
            step={1}
            className="w-full"
            disabled={!videoReady}
          />
        </div>

        <div className="grid grid-cols-3 gap-2 text-xs text-muted-foreground">
          <div>
            <span className="font-medium">Time:</span> {currentTime.toFixed(2)}s / {duration.toFixed(2)}s
          </div>
          <div>
            <span className="font-medium">FPS:</span> {fps}
          </div>
          <div>
            <span className="font-medium">Event:</span> {currentFrame.gait_event || 'None'}
          </div>
        </div>

        <div className="text-xs text-muted-foreground text-center">
          <span className="font-medium">Annotated Range:</span> Frames {firstAnnotatedFrame}-{lastAnnotatedFrame} ({annotatedStartTime.toFixed(1)}s - {annotatedEndTime.toFixed(1)}s)
        </div>

        <div className="text-xs text-muted-foreground text-center pt-2 border-t">
          <span className="font-medium">Keyboard:</span> Space = Play/Pause • ← → = Previous/Next Frame
        </div>
      </div>
    </div>
  );
}
