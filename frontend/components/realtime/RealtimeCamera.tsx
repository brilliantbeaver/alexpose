/**
 * Realtime Camera Component
 * 
 * Handles webcam access, video display, and pose overlay rendering
 * with optimized performance for real-time processing.
 */

'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import {
    Camera,
    CameraOff,
    AlertCircle,
    Eye,
    EyeOff,
    Maximize,
    Minimize,
    Maximize2,
    Minimize2
} from 'lucide-react';

interface PoseKeypoint {
    x: number;
    y: number;
    confidence: number;
    id: number;
}

interface PoseResult {
    keypoints: PoseKeypoint[];
    confidence_scores: number[];
    processing_time_ms: number;
    frame_id: number;
    timestamp: number;
    estimator_info: any;
}

interface RealtimeCameraProps {
    isActive: boolean;
    currentPose?: PoseResult | null;
    onFrame?: (frameData: string) => void;
    config?: {
        show_keypoints?: boolean;
        show_skeleton?: boolean;
        confidence_threshold?: number;
        processing_mode?: string;
    };
    onPermissionChange?: (permission: 'granted' | 'denied' | 'prompt') => void;
}

export function RealtimeCamera({
    isActive,
    currentPose,
    onFrame,
    config = {},
    onPermissionChange
}: RealtimeCameraProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const animationFrameRef = useRef<number | undefined>(undefined);
    const lastFrameTimeRef = useRef<number>(0);

    // Reusable canvas for frame capture to avoid creating new canvas each time
    const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const captureCtxRef = useRef<CanvasRenderingContext2D | null>(null);

    const [cameraError, setCameraError] = useState<string | null>(null);
    const [isInitializing, setIsInitializing] = useState(false);
    const [isCameraReady, setIsCameraReady] = useState(false);
    const [videoSize, setVideoSize] = useState({ width: 640, height: 480 });
    const [showOverlay, setShowOverlay] = useState(true);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [isMaximized, setIsMaximized] = useState(false);

    // Configuration with defaults
    const {
        show_keypoints = true,
        show_skeleton = true,
        confidence_threshold = 0.5,
        processing_mode = 'balanced'
    } = config;

    // Target FPS for frame capture - aim for 30 FPS for smooth real-time tracking
    const targetFPS = processing_mode === 'fast' ? 30 : processing_mode === 'accurate' ? 20 : 30;
    const frameInterval = 1000 / targetFPS;

    // Utility functions
    const stopFrameCapture = useCallback(() => {
        if (animationFrameRef.current !== undefined) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = undefined;
        }
    }, []);

    const clearOverlay = useCallback(() => {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');

        if (canvas && ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    }, []);

    const stopCamera = useCallback(() => {
        stopFrameCapture();

        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }

        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }

        setIsCameraReady(false);
        clearOverlay();
    }, [stopFrameCapture, clearOverlay]);

    const captureAndSendFrame = useCallback(() => {
        if (!videoRef.current || !onFrame) return;

        try {
            // Initialize reusable canvas on first use
            if (!captureCanvasRef.current) {
                captureCanvasRef.current = document.createElement('canvas');
                captureCanvasRef.current.width = 640;
                captureCanvasRef.current.height = 480;
                captureCtxRef.current = captureCanvasRef.current.getContext('2d', {
                    alpha: false,  // Disable alpha channel for better performance
                    willReadFrequently: true  // Optimize for frequent reads
                });
            }

            const canvas = captureCanvasRef.current;
            const ctx = captureCtxRef.current;

            if (!ctx) return;

            // Draw current video frame to canvas
            ctx.drawImage(videoRef.current, 0, 0, 640, 480);

            // Use JPEG encoding for fast transmission
            const frameData = canvas.toDataURL('image/jpeg', 0.6);
            onFrame(frameData);
        } catch (error) {
            console.error('Frame capture failed:', error);
        }
    }, [onFrame]);

    const startFrameCapture = useCallback(() => {
        let frameCount = 0;

        const captureFrame = (timestamp: number) => {
            if (!isActive || !videoRef.current || !onFrame) {
                return;
            }

            if (timestamp - lastFrameTimeRef.current >= frameInterval) {
                captureAndSendFrame();
                lastFrameTimeRef.current = timestamp;
                frameCount++;
            }

            animationFrameRef.current = requestAnimationFrame(captureFrame);
        };

        animationFrameRef.current = requestAnimationFrame(captureFrame);
    }, [isActive, onFrame, frameInterval, captureAndSendFrame]);

    // Drawing functions
    const drawKeypoints = useCallback((ctx: CanvasRenderingContext2D, keypoints: PoseKeypoint[]) => {
        keypoints.forEach(kp => {
            const radius = 8; // Increased from 4 to 8 for better visibility
            const alpha = Math.min(kp.confidence, 1.0);
            const hue = kp.confidence * 120;

            ctx.fillStyle = `hsla(${hue}, 100%, 50%, ${alpha})`;
            ctx.strokeStyle = `hsla(${hue}, 100%, 30%, ${alpha})`;
            ctx.lineWidth = 3; // Increased from 2 to 3

            ctx.beginPath();
            ctx.arc(kp.x, kp.y, radius, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
        });
    }, []);

    const drawSkeleton = useCallback((ctx: CanvasRenderingContext2D, keypoints: PoseKeypoint[]) => {
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
            [11, 12], [11, 23], [12, 24], [23, 24],
            [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
            [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
            [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
            [24, 26], [26, 28], [28, 30], [28, 32], [30, 32]
        ];

        const keypointMap = new Map(keypoints.map(kp => [kp.id, kp]));

        connections.forEach(([startId, endId]) => {
            const startKp = keypointMap.get(startId);
            const endKp = keypointMap.get(endId);

            if (startKp && endKp &&
                startKp.confidence >= confidence_threshold &&
                endKp.confidence >= confidence_threshold) {

                let color = 'rgba(0, 255, 0, 0.7)';
                if (startId <= 10 && endId <= 10) {
                    color = 'rgba(255, 255, 0, 0.7)';
                } else if ((startId % 2 === 1 && startId >= 11) || (endId % 2 === 1 && endId >= 11)) {
                    color = 'rgba(0, 150, 255, 0.7)';
                } else if ((startId % 2 === 0 && startId >= 12) || (endId % 2 === 0 && endId >= 12)) {
                    color = 'rgba(255, 100, 100, 0.7)';
                }

                ctx.strokeStyle = color;
                ctx.lineWidth = 6; // Increased from 3 to 6 for much better visibility
                ctx.lineCap = 'round';

                ctx.beginPath();
                ctx.moveTo(startKp.x, startKp.y);
                ctx.lineTo(endKp.x, endKp.y);
                ctx.stroke();
            }
        });
    }, [confidence_threshold]);

    const drawPoseOverlay = useCallback((pose: PoseResult) => {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');

        if (!canvas || !ctx) return;

        // Clear previous frame
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!pose.keypoints || pose.keypoints.length === 0) return;

        // Filter keypoints by confidence threshold
        const validKeypoints = pose.keypoints.filter(
            kp => kp.confidence >= confidence_threshold
        );

        if (validKeypoints.length === 0) return;

        // Draw skeleton first (behind keypoints)
        if (show_skeleton) {
            drawSkeleton(ctx, validKeypoints);
        }

        // Draw keypoints on top
        if (show_keypoints) {
            drawKeypoints(ctx, validKeypoints);
        }
    }, [confidence_threshold, show_skeleton, show_keypoints, drawSkeleton, drawKeypoints]);

    const toggleOverlay = useCallback(() => {
        setShowOverlay(prev => {
            if (prev) {
                clearOverlay();
            }
            return !prev;
        });
    }, [clearOverlay]);

    // Camera initialization
    const initializeCamera = async () => {
        setIsInitializing(true);
        setCameraError(null);
        setIsCameraReady(false);

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280, max: 1920 },
                    height: { ideal: 720, max: 1080 },
                    frameRate: { ideal: 30, max: 60 },
                    facingMode: 'user'
                },
                audio: false
            });

            streamRef.current = stream;

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.onloadedmetadata = () => {
                    if (videoRef.current) {
                        const { videoWidth, videoHeight } = videoRef.current;

                        // Set canvas to match the PROCESSED frame dimensions (640x480)
                        // not the video's natural dimensions, since keypoints are scaled to 640x480
                        const processedWidth = 640;
                        const processedHeight = 480;

                        setVideoSize({ width: processedWidth, height: processedHeight });

                        if (canvasRef.current) {
                            canvasRef.current.width = processedWidth;
                            canvasRef.current.height = processedHeight;
                        }

                        setIsCameraReady(true);
                    }
                };

                await videoRef.current.play();
            }

            onPermissionChange?.('granted');

        } catch (error) {
            console.error('Camera initialization failed:', error);

            let errorMessage = 'Failed to access camera';
            if (error instanceof Error) {
                if (error.name === 'NotAllowedError') {
                    errorMessage = 'Camera access denied. Please allow camera permissions.';
                    onPermissionChange?.('denied');
                } else if (error.name === 'NotFoundError') {
                    errorMessage = 'No camera found. Please connect a camera device.';
                } else if (error.name === 'NotReadableError') {
                    errorMessage = 'Camera is already in use by another application.';
                }
            }

            setCameraError(errorMessage);
            setIsCameraReady(false);
        } finally {
            setIsInitializing(false);
        }
    };

    const toggleFullscreen = async () => {
        if (!containerRef.current) return;

        try {
            if (!document.fullscreenElement) {
                await containerRef.current.requestFullscreen();
            } else {
                await document.exitFullscreen();
            }
        } catch (error) {
            console.error('Fullscreen toggle failed:', error);
        }
    };

    const toggleMaximize = () => {
        setIsMaximized(!isMaximized);
    };

    // Effects
    useEffect(() => {
        if (isActive) {
            initializeCamera();
        } else {
            stopCamera();
        }

        return () => {
            stopCamera();
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isActive, stopCamera]);

    useEffect(() => {
        if (isActive && isCameraReady && onFrame) {
            startFrameCapture();
        } else {
            stopFrameCapture();
        }

        return () => {
            stopFrameCapture();
        };
    }, [isActive, isCameraReady, onFrame, startFrameCapture, stopFrameCapture]);

    useEffect(() => {
        if (currentPose && showOverlay) {
            console.log('[DEBUG] Drawing pose overlay with', currentPose.keypoints?.length || 0, 'keypoints');
            drawPoseOverlay(currentPose);
        } else {
            if (!currentPose) {
                console.log('[DEBUG] No currentPose to draw');
            }
            if (!showOverlay) {
                console.log('[DEBUG] Overlay hidden');
            }
            clearOverlay();
        }
    }, [currentPose, showOverlay, drawPoseOverlay, clearOverlay]);

    useEffect(() => {
        const handleFullscreenChange = () => {
            setIsFullscreen(!!document.fullscreenElement);
        };

        const handleKeyDown = (e: KeyboardEvent) => {
            // Escape key exits maximize or fullscreen
            if (e.key === 'Escape') {
                if (isMaximized) {
                    setIsMaximized(false);
                }
                if (isFullscreen && document.fullscreenElement) {
                    document.exitFullscreen();
                }
            }
        };

        document.addEventListener('fullscreenchange', handleFullscreenChange);
        document.addEventListener('keydown', handleKeyDown);

        return () => {
            document.removeEventListener('fullscreenchange', handleFullscreenChange);
            document.removeEventListener('keydown', handleKeyDown);
        };
    }, [isMaximized, isFullscreen]);

    // Lock body scroll when maximized
    useEffect(() => {
        if (isMaximized) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = '';
        }

        return () => {
            document.body.style.overflow = '';
        };
    }, [isMaximized]);

    if (cameraError) {
        return (
            <div className="relative aspect-video bg-muted rounded-lg flex items-center justify-center">
                <Alert className="max-w-md">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{cameraError}</AlertDescription>
                </Alert>
            </div>
        );
    }

    // Video player content
    const videoPlayerContent = (
        <div className={`relative bg-black overflow-hidden ${isMaximized
            ? 'w-screen h-screen'
            : 'aspect-video rounded-lg'
            }`}>
            <video
                ref={videoRef}
                className={`${isMaximized ? 'w-full h-full object-contain' : 'w-full h-full object-cover'}`}
                playsInline
                muted
            />

            <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full pointer-events-none"
                style={{ display: showOverlay ? 'block' : 'none' }}
            />

            {isInitializing && (
                <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                    <div className="text-white text-center">
                        <Camera className="w-8 h-8 mx-auto mb-2 animate-pulse" />
                        <p>Initializing camera...</p>
                    </div>
                </div>
            )}

            <div className="absolute top-4 left-4 flex items-center gap-2">
                {isActive ? (
                    <Badge variant="default" className="bg-green-500">
                        <Camera className="w-3 h-3 mr-1" />
                        Live
                    </Badge>
                ) : (
                    <Badge variant="secondary">
                        <CameraOff className="w-3 h-3 mr-1" />
                        Inactive
                    </Badge>
                )}
            </div>

            <div className="absolute top-4 right-4 flex items-center gap-2">
                <Button
                    variant="secondary"
                    size="sm"
                    onClick={toggleOverlay}
                    className="bg-black/50 hover:bg-black/70 text-white border-white/20"
                    title={showOverlay ? "Hide overlay" : "Show overlay"}
                >
                    {showOverlay ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                </Button>

                <Button
                    variant="secondary"
                    size="sm"
                    onClick={toggleMaximize}
                    className="bg-black/50 hover:bg-black/70 text-white border-white/20"
                    title={isMaximized ? "Exit maximize" : "Maximize"}
                >
                    {isMaximized ? <Minimize className="w-4 h-4" /> : <Maximize className="w-4 h-4" />}
                </Button>

                <Button
                    variant="secondary"
                    size="sm"
                    onClick={toggleFullscreen}
                    className="bg-black/50 hover:bg-black/70 text-white border-white/20"
                    title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
                >
                    {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
                </Button>
            </div>

            {currentPose && showOverlay && currentPose.keypoints && currentPose.keypoints.length > 0 && (
                <div className="absolute bottom-4 left-4 bg-black/70 text-white px-3 py-2 rounded text-sm">
                    <div>Keypoints: {currentPose.keypoints.length}</div>
                    <div>Confidence: {currentPose.confidence_scores && currentPose.confidence_scores.length > 0
                        ? (currentPose.confidence_scores.reduce((a, b) => a + b, 0) / currentPose.confidence_scores.length * 100).toFixed(1)
                        : '0.0'}%</div>
                    <div>Processing: {currentPose.processing_time_ms.toFixed(1)}ms</div>
                </div>
            )}
        </div>
    );

    return (
        <>
            <div ref={containerRef} className="relative">
                {!isMaximized && videoPlayerContent}

                {!isMaximized && (
                    <div className="mt-4 flex items-center justify-between text-sm text-muted-foreground">
                        <div>Resolution: {videoSize.width} × {videoSize.height}</div>
                        <div>Target FPS: {targetFPS}</div>
                    </div>
                )}
            </div>

            {/* Render maximized view in a portal */}
            {isMaximized && typeof document !== 'undefined' && createPortal(
                <div className="fixed inset-0 z-50 bg-black flex items-center justify-center">
                    {videoPlayerContent}
                </div>,
                document.body
            )}
        </>
    );
}
