/**
 * Realtime Gait Analysis Page
 * 
 * This page provides live webcam-based gait analysis with real-time pose estimation
 * and immediate visual feedback through keypoint and skeletal overlays.
 */

'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
    Video,
    VideoOff,
    Settings,
    Activity,
    Zap,
    AlertCircle,
    CheckCircle,
    Loader2
} from 'lucide-react';

import { RealtimeCamera } from '@/components/realtime/RealtimeCamera';
import { RealtimeControls } from '@/components/realtime/RealtimeControls';
import { RealtimeStats } from '@/components/realtime/RealtimeStats';
import { RealtimeMetrics } from '@/components/realtime/RealtimeMetrics';
import { useRealtimeAnalysis } from '@/hooks/useRealtimeAnalysis';

export default function RealtimePage() {
    const {
        isConnected,
        isProcessing,
        currentPose,
        gaitMetrics,
        statistics,
        config,
        error,
        connect,
        disconnect,
        updateConfig,
        sendFrame
    } = useRealtimeAnalysis();

    const [cameraPermission, setCameraPermission] = useState<'granted' | 'denied' | 'prompt'>('prompt');
    const [showSettings, setShowSettings] = useState(false);

    // Check camera permission on mount
    useEffect(() => {
        checkCameraPermission();
    }, []);

    const checkCameraPermission = async () => {
        try {
            const result = await navigator.permissions.query({ name: 'camera' as PermissionName });
            setCameraPermission(result.state);

            result.addEventListener('change', () => {
                setCameraPermission(result.state);
            });
        } catch (error) {
            console.warn('Permission API not supported, will request camera access directly');
        }
    };

    const handleStartAnalysis = useCallback(async () => {
        try {
            await connect();
        } catch (error) {
            console.error('Failed to start analysis:', error);
        }
    }, [connect]);

    const handleStopAnalysis = useCallback(() => {
        disconnect();
    }, [disconnect]);

    const handleConfigChange = useCallback((newConfig: any) => {
        updateConfig(newConfig);
    }, [updateConfig]);

    const renderConnectionStatus = () => {
        if (error) {
            return (
                <Badge variant="destructive" className="flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" />
                    Error
                </Badge>
            );
        }

        if (isConnected && isProcessing) {
            return (
                <Badge variant="default" className="flex items-center gap-2 bg-green-500">
                    <CheckCircle className="w-4 h-4" />
                    Active
                </Badge>
            );
        }

        if (isConnected) {
            return (
                <Badge variant="secondary" className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Connected
                </Badge>
            );
        }

        return (
            <Badge variant="outline" className="flex items-center gap-2">
                <VideoOff className="w-4 h-4" />
                Disconnected
            </Badge>
        );
    };

    const renderCameraPermissionAlert = () => {
        if (cameraPermission === 'denied') {
            return (
                <Alert className="mb-6">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                        Camera access is required for realtime analysis. Please enable camera permissions in your browser settings.
                    </AlertDescription>
                </Alert>
            );
        }

        return null;
    };

    return (
        <div className="container mx-auto px-4 py-6 space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">See Your Movement, Live</h1>
                    <p className="text-muted-foreground mt-2">
                        Live webcam-based pose estimation and gait analysis, with feedback the moment you move
                    </p>
                </div>

                <div className="flex items-center gap-4">
                    {renderConnectionStatus()}

                    <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setShowSettings(!showSettings)}
                    >
                        <Settings className="w-4 h-4 mr-2" />
                        Settings
                    </Button>
                </div>
            </div>

            {/* Camera Permission Alert */}
            {renderCameraPermissionAlert()}

            {/* Error Alert */}
            {error && (
                <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                        {error}
                    </AlertDescription>
                </Alert>
            )}

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Camera and Controls - Left Column */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Camera Feed */}
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <Video className="w-5 h-5" />
                                Live Camera Feed
                            </CardTitle>
                            <CardDescription>
                                Webcam stream with real-time pose estimation overlay
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <RealtimeCamera
                                isActive={isConnected && isProcessing}
                                currentPose={currentPose}
                                onFrame={sendFrame}
                                config={config}
                                onPermissionChange={setCameraPermission}
                            />
                        </CardContent>
                    </Card>

                    {/* Control Panel */}
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <Settings className="w-5 h-5" />
                                Analysis Controls
                            </CardTitle>
                            <CardDescription>
                                Start/stop analysis and configure processing parameters
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="flex items-center gap-4 mb-6">
                                {!isConnected ? (
                                    <Button
                                        onClick={handleStartAnalysis}
                                        disabled={cameraPermission === 'denied'}
                                        className="flex items-center gap-2"
                                    >
                                        <Activity className="w-4 h-4" />
                                        Start Analysis
                                    </Button>
                                ) : (
                                    <Button
                                        onClick={handleStopAnalysis}
                                        variant="destructive"
                                        className="flex items-center gap-2"
                                    >
                                        <VideoOff className="w-4 h-4" />
                                        Stop Analysis
                                    </Button>
                                )}

                                <Separator orientation="vertical" className="h-6" />

                                <div className="text-sm text-muted-foreground">
                                    {isProcessing ? 'Processing frames...' : 'Ready to start'}
                                </div>
                            </div>

                            {showSettings && (
                                <RealtimeControls
                                    config={config}
                                    onConfigChange={handleConfigChange}
                                    isProcessing={isProcessing}
                                />
                            )}
                        </CardContent>
                    </Card>
                </div>

                {/* Statistics and Metrics - Right Column */}
                <div className="space-y-6">
                    {/* Current Gait Metrics */}
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <Zap className="w-5 h-5" />
                                Gait Metrics
                            </CardTitle>
                            <CardDescription>
                                Real-time gait analysis results
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <RealtimeMetrics
                                metrics={gaitMetrics}
                                isActive={isProcessing}
                            />
                        </CardContent>
                    </Card>

                    {/* Processing Statistics */}
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <Activity className="w-5 h-5" />
                                Performance Stats
                            </CardTitle>
                            <CardDescription>
                                Processing performance and system metrics
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <RealtimeStats
                                statistics={statistics}
                                isActive={isProcessing}
                            />
                        </CardContent>
                    </Card>

                    {/* Quick Info */}
                    <Card>
                        <CardHeader>
                            <CardTitle>About Realtime Analysis</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-3 text-sm text-muted-foreground">
                            <p>
                                This feature provides live gait analysis using your webcam with immediate visual feedback.
                            </p>
                            <p>
                                <strong>Features:</strong>
                            </p>
                            <ul className="list-disc list-inside space-y-1 ml-2">
                                <li>Real-time pose estimation</li>
                                <li>Live gait metrics calculation</li>
                                <li>Adjustable processing quality</li>
                                <li>Visual pose overlays</li>
                                <li>Performance monitoring</li>
                            </ul>
                            <p>
                                <strong>Requirements:</strong>
                            </p>
                            <ul className="list-disc list-inside space-y-1 ml-2">
                                <li>Webcam access permission</li>
                                <li>Good lighting conditions</li>
                                <li>Clear view of full body</li>
                                <li>Stable internet connection</li>
                            </ul>
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    );
}