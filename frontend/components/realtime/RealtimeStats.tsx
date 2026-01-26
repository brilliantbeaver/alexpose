/**
 * Realtime Statistics Component
 * 
 * Displays processing performance metrics and system statistics
 * for the realtime gait analysis system.
 */

'use client';

import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
    Activity,
    Clock,
    Cpu,
    Zap,
    Target,
    TrendingUp,
    AlertTriangle
} from 'lucide-react';

interface ProcessingStats {
    frames_received?: number;
    frames_processed?: number;
    frames_failed?: number;
    average_processing_time_ms?: number;
    poses_detected?: number;
    gait_analyses_completed?: number;
    session_duration_seconds?: number;
    fps?: number;
    frames_skipped?: number;
}

interface RealtimeStatsProps {
    statistics: ProcessingStats | null;
    isActive: boolean;
}

export function RealtimeStats({ statistics, isActive }: RealtimeStatsProps) {
    if (!statistics) {
        return (
            <div className="space-y-4">
                <div className="text-center text-muted-foreground py-8">
                    {isActive ? (
                        <div className="space-y-2">
                            <Activity className="w-8 h-8 mx-auto animate-pulse" />
                            <p>Collecting statistics...</p>
                        </div>
                    ) : (
                        <div className="space-y-2">
                            <Target className="w-8 h-8 mx-auto" />
                            <p>Start analysis to view statistics</p>
                        </div>
                    )}
                </div>
            </div>
        );
    }

    const {
        frames_received = 0,
        frames_processed = 0,
        frames_failed = 0,
        average_processing_time_ms = 0,
        poses_detected = 0,
        gait_analyses_completed = 0,
        session_duration_seconds = 0,
        fps = 0,
        frames_skipped = 0
    } = statistics;

    // Calculate derived metrics
    const successRate = frames_received > 0 ? (frames_processed / frames_received) * 100 : 0;
    const poseDetectionRate = frames_processed > 0 ? (poses_detected / frames_processed) * 100 : 0;
    const skipRate = frames_received > 0 ? (frames_skipped / frames_received) * 100 : 0;

    // Format duration
    const formatDuration = (seconds: number): string => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    // Get performance status
    const getPerformanceStatus = () => {
        if (average_processing_time_ms < 33) return { status: 'excellent', color: 'bg-green-500' };
        if (average_processing_time_ms < 50) return { status: 'good', color: 'bg-blue-500' };
        if (average_processing_time_ms < 100) return { status: 'fair', color: 'bg-yellow-500' };
        return { status: 'poor', color: 'bg-red-500' };
    };

    const performanceStatus = getPerformanceStatus();

    return (
        <div className="space-y-4">
            {/* Session Overview */}
            <div className="grid grid-cols-2 gap-3">
                <Card className="p-3">
                    <div className="flex items-center gap-2">
                        <Clock className="w-4 h-4 text-muted-foreground" />
                        <div>
                            <p className="text-sm font-medium">Duration</p>
                            <p className="text-lg font-bold">{formatDuration(session_duration_seconds)}</p>
                        </div>
                    </div>
                </Card>

                <Card className="p-3">
                    <div className="flex items-center gap-2">
                        <Zap className="w-4 h-4 text-muted-foreground" />
                        <div>
                            <p className="text-sm font-medium">FPS</p>
                            <p className="text-lg font-bold">{fps.toFixed(1)}</p>
                        </div>
                    </div>
                </Card>
            </div>

            {/* Processing Performance */}
            <Card className="p-3">
                <div className="space-y-3">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <Cpu className="w-4 h-4 text-muted-foreground" />
                            <span className="text-sm font-medium">Processing Time</span>
                        </div>
                        <Badge
                            variant="outline"
                            className={`${performanceStatus.color} text-white border-transparent`}
                        >
                            {performanceStatus.status}
                        </Badge>
                    </div>

                    <div>
                        <div className="flex justify-between text-sm mb-1">
                            <span>Avg: {average_processing_time_ms.toFixed(1)}ms</span>
                            <span>Target: &lt;33ms</span>
                        </div>
                        <Progress
                            value={Math.min((average_processing_time_ms / 100) * 100, 100)}
                            className="h-2"
                        />
                    </div>
                </div>
            </Card>

            {/* Frame Statistics */}
            <div className="space-y-3">
                <h4 className="text-sm font-medium flex items-center gap-2">
                    <Activity className="w-4 h-4" />
                    Frame Processing
                </h4>

                <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                        <span>Frames Received</span>
                        <span className="font-mono">{frames_received.toLocaleString()}</span>
                    </div>

                    <div className="flex justify-between">
                        <span>Frames Processed</span>
                        <span className="font-mono">{frames_processed.toLocaleString()}</span>
                    </div>

                    <div className="flex justify-between">
                        <span>Frames Skipped</span>
                        <span className="font-mono">{frames_skipped.toLocaleString()}</span>
                    </div>

                    <div className="flex justify-between">
                        <span>Frames Failed</span>
                        <span className="font-mono text-red-500">{frames_failed.toLocaleString()}</span>
                    </div>
                </div>

                {/* Success Rate Progress */}
                <div>
                    <div className="flex justify-between text-sm mb-1">
                        <span>Success Rate</span>
                        <span>{successRate.toFixed(1)}%</span>
                    </div>
                    <Progress value={successRate} className="h-2" />
                </div>
            </div>

            {/* Detection Statistics */}
            <div className="space-y-3">
                <h4 className="text-sm font-medium flex items-center gap-2">
                    <Target className="w-4 h-4" />
                    Detection Results
                </h4>

                <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                        <span>Poses Detected</span>
                        <span className="font-mono">{poses_detected.toLocaleString()}</span>
                    </div>

                    <div className="flex justify-between">
                        <span>Gait Analyses</span>
                        <span className="font-mono">{gait_analyses_completed.toLocaleString()}</span>
                    </div>
                </div>

                {/* Detection Rate Progress */}
                <div>
                    <div className="flex justify-between text-sm mb-1">
                        <span>Pose Detection Rate</span>
                        <span>{poseDetectionRate.toFixed(1)}%</span>
                    </div>
                    <Progress value={poseDetectionRate} className="h-2" />
                </div>
            </div>

            {/* Performance Warnings */}
            {(skipRate > 20 || average_processing_time_ms > 50) && (
                <Card className="p-3 border-yellow-200 bg-yellow-50">
                    <div className="flex items-start gap-2">
                        <AlertTriangle className="w-4 h-4 text-yellow-600 mt-0.5" />
                        <div className="text-sm">
                            <p className="font-medium text-yellow-800">Performance Notice</p>
                            <div className="text-yellow-700 mt-1 space-y-1">
                                {skipRate > 20 && (
                                    <p>• High frame skip rate ({skipRate.toFixed(1)}%) - consider reducing quality</p>
                                )}
                                {average_processing_time_ms > 50 && (
                                    <p>• Slow processing ({average_processing_time_ms.toFixed(1)}ms) - try Fast mode</p>
                                )}
                            </div>
                        </div>
                    </div>
                </Card>
            )}

            {/* Real-time Indicators */}
            {isActive && (
                <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                    <span>Live updating every 2 seconds</span>
                </div>
            )}
        </div>
    );
}