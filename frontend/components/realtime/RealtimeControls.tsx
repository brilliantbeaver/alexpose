/**
 * Realtime Controls Component
 * 
 * Provides user interface controls for configuring realtime analysis parameters
 * including processing mode, overlay settings, and performance options.
 */

'use client';

import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
    Settings,
    Zap,
    Eye,
    Target,
    Gauge,
    RotateCcw
} from 'lucide-react';

interface RealtimeConfig {
    processing_mode: 'fast' | 'balanced' | 'accurate';
    buffer_size: number;
    enable_tracking: boolean;
    confidence_threshold: number;
    show_keypoints: boolean;
    show_skeleton: boolean;
    target_fps: number;
}

interface RealtimeControlsProps {
    config: RealtimeConfig;
    onConfigChange: (newConfig: Partial<RealtimeConfig>) => void;
    isProcessing: boolean;
}

export function RealtimeControls({
    config,
    onConfigChange,
    isProcessing
}: RealtimeControlsProps) {

    const handleProcessingModeChange = (mode: string) => {
        onConfigChange({
            processing_mode: mode as 'fast' | 'balanced' | 'accurate'
        });
    };

    const handleConfidenceThresholdChange = (values: number[]) => {
        onConfigChange({ confidence_threshold: values[0] });
    };

    const handleBufferSizeChange = (values: number[]) => {
        onConfigChange({ buffer_size: values[0] });
    };

    const handleTargetFpsChange = (values: number[]) => {
        onConfigChange({ target_fps: values[0] });
    };

    const resetToDefaults = () => {
        onConfigChange({
            processing_mode: 'balanced',
            buffer_size: 30,
            enable_tracking: true,
            confidence_threshold: 0.5,
            show_keypoints: true,
            show_skeleton: true,
            target_fps: 25
        });
    };

    const getProcessingModeInfo = (mode: string) => {
        switch (mode) {
            case 'fast':
                return {
                    description: 'Optimized for speed, lower accuracy',
                    fps: '~30 FPS',
                    cpu: 'Low',
                    color: 'bg-green-500'
                };
            case 'balanced':
                return {
                    description: 'Balance between speed and accuracy',
                    fps: '~25 FPS',
                    cpu: 'Medium',
                    color: 'bg-blue-500'
                };
            case 'accurate':
                return {
                    description: 'Highest accuracy, slower processing',
                    fps: '~20 FPS',
                    cpu: 'High',
                    color: 'bg-orange-500'
                };
            default:
                return {
                    description: 'Unknown mode',
                    fps: 'Unknown',
                    cpu: 'Unknown',
                    color: 'bg-gray-500'
                };
        }
    };

    const modeInfo = getProcessingModeInfo(config.processing_mode);

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Settings className="w-5 h-5" />
                    <h3 className="text-lg font-semibold">Analysis Configuration</h3>
                </div>

                <Button
                    variant="outline"
                    size="sm"
                    onClick={resetToDefaults}
                    disabled={isProcessing}
                >
                    <RotateCcw className="w-4 h-4 mr-2" />
                    Reset
                </Button>
            </div>

            {/* Processing Mode */}
            <div className="space-y-3">
                <div className="flex items-center gap-2">
                    <Zap className="w-4 h-4" />
                    <Label className="text-sm font-medium">Processing Mode</Label>
                </div>

                <Select
                    value={config.processing_mode}
                    onValueChange={handleProcessingModeChange}
                    disabled={isProcessing}
                >
                    <SelectTrigger>
                        <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                        <SelectItem value="fast">
                            <div className="flex items-center gap-2">
                                <div className={`w-2 h-2 rounded-full bg-green-500`} />
                                Fast
                            </div>
                        </SelectItem>
                        <SelectItem value="balanced">
                            <div className="flex items-center gap-2">
                                <div className={`w-2 h-2 rounded-full bg-blue-500`} />
                                Balanced
                            </div>
                        </SelectItem>
                        <SelectItem value="accurate">
                            <div className="flex items-center gap-2">
                                <div className={`w-2 h-2 rounded-full bg-orange-500`} />
                                Accurate
                            </div>
                        </SelectItem>
                    </SelectContent>
                </Select>

                <div className="text-xs text-muted-foreground space-y-1">
                    <p>{modeInfo.description}</p>
                    <div className="flex items-center gap-4">
                        <span>Target: {modeInfo.fps}</span>
                        <span>CPU Usage: {modeInfo.cpu}</span>
                    </div>
                </div>
            </div>

            <Separator />

            {/* Visual Settings */}
            <div className="space-y-4">
                <div className="flex items-center gap-2">
                    <Eye className="w-4 h-4" />
                    <Label className="text-sm font-medium">Visual Overlay</Label>
                </div>

                <div className="space-y-3">
                    <div className="flex items-center justify-between">
                        <Label htmlFor="show-keypoints" className="text-sm">
                            Show Keypoints
                        </Label>
                        <Switch
                            id="show-keypoints"
                            checked={config.show_keypoints}
                            onCheckedChange={(checked) =>
                                onConfigChange({ show_keypoints: checked })
                            }
                        />
                    </div>

                    <div className="flex items-center justify-between">
                        <Label htmlFor="show-skeleton" className="text-sm">
                            Show Skeleton
                        </Label>
                        <Switch
                            id="show-skeleton"
                            checked={config.show_skeleton}
                            onCheckedChange={(checked) =>
                                onConfigChange({ show_skeleton: checked })
                            }
                        />
                    </div>
                </div>
            </div>

            <Separator />

            {/* Performance Settings */}
            <div className="space-y-4">
                <div className="flex items-center gap-2">
                    <Gauge className="w-4 h-4" />
                    <Label className="text-sm font-medium">Performance</Label>
                </div>

                <div className="space-y-4">
                    {/* Confidence Threshold */}
                    <div className="space-y-2">
                        <div className="flex items-center justify-between">
                            <Label className="text-sm">Confidence Threshold</Label>
                            <Badge variant="outline" className="text-xs">
                                {(config.confidence_threshold * 100).toFixed(0)}%
                            </Badge>
                        </div>
                        <Slider
                            value={[config.confidence_threshold]}
                            onValueChange={handleConfidenceThresholdChange}
                            min={0.1}
                            max={0.9}
                            step={0.1}
                            disabled={isProcessing}
                            className="w-full"
                        />
                        <p className="text-xs text-muted-foreground">
                            Minimum confidence for displaying keypoints
                        </p>
                    </div>

                    {/* Buffer Size */}
                    <div className="space-y-2">
                        <div className="flex items-center justify-between">
                            <Label className="text-sm">Buffer Size</Label>
                            <Badge variant="outline" className="text-xs">
                                {config.buffer_size} frames
                            </Badge>
                        </div>
                        <Slider
                            value={[config.buffer_size]}
                            onValueChange={handleBufferSizeChange}
                            min={10}
                            max={60}
                            step={5}
                            disabled={isProcessing}
                            className="w-full"
                        />
                        <p className="text-xs text-muted-foreground">
                            Number of frames to keep in memory for analysis
                        </p>
                    </div>

                    {/* Target FPS */}
                    <div className="space-y-2">
                        <div className="flex items-center justify-between">
                            <Label className="text-sm">Target FPS</Label>
                            <Badge variant="outline" className="text-xs">
                                {config.target_fps} fps
                            </Badge>
                        </div>
                        <Slider
                            value={[config.target_fps]}
                            onValueChange={handleTargetFpsChange}
                            min={10}
                            max={30}
                            step={5}
                            disabled={isProcessing}
                            className="w-full"
                        />
                        <p className="text-xs text-muted-foreground">
                            Target frame processing rate
                        </p>
                    </div>
                </div>
            </div>

            <Separator />

            {/* Advanced Settings */}
            <div className="space-y-4">
                <div className="flex items-center gap-2">
                    <Target className="w-4 h-4" />
                    <Label className="text-sm font-medium">Advanced</Label>
                </div>

                <div className="flex items-center justify-between">
                    <div>
                        <Label htmlFor="enable-tracking" className="text-sm">
                            Pose Tracking
                        </Label>
                        <p className="text-xs text-muted-foreground">
                            Smooth pose estimates across frames
                        </p>
                    </div>
                    <Switch
                        id="enable-tracking"
                        checked={config.enable_tracking}
                        onCheckedChange={(checked) =>
                            onConfigChange({ enable_tracking: checked })
                        }
                        disabled={isProcessing}
                    />
                </div>
            </div>

            {/* Status Info */}
            {isProcessing && (
                <div className="mt-6 p-3 bg-muted rounded-lg">
                    <p className="text-sm text-muted-foreground">
                        <strong>Note:</strong> Some settings cannot be changed while processing is active.
                        Stop the analysis to modify these settings.
                    </p>
                </div>
            )}
        </div>
    );
}