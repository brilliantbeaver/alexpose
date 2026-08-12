/**
 * Realtime Metrics Component
 * 
 * Displays real-time gait analysis metrics including cadence, step length,
 * walking speed, symmetry, and stability measurements with tooltips and proper formatting.
 */

'use client';

import React from 'react';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from '@/components/ui/tooltip';
import {
    Activity,
    Footprints,
    Gauge,
    Scale,
    Shield,
    TrendingUp,
    Clock,
    AlertCircle,
    HelpCircle
} from 'lucide-react';

interface GaitMetrics {
    cadence?: number | null;
    step_length?: number | null;
    stride_length?: number | null;
    walking_speed?: number | null;
    symmetry_index?: number | null;
    stability_score?: number | null;
    confidence: number;
    timestamp: number;
}

interface RealtimeMetricsProps {
    metrics: GaitMetrics | null;
    isActive: boolean;
}

// Metric descriptions for tooltips
const METRIC_DESCRIPTIONS = {
    cadence: "Number of steps taken per minute. Normal walking cadence is typically 100-120 steps/min. Higher values indicate faster stepping rate.",
    walking_speed: "Relative measure of forward movement speed. Values are normalized and require calibration for absolute measurements (m/s or mph).",
    step_length: "Distance covered in a single step (heel strike to opposite heel strike). Measured in relative units based on body proportions.",
    stride_length: "Distance covered in one complete gait cycle (heel strike to same heel strike). Typically about twice the step length.",
    symmetry_index: "Measure of left-right gait symmetry. 100% indicates perfect symmetry. Values below 80% may indicate asymmetric gait patterns.",
    stability_score: "Overall balance and stability during walking. Higher scores indicate more stable, controlled movement with less variability."
};

export function RealtimeMetrics({ metrics, isActive }: RealtimeMetricsProps) {
    if (!metrics) {
        return (
            <div className="space-y-4">
                <div className="text-center text-muted-foreground py-8">
                    {isActive ? (
                        <div className="space-y-2">
                            <Activity className="w-8 h-8 mx-auto animate-pulse" />
                            <p className="text-sm">Analyzing gait patterns...</p>
                            <p className="text-xs">Walk in front of the camera</p>
                        </div>
                    ) : (
                        <div className="space-y-2">
                            <Footprints className="w-8 h-8 mx-auto" />
                            <p className="text-sm">Start analysis to view metrics</p>
                        </div>
                    )}
                </div>
            </div>
        );
    }

    const {
        cadence,
        step_length,
        stride_length,
        walking_speed,
        symmetry_index,
        stability_score,
        confidence,
        timestamp
    } = metrics;

    // Format timestamp
    const formatTime = (timestamp: number): string => {
        return new Date(timestamp * 1000).toLocaleTimeString();
    };

    // Format number to max 2 decimals
    const formatNumber = (value: number | null | undefined, decimals: number = 1): string => {
        if (value === null || value === undefined) return '--';
        return value.toFixed(decimals);
    };

    // Get confidence color
    const getConfidenceColor = (conf: number): string => {
        if (conf >= 0.7) return 'bg-green-500';
        if (conf >= 0.5) return 'bg-yellow-500';
        return 'bg-red-500';
    };

    // Get metric status badge
    const getStatusBadge = (value: number | null | undefined, ranges: { good: [number, number], fair: [number, number] }) => {
        if (value === null || value === undefined) return null;

        const [goodMin, goodMax] = ranges.good;
        const [fairMin, fairMax] = ranges.fair;

        if (value >= goodMin && value <= goodMax) {
            return <Badge variant="outline" className="bg-green-500 text-white border-transparent text-xs px-1.5 py-0">good</Badge>;
        }
        if (value >= fairMin && value <= fairMax) {
            return <Badge variant="outline" className="bg-yellow-500 text-white border-transparent text-xs px-1.5 py-0">fair</Badge>;
        }
        return <Badge variant="outline" className="bg-red-500 text-white border-transparent text-xs px-1.5 py-0">poor</Badge>;
    };

    // Metric item component with tooltip
    const MetricItem = ({
        icon: Icon,
        label,
        value,
        unit,
        description,
        statusBadge
    }: {
        icon: any,
        label: string,
        value: string,
        unit: string,
        description: string,
        statusBadge?: React.ReactNode
    }) => (
        <div className="flex items-center justify-between py-2">
            <TooltipProvider>
                <Tooltip delayDuration={200}>
                    <TooltipTrigger asChild>
                        <div className="flex items-center gap-2 cursor-help">
                            <Icon className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                            <div className="flex items-center gap-1">
                                <span className="text-sm font-medium">{label}</span>
                                <HelpCircle className="w-3 h-3 text-muted-foreground" />
                            </div>
                        </div>
                    </TooltipTrigger>
                    <TooltipContent side="left" className="max-w-xs">
                        <p className="text-xs">{description}</p>
                    </TooltipContent>
                </Tooltip>
            </TooltipProvider>
            <div className="flex items-center gap-2">
                <div className="text-right">
                    <span className="text-lg font-bold tabular-nums">{value}</span>
                    <span className="text-xs text-muted-foreground ml-1">{unit}</span>
                </div>
                {statusBadge}
            </div>
        </div>
    );

    // Progress metric component with tooltip
    const ProgressMetric = ({
        icon: Icon,
        label,
        value,
        description,
        statusBadge
    }: {
        icon: any,
        label: string,
        value: number | null | undefined,
        description: string,
        statusBadge?: React.ReactNode
    }) => (
        <div className="space-y-1.5">
            <div className="flex items-center justify-between">
                <TooltipProvider>
                    <Tooltip delayDuration={200}>
                        <TooltipTrigger asChild>
                            <div className="flex items-center gap-2 cursor-help">
                                <Icon className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                                <div className="flex items-center gap-1">
                                    <span className="text-sm font-medium">{label}</span>
                                    <HelpCircle className="w-3 h-3 text-muted-foreground" />
                                </div>
                            </div>
                        </TooltipTrigger>
                        <TooltipContent side="left" className="max-w-xs">
                            <p className="text-xs">{description}</p>
                        </TooltipContent>
                    </Tooltip>
                </TooltipProvider>
                <div className="flex items-center gap-2">
                    {value !== null && value !== undefined ? (
                        <>
                            <span className="text-sm font-bold tabular-nums">{formatNumber(value * 100, 0)}%</span>
                            {statusBadge}
                        </>
                    ) : (
                        <span className="text-sm text-muted-foreground">--</span>
                    )}
                </div>
            </div>
            {value !== null && value !== undefined && (
                <Progress value={value * 100} className="h-1.5" />
            )}
        </div>
    );

    return (
        <div className="space-y-4">
            {/* Confidence and Timestamp */}
            <div className="flex items-center justify-between pb-2 border-b">
                <Badge
                    variant="outline"
                    className={`${getConfidenceColor(confidence)} text-white border-transparent text-xs`}
                >
                    {formatNumber(confidence * 100, 0)}% confidence
                </Badge>
                <div className="text-xs text-muted-foreground flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {formatTime(timestamp)}
                </div>
            </div>

            {/* Primary Metrics */}
            <div className="space-y-1 divide-y">
                <MetricItem
                    icon={Activity}
                    label="Cadence"
                    value={formatNumber(cadence, 0)}
                    unit="steps/min"
                    description={METRIC_DESCRIPTIONS.cadence}
                    statusBadge={getStatusBadge(cadence, { good: [100, 120], fair: [80, 140] })}
                />

                <MetricItem
                    icon={Gauge}
                    label="Walking Speed"
                    value={formatNumber(walking_speed, 2)}
                    unit="rel. units"
                    description={METRIC_DESCRIPTIONS.walking_speed}
                    statusBadge={getStatusBadge(walking_speed, { good: [0.8, 1.5], fair: [0.4, 2.5] })}
                />

                <MetricItem
                    icon={Footprints}
                    label="Step Length"
                    value={formatNumber(step_length, 2)}
                    unit="rel. units"
                    description={METRIC_DESCRIPTIONS.step_length}
                    statusBadge={getStatusBadge(step_length, { good: [0.4, 0.7], fair: [0.2, 1.0] })}
                />

                <MetricItem
                    icon={TrendingUp}
                    label="Stride Length"
                    value={formatNumber(stride_length, 2)}
                    unit="rel. units"
                    description={METRIC_DESCRIPTIONS.stride_length}
                    statusBadge={getStatusBadge(stride_length, { good: [0.8, 1.4], fair: [0.4, 2.0] })}
                />
            </div>

            {/* Secondary Metrics with Progress Bars */}
            <div className="space-y-3 pt-2">
                <ProgressMetric
                    icon={Scale}
                    label="Symmetry Index"
                    value={symmetry_index}
                    description={METRIC_DESCRIPTIONS.symmetry_index}
                    statusBadge={getStatusBadge(symmetry_index, { good: [0.8, 1.0], fair: [0.6, 0.8] })}
                />

                <ProgressMetric
                    icon={Shield}
                    label="Stability Score"
                    value={stability_score}
                    description={METRIC_DESCRIPTIONS.stability_score}
                    statusBadge={getStatusBadge(stability_score, { good: [0.7, 1.0], fair: [0.5, 0.7] })}
                />
            </div>

            {/* Low Confidence Warning */}
            {confidence < 0.5 && (
                <div className="flex items-start gap-2 p-2 rounded-md bg-yellow-50 border border-yellow-200">
                    <AlertCircle className="w-4 h-4 text-yellow-600 mt-0.5 flex-shrink-0" />
                    <div className="text-xs">
                        <p className="font-medium text-yellow-800">Low Confidence</p>
                        <p className="text-yellow-700 mt-0.5">
                            Ensure good lighting and clear full body view.
                        </p>
                    </div>
                </div>
            )}

            {/* Real-time Indicator */}
            {isActive && (
                <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground pt-2 border-t">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                    <span>Updating in real-time</span>
                </div>
            )}
        </div>
    );
}