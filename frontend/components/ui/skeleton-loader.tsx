/**
 * Skeleton Loading Components
 * Provides skeleton placeholders that match the actual content layout
 */

'use client';

import React from 'react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader } from '@/components/ui/card';

interface SkeletonProps {
  className?: string;
  variant?: 'default' | 'pulse' | 'wave';
}

export function Skeleton({ className, variant = 'pulse' }: SkeletonProps) {
  return (
    <div
      className={cn(
        'bg-muted rounded-md',
        variant === 'pulse' && 'animate-pulse',
        variant === 'wave' && 'animate-pulse bg-gradient-to-r from-muted via-muted/50 to-muted bg-[length:200%_100%] animate-[wave_2s_ease-in-out_infinite]',
        className
      )}
    />
  );
}

// Skeleton for dataset statistics cards
export function DatasetStatsSkeleton() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
      {[...Array(4)].map((_, i) => (
        <Card key={i}>
          <CardHeader className="pb-3">
            <Skeleton className="h-4 w-24" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-8 w-16 mb-2" />
            <Skeleton className="h-3 w-20" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// Skeleton for sequence list
export function SequenceListSkeleton({ count = 5 }: { count?: number }) {
  return (
    <div className="space-y-2">
      {[...Array(count)].map((_, i) => (
        <div key={i} className="flex items-center justify-between p-3 border rounded-lg">
          <div className="space-y-2 flex-1">
            <Skeleton className="h-4 w-48" />
            <Skeleton className="h-3 w-32" />
          </div>
          <Skeleton className="h-8 w-16" />
        </div>
      ))}
    </div>
  );
}

// Skeleton for pose analysis overview
export function PoseAnalysisSkeleton() {
  return (
    <div className="space-y-6">
      {/* Overall Assessment Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Skeleton className="h-5 w-5" />
              <Skeleton className="h-6 w-40" />
              <Skeleton className="h-4 w-4" />
            </div>
            <Skeleton className="h-4 w-20" />
          </div>
          <Skeleton className="h-4 w-64" />
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[...Array(2)].map((_, i) => (
              <div key={i} className="border rounded-lg p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Skeleton className="h-4 w-20" />
                    <Skeleton className="h-3.5 w-3.5" />
                  </div>
                  <Skeleton className="h-5 w-5" />
                </div>
                <Skeleton className="h-6 w-24" />
                <Skeleton className="h-3 w-32" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <Card key={i}>
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                <Skeleton className="h-4 w-4" />
                <Skeleton className="h-4 w-16" />
                <Skeleton className="h-3.5 w-3.5" />
              </div>
            </CardHeader>
            <CardContent className="space-y-2">
              <Skeleton className="h-8 w-12" />
              <Skeleton className="h-3 w-20" />
              <Skeleton className="h-5 w-16" />
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Recommendations */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Skeleton className="h-5 w-5" />
            <Skeleton className="h-6 w-32" />
          </div>
          <Skeleton className="h-4 w-48" />
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="flex items-start space-x-3 p-3 rounded-lg bg-muted/50">
                <Skeleton className="h-4 w-4 mt-0.5 flex-shrink-0" />
                <Skeleton className="h-4 flex-1" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Skeleton for video player
export function VideoPlayerSkeleton() {
  return (
    <div className="space-y-4">
      <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
        <div className="text-center space-y-2">
          <Skeleton className="h-12 w-12 rounded-full mx-auto" />
          <Skeleton className="h-4 w-32" />
        </div>
      </div>
      
      {/* Controls */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-4 w-16" />
        </div>
        <Skeleton className="h-2 w-full" />
        <div className="flex items-center justify-between text-xs">
          <Skeleton className="h-3 w-8" />
          <Skeleton className="h-3 w-16" />
          <Skeleton className="h-3 w-8" />
        </div>
      </div>
    </div>
  );
}

// Skeleton for frame information
export function FrameInfoSkeleton() {
  return (
    <div className="grid grid-cols-2 gap-4 text-sm border rounded-lg p-4">
      {[...Array(4)].map((_, i) => (
        <div key={i}>
          <Skeleton className="h-3 w-20 mb-1" />
          <Skeleton className="h-4 w-16" />
        </div>
      ))}
    </div>
  );
}