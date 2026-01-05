/**
 * Multi-Stage Loading Component
 * Shows progress through multiple loading stages with animations
 */

'use client';

import React, { useEffect, useState } from 'react';
import { cn } from '@/lib/utils';
import { CheckCircle2, Loader2 } from 'lucide-react';

interface LoadingStage {
  id: string;
  title: string;
  description?: string;
  icon?: React.ReactNode;
  duration?: number; // Optional duration in ms for auto-progression
}

interface MultiStageLoadingProps {
  stages: LoadingStage[];
  currentStage: string;
  className?: string;
  variant?: 'vertical' | 'horizontal';
  showProgress?: boolean;
}

export function MultiStageLoading({
  stages,
  currentStage,
  className,
  variant = 'vertical',
  showProgress = true
}: MultiStageLoadingProps) {
  const currentIndex = stages.findIndex(stage => stage.id === currentStage);
  const progress = currentIndex >= 0 ? ((currentIndex + 1) / stages.length) * 100 : 0;

  if (variant === 'horizontal') {
    return (
      <div className={cn('space-y-6', className)}>
        {showProgress && (
          <div className="w-full max-w-md mx-auto">
            <div className="flex justify-between text-xs text-muted-foreground mb-2">
              <span>Loading Progress</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div className="w-full bg-secondary rounded-full h-2">
              <div
                className="bg-primary h-2 rounded-full transition-all duration-500 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        <div className="flex items-center justify-center space-x-4 overflow-x-auto pb-2">
          {stages.map((stage, index) => {
            const isActive = stage.id === currentStage;
            const isCompleted = index < currentIndex;
            const isPending = index > currentIndex;

            return (
              <div
                key={stage.id}
                className={cn(
                  'flex flex-col items-center space-y-2 min-w-0 flex-shrink-0',
                  'transition-all duration-300'
                )}
              >
                <div
                  className={cn(
                    'w-12 h-12 rounded-full flex items-center justify-center border-2 transition-all duration-300',
                    isCompleted && 'bg-green-100 border-green-500 text-green-700',
                    isActive && 'bg-primary/10 border-primary text-primary animate-pulse',
                    isPending && 'bg-muted border-muted-foreground/20 text-muted-foreground'
                  )}
                >
                  {isCompleted ? (
                    <CheckCircle2 className="w-6 h-6" />
                  ) : isActive ? (
                    stage.icon || <Loader2 className="w-6 h-6 animate-spin" />
                  ) : (
                    stage.icon || <div className="w-3 h-3 rounded-full bg-current" />
                  )}
                </div>
                <div className="text-center max-w-24">
                  <p
                    className={cn(
                      'text-xs font-medium transition-colors duration-300',
                      isActive && 'text-primary',
                      isCompleted && 'text-green-700',
                      isPending && 'text-muted-foreground'
                    )}
                  >
                    {stage.title}
                  </p>
                  {stage.description && isActive && (
                    <p className="text-xs text-muted-foreground mt-1">
                      {stage.description}
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Current stage details */}
        {currentIndex >= 0 && stages[currentIndex].description && (
          <div className="text-center">
            <p className="text-sm text-muted-foreground">
              {stages[currentIndex].description}
            </p>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={cn('space-y-4', className)}>
      {showProgress && (
        <div className="w-full max-w-md mx-auto">
          <div className="flex justify-between text-xs text-muted-foreground mb-2">
            <span>Loading Progress</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-secondary rounded-full h-2">
            <div
              className="bg-primary h-2 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      <div className="space-y-3">
        {stages.map((stage, index) => {
          const isActive = stage.id === currentStage;
          const isCompleted = index < currentIndex;
          const isPending = index > currentIndex;

          return (
            <div
              key={stage.id}
              className={cn(
                'flex items-center space-x-3 p-3 rounded-lg transition-all duration-300',
                isActive && 'bg-primary/5 border border-primary/20',
                isCompleted && 'bg-green-50 border border-green-200',
                isPending && 'bg-muted/30'
              )}
            >
              <div
                className={cn(
                  'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 transition-all duration-300',
                  isCompleted && 'bg-green-500 text-white',
                  isActive && 'bg-primary text-primary-foreground animate-pulse',
                  isPending && 'bg-muted text-muted-foreground'
                )}
              >
                {isCompleted ? (
                  <CheckCircle2 className="w-5 h-5" />
                ) : isActive ? (
                  stage.icon || <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <div className="w-2 h-2 rounded-full bg-current" />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <p
                  className={cn(
                    'text-sm font-medium transition-colors duration-300',
                    isActive && 'text-primary',
                    isCompleted && 'text-green-700',
                    isPending && 'text-muted-foreground'
                  )}
                >
                  {stage.title}
                </p>
                {stage.description && (
                  <p
                    className={cn(
                      'text-xs mt-1 transition-colors duration-300',
                      isActive && 'text-primary/70',
                      isCompleted && 'text-green-600',
                      isPending && 'text-muted-foreground'
                    )}
                  >
                    {stage.description}
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Specialized multi-stage loading for sequence analysis
export function SequenceAnalysisLoading({ currentStage }: { currentStage: string }) {
  const stages: LoadingStage[] = [
    {
      id: 'loading-metadata',
      title: 'Loading Metadata',
      description: 'Fetching sequence information and properties'
    },
    {
      id: 'loading-frames',
      title: 'Loading Frames',
      description: 'Retrieving video frames and bounding boxes'
    },
    {
      id: 'loading-pose',
      title: 'Loading Pose Data',
      description: 'Fetching pose keypoints for each frame'
    },
    {
      id: 'processing-analysis',
      title: 'Processing Analysis',
      description: 'Analyzing gait patterns and generating insights'
    },
    {
      id: 'complete',
      title: 'Complete',
      description: 'Analysis ready for visualization'
    }
  ];

  return (
    <div className="flex items-center justify-center min-h-[400px]">
      <div className="w-full max-w-md">
        <div className="text-center mb-6">
          <h3 className="text-lg font-semibold mb-2">Loading Sequence Analysis</h3>
          <p className="text-sm text-muted-foreground">
            Please wait while we prepare your gait analysis data
          </p>
        </div>
        <MultiStageLoading
          stages={stages}
          currentStage={currentStage}
          variant="vertical"
          showProgress={true}
        />
      </div>
    </div>
  );
}