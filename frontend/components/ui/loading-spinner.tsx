/**
 * Enhanced Loading Spinner Components
 * Provides various loading states with animations and contextual messages
 */

'use client';

import React from 'react';
import { cn } from '@/lib/utils';
import { Activity, Brain, Database, FileVideo, Loader2, Zap } from 'lucide-react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  variant?: 'default' | 'pulse' | 'dots' | 'bars';
  className?: string;
}

export function LoadingSpinner({ size = 'md', variant = 'default', className }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  };

  if (variant === 'pulse') {
    return (
      <div className={cn('relative', sizeClasses[size], className)}>
        <div className="absolute inset-0 rounded-full bg-primary/20 animate-ping"></div>
        <div className="relative rounded-full bg-primary animate-pulse"></div>
      </div>
    );
  }

  if (variant === 'dots') {
    return (
      <div className={cn('flex space-x-1', className)}>
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className={cn(
              'rounded-full bg-primary animate-bounce',
              size === 'sm' ? 'w-1 h-1' : size === 'md' ? 'w-2 h-2' : size === 'lg' ? 'w-3 h-3' : 'w-4 h-4'
            )}
            style={{ animationDelay: `${i * 0.1}s` }}
          />
        ))}
      </div>
    );
  }

  if (variant === 'bars') {
    return (
      <div className={cn('flex space-x-1', className)}>
        {[0, 1, 2, 3].map((i) => (
          <div
            key={i}
            className={cn(
              'bg-primary animate-pulse',
              size === 'sm' ? 'w-1 h-4' : size === 'md' ? 'w-1 h-6' : size === 'lg' ? 'w-2 h-8' : 'w-2 h-10'
            )}
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>
    );
  }

  return (
    <Loader2 className={cn('animate-spin text-primary', sizeClasses[size], className)} />
  );
}

interface LoadingStateProps {
  title: string;
  description?: string;
  progress?: number;
  icon?: React.ReactNode;
  variant?: 'card' | 'fullscreen' | 'inline';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  children?: React.ReactNode;
}

export function LoadingState({
  title,
  description,
  progress,
  icon,
  variant = 'card',
  size = 'md',
  className
}: LoadingStateProps) {
  const content = (
    <div className="text-center space-y-4">
      <div className="relative mx-auto">
        {icon ? (
          <div className="relative">
            <div className={cn(
              'rounded-full border-4 border-primary/20 animate-spin',
              size === 'sm' ? 'w-12 h-12' : size === 'md' ? 'w-16 h-16' : 'w-20 h-20'
            )}>
              <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-primary animate-spin"></div>
            </div>
            <div className={cn(
              'absolute inset-0 flex items-center justify-center text-primary',
              size === 'sm' ? 'text-lg' : size === 'md' ? 'text-xl' : 'text-2xl'
            )}>
              {icon}
            </div>
          </div>
        ) : (
          <LoadingSpinner size={size === 'sm' ? 'lg' : size === 'md' ? 'xl' : 'xl'} />
        )}
      </div>
      
      <div className="space-y-2">
        <h3 className={cn(
          'font-semibold text-foreground',
          size === 'sm' ? 'text-sm' : size === 'md' ? 'text-lg' : 'text-xl'
        )}>
          {title}
        </h3>
        {description && (
          <p className={cn(
            'text-muted-foreground',
            size === 'sm' ? 'text-xs' : 'text-sm'
          )}>
            {description}
          </p>
        )}
      </div>

      {progress !== undefined && (
        <div className="w-full max-w-xs mx-auto">
          <div className="flex justify-between text-xs text-muted-foreground mb-1">
            <span>Progress</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-secondary rounded-full h-2">
            <div
              className="bg-primary h-2 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );

  if (variant === 'fullscreen') {
    return (
      <div className={cn('flex items-center justify-center min-h-screen', className)}>
        {content}
      </div>
    );
  }

  if (variant === 'inline') {
    return (
      <div className={cn('flex items-center justify-center py-8', className)}>
        {content}
      </div>
    );
  }

  return (
    <div className={cn('flex items-center justify-center py-12', className)}>
      {content}
    </div>
  );
}

// Specialized loading components for different contexts
export function SequenceLoadingState({ sequenceId }: { sequenceId?: string }) {
  return (
    <LoadingState
      title="Loading Sequence Data"
      description={
        sequenceId 
          ? `Fetching frames and pose data for sequence ${sequenceId}...`
          : "Preparing sequence visualization..."
      }
      icon={<FileVideo className="w-6 h-6" />}
      variant="card"
    />
  );
}

export function PoseAnalysisLoadingState({ sequenceId }: { sequenceId?: string }) {
  return (
    <LoadingState
      title="Analyzing Gait Sequence"
      description={
        sequenceId
          ? `Extracting features, detecting cycles, and analyzing symmetry for ${sequenceId}...`
          : "Processing pose data and generating analysis..."
      }
      icon={<Brain className="w-6 h-6" />}
      variant="card"
    />
  );
}

export function DatasetLoadingState() {
  return (
    <LoadingState
      title="Loading Dataset"
      description="Fetching dataset metadata and sequence information..."
      icon={<Database className="w-6 h-6" />}
      variant="fullscreen"
    />
  );
}

export function FrameProcessingState({ current, total }: { current?: number; total?: number }) {
  const progress = current && total ? (current / total) * 100 : undefined;
  
  return (
    <LoadingState
      title="Processing Frames"
      description={
        current && total 
          ? `Processing frame ${current} of ${total}...`
          : "Loading and processing video frames..."
      }
      progress={progress}
      icon={<Activity className="w-6 h-6" />}
      variant="card"
    />
  );
}

export function GaitAnalysisLoadingState() {
  return (
    <LoadingState
      title="Performing Gait Analysis"
      description="Analyzing movement patterns, detecting gait cycles, and calculating symmetry metrics..."
      icon={<Zap className="w-6 h-6" />}
      variant="card"
    />
  );
}