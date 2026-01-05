/**
 * Video Player Component
 * Custom video player with controls for gait analysis videos
 * Features: play/pause, seek, frame-by-frame navigation, speed control, fullscreen
 */

'use client';

import React, { useRef, useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Card } from '@/components/ui/card';

interface VideoPlayerProps {
  videoUrl: string;
  videoName: string;
  frameRate?: number;
  onTimeUpdate?: (currentTime: number) => void;
  onFrameChange?: (frameNumber: number) => void;
}

export function VideoPlayer({
  videoUrl,
  videoName,
  frameRate = 30,
  onTimeUpdate,
  onFrameChange,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Calculate frame from time
  const timeToFrame = (time: number) => Math.floor(time * frameRate);
  const frameToTime = (frame: number) => frame / frameRate;

  // Initialize video
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadedMetadata = () => {
      setDuration(video.duration);
      setTotalFrames(Math.floor(video.duration * frameRate));
      setIsLoading(false);
    };

    const handleTimeUpdate = () => {
      const time = video.currentTime;
      setCurrentTime(time);
      const frame = timeToFrame(time);
      setCurrentFrame(frame);
      onTimeUpdate?.(time);
      onFrameChange?.(frame);
    };

    const handleEnded = () => {
      setIsPlaying(false);
    };

    const handleError = () => {
      setError('Failed to load video. Please check the video URL.');
      setIsLoading(false);
    };

    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('ended', handleEnded);
    video.addEventListener('error', handleError);

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('ended', handleEnded);
      video.removeEventListener('error', handleError);
    };
  }, [frameRate, onTimeUpdate, onFrameChange]);

  // Play/Pause toggle
  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.pause();
    } else {
      video.play();
    }
    setIsPlaying(!isPlaying);
  };

  // Seek to specific time
  const seekTo = (time: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Math.max(0, Math.min(time, duration));
  };

  // Frame navigation
  const previousFrame = () => {
    const newFrame = Math.max(0, currentFrame - 1);
    seekTo(frameToTime(newFrame));
  };

  const nextFrame = () => {
    const newFrame = Math.min(totalFrames - 1, currentFrame + 1);
    seekTo(frameToTime(newFrame));
  };

  // Skip forward/backward
  const skip = (seconds: number) => {
    seekTo(currentTime + seconds);
  };

  // Volume control
  const handleVolumeChange = (value: number[]) => {
    const video = videoRef.current;
    if (!video) return;
    const newVolume = value[0];
    video.volume = newVolume;
    setVolume(newVolume);
    setIsMuted(newVolume === 0);
  };

  const toggleMute = () => {
    const video = videoRef.current;
    if (!video) return;
    video.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  // Playback speed
  const handlePlaybackRateChange = (value: string) => {
    const video = videoRef.current;
    if (!video) return;
    const rate = parseFloat(value);
    video.playbackRate = rate;
    setPlaybackRate(rate);
  };

  // Fullscreen
  const toggleFullscreen = () => {
    const video = videoRef.current;
    if (!video) return;

    if (!isFullscreen) {
      if (video.requestFullscreen) {
        video.requestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
    setIsFullscreen(!isFullscreen);
  };

  // Format time display
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Handle seek bar change
  const handleSeek = (value: number[]) => {
    seekTo(value[0]);
  };

  if (error) {
    return (
      <Card className="aspect-video flex items-center justify-center bg-muted">
        <div className="text-center space-y-2">
          <div className="text-4xl">⚠️</div>
          <div className="text-muted-foreground">{error}</div>
          <div className="text-sm text-muted-foreground">{videoUrl}</div>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Video Container */}
      <div className="relative aspect-video bg-black rounded-lg overflow-hidden group">
        <video
          ref={videoRef}
          className="w-full h-full"
          src={videoUrl}
          preload="metadata"
        >
          Your browser does not support the video tag.
        </video>

        {/* Loading Overlay */}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50">
            <div className="text-white text-center space-y-2">
              <div className="text-4xl animate-pulse">⏳</div>
              <div>Loading video...</div>
            </div>
          </div>
        )}

        {/* Play/Pause Overlay (appears on hover) */}
        <div
          className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
          onClick={togglePlay}
        >
          <div className="bg-black/50 rounded-full p-4">
            <div className="text-white text-4xl">
              {isPlaying ? '⏸' : '▶️'}
            </div>
          </div>
        </div>

        {/* Video Info Overlay */}
        <div className="absolute top-4 left-4 bg-black/70 text-white px-3 py-1 rounded text-sm">
          {videoName}
        </div>

        {/* Frame Counter */}
        <div className="absolute top-4 right-4 bg-black/70 text-white px-3 py-1 rounded text-sm font-mono">
          Frame: {currentFrame} / {totalFrames}
        </div>
      </div>

      {/* Controls */}
      <div className="space-y-3">
        {/* Progress Bar */}
        <div className="space-y-1">
          <Slider
            value={[currentTime]}
            max={duration || 100}
            step={0.01}
            onValueChange={handleSeek}
            className="cursor-pointer"
          />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>

        {/* Control Buttons */}
        <div className="flex items-center justify-between">
          {/* Left Controls */}
          <div className="flex items-center space-x-2">
            {/* Play/Pause */}
            <Button
              variant="outline"
              size="icon"
              onClick={togglePlay}
              disabled={isLoading}
            >
              {isPlaying ? '⏸' : '▶️'}
            </Button>

            {/* Skip Backward */}
            <Button
              variant="outline"
              size="icon"
              onClick={() => skip(-5)}
              disabled={isLoading}
              title="Skip backward 5s"
            >
              ⏪
            </Button>

            {/* Previous Frame */}
            <Button
              variant="outline"
              size="icon"
              onClick={previousFrame}
              disabled={isLoading}
              title="Previous frame"
            >
              ⏮
            </Button>

            {/* Next Frame */}
            <Button
              variant="outline"
              size="icon"
              onClick={nextFrame}
              disabled={isLoading}
              title="Next frame"
            >
              ⏭
            </Button>

            {/* Skip Forward */}
            <Button
              variant="outline"
              size="icon"
              onClick={() => skip(5)}
              disabled={isLoading}
              title="Skip forward 5s"
            >
              ⏩
            </Button>

            {/* Volume */}
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="icon"
                onClick={toggleMute}
                disabled={isLoading}
              >
                {isMuted || volume === 0 ? '🔇' : volume < 0.5 ? '🔉' : '🔊'}
              </Button>
              <div className="w-24 hidden md:block">
                <Slider
                  value={[isMuted ? 0 : volume]}
                  max={1}
                  step={0.01}
                  onValueChange={handleVolumeChange}
                  disabled={isLoading}
                />
              </div>
            </div>
          </div>

          {/* Right Controls */}
          <div className="flex items-center space-x-2">
            {/* Playback Speed */}
            <Select
              value={playbackRate.toString()}
              onValueChange={handlePlaybackRateChange}
              disabled={isLoading}
            >
              <SelectTrigger className="w-24">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0.25">0.25x</SelectItem>
                <SelectItem value="0.5">0.5x</SelectItem>
                <SelectItem value="0.75">0.75x</SelectItem>
                <SelectItem value="1">1x</SelectItem>
                <SelectItem value="1.25">1.25x</SelectItem>
                <SelectItem value="1.5">1.5x</SelectItem>
                <SelectItem value="2">2x</SelectItem>
              </SelectContent>
            </Select>

            {/* Fullscreen */}
            <Button
              variant="outline"
              size="icon"
              onClick={toggleFullscreen}
              disabled={isLoading}
              title="Fullscreen"
            >
              {isFullscreen ? '⛶' : '⛶'}
            </Button>
          </div>
        </div>

        {/* Additional Info */}
        <div className="flex justify-between text-xs text-muted-foreground">
          <div>
            Frame Rate: {frameRate} fps • Duration: {formatTime(duration)}
          </div>
          <div>
            Speed: {playbackRate}x • Frame: {currentFrame}/{totalFrames}
          </div>
        </div>
      </div>
    </div>
  );
}
