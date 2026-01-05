/**
 * YouTube URL Analysis Page
 * Input YouTube URL for gait analysis
 */

'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';

export default function YouTubePage() {
  const [url, setUrl] = useState('');
  const [isValidating, setIsValidating] = useState(false);
  const [isValid, setIsValid] = useState<boolean | null>(null);
  const [videoInfo, setVideoInfo] = useState<any>(null);

  const validateYouTubeUrl = (url: string) => {
    const patterns = [
      /^https?:\/\/(www\.)?youtube\.com\/watch\?v=[\w-]+/,
      /^https?:\/\/youtu\.be\/[\w-]+/,
      /^https?:\/\/(www\.)?youtube\.com\/shorts\/[\w-]+/,
      /^https?:\/\/(www\.)?youtube\.com\/embed\/[\w-]+/,
      /^https?:\/\/(www\.)?youtube\.com\/live\/[\w-]+/,
    ];
    return patterns.some(pattern => pattern.test(url));
  };

  const handleValidate = async () => {
    setIsValidating(true);
    
    // Simulate validation
    setTimeout(() => {
      const valid = validateYouTubeUrl(url);
      setIsValid(valid);
      
      if (valid) {
        setVideoInfo({
          title: 'Sample Gait Analysis Video',
          duration: '0:45',
          thumbnail: 'https://via.placeholder.com/320x180',
        });
      }
      
      setIsValidating(false);
    }, 1000);
  };

  const handleAnalyze = () => {
    // Navigate to analysis results
    console.log('Starting analysis for:', url);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold">YouTube URL Analysis</h1>
        <p className="text-muted-foreground">Analyze gait videos directly from YouTube</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Enter YouTube URL</CardTitle>
          <CardDescription>
            Paste a YouTube video URL containing gait footage
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex space-x-2">
            <Input
              type="url"
              placeholder="https://www.youtube.com/watch?v=..."
              value={url}
              onChange={(e) => {
                setUrl(e.target.value);
                setIsValid(null);
                setVideoInfo(null);
              }}
              className="flex-1"
            />
            <Button 
              onClick={handleValidate} 
              disabled={!url || isValidating}
            >
              {isValidating ? '⏳ Validating...' : '🔍 Validate'}
            </Button>
          </div>

          {isValid === false && (
            <Alert variant="destructive">
              <AlertDescription>
                Invalid YouTube URL. Please check the URL and try again.
              </AlertDescription>
            </Alert>
          )}

          {isValid === true && videoInfo && (
            <div className="space-y-4">
              <Alert className="bg-green-50 dark:bg-green-950 border-green-200">
                <AlertDescription>
                  ✅ Valid YouTube URL detected
                </AlertDescription>
              </Alert>

              <div className="border rounded-lg p-4 space-y-4">
                <div className="flex items-start space-x-4">
                  <div className="w-40 h-24 bg-gray-200 dark:bg-gray-800 rounded-lg flex items-center justify-center">
                    <span className="text-4xl">🎬</span>
                  </div>
                  <div className="flex-1">
                    <h3 className="font-medium">{videoInfo.title}</h3>
                    <div className="flex items-center space-x-4 mt-2 text-sm text-muted-foreground">
                      <Badge variant="secondary">Duration: {videoInfo.duration}</Badge>
                      <Badge variant="secondary">YouTube</Badge>
                    </div>
                  </div>
                </div>

                <Button onClick={handleAnalyze} className="w-full" size="lg">
                  🚀 Start Analysis
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Supported URL Formats */}
      <Card>
        <CardHeader>
          <CardTitle>Supported URL Formats</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li>✓ https://www.youtube.com/watch?v=VIDEO_ID</li>
            <li>✓ https://youtu.be/VIDEO_ID</li>
            <li>✓ https://www.youtube.com/shorts/VIDEO_ID</li>
            <li>✓ https://www.youtube.com/embed/VIDEO_ID</li>
            <li>✓ https://www.youtube.com/watch?v=VIDEO_ID&t=10s</li>
          </ul>
        </CardContent>
      </Card>

      {/* Usage Tips */}
      <Card>
        <CardHeader>
          <CardTitle>Usage Tips</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li>✓ Ensure the video is publicly accessible</li>
            <li>✓ Videos with clear gait footage work best</li>
            <li>✓ Processing may take a few minutes</li>
            <li>✓ You'll receive a notification when complete</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
