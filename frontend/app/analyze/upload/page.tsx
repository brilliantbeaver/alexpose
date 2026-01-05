/**
 * Video Upload Page
 * Drag-and-drop interface for video file uploads
 */

'use client';

import React, { useState, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';

export default function UploadPage() {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadComplete, setUploadComplete] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    const videoFile = files.find(file => 
      file.type.startsWith('video/') || 
      ['mp4', 'avi', 'mov', 'webm'].some(ext => file.name.toLowerCase().endsWith(ext))
    );
    
    if (videoFile) {
      setSelectedFile(videoFile);
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      setSelectedFile(files[0]);
    }
  }, []);

  const handleUpload = useCallback(async () => {
    if (!selectedFile) return;
    
    setIsUploading(true);
    setUploadProgress(0);
    
    // Simulate upload progress
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsUploading(false);
          setUploadComplete(true);
          return 100;
        }
        return prev + 10;
      });
    }, 300);
  }, [selectedFile]);

  const handleReset = useCallback(() => {
    setSelectedFile(null);
    setUploadProgress(0);
    setIsUploading(false);
    setUploadComplete(false);
  }, []);

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold">Upload Video</h1>
        <p className="text-muted-foreground">Upload a gait video for analysis</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Video File Upload</CardTitle>
          <CardDescription>
            Supported formats: MP4, AVI, MOV, WebM (Max size: 500MB)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Drag and Drop Zone */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-lg p-12 text-center transition-all ${
              isDragging
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-950'
                : 'border-border hover:border-blue-400'
            }`}
          >
            <div className="space-y-4">
              <div className="text-6xl">📹</div>
              <div>
                <p className="text-lg font-medium">
                  {selectedFile ? selectedFile.name : 'Drag and drop your video here'}
                </p>
                <p className="text-sm text-muted-foreground mt-2">
                  or click to browse files
                </p>
              </div>
              <input
                type="file"
                accept="video/*,.mp4,.avi,.mov,.webm"
                onChange={handleFileSelect}
                className="hidden"
                id="file-upload"
              />
              <Button asChild variant="outline">
                <label htmlFor="file-upload" className="cursor-pointer">
                  Browse Files
                </label>
              </Button>
            </div>
          </div>

          {/* File Info */}
          {selectedFile && !uploadComplete && (
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center space-x-4">
                  <div className="text-3xl">🎬</div>
                  <div>
                    <div className="font-medium">{selectedFile.name}</div>
                    <div className="text-sm text-muted-foreground">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Badge variant="secondary">
                    {selectedFile.type || 'video'}
                  </Badge>
                  <Button variant="ghost" size="sm" onClick={handleReset}>
                    ✕
                  </Button>
                </div>
              </div>

              {isUploading && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span>Uploading...</span>
                    <span>{uploadProgress}%</span>
                  </div>
                  <Progress value={uploadProgress} />
                </div>
              )}

              {!isUploading && (
                <Button onClick={handleUpload} className="w-full" size="lg">
                  📤 Upload and Analyze
                </Button>
              )}
            </div>
          )}

          {/* Upload Complete */}
          {uploadComplete && (
            <Alert className="bg-green-50 dark:bg-green-950 border-green-200">
              <AlertDescription className="flex items-center justify-between">
                <span className="flex items-center space-x-2">
                  <span>✅</span>
                  <span>Upload complete! Starting analysis...</span>
                </span>
                <Button variant="outline" size="sm" onClick={handleReset}>
                  Upload Another
                </Button>
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Upload Tips */}
      <Card>
        <CardHeader>
          <CardTitle>Upload Tips</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li>✓ Ensure the video clearly shows the person walking</li>
            <li>✓ Side view provides the best results</li>
            <li>✓ Good lighting improves accuracy</li>
            <li>✓ Minimum 5 seconds of walking recommended</li>
            <li>✓ Higher resolution videos yield better results</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
