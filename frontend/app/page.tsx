/**
 * AlexPose Homepage
 * Modern landing page with glass morphism design
 */

import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-200px)] space-y-12">
      {/* Hero Section */}
      <div className="text-center space-y-6 max-w-3xl">
        <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          AlexPose Gait Analysis
        </h1>
        <p className="text-xl text-muted-foreground">
          AI-powered gait analysis for health condition identification
        </p>
        <div className="flex flex-wrap gap-4 justify-center">
          <Button asChild size="lg" className="rounded-full">
            <Link href="/analyze/upload">
              📤 Upload Video
            </Link>
          </Button>
          <Button asChild size="lg" variant="outline" className="rounded-full">
            <Link href="/analyze/youtube">
              🔗 Analyze YouTube URL
            </Link>
          </Button>
        </div>
      </div>

      {/* Analysis Type Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-5xl">
        <Card className="hover:shadow-xl transition-all hover:scale-105 cursor-pointer border-2 hover:border-blue-500">
          <CardHeader>
            <div className="flex items-center space-x-3">
              <div className="text-5xl">📹</div>
              <div>
                <CardTitle className="text-xl">Single Video Analysis</CardTitle>
                <CardDescription>
                  Upload or link to individual gait videos
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground">
              Perfect for analyzing individual patient videos or YouTube URLs
            </p>
            <div className="flex flex-wrap gap-2">
              <Button asChild size="sm" className="rounded-full">
                <Link href="/analyze/upload">
                  📤 Upload Video
                </Link>
              </Button>
              <Button asChild size="sm" variant="outline" className="rounded-full">
                <Link href="/analyze/youtube">
                  🔗 YouTube URL
                </Link>
              </Button>
            </div>
            <div className="pt-2 border-t">
              <p className="text-xs text-muted-foreground">
                ✓ Pose estimation • ✓ Gait metrics • ✓ Condition detection
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="hover:shadow-xl transition-all hover:scale-105 cursor-pointer border-2 hover:border-purple-500 bg-gradient-to-br from-purple-50 to-blue-50">
          <CardHeader>
            <div className="flex items-center space-x-3">
              <div className="text-5xl">📊</div>
              <div>
                <CardTitle className="text-xl">GAVD Dataset Analysis</CardTitle>
                <CardDescription>
                  Process training datasets with annotations
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground">
              Upload GAVD CSV files for batch processing and model training
            </p>
            <Button asChild size="sm" className="rounded-full w-full bg-purple-600 hover:bg-purple-700">
              <Link href="/training/gavd">
                🚀 Upload GAVD Dataset
              </Link>
            </Button>
            <div className="pt-2 border-t">
              <p className="text-xs text-muted-foreground">
                ✓ Batch processing • ✓ Sequence analysis • ✓ Frame visualization
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Feature Highlights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-5xl">
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <div className="text-4xl mb-2">🎯</div>
            <CardTitle>Accurate Analysis</CardTitle>
            <CardDescription>
              Advanced AI models for precise gait pattern recognition
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Powered by state-of-the-art pose estimation and LLM classification
            </p>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <div className="text-4xl mb-2">⚡</div>
            <CardTitle>Fast Processing</CardTitle>
            <CardDescription>
              Quick analysis with real-time progress tracking
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Optimized pipeline for efficient video processing
            </p>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <div className="text-4xl mb-2">📊</div>
            <CardTitle>Detailed Insights</CardTitle>
            <CardDescription>
              Comprehensive reports with visualizations
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Interactive charts and condition identification
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-8 w-full max-w-4xl text-center">
        <div>
          <div className="text-3xl font-bold text-blue-600">4+</div>
          <div className="text-sm text-muted-foreground">Pose Estimators</div>
        </div>
        <div>
          <div className="text-3xl font-bold text-purple-600">10+</div>
          <div className="text-sm text-muted-foreground">Conditions Detected</div>
        </div>
        <div>
          <div className="text-3xl font-bold text-green-600">95%</div>
          <div className="text-sm text-muted-foreground">Accuracy</div>
        </div>
        <div>
          <div className="text-3xl font-bold text-orange-600">24/7</div>
          <div className="text-sm text-muted-foreground">Available</div>
        </div>
      </div>
    </div>
  );
}
