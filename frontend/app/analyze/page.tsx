/**
 * Analyze Landing Page
 * Hub for all gait analysis options with modern, consistent styling
 */

'use client';

import Link from 'next/link';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

interface AnalysisOption {
  id: string;
  title: string;
  description: string;
  href: string;
  icon: string;
  badge?: string;
  badgeVariant?: 'default' | 'secondary' | 'outline';
  features: string[];
}

const analysisOptions: AnalysisOption[] = [
  {
    id: 'gavd',
    title: 'GAVD Dataset Analysis',
    description: 'Analyze videos from the Gait Analysis Video Database - a clinical dataset with annotated gait sequences.',
    href: '/gavd',
    icon: '🏥',
    badge: 'Clinical',
    badgeVariant: 'default',
    features: [
      'Pre-annotated clinical sequences',
      'Multiple gait patterns',
      'Ground truth comparisons',
      'Research-grade analysis',
    ],
  },
  {
    id: 'upload',
    title: 'Upload Video',
    description: 'Upload your own video files for gait analysis. Supports MP4, AVI, MOV, and WebM formats.',
    href: '/analyze/upload',
    icon: '📤',
    features: [
      'Drag and drop upload',
      'Multiple video formats',
      'Real-time processing',
      'Detailed reports',
    ],
  },
  {
    id: 'youtube',
    title: 'YouTube Analysis',
    description: 'Analyze gait videos directly from YouTube URLs. Perfect for analyzing publicly available footage.',
    href: '/analyze/youtube',
    icon: '🎬',
    badge: 'Popular',
    badgeVariant: 'secondary',
    features: [
      'Direct URL input',
      'No download required',
      'Supports various formats',
      'Quick processing',
    ],
  },
];

export default function AnalyzePage() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Analyze Gait Videos</h1>
        <p className="text-muted-foreground text-lg">
          Choose an analysis method to get started with comprehensive gait assessment
        </p>
      </div>

      {/* Analysis Options Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {analysisOptions.map((option) => (
          <Card 
            key={option.id} 
            className="group hover:shadow-lg transition-all duration-300 hover:border-primary/50 flex flex-col"
          >
            <CardHeader className="pb-4">
              <div className="flex items-start justify-between">
                <div className="text-4xl mb-2">{option.icon}</div>
                {option.badge && (
                  <Badge variant={option.badgeVariant || 'default'}>
                    {option.badge}
                  </Badge>
                )}
              </div>
              <CardTitle className="text-xl group-hover:text-primary transition-colors">
                {option.title}
              </CardTitle>
              <CardDescription className="text-sm">
                {option.description}
              </CardDescription>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col">
              <ul className="space-y-2 mb-6 flex-1">
                {option.features.map((feature, index) => (
                  <li key={index} className="flex items-center text-sm text-muted-foreground">
                    <span className="text-green-500 mr-2">✓</span>
                    {feature}
                  </li>
                ))}
              </ul>
              <Button asChild className="w-full group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                <Link href={option.href}>
                  Get Started →
                </Link>
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Quick Tips Section */}
      <Card className="bg-muted/50">
        <CardHeader>
          <CardTitle className="text-lg">💡 Quick Tips for Best Results</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="space-y-1">
              <div className="font-medium text-sm">Video Quality</div>
              <p className="text-xs text-muted-foreground">
                Higher resolution videos (720p+) provide more accurate pose estimation
              </p>
            </div>
            <div className="space-y-1">
              <div className="font-medium text-sm">Camera Angle</div>
              <p className="text-xs text-muted-foreground">
                Side view (sagittal plane) captures the most gait information
              </p>
            </div>
            <div className="space-y-1">
              <div className="font-medium text-sm">Duration</div>
              <p className="text-xs text-muted-foreground">
                At least 5-10 seconds of continuous walking for reliable analysis
              </p>
            </div>
            <div className="space-y-1">
              <div className="font-medium text-sm">Lighting</div>
              <p className="text-xs text-muted-foreground">
                Good, even lighting improves keypoint detection accuracy
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Recent Activity Section */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">Recent Analyses</CardTitle>
            <Button variant="outline" size="sm" asChild>
              <Link href="/results">View All Results →</Link>
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <div className="text-4xl mb-2">📊</div>
            <p>No recent analyses yet</p>
            <p className="text-sm">Start by selecting an analysis method above</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
