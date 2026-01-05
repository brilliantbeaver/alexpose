/**
 * Help Center Landing Page
 * Comprehensive documentation and support hub
 */

'use client';

import Link from 'next/link';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

interface HelpSection {
  id: string;
  title: string;
  description: string;
  href: string;
  icon: string;
  badge?: string;
  topics: string[];
}

const helpSections: HelpSection[] = [
  {
    id: 'pose-analysis',
    title: 'Pose Analysis Guide',
    description: 'Comprehensive guide to understanding gait analysis results, metrics, and clinical interpretations.',
    href: '/help/pose-analysis',
    icon: '🎯',
    badge: 'Popular',
    topics: [
      'Understanding gait metrics',
      'Clinical interpretations',
      'Pose estimation models',
      'Analysis workflows',
    ],
  },
  {
    id: 'getting-started',
    title: 'Getting Started',
    description: 'Learn the basics of using AlexPose for gait analysis and health condition identification.',
    href: '#getting-started',
    icon: '🚀',
    topics: [
      'Quick start guide',
      'Upload your first video',
      'Navigate the interface',
      'Interpret results',
    ],
  },
  {
    id: 'video-requirements',
    title: 'Video Requirements',
    description: 'Best practices for recording and uploading videos to ensure accurate gait analysis.',
    href: '#video-requirements',
    icon: '📹',
    topics: [
      'Camera positioning',
      'Lighting conditions',
      'Video quality',
      'Duration and format',
    ],
  },
  {
    id: 'models',
    title: 'AI Models',
    description: 'Learn about the pose estimation and LLM models powering AlexPose analysis.',
    href: '#models',
    icon: '🤖',
    topics: [
      'Pose estimation models',
      'LLM integration',
      'Model accuracy',
      'Training datasets',
    ],
  },
];

const quickLinks = [
  { label: 'GAVD Dataset', href: '/gavd', icon: '🏥' },
  { label: 'Upload Video', href: '/analyze/upload', icon: '📤' },
  { label: 'YouTube Analysis', href: '/analyze/youtube', icon: '🎬' },
  { label: 'Browse Models', href: '/models/browse', icon: '🧠' },
];

export default function HelpPage() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Help Center</h1>
        <p className="text-muted-foreground text-lg">
          Everything you need to know about using AlexPose for gait analysis
        </p>
      </div>

      {/* Search Bar Placeholder */}
      <Card className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 border-blue-200 dark:border-blue-800">
        <CardContent className="pt-6">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🔍</span>
            <input
              type="text"
              placeholder="Search documentation..."
              className="flex-1 px-4 py-2 rounded-lg border border-input bg-background focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
        </CardContent>
      </Card>

      {/* Help Sections Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {helpSections.map((section) => (
          <Card 
            key={section.id} 
            className="group hover:shadow-lg transition-all duration-300 hover:border-primary/50 flex flex-col"
          >
            <CardHeader className="pb-4">
              <div className="flex items-start justify-between">
                <div className="text-4xl mb-2">{section.icon}</div>
                {section.badge && (
                  <Badge variant="secondary">
                    {section.badge}
                  </Badge>
                )}
              </div>
              <CardTitle className="text-xl group-hover:text-primary transition-colors">
                {section.title}
              </CardTitle>
              <CardDescription className="text-sm">
                {section.description}
              </CardDescription>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col">
              <ul className="space-y-2 mb-6 flex-1">
                {section.topics.map((topic, index) => (
                  <li key={index} className="flex items-center text-sm text-muted-foreground">
                    <span className="text-primary mr-2">•</span>
                    {topic}
                  </li>
                ))}
              </ul>
              <Button asChild className="w-full group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                <Link href={section.href}>
                  Learn More →
                </Link>
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Quick Links */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">⚡ Quick Links</CardTitle>
          <CardDescription>Jump directly to common tasks</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {quickLinks.map((link) => (
              <Button
                key={link.href}
                asChild
                variant="outline"
                className="h-auto py-4 flex flex-col items-center gap-2 hover:bg-accent hover:scale-105 transition-all"
              >
                <Link href={link.href}>
                  <span className="text-2xl">{link.icon}</span>
                  <span className="text-sm">{link.label}</span>
                </Link>
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Getting Started Section */}
      <Card id="getting-started">
        <CardHeader>
          <CardTitle className="text-xl">🚀 Getting Started</CardTitle>
          <CardDescription>Your first steps with AlexPose</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3">
            <div className="flex gap-3">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold">
                1
              </div>
              <div>
                <h3 className="font-medium">Choose Your Analysis Method</h3>
                <p className="text-sm text-muted-foreground">
                  Select from GAVD clinical dataset, upload your own video, or analyze YouTube content
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold">
                2
              </div>
              <div>
                <h3 className="font-medium">Process Your Video</h3>
                <p className="text-sm text-muted-foreground">
                  Our AI models will extract pose keypoints and analyze gait patterns automatically
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold">
                3
              </div>
              <div>
                <h3 className="font-medium">Review Results</h3>
                <p className="text-sm text-muted-foreground">
                  Explore detailed metrics, visualizations, and clinical interpretations
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Video Requirements Section */}
      <Card id="video-requirements">
        <CardHeader>
          <CardTitle className="text-xl">📹 Video Requirements</CardTitle>
          <CardDescription>Best practices for optimal analysis results</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <h3 className="font-medium flex items-center gap-2">
                <span className="text-green-500">✓</span>
                Recommended
              </h3>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• Side view (sagittal plane) camera angle</li>
                <li>• 720p or higher resolution</li>
                <li>• Good, even lighting conditions</li>
                <li>• 5-10 seconds of continuous walking</li>
                <li>• Subject clearly visible, unobstructed</li>
                <li>• Stable camera position</li>
              </ul>
            </div>
            <div className="space-y-2">
              <h3 className="font-medium flex items-center gap-2">
                <span className="text-red-500">✗</span>
                Avoid
              </h3>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• Low resolution or blurry footage</li>
                <li>• Poor lighting or shadows</li>
                <li>• Obstructed view of subject</li>
                <li>• Shaky or moving camera</li>
                <li>• Very short clips (less than 3 seconds)</li>
                <li>• Multiple people in frame</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Models Section */}
      <Card id="models">
        <CardHeader>
          <CardTitle className="text-xl">🤖 AI Models</CardTitle>
          <CardDescription>Understanding the technology behind AlexPose</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3">
            <div>
              <h3 className="font-medium mb-1">Pose Estimation Models</h3>
              <p className="text-sm text-muted-foreground">
                AlexPose uses state-of-the-art pose estimation models to detect human keypoints in video frames. 
                These models identify joint positions with high accuracy, enabling precise gait analysis.
              </p>
            </div>
            <div>
              <h3 className="font-medium mb-1">LLM Integration</h3>
              <p className="text-sm text-muted-foreground">
                Large Language Models analyze gait patterns and generate clinical interpretations, 
                providing insights into potential health conditions based on movement characteristics.
              </p>
            </div>
            <div>
              <h3 className="font-medium mb-1">Training Data</h3>
              <p className="text-sm text-muted-foreground">
                Our models are trained on the GAVD (Gait Analysis Video Database) clinical dataset, 
                ensuring accuracy across various gait patterns and health conditions.
              </p>
            </div>
          </div>
          <Button asChild variant="outline" className="w-full">
            <Link href="/models/browse">
              Explore Available Models →
            </Link>
          </Button>
        </CardContent>
      </Card>

      {/* Support Section */}
      <Card className="bg-muted/50">
        <CardHeader>
          <CardTitle className="text-lg">💬 Need More Help?</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1 space-y-1">
              <h3 className="font-medium text-sm">Documentation</h3>
              <p className="text-xs text-muted-foreground">
                Browse our comprehensive guides and tutorials
              </p>
            </div>
            <div className="flex-1 space-y-1">
              <h3 className="font-medium text-sm">Community</h3>
              <p className="text-xs text-muted-foreground">
                Join discussions and share insights with other users
              </p>
            </div>
            <div className="flex-1 space-y-1">
              <h3 className="font-medium text-sm">Contact Support</h3>
              <p className="text-xs text-muted-foreground">
                Get in touch with our team for technical assistance
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
