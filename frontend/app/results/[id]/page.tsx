/**
 * Individual Result Detail Page
 * Display detailed gait analysis results for a specific analysis
 */

'use client';

import { use } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { VideoPlayer } from '@/components/video/VideoPlayer';

// Mock data - In production, this would come from API
const mockAnalysisData = {
  '1': {
    id: 1,
    name: 'Walking Test 1',
    date: '2024-01-03',
    time: '14:30:00',
    status: 'Normal',
    confidence: 95,
    videoUrl: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
    duration: '00:00:45',
    frameCount: 1350,
    conditions: [],
    metrics: {
      cadence: { value: 112, unit: 'steps/min', normal: true, range: '100-120' },
      strideLength: { value: 1.42, unit: 'm', normal: true, range: '1.2-1.6' },
      walkingSpeed: { value: 1.35, unit: 'm/s', normal: true, range: '1.2-1.5' },
      stepWidth: { value: 0.12, unit: 'm', normal: true, range: '0.08-0.15' },
      symmetry: { value: 98, unit: '%', normal: true, range: '>95' },
      stability: { value: 94, unit: '%', normal: true, range: '>90' },
    },
    temporalAnalysis: {
      gaitCycles: 42,
      stancePhase: { left: 62, right: 63, unit: '%' },
      swingPhase: { left: 38, right: 37, unit: '%' },
      doubleSupportTime: 12,
    },
    spatialAnalysis: {
      leftStepLength: 0.71,
      rightStepLength: 0.71,
      stepLengthVariability: 2.3,
      baseOfSupport: 0.12,
    },
    aiAnalysis: {
      model: 'GPT-4.1',
      classification: 'Normal Gait Pattern',
      confidence: 95,
      reasoning: 'The gait pattern shows consistent cadence, symmetric step lengths, and normal temporal characteristics. All measured parameters fall within expected ranges for healthy adult walking.',
      recommendations: [
        'Continue regular physical activity',
        'Maintain current fitness level',
        'No immediate concerns identified',
      ],
    },
  },
  '2': {
    id: 2,
    name: 'Gait Analysis 2',
    date: '2024-01-03',
    time: '11:15:00',
    status: 'Abnormal',
    confidence: 88,
    videoUrl: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4',
    duration: '00:01:12',
    frameCount: 2160,
    conditions: ['Limping', 'Asymmetry', 'Reduced Cadence'],
    metrics: {
      cadence: { value: 85, unit: 'steps/min', normal: false, range: '100-120' },
      strideLength: { value: 1.15, unit: 'm', normal: false, range: '1.2-1.6' },
      walkingSpeed: { value: 0.98, unit: 'm/s', normal: false, range: '1.2-1.5' },
      stepWidth: { value: 0.18, unit: 'm', normal: false, range: '0.08-0.15' },
      symmetry: { value: 78, unit: '%', normal: false, range: '>95' },
      stability: { value: 82, unit: '%', normal: false, range: '>90' },
    },
    temporalAnalysis: {
      gaitCycles: 51,
      stancePhase: { left: 68, right: 58, unit: '%' },
      swingPhase: { left: 32, right: 42, unit: '%' },
      doubleSupportTime: 18,
    },
    spatialAnalysis: {
      leftStepLength: 0.62,
      rightStepLength: 0.53,
      stepLengthVariability: 8.7,
      baseOfSupport: 0.18,
    },
    aiAnalysis: {
      model: 'GPT-4.1',
      classification: 'Abnormal Gait Pattern',
      confidence: 88,
      reasoning: 'The analysis reveals significant asymmetry between left and right steps, reduced cadence, and increased step width. The stance phase shows notable differences between sides, suggesting possible compensation for discomfort or weakness on the right side.',
      recommendations: [
        'Consult with a healthcare professional for detailed evaluation',
        'Consider physical therapy assessment',
        'Monitor for pain or discomfort during walking',
        'Avoid high-impact activities until evaluated',
      ],
      possibleConditions: [
        { name: 'Antalgic Gait', probability: 72, description: 'Gait pattern suggesting pain avoidance' },
        { name: 'Hip Weakness', probability: 65, description: 'Reduced strength in hip abductors' },
        { name: 'Leg Length Discrepancy', probability: 45, description: 'Possible difference in leg lengths' },
      ],
    },
  },
  '3': {
    id: 3,
    name: 'Patient Video 3',
    date: '2024-01-02',
    time: '09:45:00',
    status: 'Normal',
    confidence: 92,
    videoUrl: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4',
    duration: '00:00:38',
    frameCount: 1140,
    conditions: [],
    metrics: {
      cadence: { value: 108, unit: 'steps/min', normal: true, range: '100-120' },
      strideLength: { value: 1.38, unit: 'm', normal: true, range: '1.2-1.6' },
      walkingSpeed: { value: 1.28, unit: 'm/s', normal: true, range: '1.2-1.5' },
      stepWidth: { value: 0.11, unit: 'm', normal: true, range: '0.08-0.15' },
      symmetry: { value: 96, unit: '%', normal: true, range: '>95' },
      stability: { value: 93, unit: '%', normal: true, range: '>90' },
    },
    temporalAnalysis: {
      gaitCycles: 34,
      stancePhase: { left: 61, right: 62, unit: '%' },
      swingPhase: { left: 39, right: 38, unit: '%' },
      doubleSupportTime: 11,
    },
    spatialAnalysis: {
      leftStepLength: 0.69,
      rightStepLength: 0.69,
      stepLengthVariability: 2.8,
      baseOfSupport: 0.11,
    },
    aiAnalysis: {
      model: 'GPT-4.1',
      classification: 'Normal Gait Pattern',
      confidence: 92,
      reasoning: 'Gait parameters are within normal ranges with good symmetry and stability. Minor variations are within acceptable limits for healthy walking.',
      recommendations: [
        'Maintain current activity level',
        'Continue regular exercise routine',
      ],
    },
  },
};

interface ResultDetailPageProps {
  params: Promise<{ id: string }>;
}

export default function ResultDetailPage({ params }: ResultDetailPageProps) {
  const { id } = use(params);
  const analysis = mockAnalysisData[id as keyof typeof mockAnalysisData];

  if (!analysis) {
    return (
      <div className="space-y-8">
        <Card>
          <CardContent className="pt-6">
            <div className="text-center space-y-4">
              <div className="text-6xl">❌</div>
              <h2 className="text-2xl font-bold">Analysis Not Found</h2>
              <p className="text-muted-foreground">
                The analysis with ID {id} could not be found.
              </p>
              <Button asChild>
                <Link href="/results">← Back to Results</Link>
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <Button variant="ghost" size="sm" asChild>
              <Link href="/results">← Back</Link>
            </Button>
          </div>
          <h1 className="text-3xl font-bold">{analysis.name}</h1>
          <p className="text-muted-foreground">
            Analysis #{analysis.id} • {analysis.date} at {analysis.time}
          </p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline">
            💾 Export Report
          </Button>
          <Button variant="outline">
            📊 Compare
          </Button>
          <Button>
            🔄 Re-analyze
          </Button>
        </div>
      </div>

      {/* Status Overview */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Analysis Summary</CardTitle>
              <CardDescription>Overall classification and confidence</CardDescription>
            </div>
            <Badge
              variant={analysis.status === 'Normal' ? 'default' : 'destructive'}
              className="text-lg px-4 py-2"
            >
              {analysis.status}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <div className="text-sm text-muted-foreground mb-1">Confidence Score</div>
              <div className="text-2xl font-bold">{analysis.confidence}%</div>
              <Progress value={analysis.confidence} className="mt-2" />
            </div>
            <div>
              <div className="text-sm text-muted-foreground mb-1">Video Duration</div>
              <div className="text-2xl font-bold">{analysis.duration}</div>
              <div className="text-sm text-muted-foreground mt-1">
                {analysis.frameCount} frames analyzed
              </div>
            </div>
            <div>
              <div className="text-sm text-muted-foreground mb-1">AI Model</div>
              <div className="text-2xl font-bold">{analysis.aiAnalysis.model}</div>
              <div className="text-sm text-muted-foreground mt-1">
                Latest generation model
              </div>
            </div>
          </div>

          {analysis.conditions.length > 0 && (
            <>
              <Separator />
              <div>
                <div className="text-sm font-medium mb-2">Detected Conditions</div>
                <div className="flex flex-wrap gap-2">
                  {analysis.conditions.map((condition, idx) => (
                    <Badge key={idx} variant="outline" className="text-sm">
                      ⚠️ {condition}
                    </Badge>
                  ))}
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Detailed Analysis Tabs */}
      <Tabs defaultValue="metrics" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="metrics">Gait Metrics</TabsTrigger>
          <TabsTrigger value="temporal">Temporal</TabsTrigger>
          <TabsTrigger value="spatial">Spatial</TabsTrigger>
          <TabsTrigger value="ai">AI Analysis</TabsTrigger>
          <TabsTrigger value="video">Video</TabsTrigger>
        </TabsList>

        {/* Gait Metrics Tab */}
        <TabsContent value="metrics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Gait Metrics</CardTitle>
              <CardDescription>Key measurements from gait analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {Object.entries(analysis.metrics).map(([key, metric]) => (
                  <div key={key} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="font-medium capitalize">
                        {key.replace(/([A-Z])/g, ' $1').trim()}
                      </div>
                      <Badge variant={metric.normal ? 'default' : 'destructive'}>
                        {metric.normal ? '✓ Normal' : '⚠ Abnormal'}
                      </Badge>
                    </div>
                    <div className="flex items-baseline space-x-2">
                      <div className="text-3xl font-bold">
                        {metric.value}
                      </div>
                      <div className="text-muted-foreground">{metric.unit}</div>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Normal range: {metric.range}
                    </div>
                    <Progress
                      value={metric.normal ? 100 : 60}
                      className={metric.normal ? '' : 'bg-red-100'}
                    />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Temporal Analysis Tab */}
        <TabsContent value="temporal" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Temporal Analysis</CardTitle>
              <CardDescription>Time-based gait characteristics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div>
                  <div className="text-sm text-muted-foreground mb-2">Gait Cycles Detected</div>
                  <div className="text-4xl font-bold">{analysis.temporalAnalysis.gaitCycles}</div>
                </div>

                <Separator />

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <div className="font-medium mb-4">Stance Phase</div>
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Left</span>
                          <span className="font-medium">
                            {analysis.temporalAnalysis.stancePhase.left}%
                          </span>
                        </div>
                        <Progress value={analysis.temporalAnalysis.stancePhase.left} />
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Right</span>
                          <span className="font-medium">
                            {analysis.temporalAnalysis.stancePhase.right}%
                          </span>
                        </div>
                        <Progress value={analysis.temporalAnalysis.stancePhase.right} />
                      </div>
                    </div>
                  </div>

                  <div>
                    <div className="font-medium mb-4">Swing Phase</div>
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Left</span>
                          <span className="font-medium">
                            {analysis.temporalAnalysis.swingPhase.left}%
                          </span>
                        </div>
                        <Progress value={analysis.temporalAnalysis.swingPhase.left} />
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Right</span>
                          <span className="font-medium">
                            {analysis.temporalAnalysis.swingPhase.right}%
                          </span>
                        </div>
                        <Progress value={analysis.temporalAnalysis.swingPhase.right} />
                      </div>
                    </div>
                  </div>
                </div>

                <Separator />

                <div>
                  <div className="text-sm text-muted-foreground mb-2">Double Support Time</div>
                  <div className="text-2xl font-bold">
                    {analysis.temporalAnalysis.doubleSupportTime}%
                  </div>
                  <div className="text-sm text-muted-foreground mt-1">
                    Normal range: 10-12%
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Spatial Analysis Tab */}
        <TabsContent value="spatial" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Spatial Analysis</CardTitle>
              <CardDescription>Distance and position measurements</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <div className="font-medium">Left Step Length</div>
                  <div className="text-3xl font-bold">
                    {analysis.spatialAnalysis.leftStepLength} m
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="font-medium">Right Step Length</div>
                  <div className="text-3xl font-bold">
                    {analysis.spatialAnalysis.rightStepLength} m
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="font-medium">Step Length Variability</div>
                  <div className="text-3xl font-bold">
                    {analysis.spatialAnalysis.stepLengthVariability}%
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Lower is better (normal: &lt;5%)
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="font-medium">Base of Support</div>
                  <div className="text-3xl font-bold">
                    {analysis.spatialAnalysis.baseOfSupport} m
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Normal range: 0.08-0.15 m
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* AI Analysis Tab */}
        <TabsContent value="ai" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>AI-Powered Analysis</CardTitle>
              <CardDescription>
                Insights from {analysis.aiAnalysis.model}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <div className="text-sm font-medium mb-2">Classification</div>
                <div className="text-2xl font-bold">{analysis.aiAnalysis.classification}</div>
                <div className="text-sm text-muted-foreground mt-1">
                  Confidence: {analysis.aiAnalysis.confidence}%
                </div>
              </div>

              <Separator />

              <div>
                <div className="text-sm font-medium mb-2">Reasoning</div>
                <p className="text-muted-foreground leading-relaxed">
                  {analysis.aiAnalysis.reasoning}
                </p>
              </div>

              <Separator />

              <div>
                <div className="text-sm font-medium mb-3">Recommendations</div>
                <ul className="space-y-2">
                  {analysis.aiAnalysis.recommendations.map((rec, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="text-blue-500 mt-1">•</span>
                      <span className="text-muted-foreground">{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {'possibleConditions' in analysis.aiAnalysis && analysis.aiAnalysis.possibleConditions && (
                <>
                  <Separator />
                  <div>
                    <div className="text-sm font-medium mb-3">Possible Conditions</div>
                    <div className="space-y-3">
                      {analysis.aiAnalysis.possibleConditions.map((condition, idx) => (
                        <div key={idx} className="border rounded-lg p-4">
                          <div className="flex items-center justify-between mb-2">
                            <div className="font-medium">{condition.name}</div>
                            <Badge variant="outline">{condition.probability}%</Badge>
                          </div>
                          <p className="text-sm text-muted-foreground">
                            {condition.description}
                          </p>
                          <Progress value={condition.probability} className="mt-2" />
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Video Tab */}
        <TabsContent value="video" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Video Analysis</CardTitle>
              <CardDescription>Original video with pose overlay and frame-by-frame controls</CardDescription>
            </CardHeader>
            <CardContent>
              <VideoPlayer
                videoUrl={analysis.videoUrl}
                videoName={analysis.name}
                frameRate={30}
                onTimeUpdate={(time) => {
                  // Handle time updates if needed
                  console.log('Current time:', time);
                }}
                onFrameChange={(frame) => {
                  // Handle frame changes if needed
                  console.log('Current frame:', frame);
                }}
              />
              <div className="mt-4 flex justify-center space-x-2">
                <Button variant="outline" asChild>
                  <a href={analysis.videoUrl} download={`${analysis.name}.mp4`}>
                    📥 Download Video
                  </a>
                </Button>
                <Button variant="outline">
                  🎨 Toggle Pose Overlay
                </Button>
                <Button variant="outline">
                  📊 Show Metrics Overlay
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
