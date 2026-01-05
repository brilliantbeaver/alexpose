/**
 * Pose Analysis Help Page
 * Comprehensive guide to understanding gait analysis metrics
 */

'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import GaitCycleDiagram from '@/components/pose-analysis/GaitCycleDiagram';
import SymmetryDiagram from '@/components/pose-analysis/SymmetryDiagram';
import { 
  Activity, 
  TrendingUp, 
  Target, 
  Zap, 
  Info,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  BookOpen,
  BarChart3
} from 'lucide-react';

export default function PoseAnalysisHelpPage() {
  return (
    <div className="container mx-auto py-8 space-y-8 max-w-6xl">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-4xl font-bold flex items-center gap-3">
          <BookOpen className="h-8 w-8" />
          Pose Analysis Guide
        </h1>
        <p className="text-lg text-muted-foreground">
          Understanding gait analysis metrics, scores, and clinical interpretations
        </p>
      </div>

      {/* Quick Navigation */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            Quick Navigation
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <a href="#overall-assessment" className="p-4 border rounded-lg hover:bg-accent transition-colors">
              <h3 className="font-semibold mb-1">Overall Assessment</h3>
              <p className="text-sm text-muted-foreground">Understanding gait quality levels</p>
            </a>
            <a href="#cadence" className="p-4 border rounded-lg hover:bg-accent transition-colors">
              <h3 className="font-semibold mb-1">Cadence</h3>
              <p className="text-sm text-muted-foreground">Steps per minute explained</p>
            </a>
            <a href="#symmetry" className="p-4 border rounded-lg hover:bg-accent transition-colors">
              <h3 className="font-semibold mb-1">Symmetry</h3>
              <p className="text-sm text-muted-foreground">Left-right balance analysis</p>
            </a>
            <a href="#stability" className="p-4 border rounded-lg hover:bg-accent transition-colors">
              <h3 className="font-semibold mb-1">Stability</h3>
              <p className="text-sm text-muted-foreground">Center of mass control</p>
            </a>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="interpretation">Interpretation</TabsTrigger>
          <TabsTrigger value="clinical">Clinical Use</TabsTrigger>
          <TabsTrigger value="faq">FAQ</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <Card id="overall-assessment">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                What is Pose Analysis?
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-muted-foreground">
                Pose Analysis uses computer vision and biomechanical algorithms to evaluate human gait patterns.
                It extracts over 50 features from video sequences to provide comprehensive assessment of walking quality,
                symmetry, and stability.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                <div className="border rounded-lg p-4 space-y-2">
                  <div className="flex items-center gap-2">
                    <Activity className="h-5 w-5 text-primary" />
                    <h3 className="font-semibold">Feature Extraction</h3>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Analyzes joint angles, velocities, and movement patterns across the entire gait sequence
                  </p>
                </div>
                <div className="border rounded-lg p-4 space-y-2">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5 text-primary" />
                    <h3 className="font-semibold">Cycle Detection</h3>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Identifies heel strikes and toe-offs to segment individual gait cycles
                  </p>
                </div>
                <div className="border rounded-lg p-4 space-y-2">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5 text-primary" />
                    <h3 className="font-semibold">Symmetry Analysis</h3>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Compares left and right side movements to detect imbalances
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Metrics Tab */}
        <TabsContent value="metrics" className="space-y-6">

          {/* Cadence Section */}
          <Card id="cadence">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Cadence (Steps per Minute)
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <p className="text-muted-foreground">
                  Cadence measures the number of steps taken per minute. It's a fundamental indicator of walking speed and efficiency.
                </p>

                <div className="border-l-4 border-primary pl-4 py-2 bg-muted/50 rounded-r">
                  <p className="font-semibold mb-1">Normal Range</p>
                  <p className="text-sm text-muted-foreground">100-120 steps/minute for typical adult walking</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                  <div className="border rounded-lg p-4">
                    <Badge className="bg-red-100 text-red-800 mb-2">Slow (&lt; 100)</Badge>
                    <p className="text-sm text-muted-foreground">
                      May indicate caution, pain, reduced mobility, or balance concerns
                    </p>
                  </div>
                  <div className="border rounded-lg p-4">
                    <Badge className="bg-green-100 text-green-800 mb-2">Normal (100-120)</Badge>
                    <p className="text-sm text-muted-foreground">
                      Typical adult walking pace, indicates good mobility
                    </p>
                  </div>
                  <div className="border rounded-lg p-4">
                    <Badge className="bg-blue-100 text-blue-800 mb-2">Fast (&gt; 120)</Badge>
                    <p className="text-sm text-muted-foreground">
                      Rapid walking or jogging gait, indicates high mobility
                    </p>
                  </div>
                </div>

                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertTitle>Clinical Significance</AlertTitle>
                  <AlertDescription>
                    Slow cadence (&lt; 100 steps/min) may indicate increased fall risk or mobility limitations.
                    Cadence is affected by age, height, fitness level, and walking surface.
                  </AlertDescription>
                </Alert>
              </div>
            </CardContent>
          </Card>

          {/* Symmetry Section */}
          <Card id="symmetry">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                Gait Symmetry
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-muted-foreground">
                Symmetry measures the balance between left and right side movements during walking.
                It compares joint angles, stride lengths, and timing between legs.
              </p>

              <div className="border-l-4 border-primary pl-4 py-2 bg-muted/50 rounded-r">
                <p className="font-semibold mb-1">Symmetry Index</p>
                <p className="text-sm text-muted-foreground">
                  Score from 0 (perfect symmetry) to 1 (complete asymmetry). Lower is better.
                </p>
              </div>

              <div className="space-y-3 mt-4">
                <div className="flex items-start gap-3 p-3 border rounded-lg bg-green-50 dark:bg-green-950">
                  <CheckCircle2 className="h-5 w-5 text-green-600 mt-0.5" />
                  <div>
                    <p className="font-semibold text-green-900 dark:text-green-100">Symmetric (&lt; 0.10)</p>
                    <p className="text-sm text-green-700 dark:text-green-300">
                      Excellent left-right balance. Normal gait pattern.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-3 p-3 border rounded-lg bg-yellow-50 dark:bg-yellow-950">
                  <AlertCircle className="h-5 w-5 text-yellow-600 mt-0.5" />
                  <div>
                    <p className="font-semibold text-yellow-900 dark:text-yellow-100">Mildly Asymmetric (0.10-0.20)</p>
                    <p className="text-sm text-yellow-700 dark:text-yellow-300">
                      Minor imbalances. May be normal variation or require monitoring.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-3 p-3 border rounded-lg bg-orange-50 dark:bg-orange-950">
                  <AlertCircle className="h-5 w-5 text-orange-600 mt-0.5" />
                  <div>
                    <p className="font-semibold text-orange-900 dark:text-orange-100">Moderately Asymmetric (0.20-0.30)</p>
                    <p className="text-sm text-orange-700 dark:text-orange-300">
                      Noticeable imbalance. Consider clinical evaluation.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-3 p-3 border rounded-lg bg-red-50 dark:bg-red-950">
                  <AlertCircle className="h-5 w-5 text-red-600 mt-0.5" />
                  <div>
                    <p className="font-semibold text-red-900 dark:text-red-100">Severely Asymmetric (&gt; 0.30)</p>
                    <p className="text-sm text-red-700 dark:text-red-300">
                      Significant imbalance requiring clinical attention.
                    </p>
                  </div>
                </div>
              </div>

              <Alert>
                <Info className="h-4 w-4" />
                <AlertTitle>Common Causes of Asymmetry</AlertTitle>
                <AlertDescription>
                  <ul className="list-disc list-inside space-y-1 mt-2">
                    <li>Injury or pain on one side</li>
                    <li>Muscle weakness or imbalance</li>
                    <li>Limb length discrepancy</li>
                    <li>Neurological conditions</li>
                    <li>Compensation patterns from previous injury</li>
                  </ul>
                </AlertDescription>
              </Alert>

              {/* Visual Diagram */}
              <div className="mt-6">
                <SymmetryDiagram />
              </div>
            </CardContent>
          </Card>

          {/* Stability Section */}
          <Card id="stability">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5" />
                Postural Stability
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-muted-foreground">
                Stability measures control of the center of mass during walking. It analyzes trunk movement,
                hip stability, and overall balance control throughout the gait cycle.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="border rounded-lg p-4 bg-green-50 dark:bg-green-950">
                  <Badge className="bg-green-600 text-white mb-2">High Stability</Badge>
                  <p className="text-sm text-muted-foreground">
                    Excellent balance control with minimal trunk sway. Low fall risk.
                  </p>
                </div>
                <div className="border rounded-lg p-4 bg-yellow-50 dark:bg-yellow-950">
                  <Badge className="bg-yellow-600 text-white mb-2">Medium Stability</Badge>
                  <p className="text-sm text-muted-foreground">
                    Adequate stability with some compensatory movements. Moderate fall risk.
                  </p>
                </div>
                <div className="border rounded-lg p-4 bg-red-50 dark:bg-red-950">
                  <Badge className="bg-red-600 text-white mb-2">Low Stability</Badge>
                  <p className="text-sm text-muted-foreground">
                    Poor balance control. Increased fall risk requiring intervention.
                  </p>
                </div>
              </div>

              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Clinical Warning</AlertTitle>
                <AlertDescription>
                  Low stability significantly increases fall risk and may indicate vestibular, neurological,
                  or musculoskeletal issues requiring clinical evaluation.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>

          {/* Gait Cycles Section */}
          <Card id="gait-cycles">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Gait Cycles
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-muted-foreground">
                A gait cycle is one complete walking cycle, starting at heel strike and ending at the next
                heel strike of the same foot.
              </p>

              <div className="border rounded-lg p-4 bg-muted/50">
                <h4 className="font-semibold mb-3">Gait Cycle Phases</h4>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Stance Phase (foot on ground)</span>
                    <Badge>~60%</Badge>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-primary h-2 rounded-full" style={{width: '60%'}}></div>
                  </div>
                  <div className="flex items-center justify-between mt-3">
                    <span className="text-sm">Swing Phase (foot in air)</span>
                    <Badge>~40%</Badge>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-secondary h-2 rounded-full" style={{width: '40%'}}></div>
                  </div>
                </div>
              </div>

              <div className="border-l-4 border-primary pl-4 py-2 bg-muted/50 rounded-r">
                <p className="font-semibold mb-1">Normal Cycle Duration</p>
                <p className="text-sm text-muted-foreground">0.9-1.2 seconds for typical adult walking</p>
              </div>

              <Alert>
                <Info className="h-4 w-4" />
                <AlertTitle>Analysis Reliability</AlertTitle>
                <AlertDescription>
                  More gait cycles provide more reliable analysis. Minimum 2-3 complete cycles recommended
                  for accurate assessment.
                </AlertDescription>
              </Alert>

              {/* Visual Diagram */}
              <div className="mt-6">
                <GaitCycleDiagram />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Interpretation Tab */}
        <TabsContent value="interpretation" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>How to Interpret Results</CardTitle>
              <CardDescription>Understanding what the metrics mean for clinical practice</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">

              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Overall Assessment Levels</h3>
                
                <div className="space-y-3">
                  <div className="border-l-4 border-green-500 pl-4 py-3 bg-green-50 dark:bg-green-950 rounded-r">
                    <div className="flex items-center gap-2 mb-2">
                      <CheckCircle2 className="h-5 w-5 text-green-600" />
                      <h4 className="font-semibold text-green-900 dark:text-green-100">Good Gait Quality</h4>
                    </div>
                    <p className="text-sm text-green-700 dark:text-green-300 mb-2">
                      Gait pattern is within normal limits with symmetric, stable movement.
                    </p>
                    <ul className="text-sm text-green-700 dark:text-green-300 list-disc list-inside space-y-1">
                      <li>Symmetry index &lt; 0.10</li>
                      <li>Cadence 100-120 steps/min</li>
                      <li>High stability</li>
                      <li>Smooth, consistent movement</li>
                    </ul>
                    <p className="text-sm text-green-700 dark:text-green-300 mt-2 font-semibold">
                      Action: Continue normal activities. No intervention needed.
                    </p>
                  </div>

                  <div className="border-l-4 border-yellow-500 pl-4 py-3 bg-yellow-50 dark:bg-yellow-950 rounded-r">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertCircle className="h-5 w-5 text-yellow-600" />
                      <h4 className="font-semibold text-yellow-900 dark:text-yellow-100">Moderate Gait Quality</h4>
                    </div>
                    <p className="text-sm text-yellow-700 dark:text-yellow-300 mb-2">
                      Some deviations detected but generally functional gait pattern.
                    </p>
                    <ul className="text-sm text-yellow-700 dark:text-yellow-300 list-disc list-inside space-y-1">
                      <li>Symmetry index 0.10-0.20</li>
                      <li>Cadence slightly outside normal range</li>
                      <li>Medium stability</li>
                      <li>Some movement inconsistencies</li>
                    </ul>
                    <p className="text-sm text-yellow-700 dark:text-yellow-300 mt-2 font-semibold">
                      Action: Monitor for changes. Consider preventive exercises or physical therapy.
                    </p>
                  </div>

                  <div className="border-l-4 border-red-500 pl-4 py-3 bg-red-50 dark:bg-red-950 rounded-r">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertCircle className="h-5 w-5 text-red-600" />
                      <h4 className="font-semibold text-red-900 dark:text-red-100">Poor Gait Quality</h4>
                    </div>
                    <p className="text-sm text-red-700 dark:text-red-300 mb-2">
                      Significant abnormalities requiring clinical attention.
                    </p>
                    <ul className="text-sm text-red-700 dark:text-red-300 list-disc list-inside space-y-1">
                      <li>Symmetry index &gt; 0.20</li>
                      <li>Cadence significantly abnormal</li>
                      <li>Low stability</li>
                      <li>Jerky or inconsistent movement</li>
                    </ul>
                    <p className="text-sm text-red-700 dark:text-red-300 mt-2 font-semibold">
                      Action: Clinical evaluation recommended. May indicate injury, neurological issue, or fall risk.
                    </p>
                  </div>
                </div>
              </div>

              <div className="space-y-4 mt-8">
                <h3 className="text-lg font-semibold">Confidence Levels</h3>
                <p className="text-sm text-muted-foreground">
                  Confidence indicates the reliability of the analysis based on data quality.
                </p>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="border rounded-lg p-4">
                    <Badge className="bg-green-600 text-white mb-2">High Confidence</Badge>
                    <p className="text-sm text-muted-foreground">
                      High-quality pose data with consistent tracking. Results are highly reliable.
                    </p>
                  </div>
                  <div className="border rounded-lg p-4">
                    <Badge className="bg-yellow-600 text-white mb-2">Medium Confidence</Badge>
                    <p className="text-sm text-muted-foreground">
                      Adequate data quality with minor tracking issues. Results are generally reliable.
                    </p>
                  </div>
                  <div className="border rounded-lg p-4">
                    <Badge className="bg-red-600 text-white mb-2">Low Confidence</Badge>
                    <p className="text-sm text-muted-foreground">
                      Limited data quality. Results should be interpreted with caution.
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Clinical Use Tab */}
        <TabsContent value="clinical" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Clinical Applications</CardTitle>
              <CardDescription>How to use pose analysis in clinical practice</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Use Cases</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="border rounded-lg p-4 space-y-2">
                    <h4 className="font-semibold">Fall Risk Assessment</h4>
                    <p className="text-sm text-muted-foreground">
                      Low stability and slow cadence are strong indicators of increased fall risk.
                      Use for screening elderly patients or those with balance disorders.
                    </p>
                  </div>

                  <div className="border rounded-lg p-4 space-y-2">
                    <h4 className="font-semibold">Rehabilitation Monitoring</h4>
                    <p className="text-sm text-muted-foreground">
                      Track symmetry and stability improvements over time during physical therapy
                      or post-surgical rehabilitation.
                    </p>
                  </div>

                  <div className="border rounded-lg p-4 space-y-2">
                    <h4 className="font-semibold">Injury Detection</h4>
                    <p className="text-sm text-muted-foreground">
                      Asymmetry patterns can reveal compensatory gait due to injury, pain, or weakness
                      on one side of the body.
                    </p>
                  </div>

                  <div className="border rounded-lg p-4 space-y-2">
                    <h4 className="font-semibold">Neurological Assessment</h4>
                    <p className="text-sm text-muted-foreground">
                      Movement quality metrics help identify gait abnormalities associated with
                      Parkinson's, stroke, or other neurological conditions.
                    </p>
                  </div>
                </div>
              </div>

              <Alert>
                <Info className="h-4 w-4" />
                <AlertTitle>Important Note</AlertTitle>
                <AlertDescription>
                  Pose analysis is a screening and monitoring tool. It should not replace comprehensive
                  clinical examination and should be used in conjunction with other assessment methods.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        </TabsContent>

        {/* FAQ Tab */}
        <TabsContent value="faq" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Frequently Asked Questions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">

              <div className="space-y-4">
                <div className="border-b pb-4">
                  <h4 className="font-semibold mb-2 flex items-center gap-2">
                    <HelpCircle className="h-4 w-4" />
                    How many gait cycles do I need for accurate analysis?
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    Minimum 2-3 complete gait cycles are recommended. More cycles provide more reliable results.
                    A typical 5-10 second walking sequence usually contains 3-5 cycles.
                  </p>
                </div>

                <div className="border-b pb-4">
                  <h4 className="font-semibold mb-2 flex items-center gap-2">
                    <HelpCircle className="h-4 w-4" />
                    What does "cached" mean in the analysis?
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    Analysis results are stored for 1 hour to improve performance. Cached results load instantly
                    (&lt;100ms) instead of taking 1-2 seconds. Use the "Refresh Analysis" button to force a new analysis.
                  </p>
                </div>

                <div className="border-b pb-4">
                  <h4 className="font-semibold mb-2 flex items-center gap-2">
                    <HelpCircle className="h-4 w-4" />
                    Why is my symmetry score high even though the person looks symmetric?
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    Remember that higher symmetry scores indicate MORE asymmetry. A score of 0.05 is better than 0.15.
                    The score measures the difference between left and right sides, so lower is better.
                  </p>
                </div>

                <div className="border-b pb-4">
                  <h4 className="font-semibold mb-2 flex items-center gap-2">
                    <HelpCircle className="h-4 w-4" />
                    What if I get "No pose data available" error?
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    This means the sequence hasn't been processed with pose estimation yet. Make sure the GAVD dataset
                    has been fully processed and includes pose keypoint data. Try selecting a different sequence.
                  </p>
                </div>

                <div className="border-b pb-4">
                  <h4 className="font-semibold mb-2 flex items-center gap-2">
                    <HelpCircle className="h-4 w-4" />
                    Can I export the analysis results?
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    Export functionality is planned for a future update. Currently, you can view and interpret
                    results in the interface. Contact support if you need raw data access.
                  </p>
                </div>

                <div className="border-b pb-4">
                  <h4 className="font-semibold mb-2 flex items-center gap-2">
                    <HelpCircle className="h-4 w-4" />
                    How accurate is the pose analysis?
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    Accuracy depends on video quality and pose tracking quality. High confidence results are
                    generally reliable for clinical screening. Always use in conjunction with clinical judgment
                    and other assessment methods.
                  </p>
                </div>

                <div className="pb-4">
                  <h4 className="font-semibold mb-2 flex items-center gap-2">
                    <HelpCircle className="h-4 w-4" />
                    What should I do if results seem incorrect?
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    First, check the confidence level. Low confidence may indicate tracking issues. Try refreshing
                    the analysis or using a different sequence. If problems persist, the video quality may be
                    insufficient for accurate pose estimation.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Footer */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-start gap-4">
            <Info className="h-5 w-5 text-primary mt-0.5" />
            <div className="space-y-2">
              <h3 className="font-semibold">Need More Help?</h3>
              <p className="text-sm text-muted-foreground">
                For additional support, technical questions, or to report issues, please contact our support team
                or refer to the full documentation.
              </p>
              <div className="flex gap-4 mt-4">
                <a href="/help" className="text-sm text-primary hover:underline">
                  Help Center
                </a>
                <a href="/docs" className="text-sm text-primary hover:underline">
                  Documentation
                </a>
                <a href="/contact" className="text-sm text-primary hover:underline">
                  Contact Support
                </a>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
