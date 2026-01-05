/**
 * Pose Analysis Overview Component
 * 
 * Displays comprehensive gait analysis results including:
 * - Overall assessment (Good/Moderate/Poor)
 * - Key metrics (Symmetry, Cadence, Stability, Gait Cycles)
 * - Recommendations
 * - Sequence information
 * 
 * Author: AlexPose Team
 * Date: January 4, 2026
 */

'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { Activity, TrendingUp, Target, Zap, AlertCircle, CheckCircle2, Info, HelpCircle, ExternalLink } from 'lucide-react';
import { POSE_ANALYSIS_TOOLTIPS } from '@/lib/pose-analysis-tooltips';
import { PoseAnalysisLoadingState } from '@/components/ui/loading-spinner';
import { PoseAnalysisResult, ClinicalRecommendation } from '@/lib/pose-analysis-types';
import Link from 'next/link';

interface PoseAnalysisOverviewProps {
  analysis: PoseAnalysisResult | null;
  loading?: boolean;
  error?: string | null;
}

export default function PoseAnalysisOverview({ 
  analysis, 
  loading = false, 
  error = null 
}: PoseAnalysisOverviewProps) {
  // Loading state
  if (loading) {
    return <PoseAnalysisLoadingState />;
  }

  // Error state
  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Analysis Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  // No analysis available
  if (!analysis) {
    return (
      <Alert>
        <Info className="h-4 w-4" />
        <AlertTitle>No Analysis Available</AlertTitle>
        <AlertDescription>
          Select a sequence from the Sequences tab to view pose analysis results.
        </AlertDescription>
      </Alert>
    );
  }

  // Extract data from analysis
  const summary = analysis.summary || {};
  const overallAssessment = summary.overall_assessment || {};
  const symmetryAssessment = summary.symmetry_assessment || {};
  const cadenceAssessment = summary.cadence_assessment || {};
  const stabilityAssessment = summary.stability_assessment || {};
  const movementQuality = summary.movement_quality || {};
  const sequenceInfo = analysis.sequence_info || analysis.metadata || {};
  const gaitCycles = analysis.gait_cycles || [];
  const performance = analysis.performance || {};

  /**
   * Get color class for assessment level
   */
  const getAssessmentColor = (level: string): string => {
    const normalizedLevel = level?.toLowerCase() || '';
    
    if (['good', 'high', 'normal', 'symmetric', 'smooth'].includes(normalizedLevel)) {
      return 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900 dark:text-green-100';
    }
    
    if (['moderate', 'mildly_asymmetric', 'medium'].includes(normalizedLevel)) {
      return 'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900 dark:text-yellow-100';
    }
    
    if (['poor', 'low', 'severely_asymmetric', 'moderately_asymmetric', 'jerky', 'slow', 'fast'].includes(normalizedLevel)) {
      return 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900 dark:text-red-100';
    }
    
    return 'bg-gray-100 text-gray-800 border-gray-200 dark:bg-gray-800 dark:text-gray-100';
  };

  /**
   * Format level text for display
   */
  const formatLevel = (level: string): string => {
    if (!level) return 'Unknown';
    return level.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  /**
   * Get icon for assessment level
   */
  const getAssessmentIcon = (level: string) => {
    const normalizedLevel = level?.toLowerCase() || '';
    
    if (['good', 'high', 'normal', 'symmetric'].includes(normalizedLevel)) {
      return <CheckCircle2 className="h-5 w-5 text-green-600" />;
    }
    
    if (['moderate', 'mildly_asymmetric', 'medium'].includes(normalizedLevel)) {
      return <AlertCircle className="h-5 w-5 text-yellow-600" />;
    }
    
    return <AlertCircle className="h-5 w-5 text-red-600" />;
  };

  return (
    <div className="space-y-6">
      {/* Overall Assessment Card */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Overall Assessment
              <Tooltip>
                <TooltipTrigger asChild>
                  <button className="ml-1">
                    <HelpCircle className="h-4 w-4 text-muted-foreground hover:text-foreground transition-colors" />
                  </button>
                </TooltipTrigger>
                <TooltipContent className="max-w-sm">
                  <p className="font-semibold mb-1">{POSE_ANALYSIS_TOOLTIPS.overallLevel.title}</p>
                  <p className="text-xs">{POSE_ANALYSIS_TOOLTIPS.overallLevel.description}</p>
                </TooltipContent>
              </Tooltip>
            </CardTitle>
            <Link 
              href="/help/pose-analysis#overall-assessment"
              className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1"
            >
              Learn more <ExternalLink className="h-3 w-3" />
            </Link>
          </div>
          <CardDescription>Summary of gait analysis results</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Overall Level */}
            <div className="border rounded-lg p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <p className="text-sm font-medium text-muted-foreground">Overall Level</p>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button>
                        <HelpCircle className="h-3.5 w-3.5 text-muted-foreground hover:text-foreground transition-colors" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs">
                      <p className="font-semibold mb-1">Overall Gait Quality</p>
                      <p className="text-xs mb-2">{POSE_ANALYSIS_TOOLTIPS.overallLevel.description}</p>
                      <div className="space-y-1 text-xs">
                        <p><span className="font-semibold text-green-400">Good:</span> {POSE_ANALYSIS_TOOLTIPS.overallLevel.interpretation.good}</p>
                        <p><span className="font-semibold text-yellow-400">Moderate:</span> {POSE_ANALYSIS_TOOLTIPS.overallLevel.interpretation.moderate}</p>
                        <p><span className="font-semibold text-red-400">Poor:</span> {POSE_ANALYSIS_TOOLTIPS.overallLevel.interpretation.poor}</p>
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </div>
                {getAssessmentIcon(overallAssessment.overall_level || 'unknown')}
              </div>
              <Badge className={`${getAssessmentColor(overallAssessment.overall_level || 'unknown')} text-sm px-3 py-1`}>
                {formatLevel(overallAssessment.overall_level || 'Unknown')}
              </Badge>
              <p className="text-xs text-muted-foreground">
                Confidence: <span className="font-medium">{typeof overallAssessment.confidence === 'number' ? `${(overallAssessment.confidence * 100).toFixed(0)}%` : 'N/A'}</span>
              </p>
            </div>
            
            {/* Symmetry */}
            <div className="border rounded-lg p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <p className="text-sm font-medium text-muted-foreground">Symmetry</p>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button>
                        <HelpCircle className="h-3.5 w-3.5 text-muted-foreground hover:text-foreground transition-colors" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs">
                      <p className="font-semibold mb-1">{POSE_ANALYSIS_TOOLTIPS.symmetry.title}</p>
                      <p className="text-xs mb-2">{POSE_ANALYSIS_TOOLTIPS.symmetry.description}</p>
                      <p className="text-xs mb-2">{POSE_ANALYSIS_TOOLTIPS.symmetry.details}</p>
                      <div className="space-y-1 text-xs">
                        <p><span className="font-semibold">Symmetric:</span> {POSE_ANALYSIS_TOOLTIPS.symmetry.interpretation.symmetric}</p>
                        <p><span className="font-semibold">Mildly Asymmetric:</span> {POSE_ANALYSIS_TOOLTIPS.symmetry.interpretation.mildly_asymmetric}</p>
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </div>
                {getAssessmentIcon(symmetryAssessment.symmetry_classification || 'unknown')}
              </div>
              <Badge className={`${getAssessmentColor(symmetryAssessment.symmetry_classification || 'unknown')} text-sm px-3 py-1`}>
                {formatLevel(symmetryAssessment.symmetry_classification || 'Unknown')}
              </Badge>
              <p className="text-xs text-muted-foreground">
                Score: <span className="font-medium">{symmetryAssessment.symmetry_score?.toFixed(3) || 'N/A'}</span>
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Cadence */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Cadence
              <Tooltip>
                <TooltipTrigger asChild>
                  <button>
                    <HelpCircle className="h-3.5 w-3.5 text-muted-foreground hover:text-foreground transition-colors" />
                  </button>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  <p className="font-semibold mb-1">{POSE_ANALYSIS_TOOLTIPS.cadence.title}</p>
                  <p className="text-xs mb-2">{POSE_ANALYSIS_TOOLTIPS.cadence.description}</p>
                  <div className="space-y-1 text-xs">
                    <p><span className="font-semibold">Normal:</span> {POSE_ANALYSIS_TOOLTIPS.cadence.interpretation.normal}</p>
                    <p><span className="font-semibold">Slow:</span> {POSE_ANALYSIS_TOOLTIPS.cadence.interpretation.slow}</p>
                    <p><span className="font-semibold">Fast:</span> {POSE_ANALYSIS_TOOLTIPS.cadence.interpretation.fast}</p>
                  </div>
                  <p className="text-xs mt-2 pt-2 border-t">
                    <span className="font-semibold">Clinical:</span> {POSE_ANALYSIS_TOOLTIPS.cadence.clinicalSignificance}
                  </p>
                </TooltipContent>
              </Tooltip>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-3xl font-bold">
              {cadenceAssessment.cadence_value?.toFixed(1) || 'N/A'}
            </div>
            <p className="text-xs text-muted-foreground">steps/minute</p>
            <Badge className={`${getAssessmentColor(cadenceAssessment.cadence_level || 'unknown')} text-xs`}>
              {formatLevel(cadenceAssessment.cadence_level || 'Unknown')}
            </Badge>
          </CardContent>
        </Card>

        {/* Stability */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Zap className="h-4 w-4" />
              Stability
              <Tooltip>
                <TooltipTrigger asChild>
                  <button>
                    <HelpCircle className="h-3.5 w-3.5 text-muted-foreground hover:text-foreground transition-colors" />
                  </button>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  <p className="font-semibold mb-1">{POSE_ANALYSIS_TOOLTIPS.stability.title}</p>
                  <p className="text-xs mb-2">{POSE_ANALYSIS_TOOLTIPS.stability.description}</p>
                  <p className="text-xs mb-2">{POSE_ANALYSIS_TOOLTIPS.stability.details}</p>
                  <div className="space-y-1 text-xs">
                    <p><span className="font-semibold">High:</span> {POSE_ANALYSIS_TOOLTIPS.stability.interpretation.high}</p>
                    <p><span className="font-semibold">Medium:</span> {POSE_ANALYSIS_TOOLTIPS.stability.interpretation.medium}</p>
                    <p><span className="font-semibold">Low:</span> {POSE_ANALYSIS_TOOLTIPS.stability.interpretation.low}</p>
                  </div>
                </TooltipContent>
              </Tooltip>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Badge className={`${getAssessmentColor(stabilityAssessment.stability_level || 'unknown')} text-sm px-3 py-1`}>
              {formatLevel(stabilityAssessment.stability_level || 'Unknown')}
            </Badge>
            <p className="text-xs text-muted-foreground mt-2">
              Center of mass stability
            </p>
          </CardContent>
        </Card>

        {/* Gait Cycles */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Gait Cycles
              <Tooltip>
                <TooltipTrigger asChild>
                  <button>
                    <HelpCircle className="h-3.5 w-3.5 text-muted-foreground hover:text-foreground transition-colors" />
                  </button>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  <p className="font-semibold mb-1">{POSE_ANALYSIS_TOOLTIPS.gaitCycles.title}</p>
                  <p className="text-xs mb-2">{POSE_ANALYSIS_TOOLTIPS.gaitCycles.description}</p>
                  <p className="text-xs mb-2">{POSE_ANALYSIS_TOOLTIPS.gaitCycles.details}</p>
                  <p className="text-xs">{POSE_ANALYSIS_TOOLTIPS.gaitCycles.interpretation}</p>
                </TooltipContent>
              </Tooltip>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-3xl font-bold">
              {gaitCycles.length || 0}
            </div>
            <p className="text-xs text-muted-foreground">detected cycles</p>
            {gaitCycles.length > 0 && (
              <p className="text-xs text-muted-foreground">
                Avg: {(gaitCycles.reduce((sum: number, c: any) => sum + (c.duration_seconds || 0), 0) / gaitCycles.length).toFixed(2)}s
              </p>
            )}
          </CardContent>
        </Card>

        {/* Movement Quality */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Movement
              <Tooltip>
                <TooltipTrigger asChild>
                  <button>
                    <HelpCircle className="h-3.5 w-3.5 text-muted-foreground hover:text-foreground transition-colors" />
                  </button>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  <p className="font-semibold mb-1">Movement Quality</p>
                  <p className="text-xs mb-2">Assessment of walking smoothness and consistency</p>
                  <div className="space-y-2 text-xs">
                    <div>
                      <p className="font-semibold">{POSE_ANALYSIS_TOOLTIPS.velocityConsistency.title}</p>
                      <p>{POSE_ANALYSIS_TOOLTIPS.velocityConsistency.description}</p>
                    </div>
                    <div>
                      <p className="font-semibold">{POSE_ANALYSIS_TOOLTIPS.movementSmoothness.title}</p>
                      <p>{POSE_ANALYSIS_TOOLTIPS.movementSmoothness.description}</p>
                    </div>
                  </div>
                </TooltipContent>
              </Tooltip>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Consistency:</span>
                <Badge className={`${getAssessmentColor(movementQuality.velocity_consistency?.level || 'unknown')} text-xs`}>
                  {formatLevel(movementQuality.velocity_consistency?.level || 'N/A')}
                </Badge>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Smoothness:</span>
                <Badge className={`${getAssessmentColor(movementQuality.movement_smoothness?.level || 'unknown')} text-xs`}>
                  {formatLevel(movementQuality.movement_smoothness?.level || 'N/A')}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recommendations */}
      {overallAssessment.recommendations && overallAssessment.recommendations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Info className="h-5 w-5" />
              Clinical Recommendations
            </CardTitle>
            <CardDescription>Evidence-based clinical suggestions with research citations</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {overallAssessment.recommendations.map((rec: string | ClinicalRecommendation, idx: number) => {
                try {
                  // Handle both old string format and new object format for backward compatibility
                  if (typeof rec === 'string') {
                    return (
                      <div key={idx} className="flex items-start space-x-3 p-3 rounded-lg bg-muted/50">
                        <span className="text-primary mt-0.5 flex-shrink-0">
                          <CheckCircle2 className="h-4 w-4" />
                        </span>
                        <span className="text-sm">{rec}</span>
                      </div>
                    );
                  }

                  // Handle new detailed recommendation format
                  const recommendation = rec as ClinicalRecommendation;
                  
                  // Ensure recommendation has required fields
                  if (!recommendation.recommendation) {
                    return (
                      <div key={idx} className="flex items-start space-x-3 p-3 rounded-lg bg-muted/50">
                        <span className="text-primary mt-0.5 flex-shrink-0">
                          <CheckCircle2 className="h-4 w-4" />
                        </span>
                        <span className="text-sm">Invalid recommendation format</span>
                      </div>
                    );
                  }
                  
                  return (
                    <div key={idx} className="border rounded-lg p-4 space-y-3">
                      {/* Recommendation Title */}
                      <div className="flex items-start space-x-3">
                        <span className="text-primary mt-0.5 flex-shrink-0">
                          <CheckCircle2 className="h-4 w-4" />
                        </span>
                        <div className="flex-1">
                          <h4 className="font-medium text-sm">{recommendation.recommendation}</h4>
                          {recommendation.clinical_rationale && (
                            <p className="text-xs text-muted-foreground mt-1">{recommendation.clinical_rationale}</p>
                          )}
                        </div>
                      </div>

                      {/* Clinical Details */}
                      {recommendation.clinical_threshold && (
                        <div className="ml-7 space-y-2">
                          <div className="text-xs">
                            <span className="font-medium text-muted-foreground">Clinical Threshold: </span>
                            <span className="text-foreground">{recommendation.clinical_threshold}</span>
                          </div>
                          
                          {recommendation.evidence_level && (
                            <div className="text-xs">
                              <span className="font-medium text-muted-foreground">Evidence Level: </span>
                              <Badge variant="outline" className="text-xs">
                                {recommendation.evidence_level.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
                              </Badge>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Primary Source */}
                      {recommendation.primary_source && (
                        <div className="ml-7 p-3 bg-muted/30 rounded-md">
                          <p className="text-xs font-medium text-muted-foreground mb-1">Primary Evidence:</p>
                          <div className="space-y-1">
                            <p className="text-xs font-medium">{recommendation.primary_source.title}</p>
                            {recommendation.primary_source.authors && (
                              <p className="text-xs text-muted-foreground">{recommendation.primary_source.authors}</p>
                            )}
                            {recommendation.primary_source.journal && recommendation.primary_source.year && (
                              <p className="text-xs text-muted-foreground">
                                {recommendation.primary_source.journal} ({recommendation.primary_source.year})
                              </p>
                            )}
                            {recommendation.primary_source.key_finding && (
                              <p className="text-xs text-foreground mt-1">
                                <span className="font-medium">Key Finding: </span>
                                {recommendation.primary_source.key_finding}
                              </p>
                            )}
                            {recommendation.primary_source.url && (
                              <Link 
                                href={recommendation.primary_source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1 text-xs text-primary hover:underline mt-1"
                              >
                                View Source <ExternalLink className="h-3 w-3" />
                              </Link>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Supporting Evidence */}
                      {recommendation.supporting_evidence && recommendation.supporting_evidence.length > 0 && (
                        <div className="ml-7">
                          <p className="text-xs font-medium text-muted-foreground mb-2">Supporting Evidence:</p>
                          <div className="space-y-2">
                            {recommendation.supporting_evidence.slice(0, 2).map((evidence, evidenceIdx: number) => (
                              <div key={evidenceIdx} className="p-2 bg-muted/20 rounded text-xs">
                                <p className="font-medium">{evidence.title}</p>
                                {evidence.journal && evidence.year && (
                                  <p className="text-muted-foreground">{evidence.journal} ({evidence.year})</p>
                                )}
                                {evidence.key_finding && (
                                  <p className="text-foreground mt-1">{evidence.key_finding}</p>
                                )}
                                {evidence.url && (
                                  <Link 
                                    href={evidence.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1 text-primary hover:underline mt-1"
                                  >
                                    View Source <ExternalLink className="h-3 w-3" />
                                  </Link>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                } catch (error) {
                  // Fallback for any rendering errors
                  console.error('Error rendering recommendation:', error, rec);
                  return (
                    <div key={idx} className="flex items-start space-x-3 p-3 rounded-lg bg-red-50 border border-red-200">
                      <span className="text-red-500 mt-0.5 flex-shrink-0">
                        <AlertCircle className="h-4 w-4" />
                      </span>
                      <span className="text-sm text-red-700">
                        Error rendering recommendation. Please check the console for details.
                      </span>
                    </div>
                  );
                }
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Sequence Information */}
      <Card>
        <CardHeader>
          <CardTitle>Sequence Information</CardTitle>
          <CardDescription>Technical details about the analyzed sequence</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="space-y-1">
              <p className="text-muted-foreground">Frames</p>
              <p className="font-medium text-lg">{sequenceInfo.num_frames || 'N/A'}</p>
            </div>
            <div className="space-y-1">
              <p className="text-muted-foreground">Duration</p>
              <p className="font-medium text-lg">{sequenceInfo.duration_seconds?.toFixed(2) || 'N/A'}s</p>
            </div>
            <div className="space-y-1">
              <p className="text-muted-foreground">FPS</p>
              <p className="font-medium text-lg">{sequenceInfo.fps || 'N/A'}</p>
            </div>
            <div className="space-y-1">
              <p className="text-muted-foreground">Format</p>
              <p className="font-medium text-lg">{sequenceInfo.keypoint_format || 'N/A'}</p>
            </div>
          </div>
          
          {/* Performance Metrics */}
          {performance.analysis_time_seconds && (
            <div className="mt-4 pt-4 border-t">
              <p className="text-xs text-muted-foreground mb-2">Performance</p>
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <span className="text-muted-foreground">Analysis Time: </span>
                  <span className="font-medium">{performance.analysis_time_seconds.toFixed(2)}s</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Processing Speed: </span>
                  <span className="font-medium">{performance.frames_per_second?.toFixed(1) || 'N/A'} fps</span>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Most Asymmetric Joints (if available) */}
      {symmetryAssessment.most_asymmetric_joints && symmetryAssessment.most_asymmetric_joints.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Asymmetry Details</CardTitle>
            <CardDescription>Joints showing the most asymmetry</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {symmetryAssessment.most_asymmetric_joints.map((joint: any, idx: number) => (
                <div key={idx} className="flex items-center justify-between p-3 rounded-lg border">
                  <span className="font-medium capitalize">{joint.joint?.replace(/_/g, ' ')}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-muted-foreground">
                      Asymmetry: {joint.asymmetry?.toFixed(3)}
                    </span>
                    <Badge className={getAssessmentColor(joint.asymmetry > 0.2 ? 'high' : joint.asymmetry > 0.1 ? 'moderate' : 'low')}>
                      {joint.asymmetry > 0.2 ? 'High' : joint.asymmetry > 0.1 ? 'Moderate' : 'Low'}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
