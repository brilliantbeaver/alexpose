/**
 * Pose Analysis Type Definitions
 * 
 * Defines TypeScript interfaces for pose analysis results, clinical recommendations,
 * and related data structures. Follows OOP principles with clear type hierarchies.
 * 
 * @module lib/pose-analysis-types
 */

/**
 * Clinical recommendation with evidence-based sources.
 * 
 * Represents a clinical recommendation generated from gait analysis,
 * including the recommendation text, clinical thresholds, evidence level,
 * and research sources.
 */
export interface ClinicalRecommendation {
  /** The recommendation text */
  recommendation: string;
  
  /** Clinical threshold that triggered this recommendation */
  clinical_threshold?: string;
  
  /** Level of evidence supporting this recommendation */
  evidence_level?: 'systematic_review' | 'meta_analysis' | 'rct' | 'cohort_study' | 'case_study' | 'expert_opinion' | 'legacy_format' | 'unknown_format';
  
  /** Primary research source */
  primary_source?: ResearchSource;
  
  /** Supporting research evidence */
  supporting_evidence?: ResearchSource[];
  
  /** Clinical rationale for the recommendation */
  clinical_rationale?: string;
}

/**
 * Research source citation.
 * 
 * Represents a research paper or study that supports a clinical recommendation.
 */
export interface ResearchSource {
  /** Title of the research paper */
  title: string;
  
  /** Authors of the paper */
  authors?: string;
  
  /** Journal or publication venue */
  journal?: string;
  
  /** Publication year */
  year?: number;
  
  /** Digital Object Identifier */
  doi?: string;
  
  /** URL to the paper */
  url?: string;
  
  /** Key finding from the research */
  key_finding?: string;
}

/**
 * Overall assessment of gait quality.
 */
export interface OverallAssessment {
  /** Overall gait quality level */
  overall_level: 'good' | 'moderate' | 'poor' | 'unknown';
  
  /** Confidence in the assessment */
  confidence: 'high' | 'medium' | 'low';
  
  /** Clinical recommendations */
  recommendations: (string | ClinicalRecommendation)[];
  
  /** Timestamp of assessment */
  timestamp?: number;
  
  /** Assessment type */
  assessment_type?: string;
  
  /** Evidence base used */
  evidence_base?: string;
}

/**
 * Symmetry assessment results.
 */
export interface SymmetryAssessment {
  /** Symmetry classification */
  symmetry_classification: 'symmetric' | 'mildly_asymmetric' | 'moderately_asymmetric' | 'severely_asymmetric' | 'unknown';
  
  /** Numerical symmetry score (0-1, lower is more symmetric) */
  symmetry_score: number;
  
  /** Most asymmetric joints */
  most_asymmetric_joints?: AsymmetricJoint[];
}

/**
 * Asymmetric joint information.
 */
export interface AsymmetricJoint {
  /** Joint name */
  joint: string;
  
  /** Asymmetry value */
  asymmetry: number;
  
  /** Asymmetry level */
  level?: 'low' | 'moderate' | 'high';
}

/**
 * Cadence assessment results.
 */
export interface CadenceAssessment {
  /** Cadence value in steps per minute */
  cadence_value: number;
  
  /** Cadence level classification */
  cadence_level: 'normal' | 'slow' | 'fast' | 'unknown';
}

/**
 * Stability assessment results.
 */
export interface StabilityAssessment {
  /** Stability level */
  stability_level: 'high' | 'medium' | 'low' | 'unknown';
  
  /** Stability index value */
  stability_index?: number;
}

/**
 * Movement quality assessment.
 */
export interface MovementQuality {
  /** Velocity consistency */
  velocity_consistency: 'good' | 'moderate' | 'poor' | 'unknown';
  
  /** Movement smoothness */
  movement_smoothness: 'smooth' | 'moderate' | 'jerky' | 'unknown';
  
  /** Velocity coefficient of variation */
  velocity_cv?: number;
  
  /** Jerk mean value */
  jerk_mean?: number;
}

/**
 * Summary of pose analysis results.
 */
export interface PoseAnalysisSummary {
  /** Overall assessment */
  overall_assessment: OverallAssessment;
  
  /** Symmetry assessment */
  symmetry_assessment: SymmetryAssessment;
  
  /** Cadence assessment */
  cadence_assessment: CadenceAssessment;
  
  /** Stability assessment */
  stability_assessment: StabilityAssessment;
  
  /** Movement quality */
  movement_quality: MovementQuality;
  
  /** Temporal regularity */
  temporal_regularity?: {
    regularity_level: 'high' | 'moderate' | 'low';
  };
  
  /** Analysis timestamp */
  analysis_timestamp?: number;
  
  /** Analysis version */
  analysis_version?: string;
}

/**
 * Gait cycle information.
 */
export interface GaitCycle {
  /** Cycle ID */
  cycle_id?: number;
  
  /** Cycle number */
  cycle_number: number;
  
  /** Start frame */
  start_frame: number;
  
  /** End frame */
  end_frame: number;
  
  /** Duration in seconds */
  duration_seconds: number;
  
  /** Duration (alias for duration_seconds) */
  duration?: number;
  
  /** Step length in meters */
  step_length?: number;
  
  /** Stride length in meters */
  stride_length?: number;
  
  /** Cycle type */
  cycle_type?: 'left' | 'right' | 'bilateral';
}

/**
 * Sequence information.
 */
export interface SequenceInfo {
  /** Number of frames */
  num_frames: number;
  
  /** Duration in seconds */
  duration_seconds: number;
  
  /** Duration (alias for duration_seconds) */
  duration?: number;
  
  /** Frame count (alias for num_frames) */
  frame_count?: number;
  
  /** Frames per second */
  fps: number;
  
  /** Keypoint format */
  keypoint_format: string;
  
  /** Sequence ID */
  sequence_id?: string;
  
  /** Dataset ID */
  dataset_id?: string;
}

/**
 * Performance metrics.
 */
export interface PerformanceMetrics {
  /** Analysis time in seconds */
  analysis_time_seconds: number;
  
  /** Processing time (alias for analysis_time_seconds) */
  processing_time?: number;
  
  /** Processing speed in frames per second */
  frames_per_second?: number;
  
  /** Cache hit indicator */
  cache_hit?: boolean;
  
  /** Confidence score */
  confidence_score?: number;
  
  /** Quality score */
  quality_score?: number;
}

/**
 * Complete pose analysis result.
 * 
 * This is the main interface for pose analysis results returned from the backend.
 */
export interface PoseAnalysisResult {
  /** Analysis ID */
  analysis_id?: string;
  
  /** Analysis version */
  version?: string;
  
  /** Summary of analysis results */
  summary: PoseAnalysisSummary;
  
  /** Detected gait cycles */
  gait_cycles: GaitCycle[];
  
  /** Sequence information */
  sequence_info: SequenceInfo;
  
  /** Metadata */
  metadata?: Record<string, any>;
  
  /** Performance metrics */
  performance?: PerformanceMetrics;
  
  /** Features extracted */
  features?: Record<string, any>;
  
  /** Symmetry analysis details */
  symmetry_analysis?: Record<string, any>;
  
  /** Timing analysis details */
  timing_analysis?: Record<string, any>;
  
  /** Phase features */
  phase_features?: Record<string, any>;
  
  /** Analysis error if any */
  analysis_error?: string;
  
  /** Timestamp of analysis */
  timestamp?: number | string;
}

/**
 * Pose analysis status.
 */
export type PoseAnalysisStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'cached';

/**
 * Pose analysis request.
 */
export interface PoseAnalysisRequest {
  /** Dataset ID */
  dataset_id: string;
  
  /** Sequence ID */
  sequence_id: string;
  
  /** Force re-analysis (ignore cache) */
  force_reanalysis?: boolean;
  
  /** Additional options */
  options?: Record<string, any>;
}

/**
 * Pose analysis response wrapper.
 */
export interface PoseAnalysisResponse {
  /** Analysis result */
  result: PoseAnalysisResult | null;
  
  /** Status of the analysis */
  status: PoseAnalysisStatus;
  
  /** Error message if failed */
  error?: string;
  
  /** Timestamp */
  timestamp: number;
}
