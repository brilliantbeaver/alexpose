/**
 * Pose Analysis Tooltips
 * 
 * Provides tooltip content for pose analysis metrics and assessments.
 * Contains clinical explanations and interpretations for user guidance.
 * 
 * @module lib/pose-analysis-tooltips
 */

/**
 * Tooltip content structure.
 */
export interface TooltipContent {
  /** Tooltip title */
  title: string;
  
  /** Main description */
  description: string;
  
  /** Additional details */
  details?: string;
  
  /** Interpretation guidelines (can be string or object with specific interpretations) */
  interpretation?: string | Record<string, string>;
  
  /** Clinical significance */
  clinicalSignificance?: string;
  
  /** Normal ranges */
  normalRange?: string;
}

/**
 * Pose analysis tooltips configuration.
 * 
 * Provides comprehensive tooltip content for all pose analysis metrics
 * and assessments displayed in the user interface.
 */
export const POSE_ANALYSIS_TOOLTIPS: Record<string, TooltipContent> = {
  overallLevel: {
    title: 'Overall Gait Quality',
    description: 'Comprehensive assessment of gait quality based on multiple metrics including symmetry, stability, cadence, and movement quality.',
    interpretation: {
      good: 'Gait patterns are within normal ranges across all measured parameters. No significant abnormalities detected.',
      moderate: 'Some gait parameters show minor deviations from normal ranges. May benefit from monitoring or minor interventions.',
      poor: 'Multiple gait parameters show significant deviations from normal ranges. Clinical evaluation recommended.',
    },
    clinicalSignificance: 'Overall gait quality provides a quick screening tool for identifying individuals who may benefit from detailed clinical assessment.',
  },
  
  symmetry: {
    title: 'Gait Symmetry',
    description: 'Measures the bilateral symmetry of gait patterns between left and right sides.',
    details: 'Symmetry is assessed by comparing joint angles, step lengths, and temporal parameters between left and right limbs.',
    interpretation: {
      symmetric: 'Left and right sides show similar gait patterns (symmetry index < 0.10). This is typical of healthy gait.',
      mildly_asymmetric: 'Minor differences between left and right sides (symmetry index 0.10-0.20). May be within normal variation.',
      moderately_asymmetric: 'Noticeable differences between sides (symmetry index 0.20-0.30). Clinical evaluation recommended.',
      severely_asymmetric: 'Significant asymmetry detected (symmetry index > 0.30). Indicates potential pathology requiring assessment.',
    },
    clinicalSignificance: 'Gait asymmetry can indicate neurological conditions, musculoskeletal injuries, pain, or compensation patterns.',
    normalRange: 'Symmetry index < 0.10 is considered normal',
  },
  
  cadence: {
    title: 'Cadence (Steps per Minute)',
    description: 'The number of steps taken per minute, indicating walking rhythm and pace.',
    interpretation: {
      normal: '100-130 steps per minute. Typical cadence for healthy adults.',
      slow: 'Below 100 steps per minute. May indicate reduced mobility, fatigue, pain, or balance concerns.',
      fast: 'Above 130 steps per minute. May indicate hurried gait or compensation for reduced step length.',
    },
    clinicalSignificance: 'Cadence changes can indicate fatigue, pain, balance issues, or neurological conditions. Slow cadence is associated with increased fall risk.',
    normalRange: '100-130 steps/minute for adults',
  },
  
  stability: {
    title: 'Gait Stability',
    description: 'Assessment of center of mass stability during walking.',
    details: 'Stability is measured by analyzing the variability and control of the center of mass trajectory during gait.',
    interpretation: {
      high: 'Excellent postural control and balance during walking. Low fall risk.',
      medium: 'Adequate stability with some variability. May benefit from balance training.',
      low: 'Poor stability with significant center of mass fluctuations. Increased fall risk. Balance assessment recommended.',
    },
    clinicalSignificance: 'Poor stability increases fall risk and limits functional mobility. Balance training has strong evidence for improving postural control.',
    normalRange: 'Stability index < 0.20 indicates good stability',
  },
  
  gaitCycles: {
    title: 'Gait Cycles',
    description: 'Number of complete gait cycles detected in the sequence.',
    details: 'A gait cycle is the period from initial contact of one foot to the next initial contact of the same foot.',
    interpretation: 'More cycles provide more reliable analysis. Minimum 3-4 cycles recommended for accurate assessment.',
    clinicalSignificance: 'Gait cycle detection enables temporal analysis and identification of gait pattern irregularities.',
  },
  
  velocityConsistency: {
    title: 'Velocity Consistency',
    description: 'Measures how consistent walking speed is throughout the sequence.',
    interpretation: {
      good: 'Consistent walking speed with low variability (CV < 0.30). Indicates good motor control.',
      moderate: 'Some speed variability (CV 0.30-0.60). May indicate fatigue or attention issues.',
      poor: 'High speed variability (CV > 0.60). May indicate motor control problems or environmental factors.',
    },
    clinicalSignificance: 'Velocity consistency reflects motor control quality and can indicate neurological or musculoskeletal issues.',
  },
  
  movementSmoothness: {
    title: 'Movement Smoothness',
    description: 'Assesses the smoothness of movement trajectories using jerk analysis.',
    interpretation: {
      smooth: 'Low jerk values (< 100). Smooth, coordinated movements typical of healthy gait.',
      moderate: 'Moderate jerk values (100-300). Some movement irregularities present.',
      jerky: 'High jerk values (> 300). Jerky, uncoordinated movements. May indicate motor control issues.',
    },
    clinicalSignificance: 'Movement smoothness reflects coordination and motor control quality. Jerky movements may indicate neurological conditions or pain.',
  },
  
  recommendations: {
    title: 'Clinical Recommendations',
    description: 'Evidence-based clinical suggestions generated from gait analysis results.',
    details: 'Recommendations are based on established clinical thresholds and peer-reviewed research from 2021-2025.',
    clinicalSignificance: 'These are clinical suggestions, not diagnoses. Healthcare provider judgment should always supersede automated recommendations.',
  },
  
  asymmetricJoints: {
    title: 'Asymmetric Joints',
    description: 'Joints showing the greatest differences between left and right sides.',
    details: 'Identifies specific joints contributing most to overall gait asymmetry.',
    clinicalSignificance: 'Helps target specific areas for clinical evaluation and intervention.',
  },
  
  temporalRegularity: {
    title: 'Temporal Regularity',
    description: 'Consistency of timing patterns across gait cycles.',
    interpretation: {
      high: 'Highly consistent timing across cycles. Indicates good motor control.',
      moderate: 'Some timing variability between cycles. Within normal range for most individuals.',
      low: 'Significant timing irregularities. May indicate motor control or attention issues.',
    },
    clinicalSignificance: 'Temporal regularity reflects the consistency and automaticity of gait patterns.',
  },
  
  phaseFeatures: {
    title: 'Gait Phase Features',
    description: 'Characteristics of different phases of the gait cycle (stance, swing).',
    details: 'Analyzes timing and kinematics during stance phase (foot on ground) and swing phase (foot in air).',
    clinicalSignificance: 'Phase-specific analysis can identify subtle gait abnormalities and compensation patterns.',
  },
  
  evidenceLevel: {
    title: 'Evidence Level',
    description: 'Quality and type of research supporting the recommendation.',
    interpretation: {
      systematic_review: 'Highest quality evidence from comprehensive review of multiple studies.',
      meta_analysis: 'Statistical synthesis of multiple studies providing strong evidence.',
      rct: 'Randomized controlled trial - gold standard for intervention studies.',
      cohort_study: 'Observational study following groups over time.',
      case_study: 'Detailed analysis of individual cases.',
      expert_opinion: 'Professional consensus without systematic research.',
    },
    clinicalSignificance: 'Evidence level indicates the strength and reliability of the research supporting the recommendation.',
  },
};

/**
 * Get tooltip content for a specific metric.
 * 
 * @param key - Metric key
 * @returns Tooltip content or undefined if not found
 */
export function getTooltipContent(key: string): TooltipContent | undefined {
  return POSE_ANALYSIS_TOOLTIPS[key];
}

/**
 * Get interpretation text for a specific metric and value.
 * 
 * @param key - Metric key
 * @param value - Metric value
 * @returns Interpretation text or undefined
 */
export function getInterpretation(key: string, value: string): string | undefined {
  const tooltip = POSE_ANALYSIS_TOOLTIPS[key];
  if (!tooltip || !tooltip.interpretation) {
    return undefined;
  }
  
  // Check if interpretation is an object (Record<string, string>)
  if (typeof tooltip.interpretation === 'object') {
    return tooltip.interpretation[value];
  }
  
  // If interpretation is a string, return it directly
  return tooltip.interpretation;
}
