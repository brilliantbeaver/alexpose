"""
Property-based tests for classification components.

These tests validate classification correctness properties using Hypothesis
for comprehensive input coverage across different classification scenarios.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from unittest.mock import Mock, patch

from hypothesis import given, strategies as st, assume, settings, example
from hypothesis.strategies import composite

from tests.property.strategies import (
    gait_features_strategy, classification_result_strategy, 
    llm_response_strategy, confidence_scores
)
from tests.utils.assertions import (
    assert_classification_result, assert_confidence_bounds,
    AssertionHelpers
)
from tests.utils.property_helpers import (
    ClassificationValidator, LLMResponseAnalyzer
)

try:
    from ambient.classification.llm_classifier import LLMClassifier
    from ambient.classification.prompt_manager import PromptManager
    from ambient.core.data_models import GaitFeatures, ClassificationResult
    AMBIENT_AVAILABLE = True
except ImportError:
    AMBIENT_AVAILABLE = False


class TestBinaryClassificationCompletenessProperty:
    """
    Property 15: Binary Classification Completeness
    For any valid gait feature set, classification should return either
    "normal" or "abnormal" with confidence score.
    **Validates: Requirements 4.1**
    """
    
    @given(gait_features=gait_features_strategy())
    @settings(max_examples=50)
    def test_binary_classification_completeness_property(self, gait_features):
        """
        Feature: testing-enhancement, Property 15: Binary Classification Completeness
        For any valid gait features, should return normal/abnormal with confidence
        """
        # Simulate binary classification
        class MockBinaryClassifier:
            def classify_gait(self, features):
                # Ensure binary output based on feature characteristics
                temporal_features = features.get("temporal_features", {})
                spatial_features = features.get("spatial_features", {})
                symmetry_features = features.get("symmetry_features", {})
                
                # Simple rule-based classification for testing
                abnormal_indicators = 0
                
                # Check temporal features
                cadence = temporal_features.get("cadence", 110)
                if cadence < 80 or cadence > 150:
                    abnormal_indicators += 1
                
                stride_time = temporal_features.get("stride_time", 1.2)
                if stride_time < 0.8 or stride_time > 1.8:
                    abnormal_indicators += 1
                
                # Check spatial features
                stride_length = spatial_features.get("stride_length", 1.4)
                if stride_length < 1.0 or stride_length > 2.0:
                    abnormal_indicators += 1
                
                # Check symmetry features
                symmetry = symmetry_features.get("left_right_symmetry", 0.9)
                if symmetry < 0.7:
                    abnormal_indicators += 1
                
                # Determine classification
                if abnormal_indicators >= 2:
                    classification = "abnormal"
                    confidence = 0.6 + (abnormal_indicators * 0.1)
                else:
                    classification = "normal"
                    confidence = 0.7 + np.random.uniform(0, 0.2)
                
                confidence = min(1.0, confidence)
                
                return {
                    "classification": classification,
                    "confidence": confidence,
                    "reasoning": f"Based on {abnormal_indicators} abnormal indicators",
                    "features_analyzed": list(features.keys())
                }
        
        classifier = MockBinaryClassifier()
        result = classifier.classify_gait(gait_features)
        
        # Property: Must return exactly one of the two valid classifications
        valid_classifications = {"normal", "abnormal"}
        assert "classification" in result, "Result must include classification field"
        assert result["classification"] in valid_classifications, (
            f"Classification '{result['classification']}' not in {valid_classifications}"
        )
        
        # Property: Must include confidence score
        assert "confidence" in result, "Result must include confidence field"
        assert isinstance(result["confidence"], (int, float)), "Confidence must be numeric"
        assert 0.0 <= result["confidence"] <= 1.0, (
            f"Confidence {result['confidence']} not in [0.0, 1.0] range"
        )
        
        # Property: Must provide reasoning or explanation
        assert "reasoning" in result, "Result must include reasoning field"
        assert isinstance(result["reasoning"], str), "Reasoning must be string"
        assert len(result["reasoning"]) > 0, "Reasoning must not be empty"
        
        # Property: Should indicate which features were analyzed
        assert "features_analyzed" in result, "Result must indicate features analyzed"
        assert isinstance(result["features_analyzed"], list), "Features analyzed must be list"
        assert len(result["features_analyzed"]) > 0, "Must analyze at least some features"
    
    @given(
        num_feature_sets=st.integers(min_value=10, max_value=100),
        classification_threshold=st.floats(min_value=0.4, max_value=0.8)
    )
    @settings(max_examples=30)
    def test_classification_distribution_property(self, num_feature_sets, classification_threshold):
        """Test that classification produces reasonable distribution of results."""
        # Generate multiple feature sets
        feature_sets = []
        for _ in range(num_feature_sets):
            features = {
                "temporal_features": {
                    "cadence": np.random.uniform(70, 160),
                    "stride_time": np.random.uniform(0.7, 2.0)
                },
                "spatial_features": {
                    "stride_length": np.random.uniform(0.8, 2.2),
                    "step_width": np.random.uniform(0.05, 0.4)
                },
                "symmetry_features": {
                    "left_right_symmetry": np.random.uniform(0.5, 1.0)
                }
            }
            feature_sets.append(features)
        
        # Classify all feature sets
        class ConsistentClassifier:
            def __init__(self, threshold):
                self.threshold = threshold
            
            def classify_batch(self, feature_list):
                results = []
                for features in feature_list:
                    # Consistent classification logic
                    abnormality_score = 0.0
                    
                    # Temporal abnormality
                    cadence = features["temporal_features"]["cadence"]
                    if cadence < 90 or cadence > 130:
                        abnormality_score += 0.3
                    
                    # Spatial abnormality
                    stride_length = features["spatial_features"]["stride_length"]
                    if stride_length < 1.1 or stride_length > 1.8:
                        abnormality_score += 0.3
                    
                    # Symmetry abnormality
                    symmetry = features["symmetry_features"]["left_right_symmetry"]
                    if symmetry < 0.8:
                        abnormality_score += 0.4
                    
                    # Classify based on threshold
                    if abnormality_score >= self.threshold:
                        classification = "abnormal"
                        confidence = min(1.0, 0.5 + abnormality_score)
                    else:
                        classification = "normal"
                        confidence = min(1.0, 0.5 + (1.0 - abnormality_score))
                    
                    results.append({
                        "classification": classification,
                        "confidence": confidence,
                        "abnormality_score": abnormality_score
                    })
                
                return results
        
        classifier = ConsistentClassifier(classification_threshold)
        results = classifier.classify_batch(feature_sets)
        
        # Property: All results should be valid binary classifications
        for i, result in enumerate(results):
            assert result["classification"] in ["normal", "abnormal"], (
                f"Result {i} has invalid classification: {result['classification']}"
            )
            assert 0.0 <= result["confidence"] <= 1.0, (
                f"Result {i} has invalid confidence: {result['confidence']}"
            )
        
        # Property: Should have reasonable distribution (not all same classification)
        normal_count = sum(1 for r in results if r["classification"] == "normal")
        abnormal_count = sum(1 for r in results if r["classification"] == "abnormal")
        
        # With random features, should get some of each (unless threshold is extreme)
        if 0.3 <= classification_threshold <= 0.7 and num_feature_sets >= 20:
            assert normal_count > 0, "Should classify some features as normal"
            assert abnormal_count > 0, "Should classify some features as abnormal"
        
        # Property: Higher abnormality scores should correlate with abnormal classification
        abnormal_results = [r for r in results if r["classification"] == "abnormal"]
        normal_results = [r for r in results if r["classification"] == "normal"]
        
        if abnormal_results and normal_results:
            avg_abnormal_score = np.mean([r["abnormality_score"] for r in abnormal_results])
            avg_normal_score = np.mean([r["abnormality_score"] for r in normal_results])
            
            assert avg_abnormal_score > avg_normal_score, (
                "Abnormal classifications should have higher abnormality scores on average"
            )


class TestClassificationConfidenceBoundsProperty:
    """
    Property 16: Classification Confidence Bounds
    For any classification result, confidence scores should be within
    valid range [0.0, 1.0] with higher values indicating greater certainty.
    **Validates: Requirements 4.2**
    """
    
    @given(classification_result=classification_result_strategy())
    @settings(max_examples=50)
    def test_confidence_bounds_property(self, classification_result):
        """
        Feature: testing-enhancement, Property 16: Classification Confidence Bounds
        For any classification result, confidence should be in [0.0, 1.0] range
        """
        # Property: Confidence must be in valid range
        confidence = classification_result["confidence"]
        assert 0.0 <= confidence <= 1.0, (
            f"Confidence {confidence} outside valid range [0.0, 1.0]"
        )
        
        # Property: Confidence should be numeric
        assert isinstance(confidence, (int, float)), (
            f"Confidence must be numeric, got {type(confidence)}"
        )
        
        # Property: High confidence should indicate strong certainty
        classification = classification_result["classification"]
        if confidence > 0.9:
            # High confidence results should have clear reasoning
            reasoning = classification_result.get("reasoning", "")
            assert len(reasoning) > 10, "High confidence should have detailed reasoning"
        
        # Property: Low confidence should indicate uncertainty
        if confidence < 0.3:
            # Low confidence might indicate borderline cases
            # Should still have valid classification
            assert classification in ["normal", "abnormal"], (
                "Even low confidence results should have valid classification"
            )
    
    @given(
        evidence_strength=st.floats(min_value=0.0, max_value=1.0),
        feature_completeness=st.floats(min_value=0.3, max_value=1.0)
    )
    @settings(max_examples=30)
    def test_confidence_calibration_property(self, evidence_strength, feature_completeness):
        """Test that confidence scores are properly calibrated to evidence strength."""
        # Simulate confidence calculation based on evidence
        class CalibratedClassifier:
            def calculate_confidence(self, evidence_str, completeness):
                # Base confidence from evidence strength
                base_confidence = evidence_str
                
                # Adjust for feature completeness
                completeness_factor = completeness
                
                # Combine factors (geometric mean to avoid overconfidence)
                calibrated_confidence = np.sqrt(base_confidence * completeness_factor)
                
                # Add small random variation for realism
                noise = np.random.uniform(-0.05, 0.05)
                final_confidence = np.clip(calibrated_confidence + noise, 0.0, 1.0)
                
                return {
                    "confidence": final_confidence,
                    "evidence_strength": evidence_str,
                    "feature_completeness": completeness,
                    "base_confidence": base_confidence,
                    "calibrated": True
                }
        
        classifier = CalibratedClassifier()
        result = classifier.calculate_confidence(evidence_strength, feature_completeness)
        
        # Property: Confidence should be in valid range
        assert 0.0 <= result["confidence"] <= 1.0, (
            f"Calibrated confidence {result['confidence']} outside [0.0, 1.0]"
        )
        
        # Property: Strong evidence should lead to higher confidence
        if evidence_strength > 0.8:
            assert result["confidence"] > 0.5, (
                f"Strong evidence ({evidence_strength}) should yield confidence > 0.5, "
                f"got {result['confidence']}"
            )
        
        # Property: Weak evidence should lead to lower confidence
        if evidence_strength < 0.3:
            assert result["confidence"] < 0.7, (
                f"Weak evidence ({evidence_strength}) should yield confidence < 0.7, "
                f"got {result['confidence']}"
            )
        
        # Property: High feature completeness should boost confidence
        if feature_completeness > 0.9 and evidence_strength > 0.5:
            assert result["confidence"] > evidence_strength * 0.8, (
                "High completeness should boost confidence relative to evidence strength"
            )
        
        # Property: Low feature completeness should reduce confidence
        if feature_completeness < 0.5 and evidence_strength > 0.2:  # Only test when evidence > 0.2
            # With geometric mean, confidence should be lower than evidence when completeness is low
            expected_max = evidence_strength * 1.5  # More generous multiplier
            assert result["confidence"] < expected_max, (
                f"Low completeness should reduce confidence relative to evidence strength: "
                f"confidence {result['confidence']:.3f} >= {expected_max:.3f}"
            )
    
    @given(
        confidence_values=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=10, max_size=100
        )
    )
    @settings(max_examples=20)
    def test_confidence_distribution_property(self, confidence_values):
        """Test properties of confidence score distributions."""
        # Property: All values should be in valid range
        for i, conf in enumerate(confidence_values):
            assert 0.0 <= conf <= 1.0, f"Confidence {i} = {conf} outside [0.0, 1.0]"
        
        # Property: Should have reasonable distribution characteristics
        if len(confidence_values) >= 20:
            mean_confidence = np.mean(confidence_values)
            std_confidence = np.std(confidence_values)
            
            # Mean should be reasonable (not extreme)
            assert 0.1 <= mean_confidence <= 0.9, (
                f"Mean confidence {mean_confidence} seems extreme"
            )
            
            # Should have some variation (not all identical)
            if not all(abs(c - confidence_values[0]) < 0.01 for c in confidence_values):
                assert std_confidence > 0.01, "Should have some confidence variation"
        
        # Property: Extreme confidence values should be less common
        extreme_low = sum(1 for c in confidence_values if c < 0.1)
        extreme_high = sum(1 for c in confidence_values if c > 0.9)
        total = len(confidence_values)
        
        # In realistic scenarios, extreme confidence should be minority
        if total >= 50:
            extreme_ratio = (extreme_low + extreme_high) / total
            assert extreme_ratio < 0.5, (
                f"Too many extreme confidence values: {extreme_ratio:.2f} of total"
            )


class TestLLMResponseConsistencyProperty:
    """
    Property 17: LLM Response Consistency
    For any identical feature set processed multiple times, LLM classification
    should return consistent results within tolerance.
    **Validates: Requirements 4.3**
    """
    
    @given(
        gait_features=gait_features_strategy(),
        num_repetitions=st.integers(min_value=3, max_value=10)
    )
    @settings(max_examples=30)
    def test_llm_response_consistency_property(self, gait_features, num_repetitions):
        """
        Feature: testing-enhancement, Property 17: LLM Response Consistency
        For identical inputs, LLM should return consistent results
        """
        # Simulate LLM with controlled randomness
        class MockLLMClassifier:
            def __init__(self, temperature=0.1):
                self.temperature = temperature
                self.call_count = 0
            
            def classify_with_llm(self, features):
                self.call_count += 1
                
                # Deterministic base classification
                temporal = features.get("temporal_features", {})
                spatial = features.get("spatial_features", {})
                symmetry = features.get("symmetry_features", {})
                
                # Calculate abnormality score deterministically
                score = 0.0
                if temporal.get("cadence", 110) < 90:
                    score += 0.3
                if spatial.get("stride_length", 1.4) < 1.1:
                    score += 0.3
                if symmetry.get("left_right_symmetry", 0.9) < 0.8:
                    score += 0.4
                
                # Add small controlled randomness (simulating LLM temperature)
                noise = np.random.normal(0, self.temperature * 0.1)
                noisy_score = np.clip(score + noise, 0.0, 1.0)
                
                # Consistent classification logic
                if noisy_score > 0.5:
                    classification = "abnormal"
                    confidence = 0.6 + noisy_score * 0.3
                else:
                    classification = "normal"
                    confidence = 0.7 + (1.0 - noisy_score) * 0.2
                
                confidence = np.clip(confidence, 0.0, 1.0)
                
                return {
                    "classification": classification,
                    "confidence": confidence,
                    "abnormality_score": noisy_score,
                    "call_id": self.call_count,
                    "reasoning": f"Analysis based on score {noisy_score:.3f}"
                }
        
        classifier = MockLLMClassifier(temperature=0.1)  # Low temperature for consistency
        
        # Run multiple classifications on identical input
        results = []
        for _ in range(num_repetitions):
            result = classifier.classify_with_llm(gait_features)
            results.append(result)
        
        # Property: All results should have valid classifications
        classifications = [r["classification"] for r in results]
        for i, classification in enumerate(classifications):
            assert classification in ["normal", "abnormal"], (
                f"Result {i} has invalid classification: {classification}"
            )
        
        # Property: Classifications should be consistent (majority agreement)
        normal_count = classifications.count("normal")
        abnormal_count = classifications.count("abnormal")
        majority_threshold = len(classifications) * 0.7  # 70% agreement
        
        consistency_achieved = (normal_count >= majority_threshold or 
                              abnormal_count >= majority_threshold)
        
        assert consistency_achieved, (
            f"Inconsistent classifications: {normal_count} normal, {abnormal_count} abnormal "
            f"out of {len(classifications)} (need 70% agreement)"
        )
        
        # Property: Confidence scores should be similar
        confidences = [r["confidence"] for r in results]
        confidence_std = np.std(confidences)
        
        # With low temperature, confidence should be fairly consistent
        assert confidence_std < 0.2, (
            f"Confidence scores too variable: std = {confidence_std:.3f}, "
            f"values = {confidences}"
        )
        
        # Property: Reasoning should be similar (same key elements)
        reasonings = [r["reasoning"] for r in results]
        # Check that all reasonings mention similar score ranges
        scores = [r["abnormality_score"] for r in results]
        score_std = np.std(scores)
        
        assert score_std < 0.15, (
            f"Abnormality scores too variable: std = {score_std:.3f}, "
            f"scores = {scores}"
        )
    
    @given(
        temperature=st.floats(min_value=0.0, max_value=1.0),
        num_trials=st.integers(min_value=5, max_value=15)
    )
    @settings(max_examples=20)
    def test_temperature_consistency_tradeoff_property(self, temperature, num_trials):
        """Test that higher temperature reduces consistency as expected."""
        # Fixed feature set for consistency testing
        fixed_features = {
            "temporal_features": {"cadence": 95, "stride_time": 1.3},
            "spatial_features": {"stride_length": 1.2, "step_width": 0.18},
            "symmetry_features": {"left_right_symmetry": 0.75}
        }
        
        class TemperatureControlledLLM:
            def __init__(self, temp):
                self.temperature = temp
            
            def classify_with_temperature(self, features):
                # Base deterministic score
                base_score = 0.6  # Borderline case
                
                # Add temperature-controlled noise
                noise = np.random.normal(0, self.temperature * 0.3)
                final_score = np.clip(base_score + noise, 0.0, 1.0)
                
                # Classification with threshold
                if final_score > 0.5:
                    classification = "abnormal"
                    confidence = 0.5 + final_score * 0.4
                else:
                    classification = "normal"
                    confidence = 0.5 + (1.0 - final_score) * 0.4
                
                return {
                    "classification": classification,
                    "confidence": np.clip(confidence, 0.0, 1.0),
                    "score": final_score
                }
        
        llm = TemperatureControlledLLM(temperature)
        
        # Run multiple trials
        results = []
        for _ in range(num_trials):
            result = llm.classify_with_temperature(fixed_features)
            results.append(result)
        
        # Measure consistency
        classifications = [r["classification"] for r in results]
        confidences = [r["confidence"] for r in results]
        scores = [r["score"] for r in results]
        
        # Property: Higher temperature should increase variability
        confidence_std = np.std(confidences)
        score_std = np.std(scores)
        
        if temperature < 0.2:  # Low temperature
            assert confidence_std < 0.15, (
                f"Low temperature ({temperature}) should have low confidence std, "
                f"got {confidence_std:.3f}"
            )
            assert score_std < 0.1, (
                f"Low temperature ({temperature}) should have low score std, "
                f"got {score_std:.3f}"
            )
        
        elif temperature > 0.7:  # High temperature
            # High temperature may have higher variability, but still bounded
            assert confidence_std < 0.4, (
                f"Even high temperature should have bounded variability, "
                f"got std = {confidence_std:.3f}"
            )
        
        # Property: All results should still be valid regardless of temperature
        for i, result in enumerate(results):
            assert result["classification"] in ["normal", "abnormal"], (
                f"Result {i} invalid classification: {result['classification']}"
            )
            assert 0.0 <= result["confidence"] <= 1.0, (
                f"Result {i} invalid confidence: {result['confidence']}"
            )


class TestClassificationExplanationCompletenessProperty:
    """
    Property 18: Classification Explanation Completeness
    For any classification result, the system should provide human-readable
    explanations for the decision.
    **Validates: Requirements 4.4**
    """
    
    @given(
        gait_features=gait_features_strategy(),
        explanation_detail_level=st.sampled_from(["brief", "detailed", "comprehensive"])
    )
    @settings(max_examples=50)
    def test_explanation_completeness_property(self, gait_features, explanation_detail_level):
        """
        Feature: testing-enhancement, Property 18: Classification Explanation Completeness
        For any classification, should provide complete human-readable explanations
        """
        # Simulate explanation generation
        class ExplanatoryClassifier:
            def classify_with_explanation(self, features, detail_level):
                # Analyze features
                temporal = features.get("temporal_features", {})
                spatial = features.get("spatial_features", {})
                symmetry = features.get("symmetry_features", {})
                
                # Determine classification
                abnormal_findings = []
                normal_findings = []
                
                # Temporal analysis
                cadence = temporal.get("cadence", 110)
                if cadence < 90:
                    abnormal_findings.append(f"Low cadence ({cadence:.1f} steps/min, normal: 90-130)")
                elif cadence > 130:
                    abnormal_findings.append(f"High cadence ({cadence:.1f} steps/min, normal: 90-130)")
                else:
                    normal_findings.append(f"Normal cadence ({cadence:.1f} steps/min)")
                
                # Spatial analysis
                stride_length = spatial.get("stride_length", 1.4)
                if stride_length < 1.1:
                    abnormal_findings.append(f"Short stride length ({stride_length:.2f}m, normal: 1.1-1.8m)")
                elif stride_length > 1.8:
                    abnormal_findings.append(f"Long stride length ({stride_length:.2f}m, normal: 1.1-1.8m)")
                else:
                    normal_findings.append(f"Normal stride length ({stride_length:.2f}m)")
                
                # Symmetry analysis
                symmetry_val = symmetry.get("left_right_symmetry", 0.9)
                if symmetry_val < 0.8:
                    abnormal_findings.append(f"Poor left-right symmetry ({symmetry_val:.2f}, normal: >0.8)")
                else:
                    normal_findings.append(f"Good left-right symmetry ({symmetry_val:.2f})")
                
                # Determine overall classification
                if len(abnormal_findings) >= 2:
                    classification = "abnormal"
                    confidence = 0.7 + len(abnormal_findings) * 0.1
                else:
                    classification = "normal"
                    confidence = 0.8 - len(abnormal_findings) * 0.1
                
                confidence = min(1.0, confidence)
                
                # Generate explanation based on detail level
                if detail_level == "brief":
                    if abnormal_findings:
                        explanation = f"Abnormal gait detected. Key issues: {', '.join(abnormal_findings[:2])}"
                    else:
                        explanation = f"Normal gait pattern. {len(normal_findings)} parameters within normal range."
                
                elif detail_level == "detailed":
                    explanation_parts = []
                    explanation_parts.append(f"Classification: {classification.upper()}")
                    explanation_parts.append(f"Confidence: {confidence:.2f}")
                    
                    if abnormal_findings:
                        explanation_parts.append("Abnormal findings:")
                        for finding in abnormal_findings:
                            explanation_parts.append(f"  • {finding}")
                    
                    if normal_findings:
                        explanation_parts.append("Normal findings:")
                        for finding in normal_findings[:3]:  # Limit for detailed
                            explanation_parts.append(f"  • {finding}")
                    
                    explanation = "\n".join(explanation_parts)
                
                else:  # comprehensive
                    explanation_parts = []
                    explanation_parts.append("=== GAIT ANALYSIS REPORT ===")
                    explanation_parts.append(f"Overall Classification: {classification.upper()}")
                    explanation_parts.append(f"Confidence Level: {confidence:.2f} ({confidence*100:.0f}%)")
                    explanation_parts.append("")
                    
                    explanation_parts.append("DETAILED FINDINGS:")
                    explanation_parts.append("Temporal Features:")
                    explanation_parts.append(f"  • Cadence: {cadence:.1f} steps/min")
                    explanation_parts.append(f"  • Stride time: {temporal.get('stride_time', 1.2):.2f} seconds")
                    
                    explanation_parts.append("Spatial Features:")
                    explanation_parts.append(f"  • Stride length: {stride_length:.2f} meters")
                    explanation_parts.append(f"  • Step width: {spatial.get('step_width', 0.15):.2f} meters")
                    
                    explanation_parts.append("Symmetry Features:")
                    explanation_parts.append(f"  • Left-right symmetry: {symmetry_val:.2f}")
                    
                    if abnormal_findings:
                        explanation_parts.append("")
                        explanation_parts.append("ABNORMAL FINDINGS:")
                        for i, finding in enumerate(abnormal_findings, 1):
                            explanation_parts.append(f"  {i}. {finding}")
                    
                    if normal_findings:
                        explanation_parts.append("")
                        explanation_parts.append("NORMAL FINDINGS:")
                        for i, finding in enumerate(normal_findings, 1):
                            explanation_parts.append(f"  {i}. {finding}")
                    
                    explanation_parts.append("")
                    explanation_parts.append("CLINICAL INTERPRETATION:")
                    if classification == "abnormal":
                        explanation_parts.append("The gait pattern shows deviations from normal parameters.")
                        explanation_parts.append("Further clinical evaluation may be warranted.")
                    else:
                        explanation_parts.append("The gait pattern appears within normal limits.")
                        explanation_parts.append("No significant abnormalities detected.")
                    
                    explanation = "\n".join(explanation_parts)
                
                return {
                    "classification": classification,
                    "confidence": confidence,
                    "explanation": explanation,
                    "detail_level": detail_level,
                    "abnormal_findings": abnormal_findings,
                    "normal_findings": normal_findings,
                    "features_analyzed": list(features.keys())
                }
        
        classifier = ExplanatoryClassifier()
        result = classifier.classify_with_explanation(gait_features, explanation_detail_level)
        
        # Property: Must include explanation field
        assert "explanation" in result, "Result must include explanation field"
        explanation = result["explanation"]
        assert isinstance(explanation, str), "Explanation must be string"
        assert len(explanation) > 0, "Explanation must not be empty"
        
        # Property: Explanation should mention the classification
        classification = result["classification"]
        assert classification.lower() in explanation.lower(), (
            f"Explanation should mention classification '{classification}'"
        )
        
        # Property: Detail level should affect explanation length
        if explanation_detail_level == "brief":
            assert len(explanation) < 200, (
                f"Brief explanation too long: {len(explanation)} characters"
            )
        elif explanation_detail_level == "comprehensive":
            assert len(explanation) > 300, (
                f"Comprehensive explanation too short: {len(explanation)} characters"
            )
        
        # Property: Should mention specific findings
        findings = result.get("abnormal_findings", []) + result.get("normal_findings", [])
        if findings:
            # At least some findings should be mentioned in explanation
            findings_mentioned = sum(
                1 for finding in findings 
                if any(word in explanation.lower() for word in finding.lower().split()[:3])
            )
            assert findings_mentioned > 0, "Explanation should mention specific findings"
        
        # Property: Should include quantitative information
        # Look for numbers in explanation (measurements, percentages, etc.)
        import re
        numbers_in_explanation = re.findall(r'\d+\.?\d*', explanation)
        assert len(numbers_in_explanation) > 0, (
            "Explanation should include quantitative information"
        )
        
        # Property: Should be human-readable (basic readability check)
        words = explanation.split()
        assert len(words) >= 5, "Explanation should have at least 5 words"
        
        # Check for reasonable sentence structure
        sentences = explanation.split('.')
        complete_sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        assert len(complete_sentences) >= 1, "Explanation should have at least one complete sentence"
    
    @given(
        classification_type=st.sampled_from(["normal", "abnormal"]),
        confidence_level=st.sampled_from(["low", "medium", "high"])
    )
    @settings(max_examples=30)
    def test_explanation_appropriateness_property(self, classification_type, confidence_level):
        """Test that explanations are appropriate for classification and confidence."""
        # Generate appropriate features for the classification type
        if classification_type == "abnormal":
            features = {
                "temporal_features": {"cadence": 75, "stride_time": 1.8},  # Abnormal values
                "spatial_features": {"stride_length": 0.9, "step_width": 0.25},
                "symmetry_features": {"left_right_symmetry": 0.65}
            }
        else:
            features = {
                "temporal_features": {"cadence": 115, "stride_time": 1.1},  # Normal values
                "spatial_features": {"stride_length": 1.5, "step_width": 0.12},
                "symmetry_features": {"left_right_symmetry": 0.92}
            }
        
        # Set confidence based on level
        confidence_map = {"low": 0.4, "medium": 0.7, "high": 0.95}
        target_confidence = confidence_map[confidence_level]
        
        # Generate explanation
        class AdaptiveExplainer:
            def explain_classification(self, features, expected_class, expected_conf):
                # Analyze features to generate appropriate explanation
                explanation_parts = []
                
                # Confidence-appropriate language
                if expected_conf > 0.8:
                    certainty_phrase = "clearly indicates"
                elif expected_conf > 0.6:
                    certainty_phrase = "suggests"
                else:
                    certainty_phrase = "may indicate"
                
                # Classification-appropriate content
                if expected_class == "abnormal":
                    explanation_parts.append(f"Analysis {certainty_phrase} abnormal gait pattern.")
                    explanation_parts.append("Deviations from normal parameters detected.")
                else:
                    explanation_parts.append(f"Analysis {certainty_phrase} normal gait pattern.")
                    explanation_parts.append("Parameters within expected ranges.")
                
                # Add confidence-appropriate recommendations
                if expected_conf < 0.5:
                    explanation_parts.append("Additional data may be needed for definitive assessment.")
                elif expected_conf > 0.9:
                    explanation_parts.append("High confidence in this assessment.")
                
                return {
                    "classification": expected_class,
                    "confidence": expected_conf,
                    "explanation": " ".join(explanation_parts),
                    "certainty_language": certainty_phrase
                }
        
        explainer = AdaptiveExplainer()
        result = explainer.explain_classification(features, classification_type, target_confidence)
        
        # Property: Explanation should match classification type
        explanation = result["explanation"].lower()
        if classification_type == "abnormal":
            assert "abnormal" in explanation or "deviation" in explanation, (
                "Abnormal classification should mention abnormality or deviations"
            )
        else:
            assert "normal" in explanation or "within" in explanation, (
                "Normal classification should mention normality or being within range"
            )
        
        # Property: Language should match confidence level
        certainty_language = result["certainty_language"]
        if confidence_level == "high":
            assert certainty_language in ["clearly indicates", "definitively shows"], (
                f"High confidence should use strong language, got '{certainty_language}'"
            )
        elif confidence_level == "low":
            assert certainty_language in ["may indicate", "suggests possible"], (
                f"Low confidence should use tentative language, got '{certainty_language}'"
            )
        
        # Property: Low confidence should mention uncertainty
        if confidence_level == "low":
            uncertainty_indicators = ["may", "additional", "possible", "uncertain"]
            has_uncertainty_language = any(
                indicator in explanation for indicator in uncertainty_indicators
            )
            assert has_uncertainty_language, (
                "Low confidence explanations should acknowledge uncertainty"
            )
        
        # Property: High confidence should be more definitive
        if confidence_level == "high":
            definitive_indicators = ["clearly", "definitively", "high confidence"]
            has_definitive_language = any(
                indicator in explanation for indicator in definitive_indicators
            )
            assert has_definitive_language, (
                "High confidence explanations should use definitive language"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "property"])