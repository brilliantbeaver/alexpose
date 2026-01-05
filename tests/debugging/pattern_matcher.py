"""
Pattern matching for test failure analysis and historical tracking.

This module provides advanced pattern recognition for test failures,
including historical analysis and trend detection.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
from enum import Enum
import sqlite3


class PatternType(Enum):
    """Types of failure patterns."""
    ERROR_MESSAGE = "error_message"
    STACK_TRACE = "stack_trace"
    TEST_NAME = "test_name"
    ENVIRONMENT = "environment"
    TIMING = "timing"
    RESOURCE = "resource"


@dataclass
class FailurePattern:
    """A pattern identified in test failures."""
    pattern_id: str
    pattern_type: PatternType
    pattern_regex: str
    description: str
    frequency: int
    first_seen: str
    last_seen: str
    affected_tests: List[str]
    confidence: float
    suggested_action: str


@dataclass
class PatternMatch:
    """A match of a pattern against a failure."""
    pattern_id: str
    pattern_type: PatternType
    match_confidence: float
    matched_text: str
    context: Dict[str, Any]


@dataclass
class HistoricalAnalysis:
    """Historical analysis of failure patterns."""
    analysis_period: str
    total_failures: int
    unique_patterns: int
    trending_patterns: List[str]
    resolved_patterns: List[str]
    new_patterns: List[str]
    pattern_frequency: Dict[str, int]
    recommendations: List[str]


class PatternMatcher:
    """Advanced pattern matching for test failure analysis."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("test_patterns.db")
        self.patterns: Dict[str, FailurePattern] = {}
        self.failure_history: List[Dict[str, Any]] = []
        self._init_database()
        self._load_patterns()
    
    def _init_database(self):
        """Initialize SQLite database for pattern storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT,
                pattern_regex TEXT,
                description TEXT,
                frequency INTEGER,
                first_seen TEXT,
                last_seen TEXT,
                affected_tests TEXT,
                confidence REAL,
                suggested_action TEXT
            )
        """)
        
        # Create failures table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT,
                error_message TEXT,
                stack_trace TEXT,
                timestamp TEXT,
                environment TEXT,
                pattern_matches TEXT
            )
        """)
        
        # Create pattern matches table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                failure_id INTEGER,
                pattern_id TEXT,
                match_confidence REAL,
                matched_text TEXT,
                context TEXT,
                FOREIGN KEY (failure_id) REFERENCES failures (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_patterns(self):
        """Load patterns from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM patterns")
        rows = cursor.fetchall()
        
        for row in rows:
            pattern = FailurePattern(
                pattern_id=row[0],
                pattern_type=PatternType(row[1]),
                pattern_regex=row[2],
                description=row[3],
                frequency=row[4],
                first_seen=row[5],
                last_seen=row[6],
                affected_tests=json.loads(row[7]),
                confidence=row[8],
                suggested_action=row[9]
            )
            self.patterns[pattern.pattern_id] = pattern
        
        conn.close()
    
    def add_failure(
        self,
        test_name: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        environment: Optional[Dict[str, Any]] = None
    ) -> List[PatternMatch]:
        """Add a failure and find matching patterns."""
        timestamp = datetime.now().isoformat()
        
        # Store failure in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO failures (test_name, error_message, stack_trace, timestamp, environment)
            VALUES (?, ?, ?, ?, ?)
        """, (
            test_name,
            error_message,
            stack_trace or "",
            timestamp,
            json.dumps(environment or {})
        ))
        
        failure_id = cursor.lastrowid
        conn.commit()
        
        # Find matching patterns
        matches = self.find_matches(
            test_name=test_name,
            error_message=error_message,
            stack_trace=stack_trace,
            environment=environment
        )
        
        # Store pattern matches
        for match in matches:
            cursor.execute("""
                INSERT INTO pattern_matches (failure_id, pattern_id, match_confidence, matched_text, context)
                VALUES (?, ?, ?, ?, ?)
            """, (
                failure_id,
                match.pattern_id,
                match.match_confidence,
                match.matched_text,
                json.dumps(match.context)
            ))
        
        conn.commit()
        conn.close()
        
        # Update pattern frequencies and last seen
        self._update_pattern_statistics(matches, timestamp)
        
        # Check for new patterns
        if not matches:
            self._detect_new_pattern(test_name, error_message, stack_trace, timestamp)
        
        return matches
    
    def find_matches(
        self,
        test_name: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        environment: Optional[Dict[str, Any]] = None
    ) -> List[PatternMatch]:
        """Find patterns that match the given failure."""
        matches = []
        
        for pattern in self.patterns.values():
            match = self._match_pattern(
                pattern, test_name, error_message, stack_trace, environment
            )
            if match:
                matches.append(match)
        
        # Sort by confidence
        matches.sort(key=lambda m: m.match_confidence, reverse=True)
        return matches
    
    def _match_pattern(
        self,
        pattern: FailurePattern,
        test_name: str,
        error_message: str,
        stack_trace: Optional[str],
        environment: Optional[Dict[str, Any]]
    ) -> Optional[PatternMatch]:
        """Check if a pattern matches the failure."""
        confidence = 0.0
        matched_text = ""
        context = {}
        
        if pattern.pattern_type == PatternType.ERROR_MESSAGE:
            match = re.search(pattern.pattern_regex, error_message, re.IGNORECASE)
            if match:
                confidence = 0.9
                matched_text = match.group(0)
                context = {"full_message": error_message}
        
        elif pattern.pattern_type == PatternType.STACK_TRACE and stack_trace:
            match = re.search(pattern.pattern_regex, stack_trace, re.IGNORECASE)
            if match:
                confidence = 0.8
                matched_text = match.group(0)
                context = {"stack_trace_lines": stack_trace.split('\n')[:5]}
        
        elif pattern.pattern_type == PatternType.TEST_NAME:
            match = re.search(pattern.pattern_regex, test_name, re.IGNORECASE)
            if match:
                confidence = 0.7
                matched_text = match.group(0)
                context = {"test_name": test_name}
        
        elif pattern.pattern_type == PatternType.ENVIRONMENT and environment:
            # Check environment patterns
            env_str = json.dumps(environment)
            match = re.search(pattern.pattern_regex, env_str, re.IGNORECASE)
            if match:
                confidence = 0.6
                matched_text = match.group(0)
                context = {"environment": environment}
        
        # Adjust confidence based on pattern history
        if confidence > 0:
            # Higher confidence for frequently seen patterns
            frequency_boost = min(0.1, pattern.frequency / 100)
            confidence = min(1.0, confidence + frequency_boost)
            
            return PatternMatch(
                pattern_id=pattern.pattern_id,
                pattern_type=pattern.pattern_type,
                match_confidence=confidence,
                matched_text=matched_text,
                context=context
            )
        
        return None
    
    def _update_pattern_statistics(self, matches: List[PatternMatch], timestamp: str):
        """Update pattern statistics based on matches."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for match in matches:
            pattern = self.patterns.get(match.pattern_id)
            if pattern:
                pattern.frequency += 1
                pattern.last_seen = timestamp
                
                # Update in database
                cursor.execute("""
                    UPDATE patterns 
                    SET frequency = ?, last_seen = ?
                    WHERE pattern_id = ?
                """, (pattern.frequency, timestamp, pattern.pattern_id))
        
        conn.commit()
        conn.close()
    
    def _detect_new_pattern(
        self,
        test_name: str,
        error_message: str,
        stack_trace: Optional[str],
        timestamp: str
    ):
        """Detect and create new patterns from unmatched failures."""
        # Simple pattern detection - look for common error patterns
        new_patterns = []
        
        # Common error message patterns
        error_patterns = [
            (r"AssertionError: (.+)", "Assertion failure"),
            (r"TypeError: (.+)", "Type error"),
            (r"ValueError: (.+)", "Value error"),
            (r"FileNotFoundError: (.+)", "File not found"),
            (r"ConnectionError: (.+)", "Connection error"),
            (r"TimeoutError: (.+)", "Timeout error"),
            (r"MemoryError: (.+)", "Memory error")
        ]
        
        for pattern_regex, description in error_patterns:
            match = re.search(pattern_regex, error_message, re.IGNORECASE)
            if match:
                pattern_id = f"auto_{hash(pattern_regex) % 10000:04d}"
                
                if pattern_id not in self.patterns:
                    new_pattern = FailurePattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType.ERROR_MESSAGE,
                        pattern_regex=pattern_regex,
                        description=f"Auto-detected: {description}",
                        frequency=1,
                        first_seen=timestamp,
                        last_seen=timestamp,
                        affected_tests=[test_name],
                        confidence=0.7,
                        suggested_action=f"Investigate {description.lower()} in test implementation"
                    )
                    
                    self.patterns[pattern_id] = new_pattern
                    self._save_pattern(new_pattern)
                    new_patterns.append(new_pattern)
        
        return new_patterns
    
    def _save_pattern(self, pattern: FailurePattern):
        """Save pattern to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO patterns 
            (pattern_id, pattern_type, pattern_regex, description, frequency, 
             first_seen, last_seen, affected_tests, confidence, suggested_action)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pattern.pattern_id,
            pattern.pattern_type.value,
            pattern.pattern_regex,
            pattern.description,
            pattern.frequency,
            pattern.first_seen,
            pattern.last_seen,
            json.dumps(pattern.affected_tests),
            pattern.confidence,
            pattern.suggested_action
        ))
        
        conn.commit()
        conn.close()
    
    def get_historical_analysis(self, days: int = 30) -> HistoricalAnalysis:
        """Get historical analysis of failure patterns."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get failures in the period
        cursor.execute("""
            SELECT COUNT(*) FROM failures WHERE timestamp >= ?
        """, (cutoff_date,))
        total_failures = cursor.fetchone()[0]
        
        # Get pattern matches in the period
        cursor.execute("""
            SELECT pm.pattern_id, COUNT(*) as frequency
            FROM pattern_matches pm
            JOIN failures f ON pm.failure_id = f.id
            WHERE f.timestamp >= ?
            GROUP BY pm.pattern_id
            ORDER BY frequency DESC
        """, (cutoff_date,))
        
        pattern_frequency = dict(cursor.fetchall())
        unique_patterns = len(pattern_frequency)
        
        # Identify trending patterns (increased frequency in recent period)
        recent_cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        cursor.execute("""
            SELECT pm.pattern_id, COUNT(*) as recent_frequency
            FROM pattern_matches pm
            JOIN failures f ON pm.failure_id = f.id
            WHERE f.timestamp >= ?
            GROUP BY pm.pattern_id
        """, (recent_cutoff,))
        
        recent_frequency = dict(cursor.fetchall())
        
        # Calculate trending patterns
        trending_patterns = []
        for pattern_id, total_freq in pattern_frequency.items():
            recent_freq = recent_frequency.get(pattern_id, 0)
            if total_freq > 0:
                recent_ratio = recent_freq / total_freq
                if recent_ratio > 0.5 and recent_freq > 2:  # More than half in recent week
                    trending_patterns.append(pattern_id)
        
        # Find new patterns (first seen in recent period)
        new_patterns = []
        for pattern_id, pattern in self.patterns.items():
            if pattern.first_seen >= cutoff_date:
                new_patterns.append(pattern_id)
        
        # Find resolved patterns (not seen recently)
        resolved_patterns = []
        for pattern_id in pattern_frequency:
            if pattern_id not in recent_frequency:
                resolved_patterns.append(pattern_id)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            pattern_frequency, trending_patterns, new_patterns
        )
        
        conn.close()
        
        return HistoricalAnalysis(
            analysis_period=f"Last {days} days",
            total_failures=total_failures,
            unique_patterns=unique_patterns,
            trending_patterns=trending_patterns,
            resolved_patterns=resolved_patterns,
            new_patterns=new_patterns,
            pattern_frequency=pattern_frequency,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        pattern_frequency: Dict[str, int],
        trending_patterns: List[str],
        new_patterns: List[str]
    ) -> List[str]:
        """Generate recommendations based on pattern analysis."""
        recommendations = []
        
        # Most frequent patterns
        if pattern_frequency:
            most_frequent = max(pattern_frequency.items(), key=lambda x: x[1])
            pattern = self.patterns.get(most_frequent[0])
            if pattern:
                recommendations.append(
                    f"Address most frequent pattern: {pattern.description} "
                    f"({most_frequent[1]} occurrences) - {pattern.suggested_action}"
                )
        
        # Trending patterns
        if trending_patterns:
            recommendations.append(
                f"Investigate trending patterns: {', '.join(trending_patterns[:3])} "
                "- these are increasing in frequency"
            )
        
        # New patterns
        if new_patterns:
            recommendations.append(
                f"Monitor new patterns: {', '.join(new_patterns[:3])} "
                "- these are recently detected issues"
            )
        
        # General recommendations
        if len(pattern_frequency) > 10:
            recommendations.append(
                "High pattern diversity detected - consider improving test stability"
            )
        
        if not recommendations:
            recommendations.append("No specific recommendations - test patterns appear stable")
        
        return recommendations
    
    def add_custom_pattern(
        self,
        pattern_id: str,
        pattern_type: PatternType,
        pattern_regex: str,
        description: str,
        suggested_action: str,
        confidence: float = 0.8
    ) -> FailurePattern:
        """Add a custom pattern for matching."""
        pattern = FailurePattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            pattern_regex=pattern_regex,
            description=description,
            frequency=0,
            first_seen=datetime.now().isoformat(),
            last_seen=datetime.now().isoformat(),
            affected_tests=[],
            confidence=confidence,
            suggested_action=suggested_action
        )
        
        self.patterns[pattern_id] = pattern
        self._save_pattern(pattern)
        return pattern
    
    def get_pattern_report(self) -> Dict[str, Any]:
        """Get comprehensive pattern analysis report."""
        historical = self.get_historical_analysis()
        
        # Top patterns by frequency
        top_patterns = []
        for pattern_id, frequency in sorted(
            historical.pattern_frequency.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]:
            pattern = self.patterns.get(pattern_id)
            if pattern:
                top_patterns.append({
                    "pattern_id": pattern_id,
                    "description": pattern.description,
                    "frequency": frequency,
                    "suggested_action": pattern.suggested_action
                })
        
        return {
            "summary": asdict(historical),
            "top_patterns": top_patterns,
            "total_patterns": len(self.patterns),
            "pattern_types": {
                ptype.value: sum(1 for p in self.patterns.values() if p.pattern_type == ptype)
                for ptype in PatternType
            }
        }
    
    def export_patterns(self, output_path: Path):
        """Export patterns to JSON file."""
        patterns_data = {
            "patterns": [asdict(pattern) for pattern in self.patterns.values()],
            "exported_at": datetime.now().isoformat()
        }
        
        output_path.write_text(json.dumps(patterns_data, indent=2))
    
    def import_patterns(self, input_path: Path):
        """Import patterns from JSON file."""
        data = json.loads(input_path.read_text())
        
        for pattern_data in data.get("patterns", []):
            pattern = FailurePattern(
                pattern_id=pattern_data["pattern_id"],
                pattern_type=PatternType(pattern_data["pattern_type"]),
                pattern_regex=pattern_data["pattern_regex"],
                description=pattern_data["description"],
                frequency=pattern_data["frequency"],
                first_seen=pattern_data["first_seen"],
                last_seen=pattern_data["last_seen"],
                affected_tests=pattern_data["affected_tests"],
                confidence=pattern_data["confidence"],
                suggested_action=pattern_data["suggested_action"]
            )
            
            self.patterns[pattern.pattern_id] = pattern
            self._save_pattern(pattern)


# Global pattern matcher instance
pattern_matcher = PatternMatcher()


def match_failure_patterns(
    test_name: str,
    error_message: str,
    stack_trace: Optional[str] = None,
    environment: Optional[Dict[str, Any]] = None
) -> List[PatternMatch]:
    """Match failure patterns using the global matcher."""
    return pattern_matcher.add_failure(
        test_name=test_name,
        error_message=error_message,
        stack_trace=stack_trace,
        environment=environment
    )


def get_pattern_matcher() -> PatternMatcher:
    """Get the global pattern matcher instance."""
    return pattern_matcher