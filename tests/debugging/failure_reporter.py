"""
Comprehensive failure reporting and tracking system.

This module provides automated failure reporting, tracking, and
management capabilities for test failures across the system.
"""

import json
import sqlite3
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
from enum import Enum
import hashlib
import logging
import subprocess
import tempfile

from .artifact_collector import collect_test_artifacts
from .pattern_matcher import match_failure_patterns
from ..property.failure_analyzer import analyze_property_failure


class FailureStatus(Enum):
    """Status of a failure report."""
    NEW = "new"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    FIXING = "fixing"
    RESOLVED = "resolved"
    CLOSED = "closed"
    WONT_FIX = "wont_fix"


class FailurePriority(Enum):
    """Priority levels for failure reports."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FailureReport:
    """Comprehensive failure report."""
    report_id: str
    test_name: str
    failure_hash: str
    first_seen: str
    last_seen: str
    occurrence_count: int
    status: FailureStatus
    priority: FailurePriority
    error_message: str
    stack_trace: Optional[str]
    failure_category: str
    root_cause: Optional[str]
    suggested_fixes: List[str]
    assigned_to: Optional[str]
    artifacts: List[str]
    pattern_matches: List[str]
    environment_info: Dict[str, Any]
    reproduction_steps: List[str]
    related_failures: List[str]
    resolution_notes: Optional[str]
    created_at: str
    updated_at: str


@dataclass
class FailureTrend:
    """Failure trend analysis."""
    period: str
    total_failures: int
    unique_failures: int
    resolved_failures: int
    new_failures: int
    trending_up: List[str]
    trending_down: List[str]
    most_frequent: List[Tuple[str, int]]
    resolution_rate: float


class FailureReporter:
    """Comprehensive failure reporting and tracking system."""
    
    def __init__(self, db_path: Optional[Path] = None, config: Optional[Dict[str, Any]] = None):
        self.db_path = db_path or Path("failure_reports.db")
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Failure tracking
        self.active_reports: Dict[str, FailureReport] = {}
        self.failure_cache: Dict[str, str] = {}  # failure_hash -> report_id
        
        # Configuration
        self.auto_assign_rules = self.config.get("auto_assign_rules", {})
        self.priority_rules = self.config.get("priority_rules", {})
        self.notification_config = self.config.get("notifications", {})
        
        self._init_database()
        self._load_active_reports()
    
    def _init_database(self):
        """Initialize failure reporting database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Failure reports table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS failure_reports (
                report_id TEXT PRIMARY KEY,
                test_name TEXT,
                failure_hash TEXT,
                first_seen TEXT,
                last_seen TEXT,
                occurrence_count INTEGER,
                status TEXT,
                priority TEXT,
                error_message TEXT,
                stack_trace TEXT,
                failure_category TEXT,
                root_cause TEXT,
                suggested_fixes TEXT,
                assigned_to TEXT,
                artifacts TEXT,
                pattern_matches TEXT,
                environment_info TEXT,
                reproduction_steps TEXT,
                related_failures TEXT,
                resolution_notes TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Failure occurrences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS failure_occurrences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT,
                timestamp TEXT,
                test_execution_id TEXT,
                environment_snapshot TEXT,
                artifacts TEXT,
                FOREIGN KEY (report_id) REFERENCES failure_reports (report_id)
            )
        """)
        
        # Failure assignments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS failure_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT,
                assigned_to TEXT,
                assigned_by TEXT,
                assigned_at TEXT,
                notes TEXT,
                FOREIGN KEY (report_id) REFERENCES failure_reports (report_id)
            )
        """)
        
        # Failure comments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS failure_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT,
                author TEXT,
                comment TEXT,
                timestamp TEXT,
                FOREIGN KEY (report_id) REFERENCES failure_reports (report_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_active_reports(self):
        """Load active failure reports from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM failure_reports 
            WHERE status NOT IN ('resolved', 'closed', 'wont_fix')
        """)
        
        for row in cursor.fetchall():
            report = self._row_to_report(row)
            self.active_reports[report.report_id] = report
            self.failure_cache[report.failure_hash] = report.report_id
        
        conn.close()
    
    def _row_to_report(self, row) -> FailureReport:
        """Convert database row to FailureReport."""
        return FailureReport(
            report_id=row[0],
            test_name=row[1],
            failure_hash=row[2],
            first_seen=row[3],
            last_seen=row[4],
            occurrence_count=row[5],
            status=FailureStatus(row[6]),
            priority=FailurePriority(row[7]),
            error_message=row[8],
            stack_trace=row[9],
            failure_category=row[10],
            root_cause=row[11],
            suggested_fixes=json.loads(row[12]) if row[12] else [],
            assigned_to=row[13],
            artifacts=json.loads(row[14]) if row[14] else [],
            pattern_matches=json.loads(row[15]) if row[15] else [],
            environment_info=json.loads(row[16]) if row[16] else {},
            reproduction_steps=json.loads(row[17]) if row[17] else [],
            related_failures=json.loads(row[18]) if row[18] else [],
            resolution_notes=row[19],
            created_at=row[20],
            updated_at=row[21]
        )
    
    def report_failure(
        self,
        test_name: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        test_execution_id: Optional[str] = None
    ) -> str:
        """Report a test failure and return report ID."""
        timestamp = datetime.now().isoformat()
        
        # Generate failure hash for deduplication
        failure_hash = self._generate_failure_hash(test_name, error_message, stack_trace)
        
        # Check if this failure already exists
        if failure_hash in self.failure_cache:
            report_id = self.failure_cache[failure_hash]
            self._update_existing_report(report_id, timestamp, test_execution_id, environment_info)
            return report_id
        
        # Create new failure report
        report_id = f"FR_{int(datetime.now().timestamp())}_{hash(failure_hash) % 10000:04d}"
        
        # Collect artifacts
        artifacts = []
        try:
            artifact_result = collect_test_artifacts(
                test_name=test_name,
                error_info={
                    "error_message": error_message,
                    "stack_trace": stack_trace,
                    "timestamp": timestamp
                }
            )
            artifacts = artifact_result.file_artifacts
        except Exception as e:
            self.logger.warning(f"Failed to collect artifacts for {test_name}: {e}")
        
        # Analyze failure patterns
        pattern_matches = []
        try:
            matches = match_failure_patterns(
                test_name=test_name,
                error_message=error_message,
                stack_trace=stack_trace,
                environment=environment_info
            )
            pattern_matches = [match.pattern_id for match in matches]
        except Exception as e:
            self.logger.warning(f"Failed to match patterns for {test_name}: {e}")
        
        # Analyze with property failure analyzer
        root_cause = None
        suggested_fixes = []
        failure_category = "unknown"
        
        try:
            analysis = analyze_property_failure(
                property_name=test_name,
                error_message=error_message,
                exception_type=self._extract_exception_type(error_message),
                traceback_str=stack_trace
            )
            root_cause = analysis.root_cause
            suggested_fixes = analysis.suggested_fixes
            failure_category = analysis.failure_category.value
        except Exception as e:
            self.logger.warning(f"Failed to analyze failure for {test_name}: {e}")
        
        # Determine priority
        priority = self._determine_priority(test_name, error_message, failure_category)
        
        # Auto-assign if rules exist
        assigned_to = self._auto_assign(test_name, failure_category, priority)
        
        # Generate reproduction steps
        reproduction_steps = self._generate_reproduction_steps(
            test_name, error_message, stack_trace, environment_info
        )
        
        # Create failure report
        report = FailureReport(
            report_id=report_id,
            test_name=test_name,
            failure_hash=failure_hash,
            first_seen=timestamp,
            last_seen=timestamp,
            occurrence_count=1,
            status=FailureStatus.NEW,
            priority=priority,
            error_message=error_message,
            stack_trace=stack_trace,
            failure_category=failure_category,
            root_cause=root_cause,
            suggested_fixes=suggested_fixes,
            assigned_to=assigned_to,
            artifacts=artifacts,
            pattern_matches=pattern_matches,
            environment_info=environment_info or {},
            reproduction_steps=reproduction_steps,
            related_failures=[],
            resolution_notes=None,
            created_at=timestamp,
            updated_at=timestamp
        )
        
        # Store in database and cache
        self._store_report(report)
        self.active_reports[report_id] = report
        self.failure_cache[failure_hash] = report_id
        
        # Record occurrence
        self._record_occurrence(report_id, timestamp, test_execution_id, environment_info)
        
        # Find related failures
        self._find_related_failures(report)
        
        # Send notifications
        self._send_notifications(report, is_new=True)
        
        self.logger.info(f"Created failure report {report_id} for test {test_name}")
        return report_id
    
    def _generate_failure_hash(
        self,
        test_name: str,
        error_message: str,
        stack_trace: Optional[str]
    ) -> str:
        """Generate a hash for failure deduplication."""
        # Normalize error message (remove dynamic parts)
        normalized_error = self._normalize_error_message(error_message)
        
        # Use first few lines of stack trace for more specific matching
        stack_signature = ""
        if stack_trace:
            lines = stack_trace.split('\n')[:5]  # First 5 lines
            stack_signature = '\n'.join(lines)
        
        # Create hash
        content = f"{test_name}|{normalized_error}|{stack_signature}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _normalize_error_message(self, error_message: str) -> str:
        """Normalize error message by removing dynamic parts."""
        import re
        
        # Remove timestamps
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', '<TIMESTAMP>', error_message)
        
        # Remove file paths
        normalized = re.sub(r'/[^\s]+\.py', '<FILEPATH>', normalized)
        
        # Remove line numbers
        normalized = re.sub(r'line \d+', 'line <NUM>', normalized)
        
        # Remove memory addresses
        normalized = re.sub(r'0x[0-9a-fA-F]+', '<ADDR>', normalized)
        
        # Remove temporary file names
        normalized = re.sub(r'tmp\w+', '<TMPFILE>', normalized)
        
        return normalized
    
    def _extract_exception_type(self, error_message: str) -> str:
        """Extract exception type from error message."""
        import re
        match = re.match(r'^(\w+Error|\w+Exception)', error_message)
        return match.group(1) if match else "Exception"
    
    def _update_existing_report(
        self,
        report_id: str,
        timestamp: str,
        test_execution_id: Optional[str],
        environment_info: Optional[Dict[str, Any]]
    ):
        """Update an existing failure report with new occurrence."""
        if report_id not in self.active_reports:
            return
        
        report = self.active_reports[report_id]
        report.last_seen = timestamp
        report.occurrence_count += 1
        report.updated_at = timestamp
        
        # Update in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE failure_reports 
            SET last_seen = ?, occurrence_count = ?, updated_at = ?
            WHERE report_id = ?
        """, (timestamp, report.occurrence_count, timestamp, report_id))
        
        conn.commit()
        conn.close()
        
        # Record occurrence
        self._record_occurrence(report_id, timestamp, test_execution_id, environment_info)
        
        # Send notifications for recurring failures
        if report.occurrence_count % 5 == 0:  # Every 5th occurrence
            self._send_notifications(report, is_recurring=True)
    
    def _determine_priority(
        self,
        test_name: str,
        error_message: str,
        failure_category: str
    ) -> FailurePriority:
        """Determine failure priority based on rules."""
        # Check custom priority rules
        for rule in self.priority_rules.get("rules", []):
            if self._matches_rule(test_name, error_message, failure_category, rule):
                return FailurePriority(rule["priority"])
        
        # Default priority logic
        if "critical" in test_name.lower() or "critical" in error_message.lower():
            return FailurePriority.CRITICAL
        elif "integration" in test_name.lower():
            return FailurePriority.HIGH
        elif failure_category in ["timeout", "resource_error"]:
            return FailurePriority.HIGH
        elif failure_category in ["assertion_error", "type_error"]:
            return FailurePriority.MEDIUM
        else:
            return FailurePriority.LOW
    
    def _auto_assign(
        self,
        test_name: str,
        failure_category: str,
        priority: FailurePriority
    ) -> Optional[str]:
        """Auto-assign failure based on rules."""
        for rule in self.auto_assign_rules.get("rules", []):
            if self._matches_assignment_rule(test_name, failure_category, priority, rule):
                return rule["assignee"]
        
        return None
    
    def _matches_rule(
        self,
        test_name: str,
        error_message: str,
        failure_category: str,
        rule: Dict[str, Any]
    ) -> bool:
        """Check if a failure matches a priority rule."""
        if "test_pattern" in rule:
            import re
            if not re.search(rule["test_pattern"], test_name):
                return False
        
        if "error_pattern" in rule:
            import re
            if not re.search(rule["error_pattern"], error_message):
                return False
        
        if "category" in rule:
            if failure_category != rule["category"]:
                return False
        
        return True
    
    def _matches_assignment_rule(
        self,
        test_name: str,
        failure_category: str,
        priority: FailurePriority,
        rule: Dict[str, Any]
    ) -> bool:
        """Check if a failure matches an assignment rule."""
        if "test_pattern" in rule:
            import re
            if not re.search(rule["test_pattern"], test_name):
                return False
        
        if "category" in rule:
            if failure_category != rule["category"]:
                return False
        
        if "priority" in rule:
            if priority.value != rule["priority"]:
                return False
        
        return True
    
    def _generate_reproduction_steps(
        self,
        test_name: str,
        error_message: str,
        stack_trace: Optional[str],
        environment_info: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate reproduction steps for the failure."""
        steps = [
            f"1. Run test: {test_name}",
            f"2. Expected error: {error_message[:100]}..."
        ]
        
        if environment_info:
            steps.append("3. Environment requirements:")
            for key, value in environment_info.items():
                if key in ["python_version", "os", "dependencies"]:
                    steps.append(f"   - {key}: {value}")
        
        if stack_trace:
            steps.append("4. Check stack trace for specific failure location")
        
        steps.extend([
            "5. Verify test setup and dependencies",
            "6. Check for environment-specific issues"
        ])
        
        return steps
    
    def _store_report(self, report: FailureReport):
        """Store failure report in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO failure_reports 
            (report_id, test_name, failure_hash, first_seen, last_seen, occurrence_count,
             status, priority, error_message, stack_trace, failure_category, root_cause,
             suggested_fixes, assigned_to, artifacts, pattern_matches, environment_info,
             reproduction_steps, related_failures, resolution_notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report.report_id, report.test_name, report.failure_hash, report.first_seen,
            report.last_seen, report.occurrence_count, report.status.value, report.priority.value,
            report.error_message, report.stack_trace, report.failure_category, report.root_cause,
            json.dumps(report.suggested_fixes), report.assigned_to, json.dumps(report.artifacts),
            json.dumps(report.pattern_matches), json.dumps(report.environment_info),
            json.dumps(report.reproduction_steps), json.dumps(report.related_failures),
            report.resolution_notes, report.created_at, report.updated_at
        ))
        
        conn.commit()
        conn.close()
    
    def _record_occurrence(
        self,
        report_id: str,
        timestamp: str,
        test_execution_id: Optional[str],
        environment_info: Optional[Dict[str, Any]]
    ):
        """Record a failure occurrence."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO failure_occurrences 
            (report_id, timestamp, test_execution_id, environment_snapshot, artifacts)
            VALUES (?, ?, ?, ?, ?)
        """, (
            report_id, timestamp, test_execution_id,
            json.dumps(environment_info or {}), json.dumps([])
        ))
        
        conn.commit()
        conn.close()
    
    def _find_related_failures(self, report: FailureReport):
        """Find failures related to the given report."""
        related = []
        
        # Find failures with similar error messages
        for other_report in self.active_reports.values():
            if other_report.report_id == report.report_id:
                continue
            
            # Check for similar error messages
            similarity = self._calculate_similarity(report.error_message, other_report.error_message)
            if similarity > 0.7:
                related.append(other_report.report_id)
            
            # Check for same test file/module
            if self._same_test_module(report.test_name, other_report.test_name):
                related.append(other_report.report_id)
        
        if related:
            report.related_failures = related
            self._store_report(report)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        # Simple similarity based on common words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _same_test_module(self, test1: str, test2: str) -> bool:
        """Check if two tests are from the same module."""
        # Extract module name (everything before the last dot)
        module1 = '.'.join(test1.split('.')[:-1])
        module2 = '.'.join(test2.split('.')[:-1])
        
        return module1 == module2 and module1 != ""
    
    def _send_notifications(self, report: FailureReport, is_new: bool = False, is_recurring: bool = False):
        """Send notifications for failure report."""
        # This is a placeholder for notification logic
        # In practice, you'd integrate with email, Slack, etc.
        
        if is_new and report.priority in [FailurePriority.HIGH, FailurePriority.CRITICAL]:
            self.logger.warning(f"New {report.priority.value} priority failure: {report.report_id}")
        
        if is_recurring and report.occurrence_count >= 10:
            self.logger.error(f"Recurring failure {report.report_id} has occurred {report.occurrence_count} times")
    
    def update_report_status(
        self,
        report_id: str,
        status: FailureStatus,
        notes: Optional[str] = None,
        assigned_to: Optional[str] = None
    ):
        """Update failure report status."""
        if report_id not in self.active_reports:
            return False
        
        report = self.active_reports[report_id]
        report.status = status
        report.updated_at = datetime.now().isoformat()
        
        if notes:
            report.resolution_notes = notes
        
        if assigned_to:
            report.assigned_to = assigned_to
        
        # Update in database
        self._store_report(report)
        
        # Remove from active reports if resolved
        if status in [FailureStatus.RESOLVED, FailureStatus.CLOSED, FailureStatus.WONT_FIX]:
            del self.active_reports[report_id]
            if report.failure_hash in self.failure_cache:
                del self.failure_cache[report.failure_hash]
        
        return True
    
    def add_comment(self, report_id: str, author: str, comment: str):
        """Add a comment to a failure report."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO failure_comments (report_id, author, comment, timestamp)
            VALUES (?, ?, ?, ?)
        """, (report_id, author, comment, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_report(self, report_id: str) -> Optional[FailureReport]:
        """Get a failure report by ID."""
        if report_id in self.active_reports:
            return self.active_reports[report_id]
        
        # Check database for resolved reports
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM failure_reports WHERE report_id = ?", (report_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_report(row)
        
        return None
    
    def get_reports(
        self,
        status: Optional[FailureStatus] = None,
        priority: Optional[FailurePriority] = None,
        assigned_to: Optional[str] = None,
        test_name_pattern: Optional[str] = None,
        limit: int = 100
    ) -> List[FailureReport]:
        """Get failure reports with optional filtering."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM failure_reports WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        if priority:
            query += " AND priority = ?"
            params.append(priority.value)
        
        if assigned_to:
            query += " AND assigned_to = ?"
            params.append(assigned_to)
        
        if test_name_pattern:
            query += " AND test_name LIKE ?"
            params.append(f"%{test_name_pattern}%")
        
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_report(row) for row in rows]
    
    def get_failure_trends(self, days: int = 30) -> FailureTrend:
        """Get failure trend analysis."""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total failures in period
        cursor.execute("""
            SELECT COUNT(*) FROM failure_reports WHERE created_at >= ?
        """, (cutoff_date,))
        total_failures = cursor.fetchone()[0]
        
        # Unique failures
        cursor.execute("""
            SELECT COUNT(DISTINCT failure_hash) FROM failure_reports WHERE created_at >= ?
        """, (cutoff_date,))
        unique_failures = cursor.fetchone()[0]
        
        # Resolved failures
        cursor.execute("""
            SELECT COUNT(*) FROM failure_reports 
            WHERE created_at >= ? AND status IN ('resolved', 'closed')
        """, (cutoff_date,))
        resolved_failures = cursor.fetchone()[0]
        
        # New failures (first seen in period)
        cursor.execute("""
            SELECT COUNT(*) FROM failure_reports WHERE first_seen >= ?
        """, (cutoff_date,))
        new_failures = cursor.fetchone()[0]
        
        # Most frequent failures
        cursor.execute("""
            SELECT test_name, occurrence_count FROM failure_reports 
            WHERE last_seen >= ?
            ORDER BY occurrence_count DESC LIMIT 10
        """, (cutoff_date,))
        most_frequent = cursor.fetchall()
        
        conn.close()
        
        # Calculate resolution rate
        resolution_rate = resolved_failures / total_failures if total_failures > 0 else 0.0
        
        return FailureTrend(
            period=f"Last {days} days",
            total_failures=total_failures,
            unique_failures=unique_failures,
            resolved_failures=resolved_failures,
            new_failures=new_failures,
            trending_up=[],  # Would need more complex analysis
            trending_down=[],  # Would need more complex analysis
            most_frequent=most_frequent,
            resolution_rate=resolution_rate
        )
    
    def export_report_summary(self, output_path: Path):
        """Export comprehensive failure report summary."""
        trends = self.get_failure_trends()
        active_reports = list(self.active_reports.values())
        
        summary = {
            "generated_at": datetime.now().isoformat(),
            "trends": asdict(trends),
            "active_reports_count": len(active_reports),
            "active_reports_by_priority": {
                priority.value: sum(1 for r in active_reports if r.priority == priority)
                for priority in FailurePriority
            },
            "active_reports_by_status": {
                status.value: sum(1 for r in active_reports if r.status == status)
                for status in FailureStatus
            },
            "top_failing_tests": Counter(r.test_name for r in active_reports).most_common(10),
            "recent_reports": [
                {
                    "report_id": r.report_id,
                    "test_name": r.test_name,
                    "status": r.status.value,
                    "priority": r.priority.value,
                    "occurrence_count": r.occurrence_count,
                    "updated_at": r.updated_at
                }
                for r in sorted(active_reports, key=lambda x: x.updated_at, reverse=True)[:20]
            ]
        }
        
        output_path.write_text(json.dumps(summary, indent=2))


# Global failure reporter instance
failure_reporter = FailureReporter()


def report_test_failure(
    test_name: str,
    error_message: str,
    stack_trace: Optional[str] = None,
    environment_info: Optional[Dict[str, Any]] = None,
    test_execution_id: Optional[str] = None
) -> str:
    """Report a test failure using the global reporter."""
    return failure_reporter.report_failure(
        test_name=test_name,
        error_message=error_message,
        stack_trace=stack_trace,
        environment_info=environment_info,
        test_execution_id=test_execution_id
    )


def get_failure_reporter() -> FailureReporter:
    """Get the global failure reporter instance."""
    return failure_reporter