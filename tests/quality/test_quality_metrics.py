"""Test quality metrics tracking and monitoring."""

import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import subprocess
import sys


@dataclass
class TestQualityMetrics:
    """Test quality metrics for a test run."""
    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    pass_rate: float
    execution_time: float
    coverage_percentage: float
    flaky_tests: int
    slow_tests: int
    test_reliability_score: float
    code_quality_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class TestQualityTracker:
    """Track test quality metrics over time."""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Path("test_quality_metrics.db")
        self._init_database()
    
    def _init_database(self):
        """Initialize the metrics database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_tests INTEGER NOT NULL,
                passed_tests INTEGER NOT NULL,
                failed_tests INTEGER NOT NULL,
                skipped_tests INTEGER NOT NULL,
                pass_rate REAL NOT NULL,
                execution_time REAL NOT NULL,
                coverage_percentage REAL NOT NULL,
                flaky_tests INTEGER DEFAULT 0,
                slow_tests INTEGER DEFAULT 0,
                test_reliability_score REAL DEFAULT 100.0,
                code_quality_score REAL DEFAULT 100.0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                test_name TEXT NOT NULL,
                failure_message TEXT,
                failure_count INTEGER DEFAULT 1
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flaky_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT UNIQUE NOT NULL,
                failure_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                last_failure TEXT,
                flakiness_score REAL DEFAULT 0.0
            )
        """)
        
        conn.commit()
        conn.close()
    
    def collect_current_metrics(self) -> TestQualityMetrics:
        """Collect current test quality metrics."""
        start_time = time.time()
        
        # Run tests and collect metrics
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "--tb=no", "-v", "--json-report", "--json-report-file=test_results.json"
        ], capture_output=True, text=True, timeout=300)
        
        execution_time = time.time() - start_time
        
        # Parse test results
        test_results_file = Path("test_results.json")
        if test_results_file.exists():
            with open(test_results_file, 'r') as f:
                test_data = json.load(f)
            
            summary = test_data.get("summary", {})
            total_tests = summary.get("total", 0)
            passed_tests = summary.get("passed", 0)
            failed_tests = summary.get("failed", 0)
            skipped_tests = summary.get("skipped", 0)
            
            # Calculate pass rate
            pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0.0
            
            # Identify slow tests (>1 second)
            slow_tests = 0
            tests = test_data.get("tests", [])
            for test in tests:
                if test.get("duration", 0) > 1.0:
                    slow_tests += 1
            
        else:
            # Fallback parsing
            total_tests = passed_tests = failed_tests = skipped_tests = 0
            slow_tests = 0
            pass_rate = 0.0
        
        # Get coverage
        coverage_percentage = self._get_coverage_percentage()
        
        # Calculate flaky tests
        flaky_tests = self._count_flaky_tests()
        
        # Calculate reliability score
        test_reliability_score = self._calculate_reliability_score(pass_rate, flaky_tests, total_tests)
        
        # Calculate code quality score
        code_quality_score = self._calculate_code_quality_score()
        
        return TestQualityMetrics(
            timestamp=datetime.now().isoformat(),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            pass_rate=pass_rate,
            execution_time=execution_time,
            coverage_percentage=coverage_percentage,
            flaky_tests=flaky_tests,
            slow_tests=slow_tests,
            test_reliability_score=test_reliability_score,
            code_quality_score=code_quality_score
        )
    
    def _get_coverage_percentage(self) -> float:
        """Get current coverage percentage."""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "--cov=ambient", "--cov=server",
                "--cov-report=term-missing", "-q"
            ], capture_output=True, text=True, timeout=60)
            
            # Parse coverage from output
            for line in result.stdout.split('\n'):
                if "TOTAL" in line:
                    parts = line.split()
                    for part in parts:
                        if '%' in part:
                            return float(part.replace('%', ''))
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _count_flaky_tests(self) -> int:
        """Count flaky tests from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM flaky_tests 
            WHERE flakiness_score > 0.2
        """)
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def _calculate_reliability_score(self, pass_rate: float, flaky_tests: int, total_tests: int) -> float:
        """Calculate test reliability score."""
        # Base score from pass rate
        score = pass_rate
        
        # Penalize for flaky tests
        if total_tests > 0:
            flaky_penalty = (flaky_tests / total_tests) * 20  # Up to 20% penalty
            score -= flaky_penalty
        
        return max(0.0, min(100.0, score))
    
    def _calculate_code_quality_score(self) -> float:
        """Calculate code quality score."""
        score = 100.0
        
        # Check for TODO/FIXME comments
        test_files = list(Path("tests").rglob("*.py"))
        todo_count = 0
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    todo_count += content.upper().count("TODO")
                    todo_count += content.upper().count("FIXME")
            except Exception:
                continue
        
        # Penalize for TODOs
        score -= min(todo_count * 2, 30)  # Up to 30% penalty
        
        return max(0.0, min(100.0, score))
    
    def record_metrics(self, metrics: TestQualityMetrics):
        """Record metrics to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO test_metrics (
                timestamp, total_tests, passed_tests, failed_tests, skipped_tests,
                pass_rate, execution_time, coverage_percentage, flaky_tests, slow_tests,
                test_reliability_score, code_quality_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.timestamp, metrics.total_tests, metrics.passed_tests,
            metrics.failed_tests, metrics.skipped_tests, metrics.pass_rate,
            metrics.execution_time, metrics.coverage_percentage, metrics.flaky_tests,
            metrics.slow_tests, metrics.test_reliability_score, metrics.code_quality_score
        ))
        
        conn.commit()
        conn.close()
    
    def get_metrics_history(self, days: int = 30) -> List[TestQualityMetrics]:
        """Get metrics history for the last N days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT * FROM test_metrics
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """, (cutoff_date,))
        
        rows = cursor.fetchall()
        conn.close()
        
        metrics_list = []
        for row in rows:
            metrics = TestQualityMetrics(
                timestamp=row[1],
                total_tests=row[2],
                passed_tests=row[3],
                failed_tests=row[4],
                skipped_tests=row[5],
                pass_rate=row[6],
                execution_time=row[7],
                coverage_percentage=row[8],
                flaky_tests=row[9],
                slow_tests=row[10],
                test_reliability_score=row[11],
                code_quality_score=row[12]
            )
            metrics_list.append(metrics)
        
        return metrics_list
    
    def analyze_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze quality trends over time."""
        history = self.get_metrics_history(days)
        
        if len(history) < 2:
            return {
                "success": False,
                "message": "Insufficient data for trend analysis"
            }
        
        # Calculate trends
        recent = history[0]
        older = history[-1]
        
        pass_rate_trend = recent.pass_rate - older.pass_rate
        coverage_trend = recent.coverage_percentage - older.coverage_percentage
        reliability_trend = recent.test_reliability_score - older.test_reliability_score
        
        # Identify issues
        issues = []
        if pass_rate_trend < -5:
            issues.append(f"Pass rate declining: {pass_rate_trend:.1f}%")
        if coverage_trend < -2:
            issues.append(f"Coverage declining: {coverage_trend:.1f}%")
        if recent.flaky_tests > 5:
            issues.append(f"High number of flaky tests: {recent.flaky_tests}")
        if recent.slow_tests > 10:
            issues.append(f"Many slow tests: {recent.slow_tests}")
        
        # Calculate averages
        avg_pass_rate = sum(m.pass_rate for m in history) / len(history)
        avg_coverage = sum(m.coverage_percentage for m in history) / len(history)
        avg_reliability = sum(m.test_reliability_score for m in history) / len(history)
        
        return {
            "success": True,
            "period_days": days,
            "data_points": len(history),
            "current_metrics": recent.to_dict(),
            "trends": {
                "pass_rate": pass_rate_trend,
                "coverage": coverage_trend,
                "reliability": reliability_trend
            },
            "averages": {
                "pass_rate": avg_pass_rate,
                "coverage": avg_coverage,
                "reliability": avg_reliability
            },
            "issues": issues
        }
    
    def generate_quality_report(self, days: int = 30) -> str:
        """Generate a quality metrics report."""
        metrics = self.collect_current_metrics()
        trends = self.analyze_trends(days)
        
        report = []
        report.append("# Test Quality Metrics Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Current metrics
        report.append("## Current Metrics")
        report.append(f"- Total Tests: {metrics.total_tests}")
        report.append(f"- Pass Rate: {metrics.pass_rate:.1f}%")
        report.append(f"- Coverage: {metrics.coverage_percentage:.1f}%")
        report.append(f"- Execution Time: {metrics.execution_time:.1f}s")
        report.append(f"- Flaky Tests: {metrics.flaky_tests}")
        report.append(f"- Slow Tests: {metrics.slow_tests}")
        report.append(f"- Reliability Score: {metrics.test_reliability_score:.1f}/100")
        report.append(f"- Code Quality Score: {metrics.code_quality_score:.1f}/100")
        report.append("")
        
        # Trends
        if trends["success"]:
            report.append(f"## Trends (Last {days} Days)")
            trend_data = trends["trends"]
            
            def trend_indicator(value: float) -> str:
                if value > 1:
                    return f"📈 +{value:.1f}%"
                elif value < -1:
                    return f"📉 {value:.1f}%"
                else:
                    return f"➡️ {value:.1f}%"
            
            report.append(f"- Pass Rate: {trend_indicator(trend_data['pass_rate'])}")
            report.append(f"- Coverage: {trend_indicator(trend_data['coverage'])}")
            report.append(f"- Reliability: {trend_indicator(trend_data['reliability'])}")
            report.append("")
            
            # Issues
            if trends["issues"]:
                report.append("## Issues Detected")
                for issue in trends["issues"]:
                    report.append(f"- ⚠️ {issue}")
                report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if metrics.pass_rate < 95:
            report.append("- Improve test pass rate to at least 95%")
        if metrics.coverage_percentage < 80:
            report.append("- Increase test coverage to at least 80%")
        if metrics.flaky_tests > 0:
            report.append(f"- Address {metrics.flaky_tests} flaky tests")
        if metrics.slow_tests > 10:
            report.append(f"- Optimize {metrics.slow_tests} slow tests")
        if not report[-1].startswith("-"):
            report.append("- Quality metrics are good! Keep up the excellent work.")
        
        return "\n".join(report)
    
    def save_quality_report(self, output_file: Path = None) -> Path:
        """Save quality report to file."""
        output_file = output_file or Path("test_quality_report.md")
        
        report = self.generate_quality_report()
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        return output_file


def main():
    """Run quality metrics tracking from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Track test quality metrics")
    parser.add_argument("--collect", action="store_true", help="Collect and record current metrics")
    parser.add_argument("--report", type=str, help="Generate report and save to file")
    parser.add_argument("--trends", type=int, default=30, help="Analyze trends over N days")
    
    args = parser.parse_args()
    
    tracker = TestQualityTracker()
    
    if args.collect:
        print("Collecting test quality metrics...")
        metrics = tracker.collect_current_metrics()
        tracker.record_metrics(metrics)
        print(f"✅ Metrics recorded: {metrics.pass_rate:.1f}% pass rate, {metrics.coverage_percentage:.1f}% coverage")
    
    if args.report:
        print("Generating quality report...")
        report_file = tracker.save_quality_report(Path(args.report))
        print(f"✅ Report saved to: {report_file}")
    
    if not args.collect and not args.report:
        # Default: show current metrics
        metrics = tracker.collect_current_metrics()
        print(f"Pass Rate: {metrics.pass_rate:.1f}%")
        print(f"Coverage: {metrics.coverage_percentage:.1f}%")
        print(f"Reliability Score: {metrics.test_reliability_score:.1f}/100")


if __name__ == "__main__":
    main()
