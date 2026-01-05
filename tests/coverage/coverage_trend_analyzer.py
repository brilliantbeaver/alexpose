"""Coverage trend analysis and reporting."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import subprocess
import sys


@dataclass
class CoverageSnapshot:
    """Coverage snapshot at a point in time."""
    timestamp: str
    overall_coverage: float
    core_coverage: float
    domain_coverage: float
    integration_coverage: float
    lines_covered: int
    lines_total: int
    branches_covered: int
    branches_total: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class CoverageTrendAnalyzer:
    """Analyze coverage trends over time."""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Path("coverage_trends.db")
        self._init_database()
    
    def _init_database(self):
        """Initialize the coverage trends database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coverage_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                overall_coverage REAL NOT NULL,
                core_coverage REAL DEFAULT 0.0,
                domain_coverage REAL DEFAULT 0.0,
                integration_coverage REAL DEFAULT 0.0,
                lines_covered INTEGER DEFAULT 0,
                lines_total INTEGER DEFAULT 0,
                branches_covered INTEGER DEFAULT 0,
                branches_total INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_coverage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                coverage_percentage REAL NOT NULL,
                lines_covered INTEGER DEFAULT 0,
                lines_total INTEGER DEFAULT 0,
                FOREIGN KEY (snapshot_id) REFERENCES coverage_snapshots(id)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON coverage_snapshots(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_path 
            ON file_coverage(file_path)
        """)
        
        conn.commit()
        conn.close()
    
    def capture_coverage_snapshot(self) -> CoverageSnapshot:
        """Capture current coverage snapshot."""
        # Run coverage analysis
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "--cov=ambient", "--cov=server",
            "--cov-report=xml:coverage.xml",
            "--cov-report=json:coverage.json",
            "-q"
        ], capture_output=True, text=True, timeout=300)
        
        # Parse coverage data
        coverage_data = self._parse_coverage_data()
        
        snapshot = CoverageSnapshot(
            timestamp=datetime.now().isoformat(),
            overall_coverage=coverage_data["overall_coverage"],
            core_coverage=coverage_data["core_coverage"],
            domain_coverage=coverage_data["domain_coverage"],
            integration_coverage=coverage_data["integration_coverage"],
            lines_covered=coverage_data["lines_covered"],
            lines_total=coverage_data["lines_total"],
            branches_covered=coverage_data.get("branches_covered", 0),
            branches_total=coverage_data.get("branches_total", 0)
        )
        
        return snapshot
    
    def _parse_coverage_data(self) -> Dict[str, Any]:
        """Parse coverage data from JSON report."""
        coverage_file = Path("coverage.json")
        
        if not coverage_file.exists():
            return {
                "overall_coverage": 0.0,
                "core_coverage": 0.0,
                "domain_coverage": 0.0,
                "integration_coverage": 0.0,
                "lines_covered": 0,
                "lines_total": 0
            }
        
        try:
            with open(coverage_file, 'r') as f:
                data = json.load(f)
            
            # Overall coverage
            totals = data.get("totals", {})
            overall_coverage = totals.get("percent_covered", 0.0)
            lines_covered = totals.get("covered_lines", 0)
            lines_total = totals.get("num_statements", 0)
            branches_covered = totals.get("covered_branches", 0)
            branches_total = totals.get("num_branches", 0)
            
            # Component coverage
            files = data.get("files", {})
            
            core_files = []
            domain_files = []
            integration_files = []
            
            for file_path, file_data in files.items():
                coverage_pct = file_data.get("summary", {}).get("percent_covered", 0.0)
                
                if any(comp in file_path.lower() for comp in ["frame", "config", "interfaces", "core"]):
                    core_files.append(coverage_pct)
                elif any(comp in file_path.lower() for comp in ["pose", "gait", "classification", "analysis"]):
                    domain_files.append(coverage_pct)
                elif any(comp in file_path.lower() for comp in ["api", "video", "storage", "server"]):
                    integration_files.append(coverage_pct)
            
            core_coverage = sum(core_files) / len(core_files) if core_files else 0.0
            domain_coverage = sum(domain_files) / len(domain_files) if domain_files else 0.0
            integration_coverage = sum(integration_files) / len(integration_files) if integration_files else 0.0
            
            return {
                "overall_coverage": overall_coverage,
                "core_coverage": core_coverage,
                "domain_coverage": domain_coverage,
                "integration_coverage": integration_coverage,
                "lines_covered": lines_covered,
                "lines_total": lines_total,
                "branches_covered": branches_covered,
                "branches_total": branches_total,
                "files": files
            }
            
        except Exception as e:
            print(f"Error parsing coverage data: {e}")
            return {
                "overall_coverage": 0.0,
                "core_coverage": 0.0,
                "domain_coverage": 0.0,
                "integration_coverage": 0.0,
                "lines_covered": 0,
                "lines_total": 0
            }
    
    def record_snapshot(self, snapshot: CoverageSnapshot) -> int:
        """Record coverage snapshot to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO coverage_snapshots (
                timestamp, overall_coverage, core_coverage, domain_coverage,
                integration_coverage, lines_covered, lines_total,
                branches_covered, branches_total
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot.timestamp, snapshot.overall_coverage, snapshot.core_coverage,
            snapshot.domain_coverage, snapshot.integration_coverage,
            snapshot.lines_covered, snapshot.lines_total,
            snapshot.branches_covered, snapshot.branches_total
        ))
        
        snapshot_id = cursor.lastrowid
        
        # Record file-level coverage
        coverage_data = self._parse_coverage_data()
        files = coverage_data.get("files", {})
        
        for file_path, file_data in files.items():
            summary = file_data.get("summary", {})
            coverage_pct = summary.get("percent_covered", 0.0)
            lines_covered = summary.get("covered_lines", 0)
            lines_total = summary.get("num_statements", 0)
            
            cursor.execute("""
                INSERT INTO file_coverage (
                    snapshot_id, file_path, coverage_percentage,
                    lines_covered, lines_total
                ) VALUES (?, ?, ?, ?, ?)
            """, (snapshot_id, file_path, coverage_pct, lines_covered, lines_total))
        
        conn.commit()
        conn.close()
        
        return snapshot_id
    
    def get_snapshot_history(self, days: int = 30) -> List[CoverageSnapshot]:
        """Get coverage snapshot history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT * FROM coverage_snapshots
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """, (cutoff_date,))
        
        rows = cursor.fetchall()
        conn.close()
        
        snapshots = []
        for row in rows:
            snapshot = CoverageSnapshot(
                timestamp=row[1],
                overall_coverage=row[2],
                core_coverage=row[3],
                domain_coverage=row[4],
                integration_coverage=row[5],
                lines_covered=row[6],
                lines_total=row[7],
                branches_covered=row[8],
                branches_total=row[9]
            )
            snapshots.append(snapshot)
        
        return snapshots
    
    def analyze_coverage_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze coverage trends over time."""
        history = self.get_snapshot_history(days)
        
        if len(history) < 2:
            return {
                "success": False,
                "message": "Insufficient data for trend analysis (need at least 2 snapshots)"
            }
        
        # Calculate trends
        recent = history[-1]
        older = history[0]
        
        overall_trend = recent.overall_coverage - older.overall_coverage
        core_trend = recent.core_coverage - older.core_coverage
        domain_trend = recent.domain_coverage - older.domain_coverage
        integration_trend = recent.integration_coverage - older.integration_coverage
        
        # Calculate velocity (coverage change per day)
        days_elapsed = (datetime.fromisoformat(recent.timestamp) - 
                       datetime.fromisoformat(older.timestamp)).days
        
        if days_elapsed > 0:
            velocity = overall_trend / days_elapsed
        else:
            velocity = 0.0
        
        # Identify regressions
        regressions = []
        if overall_trend < -1:
            regressions.append(f"Overall coverage decreased by {abs(overall_trend):.1f}%")
        if core_trend < -2:
            regressions.append(f"Core coverage decreased by {abs(core_trend):.1f}%")
        if domain_trend < -2:
            regressions.append(f"Domain coverage decreased by {abs(domain_trend):.1f}%")
        
        # Calculate statistics
        overall_coverages = [s.overall_coverage for s in history]
        avg_coverage = sum(overall_coverages) / len(overall_coverages)
        min_coverage = min(overall_coverages)
        max_coverage = max(overall_coverages)
        
        return {
            "success": True,
            "period_days": days,
            "data_points": len(history),
            "current_snapshot": recent.to_dict(),
            "trends": {
                "overall": overall_trend,
                "core": core_trend,
                "domain": domain_trend,
                "integration": integration_trend,
                "velocity": velocity
            },
            "statistics": {
                "average": avg_coverage,
                "minimum": min_coverage,
                "maximum": max_coverage,
                "range": max_coverage - min_coverage
            },
            "regressions": regressions
        }
    
    def identify_coverage_regressions(self, threshold: float = 1.0) -> List[Dict[str, Any]]:
        """Identify files with coverage regressions."""
        # Get last two snapshots
        history = self.get_snapshot_history(days=7)
        
        if len(history) < 2:
            return []
        
        recent_snapshot = history[-1]
        previous_snapshot = history[-2]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get file coverage for both snapshots
        cursor.execute("""
            SELECT 
                r.file_path,
                r.coverage_percentage as recent_coverage,
                p.coverage_percentage as previous_coverage
            FROM file_coverage r
            JOIN file_coverage p ON r.file_path = p.file_path
            WHERE r.snapshot_id = (SELECT id FROM coverage_snapshots ORDER BY timestamp DESC LIMIT 1)
            AND p.snapshot_id = (SELECT id FROM coverage_snapshots ORDER BY timestamp DESC LIMIT 1 OFFSET 1)
            AND (r.coverage_percentage - p.coverage_percentage) < ?
            ORDER BY (r.coverage_percentage - p.coverage_percentage) ASC
        """, (-threshold,))
        
        regressions = []
        for row in cursor.fetchall():
            file_path, recent_cov, previous_cov = row
            regression = {
                "file_path": file_path,
                "previous_coverage": previous_cov,
                "current_coverage": recent_cov,
                "change": recent_cov - previous_cov
            }
            regressions.append(regression)
        
        conn.close()
        
        return regressions
    
    def generate_coverage_trend_chart(self, output_file: Path = None, days: int = 30):
        """Generate coverage trend visualization."""
        output_file = output_file or Path("coverage_trends.png")
        
        history = self.get_snapshot_history(days)
        
        if len(history) < 2:
            print("Insufficient data for chart generation")
            return None
        
        # Extract data for plotting
        timestamps = [datetime.fromisoformat(s.timestamp) for s in history]
        overall_coverage = [s.overall_coverage for s in history]
        core_coverage = [s.core_coverage for s in history]
        domain_coverage = [s.domain_coverage for s in history]
        integration_coverage = [s.integration_coverage for s in history]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        plt.plot(timestamps, overall_coverage, 'b-', label='Overall', linewidth=2)
        plt.plot(timestamps, core_coverage, 'g--', label='Core', linewidth=1.5)
        plt.plot(timestamps, domain_coverage, 'r--', label='Domain', linewidth=1.5)
        plt.plot(timestamps, integration_coverage, 'y--', label='Integration', linewidth=1.5)
        
        plt.xlabel('Date')
        plt.ylabel('Coverage (%)')
        plt.title(f'Coverage Trends (Last {days} Days)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_file, dpi=150)
        plt.close()
        
        return output_file
    
    def generate_trend_report(self, days: int = 30) -> str:
        """Generate coverage trend report."""
        trends = self.analyze_coverage_trends(days)
        
        if not trends["success"]:
            return f"Coverage trend analysis failed: {trends['message']}"
        
        report = []
        report.append("# Coverage Trend Analysis Report")
        report.append(f"Period: Last {days} days")
        report.append(f"Data Points: {trends['data_points']}")
        report.append("")
        
        # Current coverage
        current = trends["current_snapshot"]
        report.append("## Current Coverage")
        report.append(f"- Overall: {current['overall_coverage']:.1f}%")
        report.append(f"- Core: {current['core_coverage']:.1f}%")
        report.append(f"- Domain: {current['domain_coverage']:.1f}%")
        report.append(f"- Integration: {current['integration_coverage']:.1f}%")
        report.append(f"- Lines: {current['lines_covered']}/{current['lines_total']}")
        report.append("")
        
        # Trends
        trend_data = trends["trends"]
        report.append("## Trends")
        
        def trend_indicator(value: float) -> str:
            if value > 0.5:
                return f"📈 +{value:.1f}%"
            elif value < -0.5:
                return f"📉 {value:.1f}%"
            else:
                return f"➡️ {value:.1f}%"
        
        report.append(f"- Overall: {trend_indicator(trend_data['overall'])}")
        report.append(f"- Core: {trend_indicator(trend_data['core'])}")
        report.append(f"- Domain: {trend_indicator(trend_data['domain'])}")
        report.append(f"- Integration: {trend_indicator(trend_data['integration'])}")
        report.append(f"- Velocity: {trend_data['velocity']:.2f}% per day")
        report.append("")
        
        # Statistics
        stats = trends["statistics"]
        report.append("## Statistics")
        report.append(f"- Average: {stats['average']:.1f}%")
        report.append(f"- Minimum: {stats['minimum']:.1f}%")
        report.append(f"- Maximum: {stats['maximum']:.1f}%")
        report.append(f"- Range: {stats['range']:.1f}%")
        report.append("")
        
        # Regressions
        if trends["regressions"]:
            report.append("## ⚠️ Coverage Regressions Detected")
            for regression in trends["regressions"]:
                report.append(f"- {regression}")
            report.append("")
            
            # File-level regressions
            file_regressions = self.identify_coverage_regressions()
            if file_regressions:
                report.append("### Files with Coverage Regressions")
                for reg in file_regressions[:10]:  # Top 10
                    report.append(
                        f"- {reg['file_path']}: "
                        f"{reg['previous_coverage']:.1f}% → {reg['current_coverage']:.1f}% "
                        f"({reg['change']:.1f}%)"
                    )
                report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if trends["regressions"]:
            report.append("- Address coverage regressions immediately")
            report.append("- Review recent code changes for untested code")
        elif trend_data["velocity"] < 0:
            report.append("- Coverage is declining, focus on adding tests")
        elif current["overall_coverage"] < 80:
            report.append("- Increase overall coverage to at least 80%")
        else:
            report.append("- Coverage trends are positive! Keep up the good work.")
        
        return "\n".join(report)
    
    def save_trend_report(self, output_file: Path = None, days: int = 30) -> Path:
        """Save coverage trend report to file."""
        output_file = output_file or Path("coverage_trend_report.md")
        
        report = self.generate_trend_report(days)
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        return output_file


def main():
    """Run coverage trend analysis from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze coverage trends")
    parser.add_argument("--capture", action="store_true", help="Capture current coverage snapshot")
    parser.add_argument("--report", type=str, help="Generate report and save to file")
    parser.add_argument("--chart", type=str, help="Generate trend chart and save to file")
    parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    
    args = parser.parse_args()
    
    analyzer = CoverageTrendAnalyzer()
    
    if args.capture:
        print("Capturing coverage snapshot...")
        snapshot = analyzer.capture_coverage_snapshot()
        snapshot_id = analyzer.record_snapshot(snapshot)
        print(f"✅ Snapshot recorded (ID: {snapshot_id}): {snapshot.overall_coverage:.1f}% coverage")
    
    if args.report:
        print("Generating coverage trend report...")
        report_file = analyzer.save_trend_report(Path(args.report), days=args.days)
        print(f"✅ Report saved to: {report_file}")
    
    if args.chart:
        print("Generating coverage trend chart...")
        chart_file = analyzer.generate_coverage_trend_chart(Path(args.chart), days=args.days)
        if chart_file:
            print(f"✅ Chart saved to: {chart_file}")
    
    if not args.capture and not args.report and not args.chart:
        # Default: show current trends
        trends = analyzer.analyze_coverage_trends(days=args.days)
        if trends["success"]:
            current = trends["current_snapshot"]
            print(f"Current Coverage: {current['overall_coverage']:.1f}%")
            print(f"Trend: {trends['trends']['overall']:+.1f}%")
            print(f"Velocity: {trends['trends']['velocity']:.2f}% per day")


if __name__ == "__main__":
    main()
