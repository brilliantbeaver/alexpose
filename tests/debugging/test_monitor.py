"""
Test execution monitoring and alerting system.

This module provides real-time monitoring of test execution,
performance tracking, and automated alerting for test failures.
"""

import time
import json
import threading
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringStatus(Enum):
    """Test execution status."""
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class MonitoredExecution:
    """Information about a test execution."""
    test_name: str
    start_time: str
    end_time: Optional[str]
    duration: Optional[float]
    status: MonitoringStatus
    error_message: Optional[str]
    memory_usage: Optional[float]
    cpu_usage: Optional[float]
    artifacts: List[str]


@dataclass
class Alert:
    """Test monitoring alert."""
    alert_id: str
    level: AlertLevel
    message: str
    test_name: Optional[str]
    timestamp: str
    details: Dict[str, Any]
    acknowledged: bool = False


@dataclass
class MonitoringMetrics:
    """Test monitoring metrics."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    average_duration: float
    total_duration: float
    failure_rate: float
    memory_peak: float
    cpu_peak: float
    alerts_generated: int


class ExecutionMonitor:
    """Real-time test execution monitor with alerting."""
    
    def __init__(self, db_path: Optional[Path] = None, alert_config: Optional[Dict[str, Any]] = None):
        self.db_path = db_path or Path("test_monitoring.db")
        self.alert_config = alert_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Active test tracking
        self.active_tests: Dict[str, MonitoredExecution] = {}
        self.completed_tests: List[MonitoredExecution] = []
        self.alerts: List[Alert] = []
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)  # Last 1000 test results
        self.failure_patterns: Dict[str, int] = defaultdict(int)
        
        # Monitoring thresholds
        self.thresholds = {
            "max_duration": 300.0,  # 5 minutes
            "max_memory_mb": 2048,  # 2GB
            "max_cpu_percent": 95,
            "failure_rate_threshold": 0.1,  # 10%
            "consecutive_failures": 3
        }
        self.thresholds.update(self.alert_config.get("thresholds", {}))
        
        # Alert handlers
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        self._init_database()
    
    def _init_database(self):
        """Initialize monitoring database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Test executions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT,
                start_time TEXT,
                end_time TEXT,
                duration REAL,
                status TEXT,
                error_message TEXT,
                memory_usage REAL,
                cpu_usage REAL,
                artifacts TEXT
            )
        """)
        
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE,
                level TEXT,
                message TEXT,
                test_name TEXT,
                timestamp TEXT,
                details TEXT,
                acknowledged BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Metrics snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                metrics TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Test monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Test monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_active_tests()
                self._analyze_performance_trends()
                self._check_system_health()
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Wait longer on error
    
    def start_test(self, test_name: str) -> str:
        """Start monitoring a test execution."""
        execution = MonitoredExecution(
            test_name=test_name,
            start_time=datetime.now().isoformat(),
            end_time=None,
            duration=None,
            status=MonitoringStatus.RUNNING,
            error_message=None,
            memory_usage=None,
            cpu_usage=None,
            artifacts=[]
        )
        
        self.active_tests[test_name] = execution
        self.logger.debug(f"Started monitoring test: {test_name}")
        return test_name
    
    def end_test(
        self,
        test_name: str,
        status: MonitoringStatus,
        error_message: Optional[str] = None,
        artifacts: Optional[List[str]] = None
    ):
        """End monitoring a test execution."""
        if test_name not in self.active_tests:
            self.logger.warning(f"Test {test_name} not found in active tests")
            return
        
        execution = self.active_tests[test_name]
        execution.end_time = datetime.now().isoformat()
        execution.status = status
        execution.error_message = error_message
        execution.artifacts = artifacts or []
        
        # Calculate duration
        start_time = datetime.fromisoformat(execution.start_time)
        end_time = datetime.fromisoformat(execution.end_time)
        execution.duration = (end_time - start_time).total_seconds()
        
        # Capture resource usage
        try:
            import psutil
            process = psutil.Process()
            execution.memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
            execution.cpu_usage = process.cpu_percent()
        except ImportError:
            pass
        
        # Move to completed tests
        self.completed_tests.append(execution)
        del self.active_tests[test_name]
        
        # Store in database
        self._store_test_execution(execution)
        
        # Check for alerts
        self._check_test_alerts(execution)
        
        # Update performance history
        self.performance_history.append({
            "test_name": test_name,
            "duration": execution.duration,
            "status": status.value,
            "timestamp": execution.end_time,
            "memory_usage": execution.memory_usage,
            "cpu_usage": execution.cpu_usage
        })
        
        self.logger.debug(f"Completed monitoring test: {test_name} ({status.value})")
    
    def _store_test_execution(self, execution: MonitoredExecution):
        """Store test execution in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO test_executions 
            (test_name, start_time, end_time, duration, status, error_message, 
             memory_usage, cpu_usage, artifacts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            execution.test_name,
            execution.start_time,
            execution.end_time,
            execution.duration,
            execution.status.value,
            execution.error_message,
            execution.memory_usage,
            execution.cpu_usage,
            json.dumps(execution.artifacts)
        ))
        
        conn.commit()
        conn.close()
    
    def _check_active_tests(self):
        """Check active tests for timeouts and resource issues."""
        current_time = datetime.now()
        
        for test_name, execution in list(self.active_tests.items()):
            start_time = datetime.fromisoformat(execution.start_time)
            duration = (current_time - start_time).total_seconds()
            
            # Check for timeout
            if duration > self.thresholds["max_duration"]:
                self._create_alert(
                    AlertLevel.WARNING,
                    f"Test {test_name} has been running for {duration:.1f}s (threshold: {self.thresholds['max_duration']}s)",
                    test_name,
                    {"duration": duration, "threshold": self.thresholds["max_duration"]}
                )
            
            # Check resource usage
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                cpu_percent = process.cpu_percent()
                
                if memory_mb > self.thresholds["max_memory_mb"]:
                    self._create_alert(
                        AlertLevel.WARNING,
                        f"Test {test_name} using {memory_mb:.1f}MB memory (threshold: {self.thresholds['max_memory_mb']}MB)",
                        test_name,
                        {"memory_usage": memory_mb, "threshold": self.thresholds["max_memory_mb"]}
                    )
                
                if cpu_percent > self.thresholds["max_cpu_percent"]:
                    self._create_alert(
                        AlertLevel.WARNING,
                        f"Test {test_name} using {cpu_percent:.1f}% CPU (threshold: {self.thresholds['max_cpu_percent']}%)",
                        test_name,
                        {"cpu_usage": cpu_percent, "threshold": self.thresholds["max_cpu_percent"]}
                    )
                    
            except ImportError:
                pass
    
    def _check_test_alerts(self, execution: MonitoredExecution):
        """Check for alerts based on test execution results."""
        # Check for test failure
        if execution.status == MonitoringStatus.FAILED:
            self.failure_patterns[execution.test_name] += 1
            
            # Check for consecutive failures
            recent_failures = self._get_recent_failures(execution.test_name, 10)
            consecutive_failures = 0
            for result in reversed(recent_failures):
                if result["status"] == "failed":
                    consecutive_failures += 1
                else:
                    break
            
            if consecutive_failures >= self.thresholds["consecutive_failures"]:
                self._create_alert(
                    AlertLevel.ERROR,
                    f"Test {execution.test_name} has failed {consecutive_failures} consecutive times",
                    execution.test_name,
                    {"consecutive_failures": consecutive_failures, "error": execution.error_message}
                )
        
        # Check for performance degradation
        if execution.duration and execution.duration > self.thresholds["max_duration"]:
            self._create_alert(
                AlertLevel.WARNING,
                f"Test {execution.test_name} took {execution.duration:.1f}s to complete (threshold: {self.thresholds['max_duration']}s)",
                execution.test_name,
                {"duration": execution.duration, "threshold": self.thresholds["max_duration"]}
            )
        
        # Check for memory issues
        if execution.memory_usage and execution.memory_usage > self.thresholds["max_memory_mb"]:
            self._create_alert(
                AlertLevel.WARNING,
                f"Test {execution.test_name} used {execution.memory_usage:.1f}MB memory (threshold: {self.thresholds['max_memory_mb']}MB)",
                execution.test_name,
                {"memory_usage": execution.memory_usage, "threshold": self.thresholds["max_memory_mb"]}
            )
    
    def _analyze_performance_trends(self):
        """Analyze performance trends and generate alerts."""
        if len(self.performance_history) < 10:
            return
        
        # Calculate recent failure rate
        recent_tests = list(self.performance_history)[-50:]  # Last 50 tests
        failed_tests = sum(1 for test in recent_tests if test["status"] == "failed")
        failure_rate = failed_tests / len(recent_tests)
        
        if failure_rate > self.thresholds["failure_rate_threshold"]:
            self._create_alert(
                AlertLevel.ERROR,
                f"High failure rate detected: {failure_rate:.1%} (threshold: {self.thresholds['failure_rate_threshold']:.1%})",
                None,
                {"failure_rate": failure_rate, "threshold": self.thresholds["failure_rate_threshold"]}
            )
        
        # Check for performance degradation
        recent_durations = [test["duration"] for test in recent_tests if test["duration"]]
        if recent_durations:
            avg_duration = sum(recent_durations) / len(recent_durations)
            
            # Compare with historical average
            historical_tests = list(self.performance_history)[:-50]  # Exclude recent tests
            if historical_tests:
                historical_durations = [test["duration"] for test in historical_tests if test["duration"]]
                if historical_durations:
                    historical_avg = sum(historical_durations) / len(historical_durations)
                    
                    # Alert if recent average is 50% slower than historical
                    if avg_duration > historical_avg * 1.5:
                        self._create_alert(
                            AlertLevel.WARNING,
                            f"Performance degradation detected: recent average {avg_duration:.1f}s vs historical {historical_avg:.1f}s",
                            None,
                            {"recent_avg": avg_duration, "historical_avg": historical_avg}
                        )
    
    def _check_system_health(self):
        """Check overall system health."""
        try:
            import psutil
            
            # Check system memory
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                self._create_alert(
                    AlertLevel.WARNING,
                    f"System memory usage high: {memory.percent:.1f}%",
                    None,
                    {"memory_percent": memory.percent}
                )
            
            # Check disk space
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 90:
                self._create_alert(
                    AlertLevel.WARNING,
                    f"System disk usage high: {disk_percent:.1f}%",
                    None,
                    {"disk_percent": disk_percent}
                )
                
        except ImportError:
            pass
    
    def _get_recent_failures(self, test_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent test results for a specific test."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT status, end_time, error_message
            FROM test_executions
            WHERE test_name = ?
            ORDER BY end_time DESC
            LIMIT ?
        """, (test_name, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "status": row[0],
                "end_time": row[1],
                "error_message": row[2]
            })
        
        conn.close()
        return results
    
    def _create_alert(
        self,
        level: AlertLevel,
        message: str,
        test_name: Optional[str],
        details: Dict[str, Any]
    ):
        """Create and process an alert."""
        alert_id = f"{level.value}_{hash(message) % 10000:04d}_{int(time.time())}"
        
        alert = Alert(
            alert_id=alert_id,
            level=level,
            message=message,
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            details=details
        )
        
        self.alerts.append(alert)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR IGNORE INTO alerts 
            (alert_id, level, message, test_name, timestamp, details)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            alert.alert_id,
            alert.level.value,
            alert.message,
            alert.test_name,
            alert.timestamp,
            json.dumps(alert.details)
        ))
        
        conn.commit()
        conn.close()
        
        # Process alert through handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
        
        self.logger.warning(f"Alert generated: {alert.level.value} - {alert.message}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler."""
        self.alert_handlers.append(handler)
    
    def get_metrics(self) -> MonitoringMetrics:
        """Get current monitoring metrics."""
        total_tests = len(self.completed_tests)
        if total_tests == 0:
            return MonitoringMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, len(self.alerts))
        
        passed_tests = sum(1 for test in self.completed_tests if test.status == MonitoringStatus.PASSED)
        failed_tests = sum(1 for test in self.completed_tests if test.status == MonitoringStatus.FAILED)
        skipped_tests = sum(1 for test in self.completed_tests if test.status == MonitoringStatus.SKIPPED)
        
        durations = [test.duration for test in self.completed_tests if test.duration]
        average_duration = sum(durations) / len(durations) if durations else 0.0
        total_duration = sum(durations) if durations else 0.0
        
        failure_rate = failed_tests / total_tests if total_tests > 0 else 0.0
        
        memory_usages = [test.memory_usage for test in self.completed_tests if test.memory_usage]
        memory_peak = max(memory_usages) if memory_usages else 0.0
        
        cpu_usages = [test.cpu_usage for test in self.completed_tests if test.cpu_usage]
        cpu_peak = max(cpu_usages) if cpu_usages else 0.0
        
        return MonitoringMetrics(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            average_duration=average_duration,
            total_duration=total_duration,
            failure_rate=failure_rate,
            memory_peak=memory_peak,
            cpu_peak=cpu_peak,
            alerts_generated=len(self.alerts)
        )
    
    def get_alerts(self, level: Optional[AlertLevel] = None, acknowledged: bool = False) -> List[Alert]:
        """Get alerts, optionally filtered by level and acknowledgment status."""
        alerts = self.alerts
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        if not acknowledged:
            alerts = [alert for alert in alerts if not alert.acknowledged]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                
                # Update in database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE alerts SET acknowledged = TRUE WHERE alert_id = ?",
                    (alert_id,)
                )
                conn.commit()
                conn.close()
                break
    
    def export_report(self, output_path: Path):
        """Export monitoring report."""
        metrics = self.get_metrics()
        alerts = self.get_alerts()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "metrics": asdict(metrics),
            "alerts": [asdict(alert) for alert in alerts],
            "active_tests": len(self.active_tests),
            "thresholds": self.thresholds,
            "recent_test_history": list(self.performance_history)[-20:]  # Last 20 tests
        }
        
        output_path.write_text(json.dumps(report, indent=2))


# Alert handlers

def console_alert_handler(alert: Alert):
    """Simple console alert handler."""
    print(f"[{alert.level.value.upper()}] {alert.timestamp}: {alert.message}")
    if alert.test_name:
        print(f"  Test: {alert.test_name}")
    if alert.details:
        print(f"  Details: {alert.details}")


def email_alert_handler(
    smtp_server: str,
    smtp_port: int,
    username: str,
    password: str,
    recipients: List[str]
):
    """Create an email alert handler."""
    def handler(alert: Alert):
        if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            try:
                msg = MIMEMultipart()
                msg['From'] = username
                msg['To'] = ', '.join(recipients)
                msg['Subject'] = f"Test Alert: {alert.level.value.upper()} - {alert.message[:50]}..."
                
                body = f"""
Test Monitoring Alert

Level: {alert.level.value.upper()}
Time: {alert.timestamp}
Message: {alert.message}
Test: {alert.test_name or 'N/A'}

Details:
{json.dumps(alert.details, indent=2)}
"""
                msg.attach(MIMEText(body, 'plain'))
                
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
                server.quit()
                
            except Exception as e:
                logging.error(f"Failed to send email alert: {e}")
    
    return handler


# Global monitor instance
test_monitor = ExecutionMonitor()


def start_test_monitoring():
    """Start the global test monitor."""
    test_monitor.start_monitoring()


def stop_test_monitoring():
    """Stop the global test monitor."""
    test_monitor.stop_monitoring()


def monitor_test(test_name: str):
    """Context manager for monitoring a test."""
    class TestMonitorContext:
        def __init__(self, name: str):
            self.name = name
        
        def __enter__(self):
            test_monitor.start_test(self.name)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                test_monitor.end_test(self.name, MonitoringStatus.PASSED)
            else:
                test_monitor.end_test(
                    self.name, 
                    MonitoringStatus.FAILED, 
                    error_message=str(exc_val)
                )
    
    return TestMonitorContext(test_name)


def get_test_monitor() -> ExecutionMonitor:
    """Get the global test monitor instance."""
    return test_monitor