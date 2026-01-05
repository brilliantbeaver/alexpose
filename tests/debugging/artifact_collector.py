"""
Test artifact collection for debugging support.

This module captures logs, system state, environment information,
and other debugging artifacts when tests fail.
"""

import os
import sys
import json
import psutil
import platform
import traceback
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import shutil
import logging


@dataclass
class SystemState:
    """System state information at time of test failure."""
    timestamp: str
    platform_info: Dict[str, str]
    memory_usage: Dict[str, float]
    cpu_usage: float
    disk_usage: Dict[str, float]
    process_info: Dict[str, Any]
    environment_vars: Dict[str, str]
    python_info: Dict[str, str]


@dataclass
class TestArtifacts:
    """Collection of test debugging artifacts."""
    test_name: str
    failure_timestamp: str
    system_state: SystemState
    log_files: List[str]
    error_traceback: Optional[str]
    test_output: Optional[str]
    environment_snapshot: Dict[str, Any]
    file_artifacts: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert artifacts to dictionary."""
        return asdict(self)


class ArtifactCollector:
    """Collects debugging artifacts when tests fail."""
    
    def __init__(self, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = artifacts_dir or Path("test_artifacts")
        self.artifacts_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def collect_artifacts(
        self,
        test_name: str,
        error_info: Optional[Dict[str, Any]] = None,
        capture_logs: bool = True,
        capture_files: bool = True,
        capture_environment: bool = True
    ) -> TestArtifacts:
        """Collect comprehensive debugging artifacts for a failed test."""
        timestamp = datetime.now().isoformat()
        
        # Create test-specific artifact directory
        test_artifacts_dir = self.artifacts_dir / f"{test_name}_{timestamp.replace(':', '-')}"
        test_artifacts_dir.mkdir(exist_ok=True)
        
        # Collect system state
        system_state = self._capture_system_state()
        
        # Collect log files
        log_files = []
        if capture_logs:
            log_files = self._collect_log_files(test_artifacts_dir)
        
        # Collect error information
        error_traceback = None
        test_output = None
        if error_info:
            error_traceback = error_info.get('traceback')
            test_output = error_info.get('output')
            
            # Save error info to file
            error_file = test_artifacts_dir / "error_info.json"
            error_file.write_text(json.dumps(error_info, indent=2, default=str))
        
        # Collect environment snapshot
        environment_snapshot = {}
        if capture_environment:
            environment_snapshot = self._capture_environment_snapshot()
            
            # Save environment to file
            env_file = test_artifacts_dir / "environment.json"
            env_file.write_text(json.dumps(environment_snapshot, indent=2, default=str))
        
        # Collect relevant files
        file_artifacts = []
        if capture_files:
            file_artifacts = self._collect_file_artifacts(test_name, test_artifacts_dir)
        
        # Save system state
        system_state_file = test_artifacts_dir / "system_state.json"
        system_state_file.write_text(json.dumps(asdict(system_state), indent=2, default=str))
        
        artifacts = TestArtifacts(
            test_name=test_name,
            failure_timestamp=timestamp,
            system_state=system_state,
            log_files=log_files,
            error_traceback=error_traceback,
            test_output=test_output,
            environment_snapshot=environment_snapshot,
            file_artifacts=file_artifacts
        )
        
        # Save complete artifacts summary
        summary_file = test_artifacts_dir / "artifacts_summary.json"
        summary_file.write_text(json.dumps(artifacts.to_dict(), indent=2, default=str))
        
        self.logger.info(f"Collected test artifacts for {test_name} in {test_artifacts_dir}")
        return artifacts
    
    def _capture_system_state(self) -> SystemState:
        """Capture current system state."""
        try:
            # Platform information
            platform_info = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": str(platform.architecture()),
                "hostname": platform.node()
            }
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percentage": memory.percent
            }
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = {
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3),
                "percentage": (disk.used / disk.total) * 100
            }
            
            # Process information
            process = psutil.Process()
            process_info = {
                "pid": process.pid,
                "memory_mb": process.memory_info().rss / (1024**2),
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
            }
            
            # Environment variables (filtered for security)
            safe_env_vars = {
                k: v for k, v in os.environ.items()
                if not any(sensitive in k.lower() for sensitive in 
                          ['password', 'secret', 'key', 'token', 'auth'])
            }
            
            # Python information
            python_info = {
                "version": sys.version,
                "executable": sys.executable,
                "platform": sys.platform,
                "path": sys.path[:5]  # First 5 paths only
            }
            
            return SystemState(
                timestamp=datetime.now().isoformat(),
                platform_info=platform_info,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                disk_usage=disk_usage,
                process_info=process_info,
                environment_vars=safe_env_vars,
                python_info=python_info
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to capture complete system state: {e}")
            # Return minimal system state
            return SystemState(
                timestamp=datetime.now().isoformat(),
                platform_info={"system": platform.system()},
                memory_usage={},
                cpu_usage=0.0,
                disk_usage={},
                process_info={},
                environment_vars={},
                python_info={"version": sys.version}
            )
    
    def _collect_log_files(self, artifacts_dir: Path) -> List[str]:
        """Collect relevant log files."""
        log_files = []
        
        # Common log locations (Windows-compatible)
        log_locations = [
            Path("logs"),
            Path("test_logs"),
            Path("./logs"),
            Path.cwd() / "logs"
        ]
        
        # Add Windows-specific temp locations
        if os.name == 'nt':
            import tempfile
            temp_dir = Path(tempfile.gettempdir())
            log_locations.extend([
                temp_dir / "test_logs",
                temp_dir / "logs"
            ])
        else:
            # Unix-specific locations
            log_locations.extend([
                Path("/tmp/test_logs"),
                Path("/var/log")
            ])
        
        for log_dir in log_locations:
            if log_dir.exists() and log_dir.is_dir():
                try:
                    for log_file in log_dir.glob("*.log"):
                        if log_file.is_file():
                            # Copy log file to artifacts
                            dest_file = artifacts_dir / f"logs_{log_file.name}"
                            shutil.copy2(log_file, dest_file)
                            log_files.append(str(dest_file))
                            
                            # Also capture recent log entries
                            try:
                                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                                    lines = f.readlines()
                                    # Get last 100 lines
                                    recent_lines = lines[-100:] if len(lines) > 100 else lines
                                    
                                recent_file = artifacts_dir / f"recent_{log_file.name}"
                                recent_file.write_text(''.join(recent_lines), encoding='utf-8')
                                log_files.append(str(recent_file))
                                
                            except Exception as e:
                                self.logger.warning(f"Failed to read log file {log_file}: {e}")
                                
                except Exception as e:
                    self.logger.warning(f"Failed to collect logs from {log_dir}: {e}")
        
        # Capture Python logging output if available
        try:
            # Get root logger handlers
            root_logger = logging.getLogger()
            for handler in root_logger.handlers:
                if hasattr(handler, 'baseFilename'):
                    log_file = Path(handler.baseFilename)
                    if log_file.exists():
                        dest_file = artifacts_dir / f"python_{log_file.name}"
                        shutil.copy2(log_file, dest_file)
                        log_files.append(str(dest_file))
        except Exception as e:
            self.logger.warning(f"Failed to collect Python log files: {e}")
        
        return log_files
    
    def _capture_environment_snapshot(self) -> Dict[str, Any]:
        """Capture environment configuration snapshot."""
        snapshot = {}
        
        try:
            # Git information
            try:
                git_info = {}
                git_commands = [
                    ("branch", ["git", "branch", "--show-current"]),
                    ("commit", ["git", "rev-parse", "HEAD"]),
                    ("status", ["git", "status", "--porcelain"]),
                    ("remote", ["git", "remote", "-v"])
                ]
                
                for name, cmd in git_commands:
                    try:
                        result = subprocess.run(
                            cmd, capture_output=True, text=True, timeout=5
                        )
                        if result.returncode == 0:
                            git_info[name] = result.stdout.strip()
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        git_info[name] = "unavailable"
                
                snapshot["git"] = git_info
            except Exception:
                snapshot["git"] = {"error": "Git information unavailable"}
            
            # Python packages
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "list", "--format=json"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    packages = json.loads(result.stdout)
                    snapshot["python_packages"] = {
                        pkg["name"]: pkg["version"] for pkg in packages
                    }
            except Exception:
                snapshot["python_packages"] = {"error": "Package list unavailable"}
            
            # Configuration files
            config_files = [
                "pyproject.toml",
                "pytest.ini",
                ".env",
                "config.yaml",
                "config.json"
            ]
            
            snapshot["config_files"] = {}
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    try:
                        # Don't include sensitive config content, just metadata
                        stat = config_path.stat()
                        snapshot["config_files"][config_file] = {
                            "exists": True,
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        }
                    except Exception:
                        snapshot["config_files"][config_file] = {"exists": True, "error": "Cannot read metadata"}
                else:
                    snapshot["config_files"][config_file] = {"exists": False}
            
            # Test environment
            snapshot["test_environment"] = {
                "working_directory": str(Path.cwd()),
                "python_path": sys.path[:3],  # First 3 paths
                "test_runner": "pytest" if "pytest" in sys.modules else "unknown"
            }
            
        except Exception as e:
            snapshot["error"] = f"Failed to capture environment: {e}"
        
        return snapshot
    
    def _collect_file_artifacts(self, test_name: str, artifacts_dir: Path) -> List[str]:
        """Collect relevant file artifacts for debugging."""
        file_artifacts = []
        
        try:
            # Test-related files
            test_files = [
                f"test_{test_name.lower()}.py",
                f"{test_name.lower()}_test.py",
                f"tests/test_{test_name.lower()}.py"
            ]
            
            for test_file in test_files:
                test_path = Path(test_file)
                if test_path.exists():
                    dest_file = artifacts_dir / f"test_file_{test_path.name}"
                    shutil.copy2(test_path, dest_file)
                    file_artifacts.append(str(dest_file))
            
            # Configuration files
            config_files = ["pytest.ini", "pyproject.toml", "conftest.py"]
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    dest_file = artifacts_dir / f"config_{config_path.name}"
                    shutil.copy2(config_path, dest_file)
                    file_artifacts.append(str(dest_file))
            
            # Recent test output files
            temp_dir = Path(tempfile.gettempdir())
            for temp_file in temp_dir.glob("pytest-*"):
                if temp_file.is_file():
                    try:
                        dest_file = artifacts_dir / f"temp_{temp_file.name}"
                        shutil.copy2(temp_file, dest_file)
                        file_artifacts.append(str(dest_file))
                    except Exception:
                        pass  # Skip files we can't copy
            
        except Exception as e:
            self.logger.warning(f"Failed to collect file artifacts: {e}")
        
        return file_artifacts
    
    def cleanup_old_artifacts(self, days_old: int = 7):
        """Clean up artifact directories older than specified days."""
        if not self.artifacts_dir.exists():
            return
        
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)
        
        for artifact_dir in self.artifacts_dir.iterdir():
            if artifact_dir.is_dir():
                try:
                    if artifact_dir.stat().st_mtime < cutoff_time:
                        shutil.rmtree(artifact_dir)
                        self.logger.info(f"Cleaned up old artifacts: {artifact_dir}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {artifact_dir}: {e}")
    
    def get_artifact_summary(self) -> Dict[str, Any]:
        """Get summary of collected artifacts."""
        if not self.artifacts_dir.exists():
            return {"message": "No artifacts directory found"}
        
        artifact_dirs = [d for d in self.artifacts_dir.iterdir() if d.is_dir()]
        
        summary = {
            "total_artifact_sets": len(artifact_dirs),
            "artifacts_directory": str(self.artifacts_dir),
            "recent_failures": []
        }
        
        # Get recent failures (last 10)
        recent_dirs = sorted(artifact_dirs, key=lambda d: d.stat().st_mtime, reverse=True)[:10]
        
        for artifact_dir in recent_dirs:
            try:
                summary_file = artifact_dir / "artifacts_summary.json"
                if summary_file.exists():
                    with open(summary_file) as f:
                        artifact_data = json.load(f)
                        summary["recent_failures"].append({
                            "test_name": artifact_data.get("test_name"),
                            "timestamp": artifact_data.get("failure_timestamp"),
                            "directory": str(artifact_dir)
                        })
            except Exception:
                summary["recent_failures"].append({
                    "test_name": "unknown",
                    "timestamp": "unknown",
                    "directory": str(artifact_dir)
                })
        
        return summary


# Global artifact collector instance
artifact_collector = ArtifactCollector()


def collect_test_artifacts(
    test_name: str,
    error_info: Optional[Dict[str, Any]] = None,
    capture_logs: bool = True,
    capture_files: bool = True,
    capture_environment: bool = True
) -> TestArtifacts:
    """Collect test artifacts using the global collector."""
    return artifact_collector.collect_artifacts(
        test_name=test_name,
        error_info=error_info,
        capture_logs=capture_logs,
        capture_files=capture_files,
        capture_environment=capture_environment
    )


def get_artifact_collector() -> ArtifactCollector:
    """Get the global artifact collector instance."""
    return artifact_collector