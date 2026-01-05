"""
Debugging helpers and utilities for test development and troubleshooting.

This module provides comprehensive debugging support including system state capture,
minimal reproduction case generation, and interactive debugging sessions.
"""

import os
import sys
import json
import inspect
import traceback
import tempfile
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import logging
import subprocess
import shutil
from contextlib import contextmanager
import threading
import time

try:
    import hypothesis
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


@dataclass
class DebugContext:
    """Context information for debugging sessions."""
    test_name: str
    timestamp: str
    system_state: Dict[str, Any]
    test_parameters: Dict[str, Any]
    error_info: Optional[Dict[str, Any]]
    reproduction_steps: List[str]
    debug_artifacts: List[str]


class SystemStateCapture:
    """Captures comprehensive system state for debugging."""
    
    @staticmethod
    def capture_system_state() -> Dict[str, Any]:
        """Capture current system state."""
        try:
            import psutil
            
            # Memory information
            memory = psutil.virtual_memory()
            memory_info = {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "percentage": memory.percent
            }
            
            # CPU information
            cpu_info = {
                "count": psutil.cpu_count(),
                "usage_percent": psutil.cpu_percent(interval=1),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
            
            # Disk information
            disk = psutil.disk_usage('/')
            disk_info = {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percentage": round((disk.used / disk.total) * 100, 2)
            }
            
            # Process information
            process = psutil.Process()
            process_info = {
                "pid": process.pid,
                "memory_mb": round(process.memory_info().rss / (1024**2), 2),
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "open_files": len(process.open_files()) if hasattr(process, 'open_files') else 0
            }
            
        except ImportError:
            # Fallback without psutil
            memory_info = {"error": "psutil not available"}
            cpu_info = {"error": "psutil not available"}
            disk_info = {"error": "psutil not available"}
            process_info = {"error": "psutil not available"}
        
        # Python environment
        python_info = {
            "version": sys.version,
            "executable": sys.executable,
            "platform": sys.platform,
            "path": sys.path[:5],  # First 5 paths
            "modules": list(sys.modules.keys())[:20]  # First 20 modules
        }
        
        # Environment variables (filtered)
        env_vars = {
            k: v for k, v in os.environ.items()
            if not any(sensitive in k.lower() for sensitive in 
                      ['password', 'secret', 'key', 'token', 'auth'])
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "memory": memory_info,
            "cpu": cpu_info,
            "disk": disk_info,
            "process": process_info,
            "python": python_info,
            "environment": env_vars,
            "working_directory": str(Path.cwd())
        }


class MinimalReproductionGenerator:
    """Generates minimal reproduction cases for test failures."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_minimal_case(
        self,
        test_function: Callable,
        failing_input: Any,
        test_name: str
    ) -> Dict[str, Any]:
        """Generate minimal reproduction case for a failing test."""
        reproduction = {
            "test_name": test_name,
            "original_input": str(failing_input),
            "minimal_case": None,
            "reproduction_steps": [],
            "analysis": {}
        }
        
        try:
            # Analyze the failing input
            input_analysis = self._analyze_input(failing_input)
            reproduction["analysis"]["input_analysis"] = input_analysis
            
            # Generate minimal case
            minimal_input = self._minimize_input(test_function, failing_input)
            reproduction["minimal_case"] = str(minimal_input)
            
            # Generate reproduction steps
            steps = self._generate_reproduction_steps(
                test_function, minimal_input, test_name
            )
            reproduction["reproduction_steps"] = steps
            
            # Create standalone test file
            test_file = self._create_standalone_test(
                test_function, minimal_input, test_name
            )
            reproduction["standalone_test_file"] = test_file
            
        except Exception as e:
            reproduction["error"] = f"Failed to generate minimal case: {e}"
            self.logger.warning(f"Minimal case generation failed: {e}")
        
        return reproduction
    
    def _analyze_input(self, input_data: Any) -> Dict[str, Any]:
        """Analyze the structure and properties of input data."""
        analysis = {
            "type": type(input_data).__name__,
            "size": self._get_data_size(input_data),
            "properties": {}
        }
        
        if isinstance(input_data, (list, tuple)):
            analysis["properties"] = {
                "length": len(input_data),
                "element_types": [type(x).__name__ for x in input_data[:5]],
                "is_homogeneous": len(set(type(x).__name__ for x in input_data)) == 1
            }
        
        elif isinstance(input_data, dict):
            analysis["properties"] = {
                "keys": list(input_data.keys())[:10],
                "key_types": [type(k).__name__ for k in input_data.keys()],
                "value_types": [type(v).__name__ for v in input_data.values()]
            }
        
        elif isinstance(input_data, str):
            analysis["properties"] = {
                "length": len(input_data),
                "encoding": "utf-8" if input_data.isascii() else "non-ascii",
                "contains_special_chars": not input_data.isalnum()
            }
        
        elif isinstance(input_data, (int, float)):
            analysis["properties"] = {
                "value": input_data,
                "is_negative": input_data < 0,
                "magnitude": abs(input_data)
            }
        
        return analysis
    
    def _get_data_size(self, data: Any) -> int:
        """Get approximate size of data in bytes."""
        try:
            return sys.getsizeof(data)
        except:
            return 0
    
    def _minimize_input(self, test_function: Callable, failing_input: Any) -> Any:
        """Attempt to minimize the failing input while preserving the failure."""
        if not HYPOTHESIS_AVAILABLE:
            return failing_input
        
        try:
            # For different data types, try different minimization strategies
            if isinstance(failing_input, (list, tuple)):
                return self._minimize_sequence(test_function, failing_input)
            elif isinstance(failing_input, dict):
                return self._minimize_dict(test_function, failing_input)
            elif isinstance(failing_input, str):
                return self._minimize_string(test_function, failing_input)
            elif isinstance(failing_input, (int, float)):
                return self._minimize_number(test_function, failing_input)
            else:
                return failing_input
        except Exception:
            return failing_input
    
    def _minimize_sequence(self, test_function: Callable, sequence: Union[list, tuple]) -> Any:
        """Minimize a sequence while preserving failure."""
        if len(sequence) <= 1:
            return sequence
        
        # Try removing elements from the end
        for i in range(len(sequence) - 1, 0, -1):
            reduced = sequence[:i]
            if self._test_still_fails(test_function, reduced):
                return self._minimize_sequence(test_function, reduced)
        
        # Try removing elements from the beginning
        for i in range(1, len(sequence)):
            reduced = sequence[i:]
            if self._test_still_fails(test_function, reduced):
                return self._minimize_sequence(test_function, reduced)
        
        return sequence
    
    def _minimize_dict(self, test_function: Callable, data: dict) -> dict:
        """Minimize a dictionary while preserving failure."""
        if len(data) <= 1:
            return data
        
        # Try removing keys one by one
        for key in list(data.keys()):
            reduced = {k: v for k, v in data.items() if k != key}
            if self._test_still_fails(test_function, reduced):
                return self._minimize_dict(test_function, reduced)
        
        return data
    
    def _minimize_string(self, test_function: Callable, text: str) -> str:
        """Minimize a string while preserving failure."""
        if len(text) <= 1:
            return text
        
        # Try reducing length
        for i in range(len(text) // 2, 0, -1):
            reduced = text[:i]
            if self._test_still_fails(test_function, reduced):
                return self._minimize_string(test_function, reduced)
        
        return text
    
    def _minimize_number(self, test_function: Callable, number: Union[int, float]) -> Union[int, float]:
        """Minimize a number while preserving failure."""
        if number == 0:
            return number
        
        # Try smaller values
        candidates = [0, 1, -1, number // 2, -number // 2] if isinstance(number, int) else [0.0, 1.0, -1.0, number / 2]
        
        for candidate in candidates:
            if candidate != number and self._test_still_fails(test_function, candidate):
                return self._minimize_number(test_function, candidate)
        
        return number
    
    def _test_still_fails(self, test_function: Callable, input_data: Any) -> bool:
        """Check if the test still fails with the given input."""
        try:
            test_function(input_data)
            return False  # Test passed, so this input doesn't reproduce the failure
        except Exception:
            return True  # Test failed, so this input reproduces the failure
    
    def _generate_reproduction_steps(
        self,
        test_function: Callable,
        minimal_input: Any,
        test_name: str
    ) -> List[str]:
        """Generate step-by-step reproduction instructions."""
        steps = [
            f"# Minimal reproduction case for {test_name}",
            "",
            "## Steps to reproduce:",
            "1. Set up the test environment",
            "2. Import required modules",
            f"3. Call the test function with minimal input: {minimal_input}",
            "4. Observe the failure",
            "",
            "## Expected behavior:",
            "The test should pass without errors",
            "",
            "## Actual behavior:",
            "The test fails with an exception"
        ]
        
        # Add function signature information
        try:
            sig = inspect.signature(test_function)
            steps.extend([
                "",
                f"## Function signature:",
                f"{test_function.__name__}{sig}"
            ])
        except Exception:
            pass
        
        return steps
    
    def _create_standalone_test(
        self,
        test_function: Callable,
        minimal_input: Any,
        test_name: str
    ) -> str:
        """Create a standalone test file for the minimal reproduction case."""
        try:
            # Get function source
            source_lines = inspect.getsourcelines(test_function)[0]
            source_code = ''.join(source_lines)
            
            # Create standalone test
            standalone_test = f'''"""
Minimal reproduction case for {test_name}
Generated automatically for debugging purposes.
"""

import pytest
{self._get_required_imports(test_function)}

{source_code}

def test_minimal_reproduction():
    """Minimal reproduction case."""
    input_data = {repr(minimal_input)}
    
    # This should reproduce the original failure
    {test_function.__name__}(input_data)

if __name__ == "__main__":
    test_minimal_reproduction()
'''
            
            # Save to temporary file
            temp_dir = Path(tempfile.gettempdir()) / "test_reproductions"
            temp_dir.mkdir(exist_ok=True)
            
            test_file = temp_dir / f"reproduce_{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
            test_file.write_text(standalone_test)
            
            return str(test_file)
            
        except Exception as e:
            self.logger.warning(f"Failed to create standalone test: {e}")
            return f"Failed to create standalone test: {e}"
    
    def _get_required_imports(self, test_function: Callable) -> str:
        """Get required imports for the test function."""
        # This is a simplified approach - in practice, you'd need more sophisticated analysis
        module = inspect.getmodule(test_function)
        if module:
            return f"# Import from {module.__name__}\n"
        return "# Add required imports here\n"


class DebugSession:
    """Interactive debugging session manager."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = datetime.now()
        self.debug_log: List[Dict[str, Any]] = []
        self.artifacts: List[str] = []
        self.system_state = SystemStateCapture.capture_system_state()
    
    def log_event(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log a debugging event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "data": data or {}
        }
        self.debug_log.append(event)
    
    def add_artifact(self, artifact_path: str):
        """Add an artifact to the debug session."""
        self.artifacts.append(artifact_path)
    
    def capture_variable_state(self, variables: Dict[str, Any]):
        """Capture the state of variables for debugging."""
        self.log_event("variable_capture", "Variable state captured", {
            "variables": {k: str(v) for k, v in variables.items()}
        })
    
    def create_checkpoint(self, name: str, data: Optional[Dict[str, Any]] = None):
        """Create a debugging checkpoint."""
        self.log_event("checkpoint", f"Checkpoint: {name}", data)
    
    def export_session(self, output_dir: Optional[Path] = None) -> Path:
        """Export the debug session to files."""
        if output_dir is None:
            output_dir = Path("debug_sessions") / f"{self.test_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export debug log
        log_file = output_dir / "debug_log.json"
        log_file.write_text(json.dumps(self.debug_log, indent=2))
        
        # Export system state
        state_file = output_dir / "system_state.json"
        state_file.write_text(json.dumps(self.system_state, indent=2))
        
        # Export session summary
        summary = {
            "test_name": self.test_name,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "events_logged": len(self.debug_log),
            "artifacts_collected": len(self.artifacts),
            "artifacts": self.artifacts
        }
        
        summary_file = output_dir / "session_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))
        
        return output_dir


# Utility functions

def capture_system_state() -> Dict[str, Any]:
    """Capture current system state for debugging."""
    return SystemStateCapture.capture_system_state()


def generate_minimal_reproduction(
    test_function: Callable,
    failing_input: Any,
    test_name: str
) -> Dict[str, Any]:
    """Generate minimal reproduction case for a failing test."""
    generator = MinimalReproductionGenerator()
    return generator.generate_minimal_case(test_function, failing_input, test_name)


@contextmanager
def create_debug_session(test_name: str):
    """Create a debugging session context manager."""
    session = DebugSession(test_name)
    session.log_event("session_start", f"Debug session started for {test_name}")
    
    try:
        yield session
    except Exception as e:
        session.log_event("session_error", f"Error in debug session: {e}", {
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        })
        raise
    finally:
        session.log_event("session_end", f"Debug session ended for {test_name}")
        output_dir = session.export_session()
        print(f"Debug session exported to: {output_dir}")


def debug_test_failure(
    test_name: str,
    error_message: str,
    failing_input: Optional[Any] = None,
    test_function: Optional[Callable] = None
) -> DebugContext:
    """Comprehensive debugging helper for test failures."""
    timestamp = datetime.now().isoformat()
    
    # Capture system state
    system_state = capture_system_state()
    
    # Analyze error
    error_info = {
        "message": error_message,
        "traceback": traceback.format_exc(),
        "timestamp": timestamp
    }
    
    # Generate reproduction steps
    reproduction_steps = [
        f"1. Test '{test_name}' failed at {timestamp}",
        f"2. Error message: {error_message}",
        "3. Check system state for resource issues",
        "4. Review test input and expected behavior"
    ]
    
    if failing_input is not None:
        reproduction_steps.append(f"5. Failing input: {failing_input}")
    
    # Generate minimal reproduction if possible
    debug_artifacts = []
    if test_function and failing_input is not None:
        try:
            minimal_case = generate_minimal_reproduction(test_function, failing_input, test_name)
            if "standalone_test_file" in minimal_case:
                debug_artifacts.append(minimal_case["standalone_test_file"])
                reproduction_steps.append(f"6. Minimal reproduction case created: {minimal_case['standalone_test_file']}")
        except Exception as e:
            reproduction_steps.append(f"6. Failed to create minimal reproduction: {e}")
    
    return DebugContext(
        test_name=test_name,
        timestamp=timestamp,
        system_state=system_state,
        test_parameters={"failing_input": str(failing_input)} if failing_input else {},
        error_info=error_info,
        reproduction_steps=reproduction_steps,
        debug_artifacts=debug_artifacts
    )


def analyze_test_environment() -> Dict[str, Any]:
    """Analyze the current test environment for potential issues."""
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "issues": [],
        "recommendations": [],
        "system_health": "unknown"
    }
    
    try:
        import psutil
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            analysis["issues"].append("High memory usage detected")
            analysis["recommendations"].append("Consider reducing test data size or running fewer parallel tests")
        
        # Check disk space
        disk = psutil.disk_usage('/')
        if (disk.free / disk.total) < 0.1:  # Less than 10% free
            analysis["issues"].append("Low disk space detected")
            analysis["recommendations"].append("Clean up temporary files or increase available disk space")
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 95:
            analysis["issues"].append("High CPU usage detected")
            analysis["recommendations"].append("Consider reducing test parallelism or optimizing test performance")
        
        # Overall health assessment
        if not analysis["issues"]:
            analysis["system_health"] = "good"
        elif len(analysis["issues"]) <= 2:
            analysis["system_health"] = "fair"
        else:
            analysis["system_health"] = "poor"
            
    except ImportError:
        analysis["issues"].append("psutil not available for system monitoring")
        analysis["recommendations"].append("Install psutil for better system monitoring")
    
    # Check Python environment
    if sys.version_info < (3, 8):
        analysis["issues"].append("Python version is older than recommended")
        analysis["recommendations"].append("Consider upgrading to Python 3.8 or newer")
    
    # Check for common test environment issues
    if "PYTEST_CURRENT_TEST" not in os.environ:
        analysis["issues"].append("Not running in pytest environment")
        analysis["recommendations"].append("Ensure tests are run with pytest")
    
    return analysis