"""
Logging utilities for AlexPose using Loguru.

This module provides centralized logging configuration and utilities
for consistent structured logging across the AlexPose system.

Author: AlexPose Team
"""

import sys
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    rotation: str = "1 day",
    retention: str = "30 days",
    compression: str = "zip",
    format_type: str = "structured",
    enable_json_logs: bool = False,
    environment: str = "development",
    max_file_size: str = "100 MB",
    backup_count: int = 10
) -> None:
    """
    Set up Loguru logging with structured formatting and file output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        rotation: Log rotation policy (time-based: "1 day", "12 hours", size-based: "100 MB")
        retention: Log retention policy ("30 days", "1 week", or number like "10")
        compression: Compression format for rotated logs ("zip", "gz", "bz2")
        format_type: Format type ("structured" or "simple")
        enable_json_logs: Enable JSON structured logging to files
        environment: Environment name (development, production, etc.)
        max_file_size: Maximum file size before rotation (used as fallback)
        backup_count: Number of backup files to keep (used with retention)
    """
    # Remove default handler
    logger.remove()
    
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Environment-specific log level adjustments
    console_log_level = log_level
    file_log_level = "DEBUG"  # Always log DEBUG to file for troubleshooting
    
    # Adjust console log level based on environment
    if environment.lower() == "production":
        # In production, be more conservative with console output
        if log_level == "DEBUG":
            console_log_level = "INFO"  # Don't spam production console with DEBUG
    elif environment.lower() == "development":
        # In development, be more verbose
        console_log_level = log_level  # Use exactly what was requested
    
    # Define structured format templates
    if format_type == "structured":
        if environment.lower() == "development":
            # More detailed format for development
            console_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<blue>{extra[component]}</blue> | "
                "<level>{message}</level>"
            )
        else:
            # Cleaner format for production
            console_format = (
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <5}</level> | "
                "<blue>{extra[component]}</blue> | "
                "<level>{message}</level>"
            )
        
        if enable_json_logs:
            # JSON structured format for file logging
            def json_formatter(record):
                log_entry = {
                    "timestamp": record["time"].isoformat(),
                    "level": record["level"].name,
                    "logger": record["name"],
                    "function": record["function"],
                    "line": record["line"],
                    "message": record["message"],
                    "component": record["extra"].get("component", "unknown"),
                    "operation": record["extra"].get("operation", ""),
                    "duration": record["extra"].get("duration"),
                    "context": record["extra"].get("context", {}),
                    "error_type": record["extra"].get("error_type"),
                    "stack_trace": record["extra"].get("stack_trace")
                }
                # Remove None values for cleaner JSON
                log_entry = {k: v for k, v in log_entry.items() if v is not None}
                return json.dumps(log_entry)
            
            file_format = json_formatter
        else:
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{extra[component]} | "
                "{extra[operation]} | "
                "{message}"
            )
    else:
        console_format = "<level>{level}</level>: {message}"
        file_format = "{time} | {level} | {message}"
    
    # Add console handler with structured formatting
    logger.add(
        sys.stderr,
        format=console_format,
        level=console_log_level,
        colorize=True,
        filter=_add_default_context
    )
    
    # Get environment-specific rotation and retention policies
    policies = get_rotation_retention_policies(environment)
    
    # Calculate and log estimated disk usage
    disk_usage = calculate_estimated_disk_usage(policies)
    
    # Add main application log file with environment-specific policies
    main_policy = policies.get("main", {"rotation": rotation, "retention": retention, "compression": compression})
    logger.add(
        log_path / "alexpose_{time:YYYY-MM-DD}.log",
        format=file_format,
        level=file_log_level,
        rotation=main_policy.get("rotation", rotation),
        retention=main_policy.get("retention", retention),
        compression=main_policy.get("compression", compression),
        enqueue=True,
        filter=_add_default_context
    )
    
    # Add error-specific log file with enhanced retention
    error_policy = policies.get("error", {"rotation": rotation, "retention": "90 days", "compression": compression})
    logger.add(
        log_path / "errors_{time:YYYY-MM-DD}.log",
        format=file_format,
        level="ERROR",
        rotation=error_policy.get("rotation", rotation),
        retention=error_policy.get("retention", "90 days"),
        compression=error_policy.get("compression", compression),
        enqueue=True,
        filter=_add_default_context
    )
    
    # Add performance log file with specific policies
    perf_policy = policies.get("performance", {"rotation": "1 day", "retention": "30 days", "compression": compression})
    logger.add(
        log_path / "performance_{time:YYYY-MM-DD}.log",
        format=file_format,
        level="INFO",
        rotation=perf_policy.get("rotation", "1 day"),
        retention=perf_policy.get("retention", "30 days"),
        compression=perf_policy.get("compression", compression),
        enqueue=True,
        filter=lambda record: record["extra"].get("log_type") == "performance"
    )
    
    # Environment-specific logging setup with advanced rotation
    if environment.lower() == "development":
        # Add debug-specific log file for development with shorter retention
        debug_policy = policies.get("debug", {"rotation": "6 hours", "retention": "3 days", "compression": compression})
        logger.add(
            log_path / "debug_{time:YYYY-MM-DD}.log",
            format=file_format,
            level="DEBUG",
            rotation=debug_policy.get("rotation", "6 hours"),
            retention=debug_policy.get("retention", "3 days"),
            compression=debug_policy.get("compression", compression),
            enqueue=True,
            filter=lambda record: record["level"].name == "DEBUG"
        )
    
    # Add environment-specific handlers
    setup_environment_specific_handlers(
        environment, log_path, file_format, rotation, retention, compression
    )
    
    logger.info(
        "Logging system initialized with advanced rotation policies",
        extra={
            "component": "logging",
            "operation": "setup",
            "environment": environment,
            "console_log_level": console_log_level,
            "file_log_level": file_log_level,
            "log_directory": str(log_path),
            "structured_format": format_type,
            "json_enabled": enable_json_logs,
            "rotation_policies": policies,
            "estimated_disk_usage_gb": round(disk_usage["total_estimated_gb"], 2),
            "disk_usage_recommendations": disk_usage["recommendations"]
        }
    )


def _json_formatter(record):
    """Custom JSON formatter for structured logging."""
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
        "component": record["extra"].get("component", "unknown"),
        "operation": record["extra"].get("operation", ""),
        "duration": record["extra"].get("duration"),
        "context": record["extra"].get("context", {}),
        "error_type": record["extra"].get("error_type"),
        "stack_trace": record["extra"].get("stack_trace")
    }
    
    # Remove None values for cleaner JSON
    log_entry = {k: v for k, v in log_entry.items() if v is not None}
    
    return json.dumps(log_entry)


def _add_default_context(record):
    """Add default context to all log records."""
    if "component" not in record["extra"]:
        record["extra"]["component"] = "core"
    if "operation" not in record["extra"]:
        record["extra"]["operation"] = ""
    return True


def get_rotation_retention_policies(environment: str) -> Dict[str, Dict[str, str]]:
    """
    Get environment-specific rotation and retention policies.
    
    Args:
        environment: Environment name (development, production, staging, testing)
        
    Returns:
        Dictionary containing rotation and retention policies for different log types
    """
    env_lower = environment.lower()
    
    if env_lower == "development":
        return {
            "main": {
                "rotation": "6 hours",
                "retention": "3 days",
                "compression": "zip",
                "max_size": "50 MB"
            },
            "debug": {
                "rotation": "3 hours", 
                "retention": "1 day",
                "compression": "zip",
                "max_size": "25 MB"
            },
            "error": {
                "rotation": "12 hours",
                "retention": "7 days", 
                "compression": "zip",
                "max_size": "100 MB"
            },
            "performance": {
                "rotation": "1 day",
                "retention": "7 days",
                "compression": "zip",
                "max_size": "200 MB"
            }
        }
    
    elif env_lower == "production":
        return {
            "main": {
                "rotation": "1 day",
                "retention": "30 days",
                "compression": "gz",
                "max_size": "500 MB"
            },
            "debug": {
                "rotation": "12 hours",
                "retention": "7 days",
                "compression": "gz", 
                "max_size": "100 MB"
            },
            "error": {
                "rotation": "1 day",
                "retention": "90 days",
                "compression": "gz",
                "max_size": "1 GB"
            },
            "performance": {
                "rotation": "1 day", 
                "retention": "60 days",
                "compression": "gz",
                "max_size": "2 GB"
            },
            "critical": {
                "rotation": "1 week",
                "retention": "1 year",
                "compression": "gz",
                "max_size": "500 MB"
            },
            "security": {
                "rotation": "1 day",
                "retention": "1 year", 
                "compression": "gz",
                "max_size": "1 GB"
            }
        }
    
    elif env_lower in ["staging", "testing"]:
        return {
            "main": {
                "rotation": "12 hours",
                "retention": "14 days",
                "compression": "zip",
                "max_size": "200 MB"
            },
            "debug": {
                "rotation": "6 hours",
                "retention": "3 days", 
                "compression": "zip",
                "max_size": "50 MB"
            },
            "error": {
                "rotation": "1 day",
                "retention": "30 days",
                "compression": "zip",
                "max_size": "500 MB"
            },
            "performance": {
                "rotation": "1 day",
                "retention": "14 days",
                "compression": "zip", 
                "max_size": "1 GB"
            },
            "test_results": {
                "rotation": "1 day",
                "retention": "30 days",
                "compression": "zip",
                "max_size": "100 MB"
            }
        }
    
    else:
        # Default/unknown environment - use conservative settings
        return {
            "main": {
                "rotation": "1 day",
                "retention": "7 days",
                "compression": "zip",
                "max_size": "100 MB"
            },
            "debug": {
                "rotation": "12 hours",
                "retention": "2 days",
                "compression": "zip",
                "max_size": "50 MB"
            },
            "error": {
                "rotation": "1 day", 
                "retention": "14 days",
                "compression": "zip",
                "max_size": "200 MB"
            },
            "performance": {
                "rotation": "1 day",
                "retention": "7 days",
                "compression": "zip",
                "max_size": "500 MB"
            }
        }


def validate_rotation_retention_config(rotation: str, retention: str) -> Dict[str, Any]:
    """
    Validate rotation and retention configuration parameters.
    
    Args:
        rotation: Rotation policy string
        retention: Retention policy string
        
    Returns:
        Dictionary with validation results and normalized values
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "normalized_rotation": rotation,
        "normalized_retention": retention
    }
    
    # Validate rotation format
    rotation_patterns = [
        r'^\d+\s+(second|seconds|minute|minutes|hour|hours|day|days|week|weeks)$',
        r'^\d+\s+(KB|MB|GB|TB)$',
        r'^\d+:\d+$',  # Time format like "23:59"
    ]
    
    import re
    rotation_valid = any(re.match(pattern, rotation, re.IGNORECASE) for pattern in rotation_patterns)
    
    if not rotation_valid:
        validation_result["errors"].append(f"Invalid rotation format: {rotation}")
        validation_result["valid"] = False
    
    # Validate retention format
    retention_patterns = [
        r'^\d+\s+(day|days|week|weeks|month|months|year|years)$',
        r'^\d+$',  # Just a number (backup count)
    ]
    
    retention_valid = any(re.match(pattern, retention, re.IGNORECASE) for pattern in retention_patterns)
    
    if not retention_valid:
        validation_result["errors"].append(f"Invalid retention format: {retention}")
        validation_result["valid"] = False
    
    # Check for potential issues
    if "MB" in rotation and int(rotation.split()[0]) > 1000:
        validation_result["warnings"].append("Large rotation size may impact performance")
    
    if "day" in retention and int(retention.split()[0]) > 365:
        validation_result["warnings"].append("Very long retention period may consume significant disk space")
    
    return validation_result


def calculate_estimated_disk_usage(policies: Dict[str, Dict[str, str]], daily_log_volume_mb: float = 100) -> Dict[str, Any]:
    """
    Calculate estimated disk usage based on rotation and retention policies.
    
    Args:
        policies: Dictionary of rotation/retention policies
        daily_log_volume_mb: Estimated daily log volume in MB
        
    Returns:
        Dictionary with disk usage estimates
    """
    total_estimated_mb = 0
    breakdown = {}
    
    for log_type, policy in policies.items():
        retention = policy.get("retention", "30 days")
        max_size = policy.get("max_size", "100 MB")
        
        # Parse retention period
        if "day" in retention:
            days = int(retention.split()[0])
        elif "week" in retention:
            days = int(retention.split()[0]) * 7
        elif "month" in retention:
            days = int(retention.split()[0]) * 30
        elif "year" in retention:
            days = int(retention.split()[0]) * 365
        else:
            days = 30  # Default
        
        # Parse max size
        if "MB" in max_size:
            max_size_mb = int(max_size.split()[0])
        elif "GB" in max_size:
            max_size_mb = int(max_size.split()[0]) * 1024
        elif "KB" in max_size:
            max_size_mb = int(max_size.split()[0]) / 1024
        else:
            max_size_mb = 100  # Default
        
        # Estimate usage (conservative approach)
        estimated_daily = daily_log_volume_mb * 0.2  # Assume each log type gets 20% of volume
        estimated_total = min(estimated_daily * days, max_size_mb * days)
        
        breakdown[log_type] = {
            "retention_days": days,
            "max_size_mb": max_size_mb,
            "estimated_daily_mb": estimated_daily,
            "estimated_total_mb": estimated_total
        }
        
        total_estimated_mb += estimated_total
    
    return {
        "total_estimated_mb": total_estimated_mb,
        "total_estimated_gb": total_estimated_mb / 1024,
        "breakdown": breakdown,
        "recommendations": _generate_disk_usage_recommendations(total_estimated_mb, breakdown)
    }


def _generate_disk_usage_recommendations(total_mb: float, breakdown: Dict) -> List[str]:
    """Generate recommendations based on estimated disk usage."""
    recommendations = []
    
    if total_mb > 10240:  # > 10 GB
        recommendations.append("Consider reducing retention periods or implementing log archival")
    
    if total_mb > 51200:  # > 50 GB
        recommendations.append("High disk usage expected - implement log compression and archival strategy")
    
    # Check for individual log types with high usage
    for log_type, info in breakdown.items():
        if info["estimated_total_mb"] > 5120:  # > 5 GB
            recommendations.append(f"Consider reducing retention for {log_type} logs")
    
    if not recommendations:
        recommendations.append("Disk usage estimates are within reasonable limits")
    
    return recommendations


def setup_advanced_log_rotation(
    log_path: Path,
    log_type: str,
    file_format,
    environment: str,
    custom_policies: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Set up advanced log rotation with environment-specific policies.
    
    Args:
        log_path: Path to log directory
        log_type: Type of log (main, debug, error, performance, etc.)
        file_format: Log file format
        environment: Environment name
        custom_policies: Optional custom rotation/retention policies
        
    Returns:
        Dictionary with handler configuration details
    """
    # Get default policies for environment
    default_policies = get_rotation_retention_policies(environment)
    
    # Use custom policies if provided, otherwise use defaults
    if custom_policies and log_type in custom_policies:
        policy = custom_policies[log_type]
    elif log_type in default_policies:
        policy = default_policies[log_type]
    else:
        # Fallback to main policy
        policy = default_policies.get("main", {
            "rotation": "1 day",
            "retention": "7 days", 
            "compression": "zip",
            "max_size": "100 MB"
        })
    
    # Validate the policy
    validation = validate_rotation_retention_config(
        policy.get("rotation", "1 day"),
        policy.get("retention", "7 days")
    )
    
    if not validation["valid"]:
        logger.warning(
            f"Invalid rotation/retention policy for {log_type}",
            extra={
                "component": "logging",
                "operation": "setup_rotation",
                "log_type": log_type,
                "errors": validation["errors"]
            }
        )
        # Use safe defaults
        policy = {
            "rotation": "1 day",
            "retention": "7 days",
            "compression": "zip",
            "max_size": "100 MB"
        }
    
    # Create the log file path
    log_file = log_path / f"{log_type}_{'{time:YYYY-MM-DD}'}.log"
    
    # Configure rotation - support both time and size-based
    rotation_config = policy.get("rotation", "1 day")
    if "MB" in rotation_config or "GB" in rotation_config:
        # Size-based rotation
        rotation = rotation_config
    else:
        # Time-based rotation
        rotation = rotation_config
    
    # Add the handler
    handler_id = logger.add(
        str(log_file),
        format=file_format,
        level="DEBUG",  # Let individual filters control level
        rotation=rotation,
        retention=policy.get("retention", "7 days"),
        compression=policy.get("compression", "zip"),
        enqueue=True,
        backtrace=True,
        diagnose=True
    )
    
    return {
        "handler_id": handler_id,
        "log_file": str(log_file),
        "policy": policy,
        "validation": validation
    }
    """Add default context to all log records."""
    if "component" not in record["extra"]:
        record["extra"]["component"] = "core"
    if "operation" not in record["extra"]:
        record["extra"]["operation"] = ""
    return True


def get_logger(name: Optional[str] = None, component: str = "core"):
    """
    Get a structured logger instance with consistent context.
    
    Args:
        name: Optional logger name
        component: Component name for structured logging
        
    Returns:
        Logger instance with structured context
    """
    if name:
        return logger.bind(name=name, component=component)
    return logger.bind(component=component)


def log_function_call(func_name: str, component: str = "core", args: Dict[str, Any] = None, kwargs: Dict[str, Any] = None):
    """
    Log function call with structured parameters.
    
    Args:
        func_name: Function name
        component: Component name
        args: Function arguments
        kwargs: Function keyword arguments
    """
    context = {
        "function_name": func_name,
        "args": args or {},
        "kwargs": kwargs or {}
    }
    
    logger.debug(
        f"Function call: {func_name}",
        extra={
            "component": component,
            "operation": "function_call",
            "context": context
        }
    )


def log_performance(operation: str, duration: float, component: str = "core", details: Dict[str, Any] = None):
    """
    Log performance metrics with structured data.
    
    Args:
        operation: Operation name
        duration: Duration in seconds
        component: Component name
        details: Additional performance details
    """
    context = details or {}
    context.update({
        "operation_name": operation,
        "execution_time_seconds": duration,
        "performance_category": _categorize_performance(duration)
    })
    
    logger.info(
        f"Performance: {operation} completed in {duration:.3f}s",
        extra={
            "component": component,
            "operation": "performance",
            "log_type": "performance",
            "duration": duration,
            "context": context
        }
    )


def log_error_with_context(error: Exception, component: str = "core", context: Dict[str, Any] = None, operation: str = ""):
    """
    Log error with structured context and stack trace.
    
    Args:
        error: Exception object
        component: Component name
        context: Additional context information
        operation: Operation being performed when error occurred
    """
    import traceback
    
    error_context = context or {}
    error_context.update({
        "error_class": type(error).__name__,
        "error_message": str(error),
        "operation_context": operation
    })
    
    logger.error(
        f"Error in {component}: {type(error).__name__}: {str(error)}",
        extra={
            "component": component,
            "operation": operation or "error_handling",
            "error_type": type(error).__name__,
            "stack_trace": traceback.format_exc(),
            "context": error_context
        }
    )


def log_system_event(event_type: str, message: str, component: str = "core", details: Dict[str, Any] = None):
    """
    Log system events with structured data.
    
    Args:
        event_type: Type of system event (startup, shutdown, config_change, etc.)
        message: Event message
        component: Component name
        details: Additional event details
    """
    context = details or {}
    context.update({
        "event_type": event_type,
        "timestamp": datetime.now().isoformat()
    })
    
    logger.info(
        f"System event: {event_type} - {message}",
        extra={
            "component": component,
            "operation": "system_event",
            "context": context
        }
    )


def log_data_operation(operation: str, data_type: str, count: int, component: str = "data", details: Dict[str, Any] = None):
    """
    Log data operations with structured metrics.
    
    Args:
        operation: Operation type (read, write, delete, process)
        data_type: Type of data (video, frame, pose, analysis)
        count: Number of items processed
        component: Component name
        details: Additional operation details
    """
    context = details or {}
    context.update({
        "data_operation": operation,
        "data_type": data_type,
        "item_count": count,
        "batch_size": context.get("batch_size", 1)
    })
    
    logger.info(
        f"Data operation: {operation} {count} {data_type} items",
        extra={
            "component": component,
            "operation": "data_operation",
            "context": context
        }
    )


def _categorize_performance(duration: float) -> str:
    """Categorize performance based on duration."""
    if duration < 0.1:
        return "fast"
    elif duration < 1.0:
        return "normal"
    elif duration < 5.0:
        return "slow"
    else:
        return "very_slow"


class LoggingContext:
    """Context manager for structured logging with consistent formatting."""
    
    def __init__(self, operation: str, component: str = "core", **context):
        self.operation = operation
        self.component = component
        self.context = context
        self.start_time = None
        self.logger = logger.bind(component=component, operation=operation)
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        
        self.logger.info(
            f"Starting operation: {self.operation}",
            extra={
                "component": self.component,
                "operation": self.operation,
                "context": self.context,
                "phase": "start"
            }
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(
                f"Completed operation: {self.operation}",
                extra={
                    "component": self.component,
                    "operation": self.operation,
                    "duration": duration,
                    "context": self.context,
                    "phase": "complete",
                    "status": "success"
                }
            )
        else:
            self.logger.error(
                f"Failed operation: {self.operation}",
                extra={
                    "component": self.component,
                    "operation": self.operation,
                    "duration": duration,
                    "context": self.context,
                    "phase": "error",
                    "status": "failed",
                    "error_type": type(exc_val).__name__,
                    "error_message": str(exc_val)
                }
            )
        
        return False  # Don't suppress exceptions
    
    def log_progress(self, message: str, progress: float = None, **details):
        """Log progress within the operation context."""
        context = self.context.copy()
        context.update(details)
        if progress is not None:
            context["progress_percent"] = progress
        
        self.logger.info(
            f"Progress: {self.operation} - {message}",
            extra={
                "component": self.component,
                "operation": self.operation,
                "context": context,
                "phase": "progress"
            }
        )


class StructuredLogger:
    """Structured logger wrapper for consistent formatting across components."""
    
    def __init__(self, component: str):
        self.component = component
        self.logger = logger.bind(component=component)
    
    def debug(self, message: str, operation: str = "", **context):
        """Log debug message with structured context."""
        self.logger.debug(
            message,
            extra={
                "component": self.component,
                "operation": operation,
                "context": context
            }
        )
    
    def info(self, message: str, operation: str = "", **context):
        """Log info message with structured context."""
        self.logger.info(
            message,
            extra={
                "component": self.component,
                "operation": operation,
                "context": context
            }
        )
    
    def warning(self, message: str, operation: str = "", **context):
        """Log warning message with structured context."""
        self.logger.warning(
            message,
            extra={
                "component": self.component,
                "operation": operation,
                "context": context
            }
        )
    
    def error(self, message: str, operation: str = "", error: Exception = None, **context):
        """Log error message with structured context."""
        if error:
            import traceback
            context.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "stack_trace": traceback.format_exc()
            })
        
        self.logger.error(
            message,
            extra={
                "component": self.component,
                "operation": operation,
                "context": context
            }
        )
    
    def performance(self, operation: str, duration: float, **details):
        """Log performance metrics with structured data."""
        log_performance(operation, duration, self.component, details)
    
    def data_operation(self, operation: str, data_type: str, count: int, **details):
        """Log data operations with structured metrics."""
        log_data_operation(operation, data_type, count, self.component, details)
    
    def system_event(self, event_type: str, message: str, **details):
        """Log system events with structured data."""
        log_system_event(event_type, message, self.component, details)


def configure_external_logging(service: str, **config):
    """
    Configure external logging service with structured format.
    
    Args:
        service: Service name (papertrail, datadog, etc.)
        **config: Service-specific configuration
    """
    if service == "papertrail":
        # Configure Papertrail logging
        host = config.get("host")
        port = config.get("port")
        if host and port:
            try:
                import socket
                handler = logger.add(
                    lambda msg: _send_to_papertrail(msg, host, port),
                    level="INFO",
                    format=_json_formatter,
                    filter=_add_default_context
                )
                log_system_event(
                    "external_logging_configured",
                    f"Papertrail logging configured: {host}:{port}",
                    "logging",
                    {"service": "papertrail", "host": host, "port": port}
                )
            except Exception as e:
                log_error_with_context(
                    e,
                    "logging",
                    {"service": "papertrail", "host": host, "port": port},
                    "configure_papertrail"
                )
    
    elif service == "datadog":
        # Configure Datadog logging
        api_key = config.get("api_key")
        if api_key:
            try:
                # This would require datadog library
                log_system_event(
                    "external_logging_configured",
                    "Datadog logging configured",
                    "logging",
                    {"service": "datadog"}
                )
            except Exception as e:
                log_error_with_context(
                    e,
                    "logging",
                    {"service": "datadog"},
                    "configure_datadog"
                )
    
    else:
        logger.warning(
            f"Unknown external logging service: {service}",
            extra={
                "component": "logging",
                "operation": "configure_external",
                "context": {"service": service, "available_services": ["papertrail", "datadog"]}
            }
        )


def _send_to_papertrail(message: str, host: str, port: int):
    """Send structured log message to Papertrail."""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(message.encode('utf-8'), (host, port))
        sock.close()
    except Exception:
        pass  # Fail silently for external logging


def log_security_event(event_type: str, message: str, component: str = "security", details: Dict[str, Any] = None):
    """
    Log security-related events with structured data.
    
    Args:
        event_type: Type of security event (authentication, authorization, etc.)
        message: Security event message
        component: Component name
        details: Additional security event details
    """
    context = details or {}
    context.update({
        "security_event_type": event_type,
        "timestamp": datetime.now().isoformat()
    })
    
    logger.warning(
        f"Security event: {event_type} - {message}",
        extra={
            "component": component,
            "operation": "security_event",
            "log_type": "security",
            "context": context
        }
    )


def log_test_result(test_name: str, result: str, duration: float = None, component: str = "testing", details: Dict[str, Any] = None):
    """
    Log test results with structured data.
    
    Args:
        test_name: Name of the test
        result: Test result (passed, failed, skipped)
        duration: Test duration in seconds
        component: Component name
        details: Additional test details
    """
    context = details or {}
    context.update({
        "test_name": test_name,
        "test_result": result,
        "test_duration": duration,
        "timestamp": datetime.now().isoformat()
    })
    
    log_level = "INFO" if result == "passed" else "WARNING" if result == "skipped" else "ERROR"
    
    getattr(logger, log_level.lower())(
        f"Test result: {test_name} - {result}",
        extra={
            "component": component,
            "operation": "test_result",
            "log_type": "test_result",
            "duration": duration,
            "context": context
        }
    )
    """
    Create a structured logger for a specific component.
    
    Args:
        component_name: Name of the component
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(component_name)


def auto_configure_logging(environment: str = None) -> None:
    """
    Automatically configure logging based on environment detection.
    
    Args:
        environment: Optional environment override. If not provided, will detect from
                    ENVIRONMENT environment variable or default to 'development'
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    
    env_lower = environment.lower()
    
    if env_lower == "production":
        configure_production_logging()
    elif env_lower == "development":
        configure_development_logging()
    elif env_lower in ["staging", "testing"]:
        # Use production-like settings but with more verbose logging
        setup_logging(
            log_level="DEBUG",
            log_dir=f"logs/{env_lower}",
            format_type="structured",
            enable_json_logs=True,
            environment=environment,
            rotation="6 hours",
            retention="14 days"
        )
    else:
        # Unknown environment, use development settings as fallback
        logger.warning(f"Unknown environment '{environment}', using development logging configuration")
        configure_development_logging()


def get_logging_status() -> Dict[str, Any]:
    """
    Get current logging configuration status.
    
    Returns:
        Dictionary containing logging status information
    """
    import os
    
    # Get current handlers
    handlers_info = []
    for handler_id, handler in logger._core.handlers.items():
        handler_info = {
            "id": handler_id,
            "level": handler.levelno,
            "level_name": handler.levelname if hasattr(handler, 'levelname') else "UNKNOWN"
        }
        
        # Try to get additional handler information
        if hasattr(handler, '_sink'):
            sink = handler._sink
            if hasattr(sink, '_path'):
                handler_info["file_path"] = str(sink._path)
            elif hasattr(sink, 'write'):
                handler_info["type"] = "console" if sink.write == sys.stderr.write else "custom"
        
        handlers_info.append(handler_info)
    
    return {
        "environment": os.getenv("ENVIRONMENT", "development"),
        "handlers_count": len(handlers_info),
        "handlers": handlers_info,
        "log_directory": os.getenv("LOG_DIRECTORY", "logs"),
        "current_level": logger.level("INFO").no,  # Get current effective level
        "available_levels": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    }


def validate_logging_configuration(environment: str = None) -> Dict[str, Any]:
    """
    Validate current logging configuration.
    
    Args:
        environment: Environment to validate against
        
    Returns:
        Dictionary containing validation results
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "environment": environment
    }
    
    try:
        # Check if log directory exists and is writable
        log_dir = os.getenv("LOG_DIRECTORY", "logs")
        log_path = Path(log_dir)
        
        if not log_path.exists():
            try:
                log_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                validation_results["errors"].append(f"Cannot create log directory {log_dir}: {e}")
                validation_results["valid"] = False
        
        # Test write permissions
        try:
            test_file = log_path / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            validation_results["errors"].append(f"No write permission for log directory {log_dir}: {e}")
            validation_results["valid"] = False
        
        # Check handler configuration
        if len(logger._core.handlers) == 0:
            validation_results["warnings"].append("No log handlers configured")
        
        # Environment-specific validations
        env_lower = environment.lower()
        if env_lower == "production":
            # Production should have structured logging
            if not any("json" in str(handler).lower() for handler in logger._core.handlers.values()):
                validation_results["warnings"].append("Production environment should use JSON structured logging")
        
        elif env_lower == "development":
            # Development should have debug logging
            debug_handlers = [h for h in logger._core.handlers.values() if h.levelno <= 10]  # DEBUG level
            if not debug_handlers:
                validation_results["warnings"].append("Development environment should have DEBUG level logging")
        
    except Exception as e:
        validation_results["errors"].append(f"Logging validation failed: {e}")
        validation_results["valid"] = False
    
    return validation_results


def cleanup_old_logs(log_dir: str = "logs", dry_run: bool = False) -> Dict[str, Any]:
    """
    Clean up old log files based on current retention policies.
    
    Args:
        log_dir: Directory containing log files
        dry_run: If True, only report what would be deleted without actually deleting
        
    Returns:
        Dictionary with cleanup results
    """
    from datetime import datetime, timedelta
    import os
    import glob
    
    log_path = Path(log_dir)
    if not log_path.exists():
        return {"error": f"Log directory {log_dir} does not exist"}
    
    # Get current environment to determine policies
    environment = os.getenv("ENVIRONMENT", "development")
    policies = get_rotation_retention_policies(environment)
    
    cleanup_results = {
        "dry_run": dry_run,
        "environment": environment,
        "files_processed": 0,
        "files_deleted": 0,
        "space_freed_mb": 0,
        "errors": [],
        "deleted_files": [],
        "kept_files": []
    }
    
    # Process each log type
    for log_type, policy in policies.items():
        retention = policy.get("retention", "30 days")
        
        # Parse retention period to days
        if "day" in retention:
            retention_days = int(retention.split()[0])
        elif "week" in retention:
            retention_days = int(retention.split()[0]) * 7
        elif "month" in retention:
            retention_days = int(retention.split()[0]) * 30
        elif "year" in retention:
            retention_days = int(retention.split()[0]) * 365
        else:
            try:
                retention_days = int(retention)  # Just a number
            except ValueError:
                retention_days = 30  # Default
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # Find log files for this type
        pattern = str(log_path / f"{log_type}_*.log*")
        log_files = glob.glob(pattern)
        
        for log_file in log_files:
            try:
                file_path = Path(log_file)
                file_stat = file_path.stat()
                file_date = datetime.fromtimestamp(file_stat.st_mtime)
                file_size_mb = file_stat.st_size / (1024 * 1024)
                
                cleanup_results["files_processed"] += 1
                
                if file_date < cutoff_date:
                    # File is older than retention period
                    if not dry_run:
                        file_path.unlink()
                        cleanup_results["space_freed_mb"] += file_size_mb
                        cleanup_results["files_deleted"] += 1
                    
                    cleanup_results["deleted_files"].append({
                        "file": str(file_path),
                        "age_days": (datetime.now() - file_date).days,
                        "size_mb": round(file_size_mb, 2)
                    })
                else:
                    cleanup_results["kept_files"].append({
                        "file": str(file_path),
                        "age_days": (datetime.now() - file_date).days,
                        "size_mb": round(file_size_mb, 2)
                    })
                    
            except Exception as e:
                cleanup_results["errors"].append(f"Error processing {log_file}: {e}")
    
    cleanup_results["space_freed_mb"] = round(cleanup_results["space_freed_mb"], 2)
    
    # Log the cleanup operation
    logger.info(
        f"Log cleanup {'simulation' if dry_run else 'completed'}",
        extra={
            "component": "logging",
            "operation": "cleanup_logs",
            "dry_run": dry_run,
            "files_processed": cleanup_results["files_processed"],
            "files_deleted": cleanup_results["files_deleted"],
            "space_freed_mb": cleanup_results["space_freed_mb"],
            "errors_count": len(cleanup_results["errors"])
        }
    )
    
    return cleanup_results


def get_log_statistics(log_dir: str = "logs") -> Dict[str, Any]:
    """
    Get statistics about current log files.
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        Dictionary with log statistics
    """
    import os
    import glob
    from datetime import datetime
    
    log_path = Path(log_dir)
    if not log_path.exists():
        return {"error": f"Log directory {log_dir} does not exist"}
    
    stats = {
        "log_directory": str(log_path),
        "total_files": 0,
        "total_size_mb": 0,
        "oldest_file": None,
        "newest_file": None,
        "by_type": {},
        "by_date": {},
        "compression_stats": {"compressed": 0, "uncompressed": 0}
    }
    
    # Get all log files
    log_files = glob.glob(str(log_path / "*.log*"))
    
    for log_file in log_files:
        try:
            file_path = Path(log_file)
            file_stat = file_path.stat()
            file_size_mb = file_stat.st_size / (1024 * 1024)
            file_date = datetime.fromtimestamp(file_stat.st_mtime)
            
            stats["total_files"] += 1
            stats["total_size_mb"] += file_size_mb
            
            # Track oldest and newest files
            if stats["oldest_file"] is None or file_date < stats["oldest_file"]["date"]:
                stats["oldest_file"] = {"file": str(file_path), "date": file_date}
            
            if stats["newest_file"] is None or file_date > stats["newest_file"]["date"]:
                stats["newest_file"] = {"file": str(file_path), "date": file_date}
            
            # Categorize by log type
            log_type = file_path.name.split('_')[0]
            if log_type not in stats["by_type"]:
                stats["by_type"][log_type] = {"count": 0, "size_mb": 0}
            stats["by_type"][log_type]["count"] += 1
            stats["by_type"][log_type]["size_mb"] += file_size_mb
            
            # Categorize by date
            date_str = file_date.strftime("%Y-%m-%d")
            if date_str not in stats["by_date"]:
                stats["by_date"][date_str] = {"count": 0, "size_mb": 0}
            stats["by_date"][date_str]["count"] += 1
            stats["by_date"][date_str]["size_mb"] += file_size_mb
            
            # Track compression
            if file_path.suffix in ['.gz', '.zip', '.bz2']:
                stats["compression_stats"]["compressed"] += 1
            else:
                stats["compression_stats"]["uncompressed"] += 1
                
        except Exception as e:
            logger.warning(f"Error processing log file {log_file}: {e}")
    
    # Round sizes
    stats["total_size_mb"] = round(stats["total_size_mb"], 2)
    for log_type in stats["by_type"]:
        stats["by_type"][log_type]["size_mb"] = round(stats["by_type"][log_type]["size_mb"], 2)
    for date in stats["by_date"]:
        stats["by_date"][date]["size_mb"] = round(stats["by_date"][date]["size_mb"], 2)
    
    return stats


def monitor_log_disk_usage(log_dir: str = "logs", threshold_gb: float = 5.0) -> Dict[str, Any]:
    """
    Monitor log disk usage and provide alerts if thresholds are exceeded.
    
    Args:
        log_dir: Directory containing log files
        threshold_gb: Alert threshold in GB
        
    Returns:
        Dictionary with monitoring results and recommendations
    """
    import shutil
    
    log_path = Path(log_dir)
    if not log_path.exists():
        return {"error": f"Log directory {log_dir} does not exist"}
    
    # Get disk usage statistics
    total, used, free = shutil.disk_usage(log_path)
    total_gb = total / (1024**3)
    used_gb = used / (1024**3)
    free_gb = free / (1024**3)
    
    # Get log-specific statistics
    log_stats = get_log_statistics(log_dir)
    log_size_gb = log_stats.get("total_size_mb", 0) / 1024
    
    monitoring_result = {
        "log_directory": str(log_path),
        "disk_total_gb": round(total_gb, 2),
        "disk_used_gb": round(used_gb, 2),
        "disk_free_gb": round(free_gb, 2),
        "disk_usage_percent": round((used_gb / total_gb) * 100, 1),
        "logs_size_gb": round(log_size_gb, 2),
        "logs_percent_of_disk": round((log_size_gb / total_gb) * 100, 2),
        "threshold_gb": threshold_gb,
        "threshold_exceeded": log_size_gb > threshold_gb,
        "recommendations": []
    }
    
    # Generate recommendations
    if log_size_gb > threshold_gb:
        monitoring_result["recommendations"].append(f"Log size ({log_size_gb:.1f} GB) exceeds threshold ({threshold_gb} GB)")
        monitoring_result["recommendations"].append("Consider running log cleanup or reducing retention periods")
    
    if free_gb < 1.0:  # Less than 1 GB free
        monitoring_result["recommendations"].append("Low disk space - immediate cleanup recommended")
    elif free_gb < 5.0:  # Less than 5 GB free
        monitoring_result["recommendations"].append("Disk space getting low - consider cleanup")
    
    if monitoring_result["disk_usage_percent"] > 90:
        monitoring_result["recommendations"].append("Disk usage very high - urgent cleanup needed")
    elif monitoring_result["disk_usage_percent"] > 80:
        monitoring_result["recommendations"].append("Disk usage high - cleanup recommended")
    
    # Log the monitoring results
    log_level = "WARNING" if monitoring_result["threshold_exceeded"] else "INFO"
    getattr(logger, log_level.lower())(
        f"Log disk usage monitoring: {log_size_gb:.1f} GB used",
        extra={
            "component": "logging",
            "operation": "monitor_disk_usage",
            "logs_size_gb": log_size_gb,
            "threshold_exceeded": monitoring_result["threshold_exceeded"],
            "disk_free_gb": free_gb,
            "recommendations_count": len(monitoring_result["recommendations"])
        }
    )
    
    return monitoring_result


def create_component_logger(component_name: str) -> StructuredLogger:
    """
    Create a structured logger for a specific component.
    
    Args:
        component_name: Name of the component
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(component_name)


# Decorator utilities for automatic function logging
def log_function_execution(component: str = "core", log_args: bool = False, log_result: bool = False, log_duration: bool = True):
    """
    Decorator to automatically log function execution with structured data.
    
    Args:
        component: Component name for logging context
        log_args: Whether to log function arguments
        log_result: Whether to log function return value
        log_duration: Whether to log execution duration
    
    Usage:
        @log_function_execution(component="video", log_args=True, log_duration=True)
        def process_video(video_path: str, frame_rate: float = 30.0):
            # Function implementation
            return result
    """
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = create_component_logger(component)
            start_time = time.time() if log_duration else None
            
            # Log function start
            context = {"function": func.__name__}
            if log_args:
                context.update({
                    "args": args[:3] if len(args) > 3 else args,  # Limit args to prevent huge logs
                    "kwargs": {k: v for k, v in list(kwargs.items())[:5]}  # Limit kwargs
                })
            
            func_logger.debug(f"Starting function: {func.__name__}", 
                            operation="function_start", **context)
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful completion
                end_context = {"function": func.__name__, "status": "success"}
                if log_duration and start_time:
                    duration = time.time() - start_time
                    end_context["duration_seconds"] = round(duration, 3)
                    
                if log_result and result is not None:
                    # Only log simple result types to avoid huge logs
                    if isinstance(result, (str, int, float, bool, list, dict)):
                        if isinstance(result, (list, dict)) and len(str(result)) > 200:
                            end_context["result_type"] = type(result).__name__
                            end_context["result_size"] = len(result) if hasattr(result, '__len__') else None
                        else:
                            end_context["result"] = result
                    else:
                        end_context["result_type"] = type(result).__name__
                
                func_logger.info(f"Completed function: {func.__name__}", 
                               operation="function_complete", **end_context)
                
                return result
                
            except Exception as e:
                # Log error
                error_context = {"function": func.__name__, "status": "error"}
                if log_duration and start_time:
                    duration = time.time() - start_time
                    error_context["duration_seconds"] = round(duration, 3)
                
                func_logger.error(f"Function failed: {func.__name__}", 
                                operation="function_error", error=e, **error_context)
                raise
        
        return wrapper
    return decorator


def log_async_function_execution(component: str = "core", log_args: bool = False, log_result: bool = False, log_duration: bool = True):
    """
    Decorator to automatically log async function execution with structured data.
    
    Args:
        component: Component name for logging context
        log_args: Whether to log function arguments
        log_result: Whether to log function return value
        log_duration: Whether to log execution duration
    
    Usage:
        @log_async_function_execution(component="api", log_args=True)
        async def process_upload(file_data: bytes):
            # Async function implementation
            return result
    """
    def decorator(func):
        import functools
        import time
        import asyncio
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            func_logger = create_component_logger(component)
            start_time = time.time() if log_duration else None
            
            # Log function start
            context = {"function": func.__name__, "async": True}
            if log_args:
                context.update({
                    "args": args[:3] if len(args) > 3 else args,
                    "kwargs": {k: v for k, v in list(kwargs.items())[:5]}
                })
            
            func_logger.debug(f"Starting async function: {func.__name__}", 
                            operation="async_function_start", **context)
            
            try:
                result = await func(*args, **kwargs)
                
                # Log successful completion
                end_context = {"function": func.__name__, "status": "success", "async": True}
                if log_duration and start_time:
                    duration = time.time() - start_time
                    end_context["duration_seconds"] = round(duration, 3)
                    
                if log_result and result is not None:
                    if isinstance(result, (str, int, float, bool, list, dict)):
                        if isinstance(result, (list, dict)) and len(str(result)) > 200:
                            end_context["result_type"] = type(result).__name__
                            end_context["result_size"] = len(result) if hasattr(result, '__len__') else None
                        else:
                            end_context["result"] = result
                    else:
                        end_context["result_type"] = type(result).__name__
                
                func_logger.info(f"Completed async function: {func.__name__}", 
                               operation="async_function_complete", **end_context)
                
                return result
                
            except Exception as e:
                # Log error
                error_context = {"function": func.__name__, "status": "error", "async": True}
                if log_duration and start_time:
                    duration = time.time() - start_time
                    error_context["duration_seconds"] = round(duration, 3)
                
                func_logger.error(f"Async function failed: {func.__name__}", 
                                operation="async_function_error", error=e, **error_context)
                raise
        
        return wrapper
    return decorator


class BatchOperationLogger:
    """
    Utility for logging batch operations with progress tracking and statistics.
    
    Usage:
        batch_logger = BatchOperationLogger("video_processing", "process_videos", total_items=100)
        
        for i, video in enumerate(videos):
            with batch_logger.log_item(i, {"video_id": video.id}):
                process_video(video)
        
        batch_logger.log_completion()
    """
    
    def __init__(self, component: str, operation: str, total_items: int = None, log_interval: int = 10):
        self.component = component
        self.operation = operation
        self.total_items = total_items
        self.log_interval = log_interval
        self.logger = create_component_logger(component)
        
        self.processed_count = 0
        self.success_count = 0
        self.error_count = 0
        self.start_time = None
        self.errors = []
        
    def start_batch(self, **context):
        """Start the batch operation logging."""
        import time
        self.start_time = time.time()
        
        batch_context = {
            "total_items": self.total_items,
            "batch_id": id(self)
        }
        batch_context.update(context)
        
        self.logger.info(f"Starting batch operation: {self.operation}", 
                        operation="batch_start", **batch_context)
    
    def log_item(self, item_index: int = None, item_context: Dict[str, Any] = None):
        """Context manager for logging individual item processing."""
        return BatchItemLogger(self, item_index, item_context or {})
    
    def log_progress(self, force: bool = False):
        """Log progress if interval reached or forced."""
        if force or (self.processed_count % self.log_interval == 0):
            import time
            
            progress_context = {
                "processed": self.processed_count,
                "success": self.success_count,
                "errors": self.error_count,
                "batch_id": id(self)
            }
            
            if self.total_items:
                progress_context["progress_percent"] = round((self.processed_count / self.total_items) * 100, 1)
                progress_context["remaining"] = self.total_items - self.processed_count
            
            if self.start_time:
                elapsed = time.time() - self.start_time
                progress_context["elapsed_seconds"] = round(elapsed, 1)
                
                if self.processed_count > 0:
                    avg_time_per_item = elapsed / self.processed_count
                    progress_context["avg_time_per_item"] = round(avg_time_per_item, 3)
                    
                    if self.total_items:
                        remaining_items = self.total_items - self.processed_count
                        estimated_remaining = remaining_items * avg_time_per_item
                        progress_context["estimated_remaining_seconds"] = round(estimated_remaining, 1)
            
            self.logger.info(f"Batch progress: {self.operation}", 
                           operation="batch_progress", **progress_context)
    
    def log_completion(self, **context):
        """Log batch operation completion with statistics."""
        import time
        
        completion_context = {
            "total_processed": self.processed_count,
            "successful": self.success_count,
            "failed": self.error_count,
            "success_rate": round((self.success_count / max(self.processed_count, 1)) * 100, 1),
            "batch_id": id(self)
        }
        
        if self.start_time:
            total_duration = time.time() - self.start_time
            completion_context["total_duration_seconds"] = round(total_duration, 1)
            
            if self.processed_count > 0:
                completion_context["avg_time_per_item"] = round(total_duration / self.processed_count, 3)
        
        if self.errors:
            completion_context["error_summary"] = self.errors[:5]  # First 5 errors
            completion_context["total_error_types"] = len(set(error["error_type"] for error in self.errors))
        
        completion_context.update(context)
        
        log_level = "warning" if self.error_count > 0 else "info"
        getattr(self.logger, log_level)(f"Batch operation completed: {self.operation}", 
                                       operation="batch_complete", **completion_context)


class BatchItemLogger:
    """Context manager for individual batch item logging."""
    
    def __init__(self, batch_logger: BatchOperationLogger, item_index: int, item_context: Dict[str, Any]):
        self.batch_logger = batch_logger
        self.item_index = item_index
        self.item_context = item_context
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.batch_logger.processed_count += 1
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        
        if exc_type is None:
            # Success
            self.batch_logger.success_count += 1
        else:
            # Error
            self.batch_logger.error_count += 1
            
            error_info = {
                "item_index": self.item_index,
                "error_type": exc_type.__name__,
                "error_message": str(exc_val),
                "item_context": self.item_context
            }
            self.batch_logger.errors.append(error_info)
            
            # Log individual error
            self.batch_logger.logger.error(
                f"Batch item failed: {self.batch_logger.operation}",
                operation="batch_item_error",
                error=exc_val,
                item_index=self.item_index,
                **self.item_context
            )
        
        # Log progress if needed
        self.batch_logger.log_progress()
        
        return False  # Don't suppress exceptions


class APIRequestLogger:
    """
    Utility for consistent API request/response logging.
    
    Usage:
        api_logger = APIRequestLogger("upload_api")
        
        @api_logger.log_request
        async def upload_video(request: Request, file: UploadFile):
            # API endpoint implementation
            return response
    """
    
    def __init__(self, component: str):
        self.component = component
        self.logger = create_component_logger(component)
    
    def log_request(self, func):
        """Decorator for logging API requests and responses."""
        import functools
        import time
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Extract request information
            request = None
            for arg in args:
                if hasattr(arg, 'method') and hasattr(arg, 'url'):
                    request = arg
                    break
            
            request_context = {
                "function": func.__name__,
                "endpoint": func.__name__
            }
            
            if request:
                request_context.update({
                    "method": request.method,
                    "url": str(request.url),
                    "client_ip": getattr(request.client, 'host', 'unknown') if request.client else 'unknown',
                    "user_agent": request.headers.get("user-agent", "unknown")
                })
            
            self.logger.info(f"API request: {func.__name__}", 
                           operation="api_request", **request_context)
            
            try:
                result = await func(*args, **kwargs)
                
                # Log successful response
                duration = time.time() - start_time
                response_context = {
                    "function": func.__name__,
                    "endpoint": func.__name__,
                    "status": "success",
                    "duration_seconds": round(duration, 3)
                }
                
                if hasattr(result, 'status_code'):
                    response_context["status_code"] = result.status_code
                
                self.logger.info(f"API response: {func.__name__}", 
                               operation="api_response", **response_context)
                
                return result
                
            except Exception as e:
                # Log error response
                duration = time.time() - start_time
                error_context = {
                    "function": func.__name__,
                    "endpoint": func.__name__,
                    "status": "error",
                    "duration_seconds": round(duration, 3)
                }
                
                self.logger.error(f"API error: {func.__name__}", 
                                operation="api_error", error=e, **error_context)
                raise
        
        return wrapper


def log_configuration_change(component: str, config_key: str, old_value: Any, new_value: Any, reason: str = ""):
    """
    Log configuration changes with structured data.
    
    Args:
        component: Component that changed configuration
        config_key: Configuration key that changed
        old_value: Previous value
        new_value: New value
        reason: Reason for the change
    """
    change_context = {
        "config_key": config_key,
        "old_value": old_value,
        "new_value": new_value,
        "change_reason": reason,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(
        f"Configuration changed: {config_key}",
        extra={
            "component": component,
            "operation": "config_change",
            "context": change_context
        }
    )


def log_resource_usage(component: str, operation: str, **metrics):
    """
    Log resource usage metrics with structured data.
    
    Args:
        component: Component name
        operation: Operation being performed
        **metrics: Resource usage metrics (memory_mb, cpu_percent, disk_mb, etc.)
    
    Usage:
        log_resource_usage("video_processor", "extract_frames", 
                          memory_mb=512, cpu_percent=85, processing_time=30.5)
    """
    resource_context = {
        "operation": operation,
        "timestamp": datetime.now().isoformat()
    }
    resource_context.update(metrics)
    
    logger.info(
        f"Resource usage: {operation}",
        extra={
            "component": component,
            "operation": "resource_usage",
            "log_type": "performance",
            "context": resource_context
        }
    )


def log_model_operation(component: str, model_name: str, operation: str, **details):
    """
    Log machine learning model operations with structured data.
    
    Args:
        component: Component name
        model_name: Name of the ML model
        operation: Operation (load, predict, train, etc.)
        **details: Additional operation details
    
    Usage:
        log_model_operation("pose_estimator", "mediapipe", "predict", 
                           input_shape=(480, 640, 3), confidence=0.85, keypoints=17)
    """
    model_context = {
        "model_name": model_name,
        "model_operation": operation,
        "timestamp": datetime.now().isoformat()
    }
    model_context.update(details)
    
    logger.info(
        f"Model operation: {model_name} - {operation}",
        extra={
            "component": component,
            "operation": "model_operation",
            "context": model_context
        }
    )


def create_operation_logger(component: str, operation: str):
    """
    Create a logger specifically for a long-running operation with consistent context.
    
    Args:
        component: Component name
        operation: Operation name
    
    Returns:
        StructuredLogger with operation context pre-configured
    
    Usage:
        op_logger = create_operation_logger("video_processor", "extract_frames")
        op_logger.info("Starting frame extraction", frame_count=100)
        op_logger.debug("Processing frame", frame_number=50)
    """
    base_logger = create_component_logger(component)
    
    class OperationLogger:
        def __init__(self, base_logger, operation):
            self.base_logger = base_logger
            self.operation = operation
        
        def debug(self, message: str, **context):
            self.base_logger.debug(message, operation=self.operation, **context)
        
        def info(self, message: str, **context):
            self.base_logger.info(message, operation=self.operation, **context)
        
        def warning(self, message: str, **context):
            self.base_logger.warning(message, operation=self.operation, **context)
        
        def error(self, message: str, error: Exception = None, **context):
            self.base_logger.error(message, operation=self.operation, error=error, **context)
        
        def performance(self, duration: float, **details):
            self.base_logger.performance(self.operation, duration, **details)
    
    return OperationLogger(base_logger, operation)


def log_startup_info(component: str, version: str = None, config: Dict[str, Any] = None):
    """
    Log application startup information with structured data.
    
    Args:
        component: Component name
        version: Application version
        config: Configuration summary
    """
    context = {}
    if version:
        context["version"] = version
    if config:
        context["config"] = config
    
    log_system_event(
        "application_startup",
        f"{component} starting up",
        component,
        context
    )


def log_shutdown_info(component: str, reason: str = "normal"):
    """
    Log application shutdown information.
    
    Args:
        component: Component name
        reason: Shutdown reason
    """
    log_system_event(
        "application_shutdown",
        f"{component} shutting down",
        component,
        {"shutdown_reason": reason}
    )


def configure_development_logging():
    """Configure logging for development environment with optimized rotation."""
    policies = get_rotation_retention_policies("development")
    main_policy = policies["main"]
    
    setup_logging(
        log_level="DEBUG",
        log_dir="logs/dev",
        format_type="structured",
        enable_json_logs=False,
        environment="development",
        rotation=main_policy["rotation"],
        retention=main_policy["retention"]
    )


def configure_production_logging():
    """Configure logging for production environment with extended retention."""
    policies = get_rotation_retention_policies("production")
    main_policy = policies["main"]
    
    setup_logging(
        log_level="INFO",
        log_dir="logs",
        format_type="structured",
        enable_json_logs=True,
        environment="production",
        rotation=main_policy["rotation"],
        retention=main_policy["retention"]
    )


def configure_logging_from_config(config_manager) -> None:
    """
    Configure logging based on configuration manager settings.
    
    Args:
        config_manager: ConfigurationManager instance with logging configuration
    """
    try:
        logging_config = config_manager.config.logging
        environment = config_manager.environment
        
        # Determine if JSON logging should be enabled
        enable_json = logging_config.external_logging.enabled or environment.lower() == "production"
        
        # Set up logging with configuration values
        setup_logging(
            log_level=logging_config.level,
            log_dir=config_manager.config.storage.logs_directory,
            rotation=logging_config.rotation,
            retention=logging_config.retention,
            compression=logging_config.compression,
            format_type=logging_config.format,
            enable_json_logs=enable_json,
            environment=environment
        )
        
        # Configure external logging if enabled
        if logging_config.external_logging.enabled:
            configure_external_logging(
                logging_config.external_logging.service,
                # Additional config would come from environment variables
            )
        
        logger.info(
            f"Logging configured for {environment} environment",
            extra={
                "component": "logging",
                "operation": "configure_from_config",
                "environment": environment,
                "log_level": logging_config.level,
                "external_logging": logging_config.external_logging.enabled
            }
        )
        
    except Exception as e:
        # Fallback to basic logging if configuration fails
        logger.error(f"Failed to configure logging from config: {e}")
        if config_manager.environment.lower() == "production":
            configure_production_logging()
        else:
            configure_development_logging()


def get_environment_log_level(environment: str, requested_level: str = "INFO") -> str:
    """
    Get appropriate log level based on environment and requested level.
    
    Args:
        environment: Environment name (development, production, etc.)
        requested_level: Requested log level
        
    Returns:
        Appropriate log level for the environment
    """
    env_lower = environment.lower()
    
    # Environment-specific log level mappings
    if env_lower == "development":
        # Development allows all log levels
        return requested_level
    elif env_lower == "production":
        # Production has more restrictive console logging
        level_hierarchy = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if requested_level in level_hierarchy:
            # Don't allow DEBUG in production console (but files still get DEBUG)
            if requested_level == "DEBUG":
                return "INFO"
            return requested_level
        return "INFO"  # Default for production
    elif env_lower in ["staging", "testing"]:
        # Staging/testing environments use INFO as minimum
        level_hierarchy = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if requested_level in level_hierarchy:
            level_index = level_hierarchy.index(requested_level)
            info_index = level_hierarchy.index("INFO")
            if level_index < info_index:
                return "INFO"
            return requested_level
        return "INFO"
    else:
        # Unknown environment, use requested level
        return requested_level


def setup_environment_specific_handlers(environment: str, log_path: Path, file_format, rotation: str, retention: str, compression: str) -> None:
    """
    Set up environment-specific log handlers with advanced rotation policies.
    
    Args:
        environment: Environment name
        log_path: Path to log directory
        file_format: Log file format
        rotation: Default rotation policy
        retention: Default retention policy
        compression: Compression format
    """
    env_lower = environment.lower()
    policies = get_rotation_retention_policies(environment)
    
    if env_lower == "development":
        # Development-specific handlers with optimized policies
        
        # Separate debug log file with frequent rotation
        debug_policy = policies.get("debug", {"rotation": "3 hours", "retention": "1 day", "compression": compression})
        logger.add(
            log_path / "debug_{time:YYYY-MM-DD}.log",
            format=file_format,
            level="DEBUG",
            rotation=debug_policy.get("rotation", "3 hours"),
            retention=debug_policy.get("retention", "1 day"),
            compression=debug_policy.get("compression", compression),
            enqueue=True,
            filter=lambda record: record["level"].name == "DEBUG"
        )
        
        # Development info log (non-debug messages) with moderate retention
        logger.add(
            log_path / "dev_info_{time:YYYY-MM-DD}.log",
            format=file_format,
            level="INFO",
            rotation=policies["main"].get("rotation", rotation),
            retention="7 days",
            compression=compression,
            enqueue=True,
            filter=lambda record: record["level"].name in ["INFO", "WARNING"]
        )
        
    elif env_lower == "production":
        # Production-specific handlers with long retention
        
        # Critical errors log for production monitoring with extended retention
        critical_policy = policies.get("critical", {"rotation": "1 week", "retention": "1 year", "compression": compression})
        logger.add(
            log_path / "critical_{time:YYYY-MM-DD}.log",
            format=file_format,
            level="CRITICAL",
            rotation=critical_policy.get("rotation", "1 week"),
            retention=critical_policy.get("retention", "1 year"),
            compression=critical_policy.get("compression", compression),
            enqueue=True,
            filter=lambda record: record["level"].name == "CRITICAL"
        )
        
        # Security-related log for production with extended retention
        security_policy = policies.get("security", {"rotation": "1 day", "retention": "1 year", "compression": compression})
        logger.add(
            log_path / "security_{time:YYYY-MM-DD}.log",
            format=file_format,
            level="WARNING",
            rotation=security_policy.get("rotation", "1 day"),
            retention=security_policy.get("retention", "1 year"),
            compression=security_policy.get("compression", compression),
            enqueue=True,
            filter=lambda record: record["extra"].get("log_type") == "security"
        )
        
    elif env_lower in ["staging", "testing"]:
        # Staging/testing specific handlers with moderate retention
        
        # Test results log with specific retention policy
        test_policy = policies.get("test_results", {"rotation": "1 day", "retention": "30 days", "compression": compression})
        logger.add(
            log_path / "test_results_{time:YYYY-MM-DD}.log",
            format=file_format,
            level="INFO",
            rotation=test_policy.get("rotation", "1 day"),
            retention=test_policy.get("retention", "30 days"),
            compression=test_policy.get("compression", compression),
            enqueue=True,
            filter=lambda record: record["extra"].get("log_type") == "test_result"
        )
    
    # Log the handler setup completion
    logger.info(
        f"Environment-specific log handlers configured for {environment}",
        extra={
            "component": "logging",
            "operation": "setup_environment_handlers",
            "environment": environment,
            "policies_applied": list(policies.keys())
        }
    )


# Additional convenience functions for common logging patterns

def log_video_processing(component: str, video_path: str, operation: str, **details):
    """
    Log video processing operations with standardized format.
    
    Args:
        component: Component name
        video_path: Path to video being processed
        operation: Processing operation (load, extract_frames, analyze, etc.)
        **details: Additional processing details
    """
    video_context = {
        "video_path": str(video_path),
        "video_operation": operation,
        "timestamp": datetime.now().isoformat()
    }
    video_context.update(details)
    
    logger.info(
        f"Video processing: {operation} - {Path(video_path).name}",
        extra={
            "component": component,
            "operation": "video_processing",
            "context": video_context
        }
    )


def log_pose_estimation(component: str, estimator_type: str, operation: str, **details):
    """
    Log pose estimation operations with standardized format.
    
    Args:
        component: Component name
        estimator_type: Type of pose estimator (mediapipe, openpose, etc.)
        operation: Estimation operation (initialize, estimate, batch_process, etc.)
        **details: Additional estimation details
    """
    pose_context = {
        "estimator_type": estimator_type,
        "pose_operation": operation,
        "timestamp": datetime.now().isoformat()
    }
    pose_context.update(details)
    
    logger.info(
        f"Pose estimation: {estimator_type} - {operation}",
        extra={
            "component": component,
            "operation": "pose_estimation",
            "context": pose_context
        }
    )


def log_gait_analysis(component: str, analysis_type: str, operation: str, **details):
    """
    Log gait analysis operations with standardized format.
    
    Args:
        component: Component name
        analysis_type: Type of analysis (temporal, symmetry, feature_extraction, etc.)
        operation: Analysis operation (analyze, extract, classify, etc.)
        **details: Additional analysis details
    """
    gait_context = {
        "analysis_type": analysis_type,
        "gait_operation": operation,
        "timestamp": datetime.now().isoformat()
    }
    gait_context.update(details)
    
    logger.info(
        f"Gait analysis: {analysis_type} - {operation}",
        extra={
            "component": component,
            "operation": "gait_analysis",
            "context": gait_context
        }
    )


def log_classification_result(component: str, classifier_type: str, result: Dict[str, Any], **details):
    """
    Log classification results with standardized format.
    
    Args:
        component: Component name
        classifier_type: Type of classifier (binary, multiclass, llm, etc.)
        result: Classification result dictionary
        **details: Additional classification details
    """
    classification_context = {
        "classifier_type": classifier_type,
        "classification_result": result,
        "timestamp": datetime.now().isoformat()
    }
    classification_context.update(details)
    
    logger.info(
        f"Classification: {classifier_type} - {result.get('prediction', 'unknown')}",
        extra={
            "component": component,
            "operation": "classification",
            "context": classification_context
        }
    )


def log_file_operation(component: str, operation: str, file_path: str, **details):
    """
    Log file operations with standardized format.
    
    Args:
        component: Component name
        operation: File operation (read, write, delete, move, etc.)
        file_path: Path to file being operated on
        **details: Additional file operation details
    """
    file_context = {
        "file_path": str(file_path),
        "file_operation": operation,
        "file_name": Path(file_path).name,
        "file_extension": Path(file_path).suffix,
        "timestamp": datetime.now().isoformat()
    }
    file_context.update(details)
    
    logger.info(
        f"File operation: {operation} - {Path(file_path).name}",
        extra={
            "component": component,
            "operation": "file_operation",
            "context": file_context
        }
    )


def log_database_operation(component: str, operation: str, table: str = None, **details):
    """
    Log database operations with standardized format.
    
    Args:
        component: Component name
        operation: Database operation (select, insert, update, delete, etc.)
        table: Database table name
        **details: Additional database operation details
    """
    db_context = {
        "db_operation": operation,
        "table": table,
        "timestamp": datetime.now().isoformat()
    }
    db_context.update(details)
    
    logger.info(
        f"Database operation: {operation}" + (f" on {table}" if table else ""),
        extra={
            "component": component,
            "operation": "database_operation",
            "context": db_context
        }
    )


def log_api_call(component: str, api_name: str, operation: str, **details):
    """
    Log external API calls with standardized format.
    
    Args:
        component: Component name
        api_name: Name of external API (openai, youtube, etc.)
        operation: API operation (request, response, error, etc.)
        **details: Additional API call details
    """
    api_context = {
        "api_name": api_name,
        "api_operation": operation,
        "timestamp": datetime.now().isoformat()
    }
    api_context.update(details)
    
    logger.info(
        f"API call: {api_name} - {operation}",
        extra={
            "component": component,
            "operation": "api_call",
            "context": api_context
        }
    )


def log_cache_operation(component: str, operation: str, cache_key: str, **details):
    """
    Log cache operations with standardized format.
    
    Args:
        component: Component name
        operation: Cache operation (hit, miss, set, delete, etc.)
        cache_key: Cache key being operated on
        **details: Additional cache operation details
    """
    cache_context = {
        "cache_operation": operation,
        "cache_key": cache_key,
        "timestamp": datetime.now().isoformat()
    }
    cache_context.update(details)
    
    logger.debug(
        f"Cache operation: {operation} - {cache_key}",
        extra={
            "component": component,
            "operation": "cache_operation",
            "context": cache_context
        }
    )


def create_progress_logger(component: str, operation: str, total_items: int = None):
    """
    Create a progress logger for long-running operations.
    
    Args:
        component: Component name
        operation: Operation name
        total_items: Total number of items to process
    
    Returns:
        Progress logger instance
    
    Usage:
        progress = create_progress_logger("video_processor", "extract_frames", 1000)
        progress.start()
        
        for i in range(1000):
            # Process item
            progress.update(i + 1, {"frame_number": i})
        
        progress.complete()
    """
    class ProgressLogger:
        def __init__(self, component, operation, total_items):
            self.component = component
            self.operation = operation
            self.total_items = total_items
            self.logger = create_component_logger(component)
            self.start_time = None
            self.last_log_time = None
            self.log_interval = 10  # Log every 10 items by default
        
        def start(self, **context):
            import time
            self.start_time = time.time()
            self.last_log_time = self.start_time
            
            start_context = {
                "total_items": self.total_items
            }
            start_context.update(context)
            
            self.logger.info(f"Starting progress tracking: {self.operation}",
                           operation="progress_start", **start_context)
        
        def update(self, current_item: int, context: Dict[str, Any] = None, force_log: bool = False):
            import time
            current_time = time.time()
            
            # Log at intervals or when forced
            if force_log or current_item % self.log_interval == 0 or current_item == self.total_items:
                progress_context = {
                    "current_item": current_item,
                    "total_items": self.total_items
                }
                
                if self.total_items:
                    progress_context["progress_percent"] = round((current_item / self.total_items) * 100, 1)
                    progress_context["remaining_items"] = self.total_items - current_item
                
                if self.start_time:
                    elapsed = current_time - self.start_time
                    progress_context["elapsed_seconds"] = round(elapsed, 1)
                    
                    if current_item > 0:
                        avg_time = elapsed / current_item
                        progress_context["avg_time_per_item"] = round(avg_time, 3)
                        
                        if self.total_items:
                            remaining_time = (self.total_items - current_item) * avg_time
                            progress_context["estimated_remaining_seconds"] = round(remaining_time, 1)
                
                if context:
                    progress_context.update(context)
                
                self.logger.info(f"Progress update: {self.operation}",
                               operation="progress_update", **progress_context)
                
                self.last_log_time = current_time
        
        def complete(self, **context):
            import time
            
            completion_context = {
                "total_items": self.total_items,
                "status": "completed"
            }
            
            if self.start_time:
                total_time = time.time() - self.start_time
                completion_context["total_duration_seconds"] = round(total_time, 1)
                
                if self.total_items and self.total_items > 0:
                    completion_context["avg_time_per_item"] = round(total_time / self.total_items, 3)
            
            completion_context.update(context)
            
            self.logger.info(f"Progress completed: {self.operation}",
                           operation="progress_complete", **completion_context)
    
    return ProgressLogger(component, operation, total_items)


# Export commonly used functions for easy importing
__all__ = [
    'setup_logging',
    'create_component_logger',
    'log_function_execution',
    'log_async_function_execution',
    'BatchOperationLogger',
    'APIRequestLogger',
    'LoggingContext',
    'StructuredLogger',
    'log_performance',
    'log_error_with_context',
    'log_system_event',
    'log_data_operation',
    'log_security_event',
    'log_test_result',
    'log_configuration_change',
    'log_resource_usage',
    'log_model_operation',
    'create_operation_logger',
    'log_video_processing',
    'log_pose_estimation',
    'log_gait_analysis',
    'log_classification_result',
    'log_file_operation',
    'log_database_operation',
    'log_api_call',
    'log_cache_operation',
    'create_progress_logger',
    'auto_configure_logging',
    'configure_development_logging',
    'configure_production_logging',
    'get_logging_status',
    'validate_logging_configuration',
    'cleanup_old_logs',
    'get_log_statistics',
    'monitor_log_disk_usage'
]