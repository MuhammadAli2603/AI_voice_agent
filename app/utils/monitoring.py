"""
Production monitoring and health checks

Features:
- System health checks
- Performance metrics collection
- Resource usage monitoring
- Call statistics
- Alerting thresholds
"""
import time
import psutil
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class HealthStatus:
    """Health check status"""
    healthy: bool
    service: str
    status: str
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_free_gb: float
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


@dataclass
class CallMetrics:
    """Call statistics"""
    total_calls: int = 0
    active_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    average_duration_sec: float = 0.0
    average_response_time_ms: float = 0.0
    calls_per_minute: float = 0.0


class MetricsCollector:
    """Collect and aggregate metrics"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size

        # Call metrics
        self.total_calls = 0
        self.active_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0

        # Time-series data (recent metrics)
        self.call_durations = deque(maxlen=window_size)
        self.response_times = deque(maxlen=window_size)
        self.call_timestamps = deque(maxlen=window_size)

        # Error tracking
        self.errors_by_type = defaultdict(int)
        self.recent_errors = deque(maxlen=100)

    def record_call_start(self, call_id: str):
        """Record call start"""
        self.total_calls += 1
        self.active_calls += 1
        self.call_timestamps.append(time.time())

        logger.info(f"Call started", extra={
            "call_id": call_id,
            "active_calls": self.active_calls,
            "total_calls": self.total_calls
        })

    def record_call_end(self, call_id: str, duration_sec: float, success: bool = True):
        """Record call completion"""
        self.active_calls = max(0, self.active_calls - 1)

        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1

        self.call_durations.append(duration_sec)

        logger.info(f"Call ended", extra={
            "call_id": call_id,
            "duration_sec": duration_sec,
            "success": success,
            "active_calls": self.active_calls
        })

    def record_response_time(self, operation: str, duration_ms: float):
        """Record operation response time"""
        self.response_times.append(duration_ms)

    def record_error(self, error_type: str, error_message: str, context: Dict = None):
        """Record error occurrence"""
        self.errors_by_type[error_type] += 1
        self.recent_errors.append({
            "type": error_type,
            "message": error_message,
            "context": context or {},
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

        logger.error(f"Error recorded: {error_type}", extra={
            "error_type": error_type,
            "error_message": error_message,
            "total_errors": self.errors_by_type[error_type],
            **(context or {})
        })

    def get_call_metrics(self) -> CallMetrics:
        """Get current call metrics"""

        # Calculate average duration
        avg_duration = 0.0
        if self.call_durations:
            avg_duration = sum(self.call_durations) / len(self.call_durations)

        # Calculate average response time
        avg_response = 0.0
        if self.response_times:
            avg_response = sum(self.response_times) / len(self.response_times)

        # Calculate calls per minute
        calls_per_min = 0.0
        if self.call_timestamps:
            now = time.time()
            recent_calls = sum(1 for ts in self.call_timestamps if now - ts < 60)
            calls_per_min = recent_calls

        return CallMetrics(
            total_calls=self.total_calls,
            active_calls=self.active_calls,
            successful_calls=self.successful_calls,
            failed_calls=self.failed_calls,
            average_duration_sec=avg_duration,
            average_response_time_ms=avg_response,
            calls_per_minute=calls_per_min
        )

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary"""
        return {
            "errors_by_type": dict(self.errors_by_type),
            "total_errors": sum(self.errors_by_type.values()),
            "recent_errors": list(self.recent_errors)[-10:]  # Last 10 errors
        }


class HealthChecker:
    """Perform health checks on services"""

    def __init__(self):
        self.checks: Dict[str, callable] = {}

    def register_check(self, name: str, check_func: callable):
        """Register a health check function"""
        self.checks[name] = check_func

    async def check_service(self, name: str) -> HealthStatus:
        """Run a single health check"""
        if name not in self.checks:
            return HealthStatus(
                healthy=False,
                service=name,
                status="unknown",
                error="Check not registered"
            )

        try:
            start = time.perf_counter()
            result = await self.checks[name]()
            latency_ms = (time.perf_counter() - start) * 1000

            return HealthStatus(
                healthy=result.get("healthy", True),
                service=name,
                status=result.get("status", "ok"),
                latency_ms=latency_ms,
                error=result.get("error")
            )

        except Exception as e:
            logger.exception(f"Health check failed: {name}")
            return HealthStatus(
                healthy=False,
                service=name,
                status="error",
                error=str(e)
            )

    async def check_all(self) -> Dict[str, HealthStatus]:
        """Run all health checks"""
        results = {}

        for name in self.checks:
            results[name] = await self.check_service(name)

        return results


class SystemMonitor:
    """Monitor system resources"""

    @staticmethod
    def get_system_metrics() -> SystemMetrics:
        """Get current system metrics"""

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_used_gb = disk.used / (1024 * 1024 * 1024)
        disk_free_gb = disk.free / (1024 * 1024 * 1024)

        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_percent=disk.percent,
            disk_used_gb=disk_used_gb,
            disk_free_gb=disk_free_gb
        )

    @staticmethod
    def check_thresholds(metrics: SystemMetrics) -> List[str]:
        """Check if any metrics exceed thresholds"""
        warnings = []

        if metrics.cpu_percent > 80:
            warnings.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")

        if metrics.memory_percent > 85:
            warnings.append(f"High memory usage: {metrics.memory_percent:.1f}%")

        if metrics.disk_percent > 90:
            warnings.append(f"Low disk space: {metrics.disk_percent:.1f}% used")

        return warnings


class ProductionMonitor:
    """Main monitoring coordinator"""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        self.system_monitor = SystemMonitor()

        # Register default health checks
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default health checks"""

        async def system_health():
            """Check system resources"""
            metrics = self.system_monitor.get_system_metrics()
            warnings = self.system_monitor.check_thresholds(metrics)

            return {
                "healthy": len(warnings) == 0,
                "status": "ok" if len(warnings) == 0 else "warning",
                "warnings": warnings
            }

        async def call_capacity():
            """Check if system can handle more calls"""
            call_metrics = self.metrics_collector.get_call_metrics()

            # Simple threshold: max 100 concurrent calls
            max_calls = 100
            healthy = call_metrics.active_calls < max_calls

            return {
                "healthy": healthy,
                "status": "ok" if healthy else "overloaded",
                "active_calls": call_metrics.active_calls,
                "max_calls": max_calls
            }

        self.health_checker.register_check("system", system_health)
        self.health_checker.register_check("capacity", call_capacity)

    async def get_health_status(self) -> Dict[str, Any]:
        """Get complete health status"""
        health_checks = await self.health_checker.check_all()

        overall_healthy = all(check.healthy for check in health_checks.values())

        return {
            "healthy": overall_healthy,
            "status": "healthy" if overall_healthy else "degraded",
            "checks": {name: asdict(status) for name, status in health_checks.items()},
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get complete metrics summary"""
        call_metrics = self.metrics_collector.get_call_metrics()
        system_metrics = self.system_monitor.get_system_metrics()
        error_summary = self.metrics_collector.get_error_summary()

        return {
            "calls": asdict(call_metrics),
            "system": asdict(system_metrics),
            "errors": error_summary,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


# Global monitor instance
production_monitor = ProductionMonitor()


# Convenience functions
def record_call_start(call_id: str):
    """Record call start"""
    production_monitor.metrics_collector.record_call_start(call_id)


def record_call_end(call_id: str, duration_sec: float, success: bool = True):
    """Record call end"""
    production_monitor.metrics_collector.record_call_end(call_id, duration_sec, success)


def record_error(error_type: str, error_message: str, context: Dict = None):
    """Record error"""
    production_monitor.metrics_collector.record_error(error_type, error_message, context)


async def get_health():
    """Get system health"""
    return await production_monitor.get_health_status()


def get_metrics():
    """Get metrics summary"""
    return production_monitor.get_metrics_summary()
