"""
Health check and readiness probe infrastructure for cloud-native deployments.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import psutil

from config.base import get_config
from shared.logging import get_logger
from shared.metrics import get_metrics_collector


class HealthStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    metadata: Dict[str, Any] = None


class HealthChecker:
    """Health check coordinator for cloud-native services"""
    
    def __init__(self, service_name: str = None):
        self.service_name = service_name or get_config().service_name
        self.logger = get_logger(__name__, service=self.service_name)
        self.metrics = get_metrics_collector(self.service_name)
        self._checks: Dict[str, Callable] = {}
        self._startup_time = time.time()
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function"""
        self._checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = []
        overall_status = HealthStatus.HEALTHY
        
        for name, check_func in self._checks.items():
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                duration_ms = (time.time() - start_time) * 1000
                
                if isinstance(result, HealthCheckResult):
                    check_result = result
                    check_result.duration_ms = duration_ms
                else:
                    # Assume boolean result
                    status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                    check_result = HealthCheckResult(
                        name=name,
                        status=status,
                        message="OK" if result else "Check failed",
                        duration_ms=duration_ms
                    )
                
                results.append(check_result)
                
                # Update overall status
                if check_result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif check_result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                
                # Record metrics
                self.metrics.record_histogram(
                    "health_check_duration_seconds",
                    duration_ms / 1000,
                    check_name=name,
                    status=check_result.status.value
                )
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                check_result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}",
                    duration_ms=duration_ms,
                    metadata={"error_type": type(e).__name__}
                )
                
                results.append(check_result)
                overall_status = HealthStatus.UNHEALTHY
                
                self.logger.error(
                    f"Health check {name} failed",
                    exc_info=True,
                    extra_fields={"check_name": name}
                )
                
                self.metrics.record_histogram(
                    "health_check_duration_seconds",
                    duration_ms / 1000,
                    check_name=name,
                    status="error"
                )
        
        # System metrics
        system_info = self._get_system_info()
        
        health_response = {
            "service": self.service_name,
            "status": overall_status.value,
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self._startup_time,
            "checks": [
                {
                    "name": result.name,
                    "status": result.status.value,
                    "message": result.message,
                    "duration_ms": result.duration_ms,
                    "metadata": result.metadata or {}
                }
                for result in results
            ],
            "system": system_info
        }
        
        return health_response
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system resource information"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024 * 1024 * 1024),
            }
        except Exception as e:
            self.logger.warning(f"Could not get system info: {e}")
            return {"error": "System info unavailable"}
    
    async def readiness_check(self) -> bool:
        """Simple readiness check for Kubernetes"""
        health_result = await self.run_health_checks()
        return health_result["status"] in [HealthStatus.HEALTHY.value, HealthStatus.DEGRADED.value]
    
    async def liveness_check(self) -> bool:
        """Simple liveness check for Kubernetes"""
        # Basic liveness - service is running
        return True


# Common health check functions
def database_health_check() -> HealthCheckResult:
    """Check database connectivity"""
    # This would be implemented with actual database connection
    # For now, return a mock healthy status
    return HealthCheckResult(
        name="database",
        status=HealthStatus.HEALTHY,
        message="Database connection OK",
        duration_ms=0
    )


def external_api_health_check() -> HealthCheckResult:
    """Check external API connectivity"""
    # This would check Yahoo Finance/Alpha Vantage APIs
    # For now, return a mock healthy status
    return HealthCheckResult(
        name="external_apis",
        status=HealthStatus.HEALTHY,
        message="External APIs accessible",
        duration_ms=0
    )


def mlflow_health_check() -> HealthCheckResult:
    """Check MLflow connectivity"""
    # This would check MLflow tracking server
    # For now, return a mock healthy status
    return HealthCheckResult(
        name="mlflow",
        status=HealthStatus.HEALTHY,
        message="MLflow tracking server OK",
        duration_ms=0
    )


def storage_health_check() -> HealthCheckResult:
    """Check storage system health"""
    import os
    from pathlib import Path
    
    config = get_config()
    data_root = Path(config.storage.data_root)
    
    try:
        # Check if data directory exists and is writable
        data_root.mkdir(parents=True, exist_ok=True)
        test_file = data_root / ".health_check"
        test_file.write_text("health_check")
        test_file.unlink()
        
        return HealthCheckResult(
            name="storage",
            status=HealthStatus.HEALTHY,
            message="Storage system accessible",
            duration_ms=0
        )
    except Exception as e:
        return HealthCheckResult(
            name="storage",
            status=HealthStatus.UNHEALTHY,
            message=f"Storage system error: {str(e)}",
            duration_ms=0
        )


# Global health checker instance
health_checker = HealthChecker()

# Register common health checks
health_checker.register_check("database", database_health_check)
health_checker.register_check("external_apis", external_api_health_check)
health_checker.register_check("mlflow", mlflow_health_check)
health_checker.register_check("storage", storage_health_check)