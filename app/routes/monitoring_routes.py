"""
Monitoring and health check API endpoints

Endpoints:
- GET /health - Health check
- GET /health/detailed - Detailed health status
- GET /metrics - System and call metrics
- GET /metrics/calls - Call statistics only
- GET /metrics/system - System resources only
"""
from fastapi import APIRouter, Response, status
from app.utils.monitoring import production_monitor, get_health, get_metrics
from app.utils.production_logger import logger

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint

    Returns 200 if system is healthy, 503 if degraded
    """
    health = await get_health()

    status_code = (
        status.HTTP_200_OK
        if health["healthy"]
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return Response(
        content="OK" if health["healthy"] else "DEGRADED",
        status_code=status_code,
        media_type="text/plain"
    )


@router.get("/health/detailed")
async def detailed_health():
    """
    Detailed health status with all checks

    Returns comprehensive health information including:
    - Overall health status
    - Individual service checks
    - System resources
    - Capacity status
    """
    health = await get_health()

    logger.info("Health check requested", extra={
        "healthy": health["healthy"],
        "status": health["status"]
    })

    return health


@router.get("/metrics")
async def get_all_metrics():
    """
    Get all metrics

    Returns:
    - Call statistics
    - System resources
    - Error summary
    """
    metrics = get_metrics()

    logger.info("Metrics requested", extra={
        "active_calls": metrics["calls"]["active_calls"],
        "total_calls": metrics["calls"]["total_calls"]
    })

    return metrics


@router.get("/metrics/calls")
async def get_call_metrics():
    """Get call statistics only"""
    metrics = get_metrics()
    return metrics["calls"]


@router.get("/metrics/system")
async def get_system_metrics():
    """Get system resource metrics only"""
    metrics = get_metrics()
    return metrics["system"]


@router.get("/metrics/errors")
async def get_error_metrics():
    """Get error summary"""
    metrics = get_metrics()
    return metrics["errors"]


@router.get("/status")
async def system_status():
    """
    Combined health and metrics endpoint

    Useful for dashboards and monitoring tools
    """
    health = await get_health()
    metrics = get_metrics()

    return {
        "health": health,
        "metrics": metrics
    }
