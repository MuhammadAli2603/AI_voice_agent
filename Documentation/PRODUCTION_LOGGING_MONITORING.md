# Production Logging & Monitoring Guide

Complete guide to production-grade logging, monitoring, and alerting for the AI Voice Agent.

---

## 🎯 Overview

The production logging and monitoring system provides:

- **Structured JSON logging** for easy parsing and analysis
- **Call/Request ID tracking** for tracing requests across services
- **Performance monitoring** with automatic timing and metrics
- **Health checks** for system status verification
- **Resource monitoring** (CPU, memory, disk)
- **Alerting** with threshold-based notifications
- **Multiple log outputs** (console, files, audit logs)
- **Log rotation** and retention policies

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Logging](#logging)
3. [Monitoring](#monitoring)
4. [Health Checks](#health-checks)
5. [Alerting](#alerting)
6. [API Endpoints](#api-endpoints)
7. [Configuration](#configuration)
8. [Best Practices](#best-practices)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install psutil aiohttp
```

### 2. Update Your Code

Replace the old logger import:

```python
# Old
from app.utils.logger import log

# New - Production Logger
from app.utils.production_logger import log, set_call_id, perf_timer, audit_log
from app.utils.monitoring import record_call_start, record_call_end, record_error
```

### 3. Set Environment

```bash
# .env
ENVIRONMENT=production  # or development
```

### 4. Basic Usage

```python
from app.utils.production_logger import log, set_call_id, perf_timer
from app.utils.monitoring import record_call_start, record_call_end

async def handle_call(call_id: str):
    # Set call ID for tracking
    set_call_id(call_id)

    # Record call start
    record_call_start(call_id)

    # Your code with performance tracking
    with perf_timer("process_audio"):
        result = await process_audio()

    log.info("Call processed successfully")

    # Record call end
    record_call_end(call_id, duration_sec=10.5, success=True)
```

---

## 📝 Logging

### Log Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General informational messages
- **WARNING**: Warning messages (potential issues)
- **ERROR**: Error messages (failures)
- **CRITICAL**: Critical errors (system failures)

### Structured Logging

In production (`ENVIRONMENT=production`), logs are output as JSON:

```json
{
  "timestamp": "2024-01-04T10:30:45.123456Z",
  "level": "INFO",
  "logger": "app.telephony.asterisk_ari",
  "function": "handle_stasis_start",
  "line": 145,
  "message": "Call started",
  "app": "ai_voice_agent",
  "environment": "production",
  "call_id": "ast-123-456",
  "extra": {
    "caller_number": "+1234567890",
    "extension": "100"
  }
}
```

### Call ID Tracking

Track calls across the system:

```python
from app.utils.production_logger import set_call_id, clear_context

# At call start
set_call_id(call_id)

# All logs will now include call_id automatically
log.info("Processing call")  # Will include call_id

# At call end
clear_context()
```

### Performance Timing

Automatic performance tracking:

```python
from app.utils.production_logger import perf_timer

# Context manager
with perf_timer("stt_transcription"):
    text = await stt.transcribe(audio)

# Manual timing
timer = perf_timer("llm_generation")
with timer:
    response = await llm.generate(prompt)
```

### Audit Logging

Log important business events:

```python
from app.utils.production_logger import audit_log

audit_log("Call completed",
    call_id=call_id,
    caller=caller_number,
    duration_sec=duration,
    cost_usd=cost
)
```

### Log Files

Logs are organized in separate files:

```
logs/
├── app/
│   └── app_2024-01-04.log          # All application logs
├── errors/
│   └── error_2024-01-04.log        # Errors only (kept 90 days)
├── audit/
│   └── audit_2024-01-04.log        # Business events (kept 365 days)
└── performance/
    └── performance_2024-01-04.log  # Performance metrics
```

**Retention:**
- Application logs: 30 days
- Error logs: 90 days
- Audit logs: 365 days
- Performance logs: 30 days

---

## 📊 Monitoring

### Metrics Collection

The monitoring system tracks:

**Call Metrics:**
- Total calls
- Active calls
- Successful/failed calls
- Average call duration
- Average response time
- Calls per minute

**System Metrics:**
- CPU usage (%)
- Memory usage (%)
- Disk usage (%)
- Available resources

**Error Metrics:**
- Errors by type
- Total error count
- Recent errors

### Recording Metrics

```python
from app.utils.monitoring import (
    record_call_start,
    record_call_end,
    record_error
)

# Start of call
record_call_start(call_id)

# End of call
record_call_end(
    call_id=call_id,
    duration_sec=15.5,
    success=True
)

# Record error
record_error(
    error_type="STT_TIMEOUT",
    error_message="Speech recognition timed out",
    context={"call_id": call_id, "duration": 30}
)
```

### Get Metrics

```python
from app.utils.monitoring import get_metrics

metrics = get_metrics()
print(metrics)

# Output:
# {
#   "calls": {
#     "total_calls": 1250,
#     "active_calls": 5,
#     "successful_calls": 1200,
#     "failed_calls": 50,
#     "average_duration_sec": 45.2,
#     "average_response_time_ms": 850.5,
#     "calls_per_minute": 12.5
#   },
#   "system": {
#     "cpu_percent": 35.2,
#     "memory_percent": 62.5,
#     "disk_percent": 45.0,
#     ...
#   },
#   "errors": {
#     "errors_by_type": {"STT_TIMEOUT": 5, "LLM_ERROR": 3},
#     "total_errors": 8,
#     "recent_errors": [...]
#   }
# }
```

---

## 🏥 Health Checks

### Checking Health

```python
from app.utils.monitoring import get_health

health = await get_health()
print(health)

# Output:
# {
#   "healthy": true,
#   "status": "healthy",
#   "checks": {
#     "system": {
#       "healthy": true,
#       "service": "system",
#       "status": "ok",
#       "latency_ms": 2.5
#     },
#     "capacity": {
#       "healthy": true,
#       "service": "capacity",
#       "status": "ok",
#       "active_calls": 5,
#       "max_calls": 100
#     }
#   },
#   "timestamp": "2024-01-04T10:30:45.123Z"
# }
```

### Custom Health Checks

Register custom health checks:

```python
from app.utils.monitoring import production_monitor

async def check_database():
    """Check database connectivity"""
    try:
        await db.ping()
        return {"healthy": True, "status": "ok"}
    except Exception as e:
        return {"healthy": False, "status": "error", "error": str(e)}

# Register the check
production_monitor.health_checker.register_check("database", check_database)
```

---

## 🚨 Alerting

### Alert Thresholds

Default thresholds:

| Metric | Warning | Critical |
|--------|---------|----------|
| CPU | 80% | 95% |
| Memory | 85% | 95% |
| Disk | 85% | 95% |
| Error Rate | 5% | 10% |
| Active Calls | 80 | 100 |
| Response Time | 5000ms | 10000ms |

### Automatic Alerting

Alerts are automatically sent when thresholds are exceeded:

```python
from app.utils.monitoring import get_metrics
from app.utils.alerting import check_and_alert

# Get current metrics
metrics = get_metrics()

# Check and send alerts if needed
await check_and_alert(metrics)
```

### Custom Alerts

Send custom alerts:

```python
from app.utils.alerting import send_alert

await send_alert(
    severity="critical",  # critical, warning, info
    title="Database Connection Lost",
    message="Unable to connect to PostgreSQL database",
    metric="database_connection",
    value="disconnected",
    threshold="connected"
)
```

### Alert Handlers

#### Console (Default)

Enabled by default - prints alerts to console.

#### Slack

```python
from app.utils.alerting import alert_manager, slack_alert_handler

# Register Slack webhook
slack_webhook = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
alert_manager.register_handler(
    await slack_alert_handler(slack_webhook)
)
```

#### Custom Webhook

```python
from app.utils.alerting import alert_manager, webhook_alert_handler

# Register custom webhook
webhook_url = "https://your-monitoring-system.com/alerts"
alert_manager.register_handler(
    await webhook_alert_handler(webhook_url)
)
```

#### Custom Handler

```python
async def email_alert_handler(alert: Alert):
    """Send alert via email"""
    await send_email(
        to="ops@example.com",
        subject=f"[{alert.severity.upper()}] {alert.title}",
        body=alert.message
    )

alert_manager.register_handler(email_alert_handler)
```

### Alert Deduplication

Alerts are automatically deduplicated within a 15-minute window to prevent alert spam.

---

## 🌐 API Endpoints

Add monitoring routes to your FastAPI app:

```python
from app.routes.monitoring_routes import router as monitoring_router

app.include_router(monitoring_router)
```

### Available Endpoints

#### GET /monitoring/health

Basic health check (returns 200 or 503)

```bash
curl http://localhost:8000/monitoring/health
# Response: OK
```

#### GET /monitoring/health/detailed

Detailed health status with all checks

```bash
curl http://localhost:8000/monitoring/health/detailed
```

#### GET /monitoring/metrics

All metrics (calls, system, errors)

```bash
curl http://localhost:8000/monitoring/metrics
```

#### GET /monitoring/metrics/calls

Call statistics only

```bash
curl http://localhost:8000/monitoring/metrics/calls
```

#### GET /monitoring/metrics/system

System resource metrics only

```bash
curl http://localhost:8000/monitoring/metrics/system
```

#### GET /monitoring/status

Combined health and metrics

```bash
curl http://localhost:8000/monitoring/status
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# .env

# Environment mode
ENVIRONMENT=production  # production or development

# Log level
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Monitoring
ALERT_WEBHOOK_URL=https://your-webhook.com/alerts
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX
```

### Customize Thresholds

```python
from app.utils.alerting import alert_manager

# Update thresholds
alert_manager.thresholds["cpu_percent"] = {
    "warning": 70,
    "critical": 90
}

alert_manager.thresholds["active_calls"] = {
    "warning": 50,
    "critical": 75
}
```

---

## ✅ Best Practices

### 1. Always Set Call Context

```python
# At the start of every call
set_call_id(call_id)

# At the end
clear_context()
```

### 2. Use Performance Timers

```python
# Wrap expensive operations
with perf_timer("database_query"):
    result = await db.query()
```

### 3. Log Structured Data

```python
# Good - structured
log.info("User authenticated", extra={
    "user_id": user_id,
    "method": "oauth",
    "provider": "google"
})

# Bad - unstructured
log.info(f"User {user_id} authenticated via google oauth")
```

### 4. Record All Errors

```python
try:
    result = await risky_operation()
except Exception as e:
    record_error(
        error_type=type(e).__name__,
        error_message=str(e),
        context={"operation": "risky_operation"}
    )
    raise
```

### 5. Monitor Critical Paths

```python
# Record metrics for important operations
record_call_start(call_id)
try:
    await handle_call()
    record_call_end(call_id, duration, success=True)
except Exception:
    record_call_end(call_id, duration, success=False)
    raise
```

### 6. Use Audit Logs for Business Events

```python
# Log important business actions
audit_log("Payment processed",
    user_id=user_id,
    amount=amount,
    currency="USD"
)
```

---

## 📈 Monitoring Dashboard Integration

### Prometheus (Coming Soon)

Export metrics in Prometheus format for Grafana dashboards.

### ELK Stack

Send JSON logs to Elasticsearch:

```bash
# Filebeat configuration
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /path/to/logs/app/*.log
  json.keys_under_root: true
  json.add_error_key: true

output.elasticsearch:
  hosts: ["localhost:9200"]
```

### CloudWatch

Stream logs to AWS CloudWatch:

```python
# Add CloudWatch handler
import watchtower

logger.add(
    watchtower.CloudWatchLogHandler(),
    format=lambda record: json.dumps(record),
    level="INFO"
)
```

---

## 🔍 Troubleshooting

### Logs Not Appearing

1. Check `ENVIRONMENT` variable
2. Verify `LOG_LEVEL` setting
3. Check disk space for log files
4. Ensure logs directory exists and is writable

### High Memory Usage

- Reduce log retention period
- Lower log level to WARNING or ERROR
- Enable log compression

### Alerts Not Firing

1. Check thresholds configuration
2. Verify alert handlers are registered
3. Check webhook URLs are correct
4. Review alert deduplication window

---

## 📚 Complete Example

```python
from fastapi import FastAPI
from app.utils.production_logger import log, set_call_id, perf_timer, audit_log
from app.utils.monitoring import (
    record_call_start,
    record_call_end,
    record_error,
    get_metrics,
    get_health
)
from app.utils.alerting import check_and_alert
from app.routes.monitoring_routes import router as monitoring_router

app = FastAPI()
app.include_router(monitoring_router)

async def handle_voice_call(call_id: str, caller: str):
    """Complete example of production logging and monitoring"""

    # Set call context
    set_call_id(call_id)

    # Record call start
    record_call_start(call_id)
    log.info("Call started", extra={"caller": caller})

    start_time = time.time()

    try:
        # Process with performance tracking
        with perf_timer("stt_transcription"):
            text = await transcribe_audio()

        with perf_timer("llm_generation"):
            response = await generate_response(text)

        with perf_timer("tts_synthesis"):
            audio = await synthesize_speech(response)

        # Log success
        duration = time.time() - start_time
        log.info("Call completed successfully", extra={
            "duration_sec": duration,
            "response_length": len(response)
        })

        # Audit log
        audit_log("Call completed",
            call_id=call_id,
            caller=caller,
            duration_sec=duration,
            success=True
        )

        # Record metrics
        record_call_end(call_id, duration, success=True)

        return audio

    except Exception as e:
        # Log error
        log.exception("Call failed")

        # Record error
        record_error(
            error_type=type(e).__name__,
            error_message=str(e),
            context={"call_id": call_id, "caller": caller}
        )

        # Record failed call
        duration = time.time() - start_time
        record_call_end(call_id, duration, success=False)

        raise

    finally:
        # Clear context
        clear_context()

        # Check metrics and alert if needed
        metrics = get_metrics()
        await check_and_alert(metrics)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## ✅ Production Checklist

- [ ] Set `ENVIRONMENT=production` in `.env`
- [ ] Configure log retention policies
- [ ] Set up log rotation
- [ ] Register alert handlers (Slack, webhooks, etc.)
- [ ] Configure alert thresholds for your use case
- [ ] Add monitoring endpoints to your API
- [ ] Set up health check monitoring (e.g., UptimeRobot)
- [ ] Configure log aggregation (ELK, CloudWatch, etc.)
- [ ] Test alerting with simulated failures
- [ ] Document your monitoring runbook

---

## 🎉 Summary

You now have production-grade logging and monitoring with:

✅ Structured JSON logging
✅ Call/Request ID tracking
✅ Performance monitoring
✅ Health checks
✅ Resource monitoring
✅ Automatic alerting
✅ Multiple log outputs
✅ Log rotation and retention

Your AI Voice Agent is ready for production monitoring!
