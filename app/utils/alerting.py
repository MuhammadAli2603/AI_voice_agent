"""
Alerting system for production monitoring

Features:
- Threshold-based alerts
- Alert deduplication
- Multiple alert channels (email, webhook, slack)
- Alert history tracking
"""
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from loguru import logger


@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    severity: str  # critical, warning, info
    title: str
    message: str
    metric: str
    value: Any
    threshold: Any
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


class AlertManager:
    """Manage alerts and notifications"""

    def __init__(self, dedupe_window_minutes: int = 15):
        self.dedupe_window = timedelta(minutes=dedupe_window_minutes)

        # Alert history
        self.alert_history = deque(maxlen=1000)
        self.recent_alerts = {}  # For deduplication

        # Alert handlers
        self.handlers: List[Callable] = []

        # Thresholds
        self.thresholds = {
            "cpu_percent": {"warning": 80, "critical": 95},
            "memory_percent": {"warning": 85, "critical": 95},
            "disk_percent": {"warning": 85, "critical": 95},
            "error_rate": {"warning": 0.05, "critical": 0.1},  # 5% and 10%
            "active_calls": {"warning": 80, "critical": 100},
            "response_time_ms": {"warning": 5000, "critical": 10000},
        }

    def register_handler(self, handler: Callable):
        """Register alert handler"""
        self.handlers.append(handler)

    def _should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent (deduplication)"""
        key = f"{alert.severity}:{alert.metric}"

        if key in self.recent_alerts:
            last_sent = self.recent_alerts[key]
            if datetime.fromisoformat(last_sent.replace('Z', '+00:00')) + self.dedupe_window > datetime.utcnow():
                logger.debug(f"Suppressing duplicate alert: {key}")
                return False

        self.recent_alerts[key] = alert.timestamp
        return True

    async def send_alert(self, alert: Alert):
        """Send alert through all handlers"""

        if not self._should_send_alert(alert):
            return

        # Store in history
        self.alert_history.append(alert)

        # Log the alert
        logger.bind(audit=True).log(
            "ERROR" if alert.severity == "critical" else "WARNING",
            f"ALERT: {alert.title}",
            extra={
                "alert_id": alert.alert_id,
                "severity": alert.severity,
                "metric": alert.metric,
                "value": alert.value,
                "threshold": alert.threshold
            }
        )

        # Send through handlers
        for handler in self.handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.exception(f"Alert handler failed: {e}")

    def check_threshold(self, metric: str, value: float) -> Optional[Alert]:
        """Check if metric exceeds threshold"""

        if metric not in self.thresholds:
            return None

        thresholds = self.thresholds[metric]

        # Check critical first
        if "critical" in thresholds and value >= thresholds["critical"]:
            return Alert(
                alert_id=f"{metric}_critical_{int(datetime.utcnow().timestamp())}",
                severity="critical",
                title=f"Critical: {metric}",
                message=f"{metric} is critically high: {value} (threshold: {thresholds['critical']})",
                metric=metric,
                value=value,
                threshold=thresholds["critical"]
            )

        # Check warning
        if "warning" in thresholds and value >= thresholds["warning"]:
            return Alert(
                alert_id=f"{metric}_warning_{int(datetime.utcnow().timestamp())}",
                severity="warning",
                title=f"Warning: {metric}",
                message=f"{metric} is high: {value} (threshold: {thresholds['warning']})",
                metric=metric,
                value=value,
                threshold=thresholds["warning"]
            )

        return None

    async def check_metrics(self, metrics: Dict[str, Any]):
        """Check all metrics and send alerts if needed"""

        # Check system metrics
        if "system" in metrics:
            system = metrics["system"]

            for metric in ["cpu_percent", "memory_percent", "disk_percent"]:
                if metric in system:
                    alert = self.check_threshold(metric, system[metric])
                    if alert:
                        await self.send_alert(alert)

        # Check call metrics
        if "calls" in metrics:
            calls = metrics["calls"]

            # Check active calls
            if "active_calls" in calls:
                alert = self.check_threshold("active_calls", calls["active_calls"])
                if alert:
                    await self.send_alert(alert)

            # Check response time
            if "average_response_time_ms" in calls:
                alert = self.check_threshold(
                    "response_time_ms",
                    calls["average_response_time_ms"]
                )
                if alert:
                    await self.send_alert(alert)

            # Check error rate
            if "total_calls" in calls and "failed_calls" in calls:
                total = calls["total_calls"]
                if total > 0:
                    error_rate = calls["failed_calls"] / total
                    alert = self.check_threshold("error_rate", error_rate)
                    if alert:
                        await self.send_alert(alert)

    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """Get recent alerts"""
        return [asdict(alert) for alert in list(self.alert_history)[-limit:]]


# Alert Handlers

async def console_alert_handler(alert: Alert):
    """Print alert to console"""
    symbol = "🔴" if alert.severity == "critical" else "⚠️"
    logger.critical(
        f"{symbol} ALERT [{alert.severity.upper()}]: {alert.title}\n"
        f"   Message: {alert.message}\n"
        f"   Metric: {alert.metric} = {alert.value} (threshold: {alert.threshold})"
    )


async def webhook_alert_handler(webhook_url: str):
    """Create webhook alert handler"""

    async def handler(alert: Alert):
        """Send alert to webhook"""
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(
                    webhook_url,
                    json=asdict(alert),
                    timeout=aiohttp.ClientTimeout(total=5)
                )
                logger.info(f"Alert sent to webhook: {alert.alert_id}")
            except Exception as e:
                logger.error(f"Failed to send webhook alert: {e}")

    return handler


async def slack_alert_handler(slack_webhook_url: str):
    """Create Slack alert handler"""

    async def handler(alert: Alert):
        """Send alert to Slack"""

        # Map severity to color
        color = {
            "critical": "#ff0000",  # Red
            "warning": "#ffaa00",   # Orange
            "info": "#0099ff"       # Blue
        }.get(alert.severity, "#cccccc")

        # Build Slack message
        message = {
            "attachments": [
                {
                    "color": color,
                    "title": f"{alert.severity.upper()}: {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Metric",
                            "value": alert.metric,
                            "short": True
                        },
                        {
                            "title": "Value",
                            "value": str(alert.value),
                            "short": True
                        },
                        {
                            "title": "Threshold",
                            "value": str(alert.threshold),
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": alert.timestamp,
                            "short": True
                        }
                    ],
                    "footer": "AI Voice Agent Monitoring",
                    "ts": int(datetime.utcnow().timestamp())
                }
            ]
        }

        async with aiohttp.ClientSession() as session:
            try:
                await session.post(
                    slack_webhook_url,
                    json=message,
                    timeout=aiohttp.ClientTimeout(total=5)
                )
                logger.info(f"Alert sent to Slack: {alert.alert_id}")
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {e}")

    return handler


# Global alert manager
alert_manager = AlertManager()

# Register default console handler
alert_manager.register_handler(console_alert_handler)


# Convenience functions
async def send_alert(severity: str, title: str, message: str, **kwargs):
    """Send custom alert"""
    alert = Alert(
        alert_id=f"custom_{int(datetime.utcnow().timestamp())}",
        severity=severity,
        title=title,
        message=message,
        metric=kwargs.get("metric", "custom"),
        value=kwargs.get("value", "N/A"),
        threshold=kwargs.get("threshold", "N/A")
    )
    await alert_manager.send_alert(alert)


async def check_and_alert(metrics: Dict[str, Any]):
    """Check metrics and send alerts"""
    await alert_manager.check_metrics(metrics)
