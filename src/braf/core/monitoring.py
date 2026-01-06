"""
Monitoring and Observability Layer for BRAF.

This module provides Prometheus metrics, Grafana dashboard configurations,
critical event alerting, and centralized logging with ELK Stack integration.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    start_http_server
)
import structlog

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    """Represents a system alert."""
    
    id: str
    title: str
    description: str
    severity: AlertSeverity
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class PrometheusMetrics:
    """Prometheus metrics collector for BRAF components."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize Prometheus metrics."""
        self.registry = registry or CollectorRegistry()
        
        # Task execution metrics
        self.tasks_total = Counter(
            'braf_tasks_total',
            'Total number of tasks processed',
            ['status', 'priority', 'worker_id'],
            registry=self.registry
        )
        
        self.task_duration = Histogram(
            'braf_task_duration_seconds',
            'Task execution duration in seconds',
            ['task_type', 'worker_id'],
            registry=self.registry
        )
        
        self.task_queue_size = Gauge(
            'braf_task_queue_size',
            'Current task queue size',
            ['priority'],
            registry=self.registry
        )
        
        # Worker metrics
        self.workers_active = Gauge(
            'braf_workers_active',
            'Number of active workers',
            registry=self.registry
        )
        
        self.worker_cpu_usage = Gauge(
            'braf_worker_cpu_usage_percent',
            'Worker CPU usage percentage',
            ['worker_id'],
            registry=self.registry
        )
        
        self.worker_memory_usage = Gauge(
            'braf_worker_memory_usage_bytes',
            'Worker memory usage in bytes',
            ['worker_id'],
            registry=self.registry
        )
        
        # Browser metrics
        self.browser_instances = Gauge(
            'braf_browser_instances_active',
            'Number of active browser instances',
            ['worker_id'],
            registry=self.registry
        )
        
        self.browser_crashes = Counter(
            'braf_browser_crashes_total',
            'Total number of browser crashes',
            ['worker_id', 'reason'],
            registry=self.registry
        )
        
        # Detection metrics
        self.detections_total = Counter(
            'braf_detections_total',
            'Total number of bot detections',
            ['detection_type', 'severity', 'worker_id'],
            registry=self.registry
        )
        
        self.detection_score = Histogram(
            'braf_detection_score',
            'Bot detection scores',
            ['worker_id'],
            registry=self.registry
        )
        
        # CAPTCHA metrics
        self.captcha_encounters = Counter(
            'braf_captcha_encounters_total',
            'Total CAPTCHA encounters',
            ['captcha_type', 'worker_id'],
            registry=self.registry
        )
        
        self.captcha_solve_duration = Histogram(
            'braf_captcha_solve_duration_seconds',
            'CAPTCHA solving duration',
            ['captcha_type', 'solver'],
            registry=self.registry
        )
        
        self.captcha_success_rate = Gauge(
            'braf_captcha_success_rate',
            'CAPTCHA solving success rate',
            ['captcha_type', 'solver'],
            registry=self.registry
        )
        
        # Compliance metrics
        self.compliance_violations = Counter(
            'braf_compliance_violations_total',
            'Total compliance violations',
            ['violation_type', 'severity', 'worker_id'],
            registry=self.registry
        )
        
        self.compliance_lockdowns = Counter(
            'braf_compliance_lockdowns_total',
            'Total compliance lockdowns',
            ['reason'],
            registry=self.registry
        )
        
        # System metrics
        self.system_uptime = Gauge(
            'braf_system_uptime_seconds',
            'System uptime in seconds',
            registry=self.registry
        )
        
        self.api_requests = Counter(
            'braf_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.api_duration = Histogram(
            'braf_api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Component info
        self.component_info = Info(
            'braf_component_info',
            'Component information',
            registry=self.registry
        )
        
        # Set component info
        self.component_info.info({
            'version': '0.1.0',
            'component': 'braf-core',
            'build_date': datetime.now(timezone.utc).isoformat()
        })
    
    def record_task_execution(
        self,
        status: str,
        priority: str,
        worker_id: str,
        duration: float,
        task_type: str = "automation"
    ):
        """Record task execution metrics."""
        self.tasks_total.labels(
            status=status,
            priority=priority,
            worker_id=worker_id
        ).inc()
        
        self.task_duration.labels(
            task_type=task_type,
            worker_id=worker_id
        ).observe(duration)
    
    def update_queue_size(self, priority: str, size: int):
        """Update task queue size."""
        self.task_queue_size.labels(priority=priority).set(size)
    
    def update_worker_metrics(
        self,
        worker_id: str,
        cpu_usage: float,
        memory_usage: int,
        is_active: bool = True
    ):
        """Update worker metrics."""
        if is_active:
            self.workers_active.inc()
        
        self.worker_cpu_usage.labels(worker_id=worker_id).set(cpu_usage)
        self.worker_memory_usage.labels(worker_id=worker_id).set(memory_usage)
    
    def record_browser_event(
        self,
        worker_id: str,
        event_type: str,
        reason: Optional[str] = None
    ):
        """Record browser-related events."""
        if event_type == "crash":
            self.browser_crashes.labels(
                worker_id=worker_id,
                reason=reason or "unknown"
            ).inc()
    
    def record_detection(
        self,
        detection_type: str,
        severity: str,
        worker_id: str,
        score: float
    ):
        """Record bot detection event."""
        self.detections_total.labels(
            detection_type=detection_type,
            severity=severity,
            worker_id=worker_id
        ).inc()
        
        self.detection_score.labels(worker_id=worker_id).observe(score)
    
    def record_captcha_event(
        self,
        captcha_type: str,
        worker_id: str,
        solver: str,
        duration: Optional[float] = None,
        success: Optional[bool] = None
    ):
        """Record CAPTCHA-related events."""
        self.captcha_encounters.labels(
            captcha_type=captcha_type,
            worker_id=worker_id
        ).inc()
        
        if duration is not None:
            self.captcha_solve_duration.labels(
                captcha_type=captcha_type,
                solver=solver
            ).observe(duration)
        
        if success is not None:
            # Update success rate (simplified - in production, use a proper rate calculation)
            rate = 1.0 if success else 0.0
            self.captcha_success_rate.labels(
                captcha_type=captcha_type,
                solver=solver
            ).set(rate)
    
    def record_compliance_violation(
        self,
        violation_type: str,
        severity: str,
        worker_id: str
    ):
        """Record compliance violation."""
        self.compliance_violations.labels(
            violation_type=violation_type,
            severity=severity,
            worker_id=worker_id
        ).inc()
    
    def record_compliance_lockdown(self, reason: str):
        """Record compliance lockdown."""
        self.compliance_lockdowns.labels(reason=reason).inc()
    
    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ):
        """Record API request metrics."""
        self.api_requests.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        self.api_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')


class GrafanaDashboardGenerator:
    """Generates Grafana dashboard configurations."""
    
    def __init__(self):
        """Initialize dashboard generator."""
        self.dashboards = {}
    
    def generate_main_dashboard(self) -> Dict[str, Any]:
        """Generate main BRAF dashboard."""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "BRAF - Browser Automation Framework",
                "tags": ["braf", "automation"],
                "timezone": "browser",
                "panels": [
                    self._create_task_metrics_panel(),
                    self._create_worker_status_panel(),
                    self._create_detection_metrics_panel(),
                    self._create_compliance_panel(),
                    self._create_system_health_panel()
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }
        
        return dashboard
    
    def _create_task_metrics_panel(self) -> Dict[str, Any]:
        """Create task metrics panel."""
        return {
            "id": 1,
            "title": "Task Execution Metrics",
            "type": "graph",
            "targets": [
                {
                    "expr": "rate(braf_tasks_total[5m])",
                    "legendFormat": "Tasks/sec - {{status}}"
                },
                {
                    "expr": "braf_task_queue_size",
                    "legendFormat": "Queue Size - {{priority}}"
                }
            ],
            "yAxes": [
                {"label": "Tasks/sec"},
                {"label": "Queue Size"}
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
        }
    
    def _create_worker_status_panel(self) -> Dict[str, Any]:
        """Create worker status panel."""
        return {
            "id": 2,
            "title": "Worker Status",
            "type": "stat",
            "targets": [
                {
                    "expr": "braf_workers_active",
                    "legendFormat": "Active Workers"
                },
                {
                    "expr": "avg(braf_worker_cpu_usage_percent)",
                    "legendFormat": "Avg CPU Usage %"
                },
                {
                    "expr": "avg(braf_worker_memory_usage_bytes) / 1024 / 1024",
                    "legendFormat": "Avg Memory Usage MB"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
        }
    
    def _create_detection_metrics_panel(self) -> Dict[str, Any]:
        """Create detection metrics panel."""
        return {
            "id": 3,
            "title": "Bot Detection Metrics",
            "type": "graph",
            "targets": [
                {
                    "expr": "rate(braf_detections_total[5m])",
                    "legendFormat": "Detections/sec - {{detection_type}}"
                },
                {
                    "expr": "histogram_quantile(0.95, braf_detection_score)",
                    "legendFormat": "95th percentile detection score"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
        }
    
    def _create_compliance_panel(self) -> Dict[str, Any]:
        """Create compliance monitoring panel."""
        return {
            "id": 4,
            "title": "Compliance Monitoring",
            "type": "graph",
            "targets": [
                {
                    "expr": "rate(braf_compliance_violations_total[5m])",
                    "legendFormat": "Violations/sec - {{violation_type}}"
                },
                {
                    "expr": "braf_compliance_lockdowns_total",
                    "legendFormat": "Total Lockdowns"
                }
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
        }
    
    def _create_system_health_panel(self) -> Dict[str, Any]:
        """Create system health panel."""
        return {
            "id": 5,
            "title": "System Health",
            "type": "graph",
            "targets": [
                {
                    "expr": "braf_system_uptime_seconds / 3600",
                    "legendFormat": "Uptime (hours)"
                },
                {
                    "expr": "rate(braf_api_requests_total[5m])",
                    "legendFormat": "API Requests/sec"
                }
            ],
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16}
        }
    
    def save_dashboard(self, dashboard: Dict[str, Any], filename: str):
        """Save dashboard configuration to file."""
        with open(filename, 'w') as f:
            json.dump(dashboard, f, indent=2)


class AlertManager:
    """Manages critical event alerting."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alerts: List[Alert] = []
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = {
            AlertSeverity.INFO: [],
            AlertSeverity.WARNING: [],
            AlertSeverity.CRITICAL: [],
            AlertSeverity.EMERGENCY: []
        }
        self.alert_rules: List[Dict[str, Any]] = []
    
    def add_alert_handler(self, severity: AlertSeverity, handler: Callable):
        """Add alert handler for specific severity."""
        self.alert_handlers[severity].append(handler)
    
    def add_alert_rule(
        self,
        name: str,
        condition: str,
        severity: AlertSeverity,
        description: str,
        threshold: Optional[float] = None
    ):
        """Add alert rule."""
        rule = {
            'name': name,
            'condition': condition,
            'severity': severity,
            'description': description,
            'threshold': threshold
        }
        self.alert_rules.append(rule)
    
    async def create_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create and process new alert."""
        alert = Alert(
            id=f"alert_{int(time.time())}_{len(self.alerts)}",
            title=title,
            description=description,
            severity=severity,
            timestamp=datetime.now(timezone.utc),
            source=source,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Trigger alert handlers
        handlers = self.alert_handlers.get(severity, [])
        for handler in handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.info(f"Alert created: {alert.title} ({severity.value})")
        return alert
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now(timezone.utc)
                logger.info(f"Alert resolved: {alert.title}")
                return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity level."""
        return [alert for alert in self.alerts if alert.severity == severity]


class ELKLogger:
    """Centralized logging with ELK Stack integration."""
    
    def __init__(
        self,
        elasticsearch_host: str = "localhost:9200",
        index_prefix: str = "braf"
    ):
        """Initialize ELK logger."""
        self.elasticsearch_host = elasticsearch_host
        self.index_prefix = index_prefix
        self.structured_logger = structlog.get_logger()
        
        # Configure structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def log_task_execution(
        self,
        task_id: str,
        worker_id: str,
        status: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log task execution event."""
        self.structured_logger.info(
            "task_execution",
            task_id=task_id,
            worker_id=worker_id,
            status=status,
            duration=duration,
            metadata=metadata or {}
        )
    
    def log_compliance_event(
        self,
        event_type: str,
        worker_id: str,
        severity: str,
        details: Dict[str, Any]
    ):
        """Log compliance event."""
        self.structured_logger.warning(
            "compliance_event",
            event_type=event_type,
            worker_id=worker_id,
            severity=severity,
            details=details
        )
    
    def log_detection_event(
        self,
        detection_type: str,
        worker_id: str,
        score: float,
        details: Dict[str, Any]
    ):
        """Log bot detection event."""
        self.structured_logger.warning(
            "detection_event",
            detection_type=detection_type,
            worker_id=worker_id,
            score=score,
            details=details
        )
    
    def log_system_event(
        self,
        event_type: str,
        component: str,
        message: str,
        level: str = "info",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log system event."""
        log_func = getattr(self.structured_logger, level.lower(), self.structured_logger.info)
        log_func(
            "system_event",
            event_type=event_type,
            component=component,
            message=message,
            metadata=metadata or {}
        )


class MonitoringManager:
    """Main monitoring and observability manager."""
    
    def __init__(
        self,
        prometheus_port: int = 8000,
        elasticsearch_host: str = "localhost:9200"
    ):
        """Initialize monitoring manager."""
        self.prometheus_port = prometheus_port
        self.metrics = PrometheusMetrics()
        self.dashboard_generator = GrafanaDashboardGenerator()
        self.alert_manager = AlertManager()
        self.elk_logger = ELKLogger(elasticsearch_host)
        self.prometheus_server = None
        
        # Set up default alert rules
        self._setup_default_alert_rules()
        
        # Set up default alert handlers
        self._setup_default_alert_handlers()
    
    def _setup_default_alert_rules(self):
        """Set up default alerting rules."""
        self.alert_manager.add_alert_rule(
            name="high_detection_rate",
            condition="rate(braf_detections_total[5m]) > 0.1",
            severity=AlertSeverity.WARNING,
            description="High bot detection rate detected",
            threshold=0.1
        )
        
        self.alert_manager.add_alert_rule(
            name="compliance_violation",
            condition="braf_compliance_violations_total > 0",
            severity=AlertSeverity.CRITICAL,
            description="Compliance violation detected"
        )
        
        self.alert_manager.add_alert_rule(
            name="worker_down",
            condition="braf_workers_active == 0",
            severity=AlertSeverity.EMERGENCY,
            description="No active workers available"
        )
    
    def _setup_default_alert_handlers(self):
        """Set up default alert handlers."""
        async def log_alert(alert: Alert):
            """Log alert to ELK."""
            self.elk_logger.log_system_event(
                event_type="alert",
                component="monitoring",
                message=f"Alert: {alert.title}",
                level=alert.severity.value,
                metadata={
                    'alert_id': alert.id,
                    'description': alert.description,
                    'source': alert.source,
                    'metadata': alert.metadata
                }
            )
        
        # Add log handler for all severities
        for severity in AlertSeverity:
            self.alert_manager.add_alert_handler(severity, log_alert)
    
    def start_prometheus_server(self):
        """Start Prometheus metrics server."""
        self.prometheus_server = start_http_server(self.prometheus_port, registry=self.metrics.registry)
        logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
    
    def generate_dashboards(self, output_dir: str = "./dashboards"):
        """Generate Grafana dashboards."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate main dashboard
        main_dashboard = self.dashboard_generator.generate_main_dashboard()
        self.dashboard_generator.save_dashboard(
            main_dashboard,
            f"{output_dir}/braf-main-dashboard.json"
        )
        
        logger.info(f"Grafana dashboards generated in {output_dir}")
    
    async def check_alert_conditions(self):
        """Check alert conditions and trigger alerts if needed."""
        # This would typically query Prometheus for metrics
        # For now, we'll implement basic checks
        
        # Check if any workers are active
        # In a real implementation, this would query the metrics
        active_workers = 0  # This would come from metrics
        
        if active_workers == 0:
            await self.alert_manager.create_alert(
                title="No Active Workers",
                description="All workers are offline or unavailable",
                severity=AlertSeverity.EMERGENCY,
                source="monitoring_manager"
            )
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Determine overall health based on alerts
        health_status = "healthy"
        if any(alert.severity == AlertSeverity.EMERGENCY for alert in active_alerts):
            health_status = "critical"
        elif any(alert.severity == AlertSeverity.CRITICAL for alert in active_alerts):
            health_status = "degraded"
        elif any(alert.severity == AlertSeverity.WARNING for alert in active_alerts):
            health_status = "warning"
        
        return {
            "status": health_status,
            "active_alerts": len(active_alerts),
            "alerts_by_severity": {
                severity.value: len(self.alert_manager.get_alerts_by_severity(severity))
                for severity in AlertSeverity
            },
            "uptime_seconds": time.time(),  # This would be actual uptime
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global monitoring manager instance
_monitoring_manager: Optional[MonitoringManager] = None


def get_monitoring_manager() -> Optional[MonitoringManager]:
    """
    Get global monitoring manager instance.
    
    Returns:
        Monitoring manager instance or None if not initialized
    """
    return _monitoring_manager


def init_monitoring_manager(
    prometheus_port: int = 8000,
    elasticsearch_host: str = "localhost:9200"
) -> MonitoringManager:
    """
    Initialize global monitoring manager.
    
    Args:
        prometheus_port: Port for Prometheus metrics server
        elasticsearch_host: Elasticsearch host for logging
        
    Returns:
        Initialized monitoring manager
    """
    global _monitoring_manager
    
    _monitoring_manager = MonitoringManager(
        prometheus_port=prometheus_port,
        elasticsearch_host=elasticsearch_host
    )
    
    return _monitoring_manager
