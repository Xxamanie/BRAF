"""
RESILIENCE ORCHESTRATOR

Coordinates fallback systems and adaptive recovery
"""

from __future__ import annotations

import asyncio
import random
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

import psutil
import signal
import sys
import os


class ResilienceOrchestrator:
    """Orchestrates system resilience with automatic failover."""

    def __init__(self) -> None:
        self.fallback_systems: List[Dict[str, Any]] = []
        self.health_checkers: List[Callable[[], Any]] = []
        self.recovery_protocols: Dict[str, Callable[..., Any]] = {}
        self.system_state: str = "operational"
        self.failure_count: int = 0

    def register_fallback_system(
        self, system_name: str, activation_condition: Callable[[Dict[str, Any]], bool], recovery_protocol: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """Register a fallback system with activation conditions."""
        self.fallback_systems.append(
            {
                "name": system_name,
                "condition": activation_condition,
                "protocol": recovery_protocol,
                "last_activated": None,
                "success_rate": 1.0,
            }
        )

    async def monitor_system_health(self, check_interval: int = 30) -> None:
        """Continuously monitor system health and trigger failovers."""
        while True:
            try:
                system_health = await self._perform_health_check()
                if system_health["overall_status"] != "healthy":
                    print(f"System health degraded: {system_health}")
                    await self._activate_fallback_systems(system_health)
                # Adaptive adjustment based on failure patterns
                if self.failure_count > 3:
                    await self._adjust_operational_parameters()
                await asyncio.sleep(check_interval)
            except Exception as e:  # pragma: no cover - defensive logging only
                print(f"Health monitoring error: {e}")
                await asyncio.sleep(5)

    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        checks: Dict[str, Any] = {
            "resource_utilization": self._check_resource_utilization(),
            "network_connectivity": await self._check_network_connectivity(),
            "component_availability": await self._check_component_availability(),
            "performance_metrics": self._check_performance_metrics(),
            "anomaly_detection": await self._detect_anomalies(),
        }
        overall_status = "healthy"
        if any(check["status"] != "healthy" for check in checks.values()):
            overall_status = "degraded"
        if any(check["status"] == "critical" for check in checks.values()):
            overall_status = "critical"
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "detailed_checks": checks,
            "failure_count": self.failure_count,
            "system_state": self.system_state,
        }

    def _check_resource_utilization(self) -> Dict[str, Any]:
        """Check system resource utilization."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        try:
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent
        except Exception:
            disk_percent = 0.0
        status = "healthy"
        if cpu_percent > 85 or memory.percent > 90 or disk_percent > 95:
            status = "critical"
        elif cpu_percent > 70 or memory.percent > 80 or disk_percent > 85:
            status = "degraded"
        return {"status": status, "cpu_percent": cpu_percent, "memory_percent": memory.percent, "disk_percent": disk_percent}

    async def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity to critical services (simulated)."""
        # Light-weight connectivity heuristic to avoid external calls in sandboxed envs
        test_endpoints = ["1.1.1.1", "8.8.8.8", "api.ipify.org"]
        # Simulate success ratio rather than real HTTP requests
        successful = sum(1 for _ in test_endpoints if random.random() > 0.2)
        status = "healthy" if successful >= 2 else "degraded"
        if successful == 0:
            status = "critical"
        return {"status": status, "successful_connections": successful, "total_endpoints": len(test_endpoints)}

    async def _check_component_availability(self) -> Dict[str, Any]:
        """Check availability of system components (simulated)."""
        components = {
            "profile_generator": self._check_component("profile_generator"),
            "research_engine": self._check_component("research_engine"),
            "data_processor": self._check_component("data_processor"),
        }
        available = sum(1 for status in components.values() if status == "available")
        total = len(components)
        status = "healthy" if available == total else "degraded"
        if available == 0:
            status = "critical"
        return {"status": status, "components": components, "available_count": available}

    def _check_component(self, component_name: str) -> str:
        """Check if a specific component is available (simulated)."""
        return "available" if random.random() > 0.1 else "unavailable"

    async def _detect_anomalies(self) -> Dict[str, Any]:
        """Detect system anomalies and unusual patterns."""
        current_time = datetime.now()
        anomalies: List[str] = []
        # Unusual system load patterns
        if psutil.cpu_percent() > 90 and current_time.hour not in [13, 14, 15]:
            anomalies.append("unusual_cpu_load")
        # Network anomalies
        try:
            network_connections = len(psutil.net_connections())
            if network_connections > 1000:
                anomalies.append("excessive_network_connections")
        except Exception:
            pass
        # Memory anomalies
        mem = psutil.virtual_memory()
        if mem.available < 100 * 1024 * 1024:  # Less than 100MB available
            anomalies.append("low_memory")
        status = "healthy" if not anomalies else "degraded"
        if "excessive_network_connections" in anomalies:
            status = "critical"
        return {"status": status, "anomalies_detected": anomalies, "anomaly_count": len(anomalies)}

    async def _activate_fallback_systems(self, health_report: Dict[str, Any]) -> None:
        """Activate appropriate fallback systems."""
        print("Activating fallback systems...")
        for system in self.fallback_systems:
            try:
                # Check if activation condition is met
                if system["condition"](health_report):
                    print(f"Activating fallback: {system['name']}")
                    # Execute recovery protocol
                    result = await system["protocol"](health_report)
                    system["last_activated"] = datetime.now().isoformat()
                    if isinstance(result, dict) and result.get("success", False):
                        system["success_rate"] = min(1.0, system["success_rate"] + 0.1)
                    else:
                        system["success_rate"] = max(0.0, system["success_rate"] - 0.2)
                    # Update system state
                    self.system_state = "fallback_active"
            except Exception as e:  # pragma: no cover - defensive logging only
                print(f"Failed to activate fallback {system['name']}: {e}")
                self.failure_count += 1

    async def _adjust_operational_parameters(self) -> None:
        """Adjust operational parameters based on failure patterns."""
        print("Adjusting operational parameters...")
        adjustments: Dict[str, Any] = {
            "increase_delays": random.uniform(1.1, 1.5),
            "reduce_concurrency": random.uniform(0.7, 0.9),
            "change_rotation_patterns": True,
            "enable_additional_logging": True,
        }
        for key, value in adjustments.items():
            os.environ[f"ADJUSTMENT_{key.upper()}"] = str(value)
        # Reset failure count after adjustment
        self.failure_count = max(0, self.failure_count - 2)

    def _check_performance_metrics(self) -> Dict[str, Any]:
        """Check system performance metrics (simulated)."""
        metrics: Dict[str, Any] = {
            "response_times": self._measure_response_times(),
            "throughput": self._measure_throughput(),
            "error_rates": self._calculate_error_rates(),
            "latency": self._measure_latency(),
        }
        degraded_metrics = sum(1 for metric in metrics.values() if metric["status"] == "degraded")
        critical_metrics = sum(1 for metric in metrics.values() if metric["status"] == "critical")
        if critical_metrics > 0:
            status = "critical"
        elif degraded_metrics > 1:
            status = "degraded"
        else:
            status = "healthy"
        return {"status": status, "metrics": metrics, "degraded_count": degraded_metrics, "critical_count": critical_metrics}

    def _measure_response_times(self) -> Dict[str, Any]:
        """Measure system response times (simulated)."""
        avg_response_time = random.uniform(50, 200)  # milliseconds
        status = "healthy"
        if avg_response_time > 500:
            status = "critical"
        elif avg_response_time > 300:
            status = "degraded"
        return {"status": status, "avg_response_time_ms": avg_response_time, "threshold_exceeded": avg_response_time > 300}

    def _measure_throughput(self) -> Dict[str, Any]:
        """Measure system throughput (simulated)."""
        operations_per_second = random.uniform(10, 100)
        status = "healthy"
        if operations_per_second < 5:
            status = "critical"
        elif operations_per_second < 20:
            status = "degraded"
        return {"status": status, "ops_per_second": operations_per_second, "threshold": 20}

    def _calculate_error_rates(self) -> Dict[str, Any]:
        """Calculate system error rates (simulated)."""
        error_rate = random.uniform(0, 0.1)
        status = "healthy"
        if error_rate > 0.05:  # 5% error rate
            status = "critical"
        elif error_rate > 0.02:  # 2% error rate
            status = "degraded"
        return {"status": status, "error_rate": error_rate, "acceptable_threshold": 0.02}

    def _measure_latency(self) -> Dict[str, Any]:
        """Measure system latency (simulated)."""
        latency = random.uniform(10, 100)  # milliseconds
        status = "healthy"
        if latency > 200:
            status = "critical"
        elif latency > 100:
            status = "degraded"
        return {"status": status, "latency_ms": latency, "threshold": 100}

    async def graceful_shutdown(self, signal_received: Optional[str] = None) -> None:
        """Handle graceful shutdown."""
        print(f"\nGraceful shutdown initiated by {signal_received}")
        # Save current state
        await self._save_system_state()
        # Notify components
        await self._notify_components_shutdown()
        # Perform cleanup
        await self._perform_shutdown_cleanup()
        print("System shutdown complete")
        # Do not call sys.exit in async context; let caller decide

    async def _save_system_state(self) -> None:
        """Save current system state for recovery."""
        state_data = {
            "timestamp": datetime.now().isoformat(),
            "system_state": self.system_state,
            "failure_count": self.failure_count,
            "fallback_systems": [
                {
                    "name": system["name"],
                    "last_activated": system["last_activated"],
                    "success_rate": system["success_rate"],
                }
                for system in self.fallback_systems
            ],
            "recovery_point": self._create_recovery_point(),
        }
        with open("system_state_backup.json", "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2)

    def _create_recovery_point(self) -> str:
        """Create a recovery point identifier."""
        return hashlib.sha256(f"recovery_{datetime.now().isoformat()}_{random.random()}".encode()).hexdigest()[:16]

    async def _notify_components_shutdown(self) -> None:
        """Notify all components of impending shutdown (simulated)."""
        print("Notifying components of shutdown...")
        await asyncio.sleep(1)

    async def _perform_shutdown_cleanup(self) -> None:
        """Perform cleanup before shutdown (simulated)."""
        print("Performing shutdown cleanup...")
        cleanup_actions = [self._clear_temporary_data(), self._close_network_connections(), self._flush_buffers(), self._archive_logs()]
        for action in cleanup_actions:
            try:
                await action
            except Exception as e:  # pragma: no cover - defensive logging only
                print(f"Cleanup action failed: {e}")

    async def _clear_temporary_data(self) -> None:
        """Clear temporary data (simulated)."""
        import tempfile

        temp_dir = tempfile.gettempdir()
        print(f"Clearing temporary data from {temp_dir}")
        await asyncio.sleep(0)

    async def _close_network_connections(self) -> None:
        """Close network connections (simulated)."""
        print("Closing network connections...")
        await asyncio.sleep(0)

    async def _flush_buffers(self) -> None:
        """Flush system buffers (simulated)."""
        print("Flushing buffers...")
        await asyncio.sleep(0)

    async def _archive_logs(self) -> None:
        """Archive system logs (simulated)."""
        print("Archiving logs...")
        await asyncio.sleep(0)

    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(self.graceful_shutdown("SIGINT")))
            loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(self.graceful_shutdown("SIGTERM")))
        except NotImplementedError:
            # Signal handlers may not be available on Windows/threads
            pass

    async def run_resilience_framework(self) -> None:
        """Run the complete resilience framework."""
        print("Starting resilience framework...")
        # Setup signal handlers
        self.setup_signal_handlers()
        # Register default fallback systems
        self._register_default_fallbacks()
        # Start health monitoring and periodic state save
        monitor_task = asyncio.create_task(self.monitor_system_health())
        state_save_task = asyncio.create_task(self._periodic_state_save())
        await asyncio.gather(monitor_task, state_save_task)

    def _register_default_fallbacks(self) -> None:
        """Register default fallback systems."""

        def high_cpu_condition(health_report: Dict[str, Any]) -> bool:
            cpu_check = health_report["detailed_checks"]["resource_utilization"]
            return cpu_check.get("cpu_percent", 0) > 85

        async def cpu_reduction_protocol(health_report: Dict[str, Any]) -> Dict[str, Any]:
            print("Activating CPU reduction protocol")
            os.environ["OPERATIONAL_MODE"] = "conservative"
            return {"success": True, "action": "cpu_reduction"}

        self.register_fallback_system("cpu_reduction", high_cpu_condition, cpu_reduction_protocol)

        def network_failure_condition(health_report: Dict[str, Any]) -> bool:
            network_check = health_report["detailed_checks"]["network_connectivity"]
            return network_check.get("status") == "critical"

        async def network_fallback_protocol(health_report: Dict[str, Any]) -> Dict[str, Any]:
            print("Activating network fallback protocol")
            os.environ["NETWORK_MODE"] = "fallback"
            return {"success": True, "action": "network_fallback"}

        self.register_fallback_system("network_fallback", network_failure_condition, network_fallback_protocol)

    async def _periodic_state_save(self, interval: int = 300) -> None:
        """Periodically save system state."""
        while True:
            try:
                await self._save_system_state()
                await asyncio.sleep(interval)
            except Exception as e:  # pragma: no cover - defensive logging only
                print(f"Periodic state save failed: {e}")
                await asyncio.sleep(60)
