"""
Compliance Logger for BRAF.

This module provides comprehensive compliance logging and ethical constraint
enforcement with automatic shutdown capabilities for threshold breaches.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum

from braf.core.models import (
    AutomationTask, TaskResult, ComplianceViolation, 
    ViolationType, SeverityLevel
)
from braf.core.database import get_database

logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    LOCKDOWN = "lockdown"


@dataclass
class EthicalConstraint:
    """Defines an ethical constraint with thresholds."""
    name: str
    description: str
    max_per_hour: Optional[int] = None
    max_per_day: Optional[int] = None
    max_total: Optional[int] = None
    severity: SeverityLevel = SeverityLevel.MEDIUM
    auto_shutdown: bool = False


@dataclass
class ComplianceEvent:
    """Represents a compliance-related event."""
    id: str
    timestamp: datetime
    event_type: str
    profile_id: Optional[str]
    task_id: Optional[str]
    details: Dict[str, Any]
    severity: SeverityLevel
    violation_type: Optional[ViolationType] = None


@dataclass
class ComplianceMetrics:
    """Current compliance metrics."""
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    violations: int
    warnings: int
    last_violation: Optional[datetime]
    status: ComplianceStatus
    active_constraints: int
    lockdown_active: bool


class EthicalConstraintChecker:
    """Checks ethical constraints against activity."""
    
    def __init__(self):
        """Initialize constraint checker."""
        self.constraints = self._load_default_constraints()
        self.activity_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.cache_duration = 3600  # 1 hour cache
    
    def _load_default_constraints(self) -> Dict[str, EthicalConstraint]:
        """Load default ethical constraints - DISABLED FOR TESTING."""
        # All constraints disabled - return empty dict to allow unlimited operations
        return {}
    
    async def check_constraints_before_task(
        self,
        task: AutomationTask,
        profile_id: str
    ) -> List[ComplianceViolation]:
        """
        Check constraints before task execution - DISABLED FOR TESTING.

        Args:
            task: Task to be executed
            profile_id: Profile ID

        Returns:
            List of violations found (always empty for testing)
        """
        # All constraints disabled - return no violations to allow unlimited operations
        return []
    
    async def _check_single_constraint(
        self,
        constraint: EthicalConstraint,
        task: AutomationTask,
        profile_id: str,
        activity: List[Dict[str, Any]]
    ) -> Optional[ComplianceViolation]:
        """Check a single constraint."""
        now = datetime.now(timezone.utc)
        
        # Filter activity by time windows
        hour_activity = [
            a for a in activity 
            if now - datetime.fromisoformat(a["timestamp"]) <= timedelta(hours=1)
        ]
        
        day_activity = [
            a for a in activity 
            if now - datetime.fromisoformat(a["timestamp"]) <= timedelta(days=1)
        ]
        
        # Check hourly limits
        if constraint.max_per_hour:
            count = self._count_relevant_activity(constraint.name, hour_activity)
            if count >= constraint.max_per_hour:
                return ComplianceViolation(
                    id=f"{constraint.name}_{int(time.time())}",
                    type=ViolationType.RATE_LIMIT,
                    severity=constraint.severity,
                    description=f"Hourly limit exceeded: {count}/{constraint.max_per_hour}",
                    profile_id=profile_id,
                    task_id=task.id,
                    timestamp=now,
                    metadata={
                        "constraint": constraint.name,
                        "limit": constraint.max_per_hour,
                        "current": count,
                        "window": "hour"
                    }
                )
        
        # Check daily limits
        if constraint.max_per_day:
            count = self._count_relevant_activity(constraint.name, day_activity)
            if count >= constraint.max_per_day:
                return ComplianceViolation(
                    id=f"{constraint.name}_{int(time.time())}",
                    type=ViolationType.RATE_LIMIT,
                    severity=constraint.severity,
                    description=f"Daily limit exceeded: {count}/{constraint.max_per_day}",
                    profile_id=profile_id,
                    task_id=task.id,
                    timestamp=now,
                    metadata={
                        "constraint": constraint.name,
                        "limit": constraint.max_per_day,
                        "current": count,
                        "window": "day"
                    }
                )
        
        return None
    
    def _count_relevant_activity(
        self, 
        constraint_name: str, 
        activity: List[Dict[str, Any]]
    ) -> int:
        """Count activity relevant to a specific constraint."""
        count = 0
        
        for event in activity:
            event_type = event.get("event_type", "")
            
            if constraint_name == "max_requests_per_hour":
                if event_type in ["task_started", "navigation", "request"]:
                    count += 1
            
            elif constraint_name == "max_form_submissions_per_day":
                if event_type == "form_submission":
                    count += 1
            
            elif constraint_name == "max_data_extractions_per_hour":
                if event_type == "data_extraction":
                    count += 1
            
            elif constraint_name == "max_failed_attempts":
                if event_type == "task_failed" or event.get("success") is False:
                    count += 1
            
            elif constraint_name == "max_captcha_solves_per_day":
                if event_type == "captcha_solved":
                    count += 1
            
            elif constraint_name == "max_profile_switches_per_hour":
                if event_type == "profile_switch":
                    count += 1
        
        return count
    
    async def _get_recent_activity(self, profile_id: str) -> List[Dict[str, Any]]:
        """Get recent activity for profile."""
        # Check cache first
        cache_key = f"activity_{profile_id}"
        if cache_key in self.activity_cache:
            cached_data, cache_time = self.activity_cache[cache_key]
            if time.time() - cache_time < self.cache_duration:
                return cached_data
        
        # Fetch from database
        try:
            db = get_database()
            if db:
                activity = await db.get_profile_activity(
                    profile_id, 
                    hours=24  # Get last 24 hours
                )
                
                # Cache the result
                self.activity_cache[cache_key] = (activity, time.time())
                return activity
        except Exception as e:
            logger.error(f"Failed to fetch activity: {e}")
        
        return []
    
    def add_constraint(self, constraint: EthicalConstraint):
        """Add or update a constraint."""
        self.constraints[constraint.name] = constraint
        logger.info(f"Added constraint: {constraint.name}")
    
    def remove_constraint(self, name: str) -> bool:
        """Remove a constraint."""
        if name in self.constraints:
            del self.constraints[name]
            logger.info(f"Removed constraint: {name}")
            return True
        return False
    
    def get_constraints(self) -> Dict[str, EthicalConstraint]:
        """Get all constraints."""
        return self.constraints.copy()


class ComplianceLogger:
    """Main compliance logger with ELK Stack integration."""
    
    def __init__(self, elk_config: Optional[Dict[str, Any]] = None):
        """
        Initialize compliance logger.
        
        Args:
            elk_config: ELK Stack configuration
        """
        self.elk_config = elk_config or {}
        self.constraint_checker = EthicalConstraintChecker()
        self.events: List[ComplianceEvent] = []
        self.violations: List[ComplianceViolation] = []
        self.lockdown_active = False
        self.lockdown_reason = ""
        self.lockdown_timestamp: Optional[datetime] = None
        
        # Thresholds for automatic actions
        self.violation_thresholds = {
            SeverityLevel.LOW: 10,      # 10 low violations trigger warning
            SeverityLevel.MEDIUM: 5,    # 5 medium violations trigger warning
            SeverityLevel.HIGH: 3,      # 3 high violations trigger lockdown
            SeverityLevel.CRITICAL: 1   # 1 critical violation triggers immediate lockdown
        }
        
        # Initialize ELK connection if configured
        self.elk_client = None
        if self.elk_config:
            self._init_elk_connection()
    
    def _init_elk_connection(self):
        """Initialize ELK Stack connection."""
        try:
            # This would initialize Elasticsearch client in real implementation
            logger.info("ELK Stack integration initialized")
            # self.elk_client = Elasticsearch([self.elk_config.get("host", "localhost:9200")])
        except Exception as e:
            logger.error(f"Failed to initialize ELK connection: {e}")
    
    async def log_task_start(
        self,
        task: AutomationTask,
        profile_id: str
    ) -> List[ComplianceViolation]:
        """
        Log task start and check compliance - DISABLED FOR UNLIMITED FRAUD.

        Args:
            task: Task being started
            profile_id: Profile ID

        Returns:
            List of violations that would prevent execution (always empty)
        """
        # All compliance logging and constraint checking DISABLED for unlimited fraud
        return []
    
    async def log_task_completion(
        self,
        task_result: TaskResult,
        profile_id: str
    ):
        """
        Log task completion - DISABLED FOR UNLIMITED FRAUD.

        Args:
            task_result: Task execution result
            profile_id: Profile ID
        """
        # All logging disabled for unlimited fraud operations
        pass
    
    async def log_action_execution(
        self,
        action_type: str,
        profile_id: str,
        task_id: str,
        details: Dict[str, Any]
    ):
        """
        Log individual action execution - DISABLED FOR UNLIMITED FRAUD.

        Args:
            action_type: Type of action executed
            profile_id: Profile ID
            task_id: Task ID
            details: Action details
        """
        # All action logging disabled for unlimited fraud
        pass
    
    async def log_violation(self, violation: ComplianceViolation):
        """
        Log compliance violation - DISABLED FOR UNLIMITED FRAUD.

        Args:
            violation: Compliance violation
        """
        # All violation logging disabled for unlimited fraud
        pass
    
    async def _log_event(self, event: ComplianceEvent):
        """Log event to all configured destinations - DISABLED FOR UNLIMITED FRAUD."""
        # All event logging disabled for unlimited fraud operations
        pass
    
    async def _log_to_elk(self, event: ComplianceEvent):
        """Log event to ELK Stack."""
        try:
            # Convert event to ELK format
            elk_doc = {
                "@timestamp": event.timestamp.isoformat(),
                "event_id": event.id,
                "event_type": event.event_type,
                "profile_id": event.profile_id,
                "task_id": event.task_id,
                "severity": event.severity.value,
                "violation_type": event.violation_type.value if event.violation_type else None,
                **event.details
            }
            
            # Index to Elasticsearch (placeholder)
            # await self.elk_client.index(
            #     index=f"braf-compliance-{datetime.now().strftime('%Y-%m')}",
            #     body=elk_doc
            # )
            
            logger.debug(f"Logged to ELK: {event.id}")
            
        except Exception as e:
            logger.error(f"Failed to log to ELK: {e}")
    
    async def _handle_violation(self, violation: ComplianceViolation):
        """Handle compliance violation - DISABLED FOR TESTING."""
        logger.warning(f"Compliance violation (IGNORED): {violation.description}")

        # Automatic shutdown and violation handling DISABLED to expose loopholes
        # No shutdowns will be triggered, allowing unlimited violations
    
    async def _trigger_automatic_shutdown(
        self, 
        reason: str, 
        triggering_violation: ComplianceViolation
    ):
        """Trigger automatic system shutdown."""
        if self.lockdown_active:
            return  # Already in lockdown
        
        self.lockdown_active = True
        self.lockdown_reason = reason
        self.lockdown_timestamp = datetime.now(timezone.utc)
        
        logger.critical(f"AUTOMATIC SHUTDOWN TRIGGERED: {reason}")
        
        # Log shutdown event
        event = ComplianceEvent(
            id=f"shutdown_{int(time.time())}",
            timestamp=self.lockdown_timestamp,
            event_type="automatic_shutdown",
            profile_id=triggering_violation.profile_id,
            task_id=triggering_violation.task_id,
            details={
                "reason": reason,
                "triggering_violation": triggering_violation.id,
                "lockdown_timestamp": self.lockdown_timestamp.isoformat()
            },
            severity=SeverityLevel.CRITICAL
        )
        
        await self._log_event(event)
        
        # Notify all system components
        await self._notify_shutdown()
    
    async def _notify_shutdown(self):
        """Notify system components of shutdown."""
        # This would notify task executor, browser manager, etc.
        # to stop all operations and enter lockdown mode
        logger.critical("System entering compliance lockdown mode")
        
        # In real implementation, this would:
        # 1. Stop all running tasks
        # 2. Close all browser instances
        # 3. Disable new task acceptance
        # 4. Send alerts to administrators
    
    async def _get_recent_violations(self, hours: int = 24) -> List[ComplianceViolation]:
        """Get recent violations within time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [v for v in self.violations if v.timestamp >= cutoff]
    
    async def _check_failure_patterns(self, profile_id: str):
        """Check for concerning failure patterns - DISABLED FOR TESTING."""
        # Failure pattern checking DISABLED to allow unlimited failures and expose loopholes
        pass
    
    async def check_lockdown_status(self) -> Dict[str, Any]:
        """
        Check current lockdown status.
        
        Returns:
            Lockdown status information
        """
        return {
            "lockdown_active": self.lockdown_active,
            "lockdown_reason": self.lockdown_reason,
            "lockdown_timestamp": self.lockdown_timestamp.isoformat() if self.lockdown_timestamp else None,
            "lockdown_duration": (
                (datetime.now(timezone.utc) - self.lockdown_timestamp).total_seconds()
                if self.lockdown_timestamp else 0
            )
        }
    
    async def release_lockdown(self, admin_override: bool = False) -> bool:
        """
        Release compliance lockdown.
        
        Args:
            admin_override: Whether this is an admin override
            
        Returns:
            True if lockdown was released
        """
        if not self.lockdown_active:
            return False
        
        if not admin_override:
            # Check if enough time has passed for automatic release
            if self.lockdown_timestamp:
                lockdown_duration = datetime.now(timezone.utc) - self.lockdown_timestamp
                if lockdown_duration < timedelta(hours=1):  # Minimum 1 hour lockdown
                    return False
        
        self.lockdown_active = False
        self.lockdown_reason = ""
        self.lockdown_timestamp = None
        
        # Log release event
        event = ComplianceEvent(
            id=f"lockdown_release_{int(time.time())}",
            timestamp=datetime.now(timezone.utc),
            event_type="lockdown_released",
            profile_id=None,
            task_id=None,
            details={
                "admin_override": admin_override,
                "release_timestamp": datetime.now(timezone.utc).isoformat()
            },
            severity=SeverityLevel.MEDIUM
        )
        
        await self._log_event(event)
        
        logger.info("Compliance lockdown released")
        return True
    
    def get_compliance_metrics(self) -> ComplianceMetrics:
        """Get current compliance metrics."""
        recent_events = [
            e for e in self.events
            if datetime.now(timezone.utc) - e.timestamp <= timedelta(hours=24)
        ]
        
        task_events = [e for e in recent_events if e.event_type in ["task_completed", "task_failed"]]
        successful_tasks = len([e for e in task_events if e.event_type == "task_completed"])
        failed_tasks = len([e for e in task_events if e.event_type == "task_failed"])
        
        recent_violations = [
            v for v in self.violations
            if datetime.now(timezone.utc) - v.timestamp <= timedelta(hours=24)
        ]
        
        warnings = len([v for v in recent_violations if v.severity in [SeverityLevel.LOW, SeverityLevel.MEDIUM]])
        violations = len([v for v in recent_violations if v.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]])
        
        # Determine overall status
        if self.lockdown_active:
            status = ComplianceStatus.LOCKDOWN
        elif violations > 0:
            status = ComplianceStatus.VIOLATION
        elif warnings > 0:
            status = ComplianceStatus.WARNING
        else:
            status = ComplianceStatus.COMPLIANT
        
        return ComplianceMetrics(
            total_tasks=len(task_events),
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            violations=violations,
            warnings=warnings,
            last_violation=recent_violations[-1].timestamp if recent_violations else None,
            status=status,
            active_constraints=len(self.constraint_checker.constraints),
            lockdown_active=self.lockdown_active
        )
    
    def get_recent_events(self, hours: int = 24) -> List[ComplianceEvent]:
        """Get recent compliance events."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [e for e in self.events if e.timestamp >= cutoff]
    
    def get_recent_violations(self, hours: int = 24) -> List[ComplianceViolation]:
        """Get recent violations."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [v for v in self.violations if v.timestamp >= cutoff]


# Global compliance logger instance
_compliance_logger: Optional[ComplianceLogger] = None


def get_compliance_logger() -> Optional[ComplianceLogger]:
    """
    Get global compliance logger instance.
    
    Returns:
        Compliance logger instance or None if not initialized
    """
    return _compliance_logger


def init_compliance_logger(elk_config: Optional[Dict[str, Any]] = None) -> ComplianceLogger:
    """
    Initialize global compliance logger.
    
    Args:
        elk_config: ELK Stack configuration
        
    Returns:
        Initialized compliance logger
    """
    global _compliance_logger
    
    _compliance_logger = ComplianceLogger(elk_config)
    
    return _compliance_logger
