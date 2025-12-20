"""
Unit tests for ComplianceLogger component.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from braf.core.models import (
    AutomationTask, AutomationAction, ActionType, TaskResult,
    ComplianceViolation, ViolationType, SeverityLevel
)
from braf.core.compliance_logger import (
    ComplianceLogger, EthicalConstraint, EthicalConstraintChecker,
    ComplianceStatus, ComplianceEvent
)


class TestEthicalConstraint:
    """Test EthicalConstraint data class."""
    
    def test_constraint_creation(self):
        """Test constraint creation with various parameters."""
        constraint = EthicalConstraint(
            name="test_constraint",
            description="Test constraint",
            max_per_hour=10,
            max_per_day=100,
            severity=SeverityLevel.HIGH,
            auto_shutdown=True
        )
        
        assert constraint.name == "test_constraint"
        assert constraint.max_per_hour == 10
        assert constraint.max_per_day == 100
        assert constraint.severity == SeverityLevel.HIGH
        assert constraint.auto_shutdown is True
    
    def test_constraint_defaults(self):
        """Test constraint with default values."""
        constraint = EthicalConstraint(
            name="default_constraint",
            description="Default constraint"
        )
        
        assert constraint.max_per_hour is None
        assert constraint.max_per_day is None
        assert constraint.max_total is None
        assert constraint.severity == SeverityLevel.MEDIUM
        assert constraint.auto_shutdown is False


class TestEthicalConstraintChecker:
    """Test EthicalConstraintChecker functionality."""
    
    @pytest.fixture
    def checker(self):
        """Create constraint checker for testing."""
        return EthicalConstraintChecker()
    
    @pytest.fixture
    def sample_task(self):
        """Create sample task for testing."""
        return AutomationTask(
            id="test_task",
            profile_id="test_profile",
            actions=[
                AutomationAction(type=ActionType.NAVIGATE, url="https://example.com"),
                AutomationAction(type=ActionType.CLICK, selector="button")
            ]
        )
    
    def test_load_default_constraints(self, checker):
        """Test loading of default constraints."""
        constraints = checker.constraints
        
        assert len(constraints) > 0
        assert "max_requests_per_hour" in constraints
        assert "max_form_submissions_per_day" in constraints
        assert "max_data_extractions_per_hour" in constraints
        
        # Check constraint properties
        req_constraint = constraints["max_requests_per_hour"]
        assert req_constraint.max_per_hour == 100
        assert req_constraint.auto_shutdown is True
    
    async def test_check_constraints_no_violations(self, checker, sample_task):
        """Test constraint checking with no violations."""
        # Mock empty activity
        with patch.object(checker, '_get_recent_activity') as mock_activity:
            mock_activity.return_value = []
            
            violations = await checker.check_constraints_before_task(
                sample_task, "test_profile"
            )
            
            assert len(violations) == 0
    
    async def test_check_constraints_with_violations(self, checker, sample_task):
        """Test constraint checking with violations."""
        # Mock activity that exceeds limits
        mock_activity_data = [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "task_started",
                "profile_id": "test_profile"
            }
            for _ in range(150)  # Exceeds max_requests_per_hour (100)
        ]
        
        with patch.object(checker, '_get_recent_activity') as mock_activity:
            mock_activity.return_value = mock_activity_data
            
            violations = await checker.check_constraints_before_task(
                sample_task, "test_profile"
            )
            
            assert len(violations) > 0
            assert violations[0].type == ViolationType.RATE_LIMIT
            assert "Hourly limit exceeded" in violations[0].description
    
    def test_count_relevant_activity(self, checker):
        """Test activity counting for different constraint types."""
        activity = [
            {"event_type": "task_started", "timestamp": datetime.now(timezone.utc).isoformat()},
            {"event_type": "form_submission", "timestamp": datetime.now(timezone.utc).isoformat()},
            {"event_type": "data_extraction", "timestamp": datetime.now(timezone.utc).isoformat()},
            {"event_type": "task_failed", "success": False, "timestamp": datetime.now(timezone.utc).isoformat()},
            {"event_type": "captcha_solved", "timestamp": datetime.now(timezone.utc).isoformat()},
        ]
        
        # Test different constraint types
        assert checker._count_relevant_activity("max_requests_per_hour", activity) == 1
        assert checker._count_relevant_activity("max_form_submissions_per_day", activity) == 1
        assert checker._count_relevant_activity("max_data_extractions_per_hour", activity) == 1
        assert checker._count_relevant_activity("max_failed_attempts", activity) == 1
        assert checker._count_relevant_activity("max_captcha_solves_per_day", activity) == 1
    
    def test_add_remove_constraints(self, checker):
        """Test adding and removing constraints."""
        initial_count = len(checker.constraints)
        
        # Add new constraint
        new_constraint = EthicalConstraint(
            name="test_new_constraint",
            description="Test constraint",
            max_per_hour=5
        )
        
        checker.add_constraint(new_constraint)
        assert len(checker.constraints) == initial_count + 1
        assert "test_new_constraint" in checker.constraints
        
        # Remove constraint
        result = checker.remove_constraint("test_new_constraint")
        assert result is True
        assert len(checker.constraints) == initial_count
        assert "test_new_constraint" not in checker.constraints
        
        # Try to remove non-existent constraint
        result = checker.remove_constraint("non_existent")
        assert result is False
    
    def test_get_constraints(self, checker):
        """Test getting constraints."""
        constraints = checker.get_constraints()
        
        assert isinstance(constraints, dict)
        assert len(constraints) > 0
        
        # Should be a copy, not the original
        constraints["test"] = "modified"
        assert "test" not in checker.constraints


class TestComplianceLogger:
    """Test ComplianceLogger functionality."""
    
    @pytest.fixture
    def logger(self):
        """Create compliance logger for testing."""
        return ComplianceLogger()
    
    @pytest.fixture
    def sample_task(self):
        """Create sample task for testing."""
        return AutomationTask(
            id="test_task",
            profile_id="test_profile",
            actions=[
                AutomationAction(type=ActionType.NAVIGATE, url="https://example.com")
            ]
        )
    
    @pytest.fixture
    def sample_task_result(self):
        """Create sample task result."""
        return TaskResult(
            task_id="test_task",
            success=True,
            execution_time=45.2,
            actions_completed=1,
            metadata={"test": True}
        )
    
    def test_logger_initialization(self, logger):
        """Test logger initialization."""
        assert logger.constraint_checker is not None
        assert len(logger.events) == 0
        assert len(logger.violations) == 0
        assert logger.lockdown_active is False
        assert logger.lockdown_reason == ""
        assert logger.lockdown_timestamp is None
    
    async def test_log_task_start(self, logger, sample_task):
        """Test logging task start."""
        profile_id = "test_profile"
        
        # Mock database to avoid actual DB calls
        with patch('braf.core.compliance_logger.get_database') as mock_db:
            mock_db.return_value = None
            
            violations = await logger.log_task_start(sample_task, profile_id)
            
            assert len(violations) == 0  # No violations expected
            assert len(logger.events) > 0
            
            # Check the logged event
            start_event = logger.events[-1]
            assert start_event.event_type == "task_started"
            assert start_event.profile_id == profile_id
            assert start_event.task_id == sample_task.id
            assert start_event.severity == SeverityLevel.LOW
    
    async def test_log_task_completion(self, logger, sample_task_result):
        """Test logging task completion."""
        profile_id = "test_profile"
        
        with patch('braf.core.compliance_logger.get_database') as mock_db:
            mock_db.return_value = None
            
            await logger.log_task_completion(sample_task_result, profile_id)
            
            assert len(logger.events) > 0
            
            # Check the logged event
            completion_event = logger.events[-1]
            assert completion_event.event_type == "task_completed"
            assert completion_event.profile_id == profile_id
            assert completion_event.task_id == sample_task_result.task_id
            assert completion_event.details["success"] is True
    
    async def test_log_action_execution(self, logger):
        """Test logging action execution."""
        with patch('braf.core.compliance_logger.get_database') as mock_db:
            mock_db.return_value = None
            
            await logger.log_action_execution(
                "click", "test_profile", "test_task", {"success": True}
            )
            
            assert len(logger.events) > 0
            
            event = logger.events[-1]
            assert event.event_type == "click"
            assert event.profile_id == "test_profile"
            assert event.task_id == "test_task"
            assert event.details["success"] is True
    
    async def test_log_violation(self, logger):
        """Test logging compliance violation."""
        violation = ComplianceViolation(
            id="test_violation",
            type=ViolationType.RATE_LIMIT,
            severity=SeverityLevel.MEDIUM,
            description="Test violation",
            profile_id="test_profile",
            timestamp=datetime.now(timezone.utc),
            metadata={"test": True}
        )
        
        with patch('braf.core.compliance_logger.get_database') as mock_db:
            mock_db.return_value = None
            
            await logger.log_violation(violation)
            
            assert len(logger.violations) > 0
            assert len(logger.events) > 0
            
            # Check violation was stored
            stored_violation = logger.violations[-1]
            assert stored_violation.id == violation.id
            
            # Check event was logged
            violation_event = logger.events[-1]
            assert violation_event.event_type == "compliance_violation"
            assert violation_event.violation_type == ViolationType.RATE_LIMIT
    
    async def test_automatic_shutdown_trigger(self, logger):
        """Test automatic shutdown triggering."""
        # Create critical violation that should trigger shutdown
        critical_violation = ComplianceViolation(
            id="critical_violation",
            type=ViolationType.SUSPICIOUS_ACTIVITY,
            severity=SeverityLevel.CRITICAL,
            description="Critical test violation",
            profile_id="test_profile",
            timestamp=datetime.now(timezone.utc),
            metadata={"constraint": "test_constraint"}
        )
        
        # Mock constraint with auto_shutdown enabled
        mock_constraint = EthicalConstraint(
            name="test_constraint",
            description="Test constraint",
            auto_shutdown=True,
            severity=SeverityLevel.CRITICAL
        )
        
        logger.constraint_checker.constraints["test_constraint"] = mock_constraint
        
        with patch('braf.core.compliance_logger.get_database') as mock_db:
            mock_db.return_value = None
            
            await logger.log_violation(critical_violation)
            
            # Should trigger lockdown
            assert logger.lockdown_active is True
            assert logger.lockdown_reason is not None
            assert logger.lockdown_timestamp is not None
    
    async def test_lockdown_status_check(self, logger):
        """Test lockdown status checking."""
        # Initially not in lockdown
        status = await logger.check_lockdown_status()
        assert status["lockdown_active"] is False
        assert status["lockdown_reason"] == ""
        assert status["lockdown_timestamp"] is None
        
        # Trigger lockdown
        logger.lockdown_active = True
        logger.lockdown_reason = "Test lockdown"
        logger.lockdown_timestamp = datetime.now(timezone.utc)
        
        status = await logger.check_lockdown_status()
        assert status["lockdown_active"] is True
        assert status["lockdown_reason"] == "Test lockdown"
        assert status["lockdown_timestamp"] is not None
        assert status["lockdown_duration"] >= 0
    
    async def test_release_lockdown(self, logger):
        """Test lockdown release."""
        # Set up lockdown
        logger.lockdown_active = True
        logger.lockdown_reason = "Test lockdown"
        logger.lockdown_timestamp = datetime.now(timezone.utc) - timedelta(hours=2)
        
        with patch('braf.core.compliance_logger.get_database') as mock_db:
            mock_db.return_value = None
            
            # Release with admin override
            result = await logger.release_lockdown(admin_override=True)
            assert result is True
            assert logger.lockdown_active is False
            
            # Check release event was logged
            assert len(logger.events) > 0
            release_event = logger.events[-1]
            assert release_event.event_type == "lockdown_released"
    
    async def test_failure_pattern_detection(self, logger):
        """Test failure pattern detection."""
        profile_id = "test_profile"
        
        # Add multiple failure events
        for i in range(6):  # More than threshold (5)
            event = ComplianceEvent(
                id=f"failure_{i}",
                timestamp=datetime.now(timezone.utc),
                event_type="task_failed",
                profile_id=profile_id,
                task_id=f"task_{i}",
                details={"success": False},
                severity=SeverityLevel.MEDIUM
            )
            logger.events.append(event)
        
        with patch('braf.core.compliance_logger.get_database') as mock_db:
            mock_db.return_value = None
            
            await logger._check_failure_patterns(profile_id)
            
            # Should detect pattern and create violation
            assert len(logger.violations) > 0
            pattern_violation = logger.violations[-1]
            assert pattern_violation.type == ViolationType.SUSPICIOUS_ACTIVITY
            assert "failure rate" in pattern_violation.description.lower()
    
    def test_get_compliance_metrics(self, logger):
        """Test compliance metrics generation."""
        # Add some test events
        now = datetime.now(timezone.utc)
        
        # Add task events
        logger.events.extend([
            ComplianceEvent(
                id="task_1", timestamp=now, event_type="task_completed",
                profile_id="p1", task_id="t1", details={"success": True},
                severity=SeverityLevel.LOW
            ),
            ComplianceEvent(
                id="task_2", timestamp=now, event_type="task_failed",
                profile_id="p1", task_id="t2", details={"success": False},
                severity=SeverityLevel.MEDIUM
            )
        ])
        
        # Add violations
        logger.violations.extend([
            ComplianceViolation(
                id="v1", type=ViolationType.RATE_LIMIT, severity=SeverityLevel.MEDIUM,
                description="Test", profile_id="p1", timestamp=now
            ),
            ComplianceViolation(
                id="v2", type=ViolationType.SUSPICIOUS_ACTIVITY, severity=SeverityLevel.HIGH,
                description="Test", profile_id="p1", timestamp=now
            )
        ])
        
        metrics = logger.get_compliance_metrics()
        
        assert metrics.total_tasks == 2
        assert metrics.successful_tasks == 1
        assert metrics.failed_tasks == 1
        assert metrics.warnings == 1  # Medium severity
        assert metrics.violations == 1  # High severity
        assert metrics.status == ComplianceStatus.VIOLATION  # Due to high severity violation
    
    def test_get_recent_events_and_violations(self, logger):
        """Test getting recent events and violations."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(hours=25)  # Older than 24 hours
        
        # Add recent and old events
        logger.events.extend([
            ComplianceEvent(
                id="recent", timestamp=now, event_type="test",
                profile_id="p1", task_id="t1", details={},
                severity=SeverityLevel.LOW
            ),
            ComplianceEvent(
                id="old", timestamp=old_time, event_type="test",
                profile_id="p1", task_id="t1", details={},
                severity=SeverityLevel.LOW
            )
        ])
        
        logger.violations.extend([
            ComplianceViolation(
                id="recent_v", type=ViolationType.RATE_LIMIT, severity=SeverityLevel.LOW,
                description="Recent", profile_id="p1", timestamp=now
            ),
            ComplianceViolation(
                id="old_v", type=ViolationType.RATE_LIMIT, severity=SeverityLevel.LOW,
                description="Old", profile_id="p1", timestamp=old_time
            )
        ])
        
        # Get recent events (24 hours)
        recent_events = logger.get_recent_events(24)
        assert len(recent_events) == 1
        assert recent_events[0].id == "recent"
        
        # Get recent violations (24 hours)
        recent_violations = logger.get_recent_violations(24)
        assert len(recent_violations) == 1
        assert recent_violations[0].id == "recent_v"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])