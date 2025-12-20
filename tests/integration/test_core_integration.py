"""
Integration tests for core BRAF components.

Tests the integration between task executor, compliance logger,
behavioral engine, and other core components.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from braf.core.models import (
    AutomationTask, AutomationAction, ActionType, TaskPriority,
    TaskResult, ComplianceViolation, ViolationType, SeverityLevel
)
from braf.core.task_executor import TaskExecutor, TaskValidator, TaskPreprocessor
from braf.core.compliance_logger import ComplianceLogger, EthicalConstraint
from braf.core.behavioral.behavioral_engine import BehavioralEngine


class TestCoreIntegration:
    """Test core component integration."""
    
    @pytest.fixture
    async def task_executor(self):
        """Create task executor for testing."""
        return TaskExecutor()
    
    @pytest.fixture
    async def compliance_logger(self):
        """Create compliance logger for testing."""
        return ComplianceLogger()
    
    @pytest.fixture
    async def behavioral_engine(self):
        """Create behavioral engine for testing."""
        return BehavioralEngine()
    
    @pytest.fixture
    def sample_task(self):
        """Create sample automation task."""
        return AutomationTask(
            id="test_task_001",
            profile_id="test_profile_001",
            actions=[
                AutomationAction(
                    type=ActionType.NAVIGATE,
                    url="https://example.com",
                    timeout=30
                ),
                AutomationAction(
                    type=ActionType.CLICK,
                    selector="button#submit",
                    timeout=10
                ),
                AutomationAction(
                    type=ActionType.TYPE,
                    selector="input[name='username']",
                    data="testuser",
                    timeout=10
                )
            ],
            priority=TaskPriority.NORMAL,
            timeout=300
        )
    
    async def test_task_validation_integration(self, task_executor, sample_task):
        """Test task validation with various scenarios."""
        validator = task_executor.validator
        
        # Test valid task
        result = await validator.validate_task(sample_task)
        assert result["valid"] is True
        assert result["action_count"] == 3
        assert "estimated_duration" in result
        
        # Test task with too many actions
        large_task = AutomationTask(
            id="large_task",
            profile_id="test_profile",
            actions=[
                AutomationAction(type=ActionType.CLICK, selector="button")
                for _ in range(150)  # Exceeds max_actions_per_task
            ]
        )
        
        with pytest.raises(Exception):  # Should raise TaskValidationError
            await validator.validate_task(large_task)
    
    async def test_task_preprocessing_integration(self, task_executor, sample_task):
        """Test task preprocessing functionality."""
        preprocessor = task_executor.preprocessor
        
        # Preprocess the task
        processed_task = await preprocessor.preprocess_task(sample_task)
        
        # Should have more actions due to implicit waits and CAPTCHA checks
        assert len(processed_task.actions) > len(sample_task.actions)
        assert processed_task.metadata["preprocessed_at"] is not None
        assert processed_task.metadata["original_action_count"] == 3
        
        # Check for implicit waits after navigation
        navigation_indices = [
            i for i, action in enumerate(processed_task.actions)
            if action.type == ActionType.NAVIGATE
        ]
        
        for nav_idx in navigation_indices:
            if nav_idx + 1 < len(processed_task.actions):
                next_action = processed_task.actions[nav_idx + 1]
                # Should have a wait or CAPTCHA check after navigation
                assert (next_action.type == ActionType.WAIT or 
                       next_action.metadata.get("captcha_check"))
    
    async def test_compliance_constraint_checking(self, compliance_logger, sample_task):
        """Test compliance constraint checking."""
        constraint_checker = compliance_logger.constraint_checker
        
        # Test with no violations
        violations = await constraint_checker.check_constraints_before_task(
            sample_task, "test_profile"
        )
        assert len(violations) == 0
        
        # Add a strict constraint
        strict_constraint = EthicalConstraint(
            name="test_strict_limit",
            description="Very strict limit for testing",
            max_per_hour=0,  # No actions allowed
            severity=SeverityLevel.HIGH,
            auto_shutdown=True
        )
        constraint_checker.add_constraint(strict_constraint)
        
        # Mock recent activity to trigger violation
        with patch.object(constraint_checker, '_get_recent_activity') as mock_activity:
            mock_activity.return_value = [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "task_started",
                    "profile_id": "test_profile"
                }
            ]
            
            violations = await constraint_checker.check_constraints_before_task(
                sample_task, "test_profile"
            )
            
            # Should detect violation due to strict limit
            assert len(violations) > 0
            assert violations[0].type == ViolationType.RATE_LIMIT
    
    async def test_compliance_logging_integration(self, compliance_logger, sample_task):
        """Test compliance logging functionality."""
        profile_id = "test_profile_001"
        
        # Mock database to avoid actual DB calls
        with patch('braf.core.compliance_logger.get_database') as mock_db:
            mock_db.return_value = None  # No database configured
            
            # Log task start
            violations = await compliance_logger.log_task_start(sample_task, profile_id)
            assert len(violations) == 0  # No violations expected
            
            # Check that event was logged
            assert len(compliance_logger.events) > 0
            start_event = compliance_logger.events[-1]
            assert start_event.event_type == "task_started"
            assert start_event.profile_id == profile_id
            assert start_event.task_id == sample_task.id
            
            # Log task completion
            task_result = TaskResult(
                task_id=sample_task.id,
                success=True,
                execution_time=45.2,
                actions_completed=3,
                metadata={"test": True}
            )
            
            await compliance_logger.log_task_completion(task_result, profile_id)
            
            # Check completion event
            completion_event = compliance_logger.events[-1]
            assert completion_event.event_type == "task_completed"
            assert completion_event.details["success"] is True
    
    async def test_lockdown_mechanism(self, compliance_logger):
        """Test automatic lockdown mechanism."""
        # Create a critical violation
        critical_violation = ComplianceViolation(
            id="critical_test_violation",
            type=ViolationType.SUSPICIOUS_ACTIVITY,
            severity=SeverityLevel.CRITICAL,
            description="Test critical violation",
            profile_id="test_profile",
            timestamp=datetime.now(timezone.utc),
            metadata={"test": True}
        )
        
        # Mock database and ELK logging
        with patch('braf.core.compliance_logger.get_database') as mock_db:
            mock_db.return_value = None
            
            # Log the violation
            await compliance_logger.log_violation(critical_violation)
            
            # Should trigger lockdown
            assert compliance_logger.lockdown_active is True
            assert "critical" in compliance_logger.lockdown_reason.lower()
            
            # Check lockdown status
            status = await compliance_logger.check_lockdown_status()
            assert status["lockdown_active"] is True
            assert status["lockdown_reason"] is not None
    
    @pytest.mark.asyncio
    async def test_behavioral_integration(self, behavioral_engine):
        """Test behavioral engine integration."""
        # Test mouse movement generation
        target_pos = (500, 300)
        movement_path = await behavioral_engine.move_mouse(target_pos)
        
        assert len(movement_path) > 0
        assert all(len(point) == 3 for point in movement_path)  # (x, y, timestamp)
        
        # Test typing simulation
        text = "Hello World"
        typing_sequence = await behavioral_engine.type_text(text)
        
        assert len(typing_sequence) > 0
        # Should have more keystrokes than characters due to errors and corrections
        keystroke_count = len([seq for seq in typing_sequence if seq[0] not in ["pause", "backspace"]])
        assert keystroke_count >= len(text)
        
        # Test delay generation
        delay = await behavioral_engine.wait_with_human_delay("click")
        assert delay > 0
        assert delay < 5.0  # Should be reasonable
    
    @pytest.mark.asyncio
    async def test_full_task_execution_flow(self, task_executor, compliance_logger):
        """Test complete task execution flow with mocked browser."""
        # Create a simple task
        task = AutomationTask(
            id="integration_test_task",
            profile_id="integration_test_profile",
            actions=[
                AutomationAction(
                    type=ActionType.NAVIGATE,
                    url="https://httpbin.org/html",
                    timeout=30
                )
            ],
            priority=TaskPriority.NORMAL,
            timeout=60
        )
        
        # Mock all external dependencies
        with patch('braf.core.task_executor.get_browser_instance_manager') as mock_browser_mgr, \
             patch('braf.core.task_executor.get_captcha_solver') as mock_captcha, \
             patch('braf.core.task_executor.get_behavioral_engine') as mock_behavioral, \
             patch('braf.core.task_executor.get_database') as mock_db:
            
            # Setup mocks
            mock_browser_instance = MagicMock()
            mock_browser_instance.id = "test_browser_001"
            mock_browser_instance.fingerprint_id = "test_fingerprint_001"
            mock_browser_instance.page = AsyncMock()
            
            mock_browser_mgr.return_value.get_instance = AsyncMock(return_value=mock_browser_instance)
            mock_browser_mgr.return_value.release_instance = AsyncMock()
            
            mock_captcha.return_value = None  # No CAPTCHA solver
            
            mock_behavioral_engine = AsyncMock()
            mock_behavioral_engine.wait_with_human_delay = AsyncMock(return_value=0.1)
            mock_behavioral_engine.check_break_needed = AsyncMock(return_value=(False, 0))
            mock_behavioral.return_value = mock_behavioral_engine
            
            mock_db.return_value = None  # No database
            
            # Mock the automation executor
            with patch('braf.core.browser.automation_utils.AutomationExecutor') as mock_executor_class:
                mock_executor = AsyncMock()
                mock_executor.execute_action = AsyncMock(return_value={
                    "success": True,
                    "action_type": "navigate",
                    "url": "https://httpbin.org/html"
                })
                mock_executor_class.return_value = mock_executor
                
                # Execute the task
                result = await task_executor.execute_task(task)
                
                # Verify result
                assert result.success is True
                assert result.task_id == task.id
                assert result.actions_completed > 0
                assert result.execution_time > 0
                
                # Verify browser manager was called
                mock_browser_mgr.return_value.get_instance.assert_called_once()
                mock_browser_mgr.return_value.release_instance.assert_called_once()
    
    async def test_error_handling_integration(self, task_executor):
        """Test error handling in task execution."""
        # Create task with invalid action
        invalid_task = AutomationTask(
            id="invalid_task",
            profile_id="test_profile",
            actions=[
                AutomationAction(
                    type=ActionType.TYPE,
                    selector="",  # Empty selector should cause validation error
                    data="test",
                    timeout=10
                )
            ]
        )
        
        # Should handle validation error gracefully
        result = await task_executor.execute_task(invalid_task)
        assert result.success is False
        assert "validation" in result.error.lower()
        assert result.actions_completed == 0
    
    def test_component_initialization(self):
        """Test that all components can be initialized properly."""
        # Test task executor initialization
        executor = TaskExecutor()
        assert executor.validator is not None
        assert executor.preprocessor is not None
        assert len(executor.active_tasks) == 0
        
        # Test compliance logger initialization
        logger = ComplianceLogger()
        assert logger.constraint_checker is not None
        assert len(logger.events) == 0
        assert logger.lockdown_active is False
        
        # Test behavioral engine initialization
        engine = BehavioralEngine()
        assert engine.mouse_controller is not None
        assert engine.typing_controller is not None
        assert engine.delay_controller is not None
    
    async def test_metrics_and_stats_integration(self, compliance_logger, task_executor):
        """Test metrics collection across components."""
        # Get initial stats
        initial_stats = task_executor.get_execution_stats()
        assert initial_stats["total_executed"] == 0
        
        initial_metrics = compliance_logger.get_compliance_metrics()
        assert initial_metrics.total_tasks == 0
        assert initial_metrics.status.value == "compliant"
        
        # Simulate some activity
        with patch('braf.core.compliance_logger.get_database') as mock_db:
            mock_db.return_value = None
            
            # Log some events
            await compliance_logger.log_action_execution(
                "click", "test_profile", "test_task", {"success": True}
            )
            
            await compliance_logger.log_action_execution(
                "data_extraction", "test_profile", "test_task", {"items": 5}
            )
            
            # Check that events were recorded
            assert len(compliance_logger.events) >= 2
            
            # Check metrics update
            updated_metrics = compliance_logger.get_compliance_metrics()
            assert len(compliance_logger.get_recent_events()) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])