"""
Unit tests for TaskExecutor component.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from braf.core.models import (
    AutomationTask, AutomationAction, ActionType, TaskPriority
)
from braf.core.task_executor import (
    TaskExecutor, TaskValidator, TaskPreprocessor, TaskValidationError
)


class TestTaskValidator:
    """Test TaskValidator functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create validator for testing."""
        return TaskValidator()
    
    @pytest.fixture
    def valid_task(self):
        """Create valid task for testing."""
        return AutomationTask(
            id="test_task",
            profile_id="test_profile",
            actions=[
                AutomationAction(
                    type=ActionType.NAVIGATE,
                    url="https://example.com",
                    timeout=30
                ),
                AutomationAction(
                    type=ActionType.CLICK,
                    selector="button",
                    timeout=10
                )
            ],
            priority=TaskPriority.NORMAL,
            timeout=300
        )
    
    async def test_validate_valid_task(self, validator, valid_task):
        """Test validation of valid task."""
        result = await validator.validate_task(valid_task)
        
        assert result["valid"] is True
        assert result["action_count"] == 2
        assert result["estimated_duration"] > 0
        assert isinstance(result["domains"], list)
    
    async def test_validate_empty_task(self, validator):
        """Test validation of task with no actions."""
        # The Pydantic model itself prevents empty tasks, so we test that
        with pytest.raises(Exception) as exc_info:
            empty_task = AutomationTask(
                id="empty_task",
                profile_id="test_profile",
                actions=[],
                priority=TaskPriority.NORMAL
            )
        
        assert "at least one action" in str(exc_info.value).lower()
    
    async def test_validate_task_too_many_actions(self, validator):
        """Test validation of task with too many actions."""
        large_task = AutomationTask(
            id="large_task",
            profile_id="test_profile",
            actions=[
                AutomationAction(type=ActionType.CLICK, selector="button")
                for _ in range(150)  # Exceeds default limit
            ]
        )
        
        with pytest.raises(TaskValidationError) as exc_info:
            await validator.validate_task(large_task)
        
        assert "too many actions" in str(exc_info.value).lower()
    
    async def test_validate_action_missing_required_fields(self, validator):
        """Test validation of actions with missing required fields."""
        # Navigate action without URL
        task_no_url = AutomationTask(
            id="no_url_task",
            profile_id="test_profile",
            actions=[
                AutomationAction(
                    type=ActionType.NAVIGATE,
                    # Missing URL
                    timeout=30
                )
            ]
        )
        
        with pytest.raises(TaskValidationError) as exc_info:
            await validator.validate_task(task_no_url)
        
        assert "requires URL" in str(exc_info.value)
        
        # Type action without data
        task_no_data = AutomationTask(
            id="no_data_task",
            profile_id="test_profile",
            actions=[
                AutomationAction(
                    type=ActionType.TYPE,
                    selector="input",
                    # Missing data
                    timeout=10
                )
            ]
        )
        
        with pytest.raises(TaskValidationError) as exc_info:
            await validator.validate_task(task_no_data)
        
        assert "requires data" in str(exc_info.value)
    
    async def test_validate_blocked_domain(self, validator):
        """Test validation of blocked domains."""
        blocked_task = AutomationTask(
            id="blocked_task",
            profile_id="test_profile",
            actions=[
                AutomationAction(
                    type=ActionType.NAVIGATE,
                    url="https://facebook.com/login",  # Blocked domain
                    timeout=30
                )
            ]
        )
        
        with pytest.raises(TaskValidationError) as exc_info:
            await validator.validate_task(blocked_task)
        
        assert "blocked" in str(exc_info.value).lower()
    
    def test_estimate_task_duration(self, validator, valid_task):
        """Test task duration estimation."""
        duration = validator._estimate_task_duration(valid_task)
        
        assert duration > 0
        assert duration < 3600  # Should be reasonable
        
        # Test with typing action
        typing_task = AutomationTask(
            id="typing_task",
            profile_id="test_profile",
            actions=[
                AutomationAction(
                    type=ActionType.TYPE,
                    selector="input",
                    data="This is a long text to type",
                    timeout=10
                )
            ]
        )
        
        typing_duration = validator._estimate_task_duration(typing_task)
        assert typing_duration > 2  # Should account for typing time


class TestTaskPreprocessor:
    """Test TaskPreprocessor functionality."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor for testing."""
        return TaskPreprocessor()
    
    @pytest.fixture
    def sample_task(self):
        """Create sample task for testing."""
        return AutomationTask(
            id="sample_task",
            profile_id="test_profile",
            actions=[
                AutomationAction(
                    type=ActionType.NAVIGATE,
                    url="https://example.com",
                    timeout=30
                ),
                AutomationAction(
                    type=ActionType.CLICK,
                    selector="button#submit",
                    timeout=10,
                    metadata={"submit_form": True}
                )
            ],
            priority=TaskPriority.NORMAL,
            timeout=300
        )
    
    async def test_preprocess_task_basic(self, preprocessor, sample_task):
        """Test basic task preprocessing."""
        processed = await preprocessor.preprocess_task(sample_task)
        
        # Should have metadata added
        assert "preprocessed_at" in processed.metadata
        assert "original_action_count" in processed.metadata
        assert processed.metadata["original_action_count"] == 2
        
        # Should have more actions due to implicit waits and CAPTCHA checks
        assert len(processed.actions) > len(sample_task.actions)
    
    async def test_add_implicit_waits(self, preprocessor, sample_task):
        """Test implicit wait addition."""
        enhanced = await preprocessor._add_implicit_waits(sample_task.actions)
        
        # Should have more actions
        assert len(enhanced) > len(sample_task.actions)
        
        # Should have wait after navigation
        nav_indices = [
            i for i, action in enumerate(enhanced)
            if action.type == ActionType.NAVIGATE
        ]
        
        for nav_idx in nav_indices:
            if nav_idx + 1 < len(enhanced):
                next_action = enhanced[nav_idx + 1]
                if next_action.type == ActionType.WAIT:
                    assert next_action.metadata.get("implicit") is True
                    assert next_action.metadata.get("reason") == "post_navigation"
    
    async def test_optimize_actions(self, preprocessor):
        """Test action optimization."""
        # Create actions with consecutive type actions
        actions = [
            AutomationAction(
                type=ActionType.TYPE,
                selector="input#username",
                data="user",
                timeout=10
            ),
            AutomationAction(
                type=ActionType.TYPE,
                selector="input#username",  # Same selector
                data="name",
                timeout=10
            ),
            AutomationAction(
                type=ActionType.WAIT,
                data="1.0",
                timeout=10
            ),
            AutomationAction(
                type=ActionType.WAIT,  # Consecutive wait
                data="2.0",
                timeout=10
            )
        ]
        
        optimized = await preprocessor._optimize_actions(actions)
        
        # Should have fewer actions due to optimization
        assert len(optimized) < len(actions)
        
        # First action should have merged data
        first_type_action = next(a for a in optimized if a.type == ActionType.TYPE)
        assert first_type_action.data == "username"  # Merged data
    
    async def test_inject_captcha_handling(self, preprocessor, sample_task):
        """Test CAPTCHA handling injection."""
        enhanced = await preprocessor._inject_captcha_handling(sample_task.actions)
        
        # Should have CAPTCHA check actions
        captcha_checks = [
            a for a in enhanced 
            if a.metadata and a.metadata.get("captcha_check")
        ]
        
        assert len(captcha_checks) > 0
    
    async def test_add_error_recovery(self, preprocessor, sample_task):
        """Test error recovery addition."""
        enhanced = await preprocessor._add_error_recovery(sample_task.actions)
        
        # Critical actions should have retry metadata
        for action in enhanced:
            if action.type in [ActionType.NAVIGATE, ActionType.CLICK]:
                assert action.metadata is not None
                assert "max_retries" in action.metadata
                assert action.metadata["max_retries"] > 0


class TestTaskExecutor:
    """Test TaskExecutor functionality."""
    
    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        return TaskExecutor()
    
    @pytest.fixture
    def simple_task(self):
        """Create simple task for testing."""
        return AutomationTask(
            id="simple_task",
            profile_id="test_profile",
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
    
    def test_executor_initialization(self, executor):
        """Test executor initialization."""
        assert executor.validator is not None
        assert executor.preprocessor is not None
        assert len(executor.active_tasks) == 0
        assert executor.execution_stats["total_executed"] == 0
    
    async def test_execute_task_validation_failure(self, executor):
        """Test task execution with validation failure."""
        invalid_task = AutomationTask(
            id="invalid_task",
            profile_id="test_profile",
            actions=[],  # No actions - should fail validation
            priority=TaskPriority.NORMAL
        )
        
        result = await executor.execute_task(invalid_task)
        
        assert result.success is False
        assert "validation" in result.error.lower()
        assert result.actions_completed == 0
        assert executor.execution_stats["failed"] == 1
    
    async def test_cancel_task(self, executor):
        """Test task cancellation."""
        # Add a mock active task
        execution_id = "test_execution_001"
        executor.active_tasks[execution_id] = {
            "task": MagicMock(),
            "start_time": 12345,
            "status": "running",
            "profile_id": "test_profile"
        }
        
        # Cancel the task
        result = executor.cancel_task(execution_id)
        assert result is True
        
        # Check that task is marked as cancelled
        from braf.core.models import TaskStatus
        assert executor.active_tasks[execution_id]["status"] == TaskStatus.CANCELLED
        
        # Try to cancel non-existent task
        result = executor.cancel_task("non_existent")
        assert result is False
    
    def test_get_active_tasks(self, executor):
        """Test getting active tasks."""
        # Initially empty
        active = executor.get_active_tasks()
        assert len(active) == 0
        
        # Add a task
        executor.active_tasks["test_001"] = {
            "task": MagicMock(),
            "start_time": 12345,
            "status": "running"
        }
        
        active = executor.get_active_tasks()
        assert len(active) == 1
        assert "test_001" in active
    
    def test_get_execution_stats(self, executor):
        """Test getting execution statistics."""
        stats = executor.get_execution_stats()
        
        assert "total_executed" in stats
        assert "successful" in stats
        assert "failed" in stats
        assert "cancelled" in stats
        
        # All should be 0 initially
        assert all(count == 0 for count in stats.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
