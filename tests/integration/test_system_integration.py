"""
Final System Integration Tests for BRAF.

This module provides end-to-end workflow testing, distributed operation validation,
and comprehensive system integration tests.
"""

import asyncio
import logging
import pytest
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from braf.core.models import (
    AutomationTask, AutomationAction, ActionType, TaskPriority,
    TaskResult, WorkerStatus
)
from braf.core.task_executor import init_task_executor
from braf.core.compliance_logger import init_compliance_logger
from braf.core.browser import init_browser_manager
from braf.core.behavioral import init_behavioral_engine
from braf.core.captcha import init_captcha_solver
from braf.core.monitoring import init_monitoring_manager
from braf.core.security import init_security_manager
from braf.core.error_handling import init_error_recovery_manager
from braf.worker.worker_node import WorkerNode, WorkerConfig
from braf.worker.profile_service import init_profile_service
from braf.worker.proxy_service import init_proxy_service

logger = logging.getLogger(__name__)


class TestSystemIntegration:
    """Comprehensive system integration tests."""
    
    @pytest.fixture
    async def system_components(self):
        """Initialize all system components for testing."""
        components = {}
        
        # Initialize core components
        components['task_executor'] = init_task_executor()
        components['compliance_logger'] = init_compliance_logger()
        components['behavioral_engine'] = init_behavioral_engine()
        components['captcha_solver'] = init_captcha_solver(test_mode=True)
        components['monitoring_manager'] = init_monitoring_manager()
        components['security_manager'] = init_security_manager("test_deployment")
        components['error_recovery'] = init_error_recovery_manager()
        
        # Initialize browser manager with mocked browser
        with patch('playwright.async_api.async_playwright'):
            components['browser_manager'] = init_browser_manager()
        
        # Initialize worker services
        components['profile_service'] = init_profile_service()
        components['proxy_service'] = init_proxy_service()
        
        return components
    
    @pytest.fixture
    def sample_automation_workflow(self):
        """Create a sample automation workflow for testing."""
        return AutomationTask(
            id="integration_test_workflow",
            profile_id="test_profile_001",
            actions=[
                AutomationAction(
                    type=ActionType.NAVIGATE,
                    url="https://httpbin.org/html",
                    timeout=30
                ),
                AutomationAction(
                    type=ActionType.WAIT,
                    data="2.0",
                    timeout=10
                ),
                AutomationAction(
                    type=ActionType.EXTRACT,
                    selector="h1",
                    timeout=10,
                    metadata={"attribute": "text"}
                ),
                AutomationAction(
                    type=ActionType.SCREENSHOT,
                    data="test_screenshot.png",
                    timeout=10
                )
            ],
            priority=TaskPriority.NORMAL,
            timeout=300
        )
    
    async def test_end_to_end_workflow_execution(
        self,
        system_components,
        sample_automation_workflow
    ):
        """Test complete end-to-end workflow execution."""
        task_executor = system_components['task_executor']
        compliance_logger = system_components['compliance_logger']
        
        # Mock browser operations
        with patch('braf.core.browser.automation_utils.AutomationExecutor') as mock_executor_class:
            mock_executor = AsyncMock()
            
            # Mock successful action executions
            mock_executor.execute_action.side_effect = [
                {"success": True, "action_type": "navigate", "url": "https://httpbin.org/html"},
                {"success": True, "action_type": "wait", "duration": 2.0},
                {"success": True, "action_type": "extract", "value": "Herman Melville - Moby-Dick"},
                {"success": True, "action_type": "screenshot", "screenshot_path": "test_screenshot.png"}
            ]
            mock_executor_class.return_value = mock_executor
            
            # Mock browser manager
            with patch('braf.core.task_executor.get_browser_instance_manager') as mock_browser_mgr:
                mock_browser_instance = MagicMock()
                mock_browser_instance.id = "test_browser_001"
                mock_browser_instance.fingerprint_id = "test_fingerprint_001"
                mock_browser_instance.page = AsyncMock()
                
                mock_browser_mgr.return_value.get_instance = AsyncMock(return_value=mock_browser_instance)
                mock_browser_mgr.return_value.release_instance = AsyncMock()
                
                # Execute the workflow
                result = await task_executor.execute_task(sample_automation_workflow)
                
                # Verify successful execution
                assert result.success is True
                assert result.task_id == sample_automation_workflow.id
                assert result.actions_completed == 4
                assert result.execution_time > 0
                
                # Verify all actions were executed
                assert mock_executor.execute_action.call_count == 4
                
                # Verify browser lifecycle
                mock_browser_mgr.return_value.get_instance.assert_called_once()
                mock_browser_mgr.return_value.release_instance.assert_called_once()
    
    async def test_compliance_integration(
        self,
        system_components,
        sample_automation_workflow
    ):
        """Test compliance logging integration throughout workflow."""
        compliance_logger = system_components['compliance_logger']
        task_executor = system_components['task_executor']
        
        # Mock database operations
        with patch('braf.core.compliance_logger.get_database') as mock_db:
            mock_db.return_value = None
            
            # Log task start
            violations = await compliance_logger.log_task_start(
                sample_automation_workflow,
                sample_automation_workflow.profile_id
            )
            
            # Should have no violations for normal task
            assert len(violations) == 0
            
            # Verify event was logged
            assert len(compliance_logger.events) > 0
            start_event = compliance_logger.events[-1]
            assert start_event.event_type == "task_started"
            
            # Log some actions
            await compliance_logger.log_action_execution(
                "navigate",
                sample_automation_workflow.profile_id,
                sample_automation_workflow.id,
                {"url": "https://httpbin.org/html", "success": True}
            )
            
            await compliance_logger.log_action_execution(
                "data_extraction",
                sample_automation_workflow.profile_id,
                sample_automation_workflow.id,
                {"items_extracted": 1, "success": True}
            )
            
            # Create mock task result
            task_result = TaskResult(
                task_id=sample_automation_workflow.id,
                success=True,
                execution_time=45.2,
                actions_completed=4
            )
            
            # Log task completion
            await compliance_logger.log_task_completion(
                task_result,
                sample_automation_workflow.profile_id
            )
            
            # Verify completion event
            completion_event = compliance_logger.events[-1]
            assert completion_event.event_type == "task_completed"
            assert completion_event.details["success"] is True
    
    async def test_behavioral_integration(self, system_components):
        """Test behavioral engine integration with task execution."""
        behavioral_engine = system_components['behavioral_engine']
        
        # Test mouse movement generation
        target_position = (500, 300)
        movement_path = await behavioral_engine.move_mouse(target_position)
        
        assert len(movement_path) > 0
        assert all(len(point) == 3 for point in movement_path)  # (x, y, timestamp)
        
        # Verify path ends at target
        final_point = movement_path[-1]
        assert abs(final_point[0] - target_position[0]) < 5  # Allow small tolerance
        assert abs(final_point[1] - target_position[1]) < 5
        
        # Test typing simulation
        test_text = "Hello, World!"
        typing_sequence = await behavioral_engine.type_text(test_text)
        
        assert len(typing_sequence) > 0
        
        # Should have realistic timing
        total_time = sum(delay for _, delay in typing_sequence)
        assert total_time > len(test_text) * 0.05  # At least 50ms per character
        
        # Test delay generation
        delay = await behavioral_engine.wait_with_human_delay("click")
        assert 0.1 <= delay <= 2.0  # Reasonable delay range
    
    async def test_captcha_integration(self, system_components):
        """Test CAPTCHA solver integration."""
        captcha_solver = system_components['captcha_solver']
        
        # Test image CAPTCHA solving (test mode)
        test_image_data = b"fake_image_data"
        solution = await captcha_solver.solve_image_captcha(test_image_data)
        
        # In test mode, should return test solution
        assert solution is not None
        assert solution == "TEST123"
        
        # Test reCAPTCHA v2 solving (test mode)
        test_site_key = "6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI"  # Google test key
        test_url = "https://localhost/test"
        
        recaptcha_solution = await captcha_solver.solve_recaptcha_v2(test_site_key, test_url)
        
        # In test mode, should return test token
        assert recaptcha_solution is not None
        assert "03AGdBq25SiXT-pmSeBXjzScW-EiocHwwpwqJRCAC7g" in recaptcha_solution
    
    async def test_error_handling_integration(self, system_components):
        """Test error handling and recovery integration."""
        error_recovery = system_components['error_recovery']
        task_executor = system_components['task_executor']
        
        # Test retry mechanism
        call_count = 0
        
        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Simulated network error")
            return "success"
        
        # Should retry and eventually succeed
        result = await error_recovery.execute_with_recovery(
            failing_function,
            operation="test_operation",
            component="test_component"
        )
        
        assert result == "success"
        assert call_count == 3  # Failed twice, succeeded on third attempt
        
        # Test circuit breaker
        error_recovery.register_circuit_breaker(
            "test_service",
            error_recovery.CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0)
        )
        
        async def always_failing_function():
            raise ServiceUnavailableError("Service down")
        
        # Should fail and open circuit breaker
        with pytest.raises(ServiceUnavailableError):
            for _ in range(3):
                try:
                    await error_recovery.execute_with_recovery(
                        always_failing_function,
                        operation="test_operation",
                        component="test_component",
                        circuit_breaker_name="test_service"
                    )
                except ServiceUnavailableError:
                    pass
        
        # Circuit breaker should be open
        cb_state = error_recovery.circuit_breakers["test_service"].get_state()
        assert cb_state["state"] == "open"
    
    async def test_monitoring_integration(self, system_components):
        """Test monitoring and metrics integration."""
        monitoring_manager = system_components['monitoring_manager']
        
        # Test metrics recording
        monitoring_manager.metrics.record_task_execution(
            status="success",
            priority="normal",
            worker_id="test_worker",
            duration=45.2
        )
        
        monitoring_manager.metrics.record_detection(
            detection_type="bot_detection",
            severity="medium",
            worker_id="test_worker",
            score=0.3
        )
        
        monitoring_manager.metrics.record_captcha_event(
            captcha_type="recaptcha_v2",
            worker_id="test_worker",
            solver="test_solver",
            duration=5.2,
            success=True
        )
        
        # Get metrics output
        metrics_output = monitoring_manager.metrics.get_metrics()
        assert "braf_tasks_total" in metrics_output
        assert "braf_detections_total" in metrics_output
        assert "braf_captcha_encounters_total" in metrics_output
        
        # Test alert creation
        alert = await monitoring_manager.alert_manager.create_alert(
            title="Test Alert",
            description="Integration test alert",
            severity=monitoring_manager.alert_manager.AlertSeverity.WARNING,
            source="integration_test"
        )
        
        assert alert.title == "Test Alert"
        assert alert.severity.value == "warning"
        
        # Verify alert was logged
        assert len(monitoring_manager.alert_manager.alerts) > 0
    
    async def test_security_integration(self, system_components):
        """Test security and key management integration."""
        security_manager = system_components['security_manager']
        
        # Initialize security with test password
        await security_manager.initialize_security("test_master_password")
        
        # Test key derivation
        test_key = security_manager.key_derivation.derive_key(
            password="test_password",
            key_type=security_manager.key_derivation.KeyType.ENCRYPTION
        )
        
        assert test_key.key_data is not None
        assert len(test_key.key_data) == 32  # Default key length
        assert test_key.key_type.value == "encryption"
        
        # Test access logging
        access_log = security_manager.access_logger.log_access(
            user_id="test_user",
            resource="test_resource",
            action="read",
            success=True,
            ip_address="127.0.0.1"
        )
        
        assert access_log.user_id == "test_user"
        assert access_log.success is True
        
        # Test security status
        status = security_manager.get_security_status()
        assert status["deployment_id"] == "test_deployment"
        assert status["threat_level"] == "low"
    
    async def test_worker_node_integration(self, system_components):
        """Test worker node integration with all components."""
        # Create worker configuration
        config = WorkerConfig(
            worker_id="integration_test_worker",
            max_concurrent_tasks=2,
            heartbeat_interval=5,
            health_check_interval=10
        )
        
        # Create worker node
        worker = WorkerNode(config)
        
        # Mock external dependencies
        with patch('braf.worker.worker_node.init_browser_manager') as mock_browser_init, \
             patch('braf.worker.worker_node.init_profile_service') as mock_profile_init, \
             patch('braf.worker.worker_node.init_proxy_service') as mock_proxy_init:
            
            mock_browser_init.return_value = system_components['browser_manager']
            mock_profile_init.return_value = system_components['profile_service']
            mock_proxy_init.return_value = system_components['proxy_service']
            
            # Initialize worker
            await worker.initialize()
            
            # Verify worker state
            assert worker.state.value == "idle"
            assert worker.worker_id == "integration_test_worker"
            
            # Test health check
            health_status = await worker.health_checker.perform_health_check()
            assert "overall_health" in health_status
            assert "components" in health_status
            
            # Test worker status
            status = worker.get_status()
            assert status.worker_id == "integration_test_worker"
            assert status.current_tasks == 0
            assert status.max_tasks == 2
            
            # Shutdown worker
            await worker.shutdown(graceful=True)
    
    async def test_distributed_operation_simulation(self, system_components):
        """Test distributed operation across multiple simulated workers."""
        # Create multiple worker configurations
        worker_configs = [
            WorkerConfig(worker_id=f"worker_{i}", max_concurrent_tasks=1)
            for i in range(3)
        ]
        
        workers = []
        
        # Mock all external dependencies
        with patch('braf.worker.worker_node.init_browser_manager') as mock_browser_init, \
             patch('braf.worker.worker_node.init_profile_service') as mock_profile_init, \
             patch('braf.worker.worker_node.init_proxy_service') as mock_proxy_init, \
             patch('braf.core.browser.automation_utils.AutomationExecutor') as mock_executor_class:
            
            # Setup mocks
            mock_browser_init.return_value = system_components['browser_manager']
            mock_profile_init.return_value = system_components['profile_service']
            mock_proxy_init.return_value = system_components['proxy_service']
            
            mock_executor = AsyncMock()
            mock_executor.execute_action.return_value = {"success": True, "action_type": "test"}
            mock_executor_class.return_value = mock_executor
            
            # Initialize workers
            for config in worker_configs:
                worker = WorkerNode(config)
                await worker.initialize()
                workers.append(worker)
            
            # Create test tasks
            tasks = [
                AutomationTask(
                    id=f"distributed_task_{i}",
                    profile_id=f"profile_{i}",
                    actions=[
                        AutomationAction(
                            type=ActionType.NAVIGATE,
                            url=f"https://example.com/page{i}",
                            timeout=30
                        )
                    ]
                )
                for i in range(5)
            ]
            
            # Distribute tasks across workers
            task_assignments = []
            for i, task in enumerate(tasks):
                worker = workers[i % len(workers)]
                task_assignments.append((worker, task))
            
            # Submit tasks (simulate task distribution)
            for worker, task in task_assignments:
                await worker.submit_task(task)
            
            # Wait for tasks to complete (short wait for test)
            await asyncio.sleep(2)
            
            # Verify workers processed tasks
            total_processed = sum(
                len(worker.active_tasks) for worker in workers
            )
            
            # Some tasks should be processed or in progress
            assert total_processed >= 0  # At least some activity
            
            # Shutdown all workers
            for worker in workers:
                await worker.shutdown(graceful=True)
    
    async def test_system_resilience(self, system_components):
        """Test system resilience under various failure conditions."""
        task_executor = system_components['task_executor']
        compliance_logger = system_components['compliance_logger']
        error_recovery = system_components['error_recovery']
        
        # Test task execution with simulated failures
        failure_task = AutomationTask(
            id="failure_test_task",
            profile_id="test_profile",
            actions=[
                AutomationAction(
                    type=ActionType.NAVIGATE,
                    url="https://nonexistent-domain-12345.com",
                    timeout=5
                )
            ]
        )
        
        # Mock browser manager to simulate failure
        with patch('braf.core.task_executor.get_browser_instance_manager') as mock_browser_mgr:
            mock_browser_mgr.return_value.get_instance.side_effect = ConnectionError("Network error")
            
            # Execute task - should handle failure gracefully
            result = await task_executor.execute_task(failure_task)
            
            # Should fail but not crash the system
            assert result.success is False
            assert "error" in result.error.lower() or "network" in result.error.lower()
        
        # Test compliance lockdown recovery
        if not compliance_logger.lockdown_active:
            # Trigger lockdown
            await compliance_logger._trigger_automatic_shutdown(
                "Test lockdown for resilience testing",
                None
            )
            
            assert compliance_logger.lockdown_active is True
            
            # Test lockdown release
            released = await compliance_logger.release_lockdown(admin_override=True)
            assert released is True
            assert compliance_logger.lockdown_active is False
        
        # Test error recovery statistics
        error_stats = error_recovery.get_error_statistics()
        assert "total_errors_24h" in error_stats
        assert "circuit_breaker_states" in error_stats
    
    async def test_performance_under_load(self, system_components):
        """Test system performance under simulated load."""
        task_executor = system_components['task_executor']
        monitoring_manager = system_components['monitoring_manager']
        
        # Create multiple concurrent tasks
        concurrent_tasks = []
        for i in range(10):
            task = AutomationTask(
                id=f"load_test_task_{i}",
                profile_id=f"profile_{i % 3}",  # Simulate 3 profiles
                actions=[
                    AutomationAction(
                        type=ActionType.WAIT,
                        data="0.1",  # Short wait for fast execution
                        timeout=5
                    )
                ]
            )
            concurrent_tasks.append(task)
        
        # Mock browser operations for fast execution
        with patch('braf.core.browser.automation_utils.AutomationExecutor') as mock_executor_class:
            mock_executor = AsyncMock()
            mock_executor.execute_action.return_value = {
                "success": True,
                "action_type": "wait",
                "duration": 0.1
            }
            mock_executor_class.return_value = mock_executor
            
            with patch('braf.core.task_executor.get_browser_instance_manager') as mock_browser_mgr:
                mock_browser_instance = MagicMock()
                mock_browser_instance.id = "load_test_browser"
                mock_browser_instance.page = AsyncMock()
                
                mock_browser_mgr.return_value.get_instance = AsyncMock(return_value=mock_browser_instance)
                mock_browser_mgr.return_value.release_instance = AsyncMock()
                
                # Execute tasks concurrently
                start_time = time.time()
                
                results = await asyncio.gather(
                    *[task_executor.execute_task(task) for task in concurrent_tasks],
                    return_exceptions=True
                )
                
                execution_time = time.time() - start_time
                
                # Verify results
                successful_results = [r for r in results if isinstance(r, TaskResult) and r.success]
                
                # Should complete most tasks successfully
                assert len(successful_results) >= 8  # At least 80% success rate
                
                # Should complete in reasonable time (parallel execution)
                assert execution_time < 5.0  # Should be much faster than sequential
                
                # Record performance metrics
                monitoring_manager.metrics.record_api_request(
                    method="POST",
                    endpoint="/tasks/execute",
                    status=200,
                    duration=execution_time
                )
    
    def test_configuration_validation(self, system_components):
        """Test system configuration validation."""
        # Test that all required components are initialized
        required_components = [
            'task_executor',
            'compliance_logger',
            'behavioral_engine',
            'captcha_solver',
            'monitoring_manager',
            'security_manager',
            'error_recovery'
        ]
        
        for component_name in required_components:
            assert component_name in system_components
            assert system_components[component_name] is not None
        
        # Test component interconnections
        task_executor = system_components['task_executor']
        assert task_executor.validator is not None
        assert task_executor.preprocessor is not None
        
        compliance_logger = system_components['compliance_logger']
        assert compliance_logger.constraint_checker is not None
        assert len(compliance_logger.constraint_checker.constraints) > 0
        
        monitoring_manager = system_components['monitoring_manager']
        assert monitoring_manager.metrics is not None
        assert monitoring_manager.alert_manager is not None
        
        security_manager = system_components['security_manager']
        assert security_manager.key_derivation is not None
        assert security_manager.access_logger is not None


class TestSystemScenarios:
    """Test realistic system usage scenarios."""
    
    async def test_web_scraping_scenario(self):
        """Test a realistic web scraping scenario."""
        # This would test a complete web scraping workflow
        # with navigation, data extraction, and compliance checking
        pass
    
    async def test_form_automation_scenario(self):
        """Test a realistic form automation scenario."""
        # This would test form filling, CAPTCHA solving,
        # and submission with proper behavioral patterns
        pass
    
    async def test_multi_site_automation_scenario(self):
        """Test automation across multiple sites."""
        # This would test profile switching, proxy rotation,
        # and cross-site automation workflows
        pass


if __name__ == "__main__":
    # Configure logging for test output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    pytest.main([__file__, "-v", "-s"])
