"""
Task Executor Engine for BRAF.

This module provides the main task execution engine that orchestrates
automation workflows with behavioral integration, compliance checking,
and result reporting.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from braf.core.models import (
    AutomationTask, AutomationAction, TaskStatus, TaskResult,
    ComplianceViolation, ActionType
)
from braf.core.browser import get_browser_instance_manager
from braf.core.browser.automation_utils import execute_automation_sequence
from braf.core.behavioral import get_behavioral_engine
from braf.core.captcha import get_captcha_solver
from braf.core.database import get_database

logger = logging.getLogger(__name__)


class TaskValidationError(Exception):
    """Raised when task validation fails."""
    pass


class TaskExecutionError(Exception):
    """Raised when task execution fails."""
    pass


class TaskValidator:
    """Validates automation tasks before execution."""
    
    def __init__(self):
        """Initialize task validator."""
        self.max_actions_per_task = 100
        self.max_task_duration = 3600  # 1 hour
        self.allowed_domains = set()  # Empty means all domains allowed
        self.blocked_domains = {
            "facebook.com", "twitter.com", "instagram.com",
            "linkedin.com", "tiktok.com", "snapchat.com"
        }
    
    async def validate_task(self, task: AutomationTask) -> Dict[str, Any]:
        """
        Validate automation task.
        
        Args:
            task: Task to validate
            
        Returns:
            Validation result with issues and warnings
            
        Raises:
            TaskValidationError: If task is invalid
        """
        issues = []
        warnings = []
        
        # Basic task validation
        if not task.actions:
            issues.append("Task has no actions")
        
        if len(task.actions) > self.max_actions_per_task:
            issues.append(f"Task has too many actions ({len(task.actions)} > {self.max_actions_per_task})")
        
        if task.timeout and task.timeout > self.max_task_duration:
            issues.append(f"Task timeout too long ({task.timeout} > {self.max_task_duration})")
        
        # Validate actions
        for i, action in enumerate(task.actions):
            action_issues = await self._validate_action(action, i)
            issues.extend(action_issues)
        
        # Domain validation
        urls = self._extract_urls_from_task(task)
        for url in urls:
            domain_issues = self._validate_domain(url)
            issues.extend(domain_issues)
        
        # Ethical constraints
        ethical_issues = await self._check_ethical_constraints(task)
        issues.extend(ethical_issues)
        
        # Performance warnings
        if len(task.actions) > 50:
            warnings.append("Large number of actions may impact performance")
        
        if task.timeout and task.timeout > 1800:  # 30 minutes
            warnings.append("Long timeout may cause resource issues")
        
        # Raise exception if critical issues found
        if issues:
            raise TaskValidationError(f"Task validation failed: {'; '.join(issues)}")
        
        return {
            "valid": True,
            "warnings": warnings,
            "action_count": len(task.actions),
            "estimated_duration": self._estimate_task_duration(task),
            "domains": list(set(self._extract_domains_from_urls(urls)))
        }
    
    async def _validate_action(self, action: AutomationAction, index: int) -> List[str]:
        """Validate individual action."""
        issues = []
        
        # Required fields
        if not action.type:
            issues.append(f"Action {index}: Missing action type")
        
        # Type-specific validation
        if action.type == ActionType.NAVIGATE:
            if not action.url:
                issues.append(f"Action {index}: Navigate action requires URL")
        
        elif action.type == ActionType.TYPE:
            if not action.data:
                issues.append(f"Action {index}: Type action requires data")
            if not action.selector:
                issues.append(f"Action {index}: Type action requires selector")
        
        elif action.type in [ActionType.CLICK, ActionType.HOVER, ActionType.EXTRACT]:
            if not action.selector:
                issues.append(f"Action {index}: {action.type.value} action requires selector")
        
        elif action.type == ActionType.UPLOAD:
            if not action.data or not action.selector:
                issues.append(f"Action {index}: Upload action requires file path and selector")
        
        # Timeout validation
        if action.timeout and action.timeout > 300:  # 5 minutes
            issues.append(f"Action {index}: Timeout too long ({action.timeout}s)")
        
        return issues
    
    def _extract_urls_from_task(self, task: AutomationTask) -> List[str]:
        """Extract all URLs from task actions."""
        urls = []
        
        for action in task.actions:
            if action.url:
                urls.append(action.url)
        
        return urls
    
    def _extract_domains_from_urls(self, urls: List[str]) -> List[str]:
        """Extract domains from URLs."""
        domains = []
        
        for url in urls:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                if parsed.netloc:
                    domains.append(parsed.netloc.lower())
            except Exception:
                continue
        
        return domains
    
    def _validate_domain(self, url: str) -> List[str]:
        """Validate domain restrictions."""
        issues = []
        
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check blocked domains
            for blocked in self.blocked_domains:
                if blocked in domain:
                    issues.append(f"Domain blocked: {domain}")
                    break
            
            # Check allowed domains (if configured)
            if self.allowed_domains:
                allowed = any(allowed_domain in domain for allowed_domain in self.allowed_domains)
                if not allowed:
                    issues.append(f"Domain not in allowed list: {domain}")
        
        except Exception as e:
            issues.append(f"Invalid URL format: {url}")
        
        return issues
    
    async def _check_ethical_constraints(self, task: AutomationTask) -> List[str]:
        """Check ethical constraints."""
        issues = []
        
        # Check for potential scraping violations
        extract_actions = [a for a in task.actions if a.type == ActionType.EXTRACT]
        if len(extract_actions) > 20:
            issues.append("Excessive data extraction may violate terms of service")
        
        # Check for rapid-fire actions
        click_actions = [a for a in task.actions if a.type == ActionType.CLICK]
        if len(click_actions) > 10:
            issues.append("Excessive clicking may appear bot-like")
        
        # Check for form spam potential
        type_actions = [a for a in task.actions if a.type == ActionType.TYPE]
        if len(type_actions) > 15:
            issues.append("Excessive form filling may be considered spam")
        
        return issues
    
    def _estimate_task_duration(self, task: AutomationTask) -> float:
        """Estimate task execution duration in seconds."""
        base_duration = 0.0
        
        for action in task.actions:
            if action.type == ActionType.NAVIGATE:
                base_duration += 3.0  # Page load time
            elif action.type == ActionType.CLICK:
                base_duration += 1.0  # Click + movement
            elif action.type == ActionType.TYPE:
                text_length = len(action.data) if action.data else 10
                base_duration += text_length * 0.1  # Typing speed
            elif action.type == ActionType.WAIT:
                wait_time = float(action.data) if action.data else 1.0
                base_duration += wait_time
            else:
                base_duration += 0.5  # Other actions
        
        # Add behavioral overhead (20-50% more time)
        behavioral_overhead = base_duration * 0.35
        
        return base_duration + behavioral_overhead


class TaskPreprocessor:
    """Preprocesses tasks before execution."""
    
    def __init__(self):
        """Initialize task preprocessor."""
        pass
    
    async def preprocess_task(self, task: AutomationTask) -> AutomationTask:
        """
        Preprocess task for execution.
        
        Args:
            task: Original task
            
        Returns:
            Preprocessed task
        """
        # Create a copy to avoid modifying original
        processed_task = AutomationTask(
            id=task.id,
            profile_id=task.profile_id,
            actions=task.actions.copy(),
            priority=task.priority,
            timeout=task.timeout,
            metadata=task.metadata.copy() if task.metadata else {}
        )
        
        # Add preprocessing metadata
        processed_task.metadata["preprocessed_at"] = datetime.now(timezone.utc).isoformat()
        processed_task.metadata["original_action_count"] = len(task.actions)
        
        # Optimize action sequence
        processed_task.actions = await self._optimize_actions(processed_task.actions)
        
        # Add implicit waits
        processed_task.actions = await self._add_implicit_waits(processed_task.actions)
        
        # Inject CAPTCHA handling
        processed_task.actions = await self._inject_captcha_handling(processed_task.actions)
        
        # Add error recovery
        processed_task.actions = await self._add_error_recovery(processed_task.actions)
        
        processed_task.metadata["processed_action_count"] = len(processed_task.actions)
        
        return processed_task
    
    async def _optimize_actions(self, actions: List[AutomationAction]) -> List[AutomationAction]:
        """Optimize action sequence."""
        optimized = []
        
        for i, action in enumerate(actions):
            # Remove redundant waits
            if action.type == ActionType.WAIT:
                # Skip if next action is also a wait
                if i + 1 < len(actions) and actions[i + 1].type == ActionType.WAIT:
                    continue
            
            # Merge consecutive type actions on same element
            if (action.type == ActionType.TYPE and 
                i + 1 < len(actions) and 
                actions[i + 1].type == ActionType.TYPE and
                action.selector == actions[i + 1].selector):
                
                # Merge the text data
                merged_data = (action.data or "") + (actions[i + 1].data or "")
                action.data = merged_data
                # Skip the next action
                actions[i + 1] = None
            
            if action is not None:
                optimized.append(action)
        
        # Filter out None values
        return [a for a in optimized if a is not None]
    
    async def _add_implicit_waits(self, actions: List[AutomationAction]) -> List[AutomationAction]:
        """Add implicit waits between actions."""
        enhanced = []
        
        for i, action in enumerate(actions):
            enhanced.append(action)
            
            # Add wait after navigation
            if action.type == ActionType.NAVIGATE:
                wait_action = AutomationAction(
                    type=ActionType.WAIT,
                    data="2.0",  # 2 second wait
                    metadata={"implicit": True, "reason": "post_navigation"}
                )
                enhanced.append(wait_action)
            
            # Add wait after form submission
            elif (action.type == ActionType.CLICK and 
                  action.metadata and 
                  action.metadata.get("submit_form")):
                wait_action = AutomationAction(
                    type=ActionType.WAIT,
                    data="3.0",  # 3 second wait
                    metadata={"implicit": True, "reason": "post_submit"}
                )
                enhanced.append(wait_action)
        
        return enhanced
    
    async def _inject_captcha_handling(self, actions: List[AutomationAction]) -> List[AutomationAction]:
        """Inject CAPTCHA detection and handling."""
        enhanced = []
        
        for action in actions:
            enhanced.append(action)
            
            # Add CAPTCHA check after navigation or form interactions
            if action.type in [ActionType.NAVIGATE, ActionType.CLICK]:
                captcha_check = AutomationAction(
                    type=ActionType.EXTRACT,  # Will be handled specially
                    selector="body",
                    metadata={
                        "captcha_check": True,
                        "implicit": True
                    }
                )
                enhanced.append(captcha_check)
        
        return enhanced
    
    async def _add_error_recovery(self, actions: List[AutomationAction]) -> List[AutomationAction]:
        """Add error recovery mechanisms."""
        for action in actions:
            # Add retry metadata for critical actions
            if action.type in [ActionType.NAVIGATE, ActionType.CLICK]:
                action.metadata = action.metadata or {}
                action.metadata["max_retries"] = 3
                action.metadata["retry_delay"] = 2.0
        
        return actions


class TaskExecutor:
    """Main task execution engine."""
    
    def __init__(self):
        """Initialize task executor."""
        self.validator = TaskValidator()
        self.preprocessor = TaskPreprocessor()
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.execution_stats = {
            "total_executed": 0,
            "successful": 0,
            "failed": 0,
            "cancelled": 0
        }
    
    async def execute_task(
        self, 
        task: AutomationTask,
        profile_id: Optional[str] = None
    ) -> TaskResult:
        """
        Execute automation task with full orchestration.
        
        Args:
            task: Task to execute
            profile_id: Optional profile ID override
            
        Returns:
            Task execution result
        """
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Starting task execution {execution_id}")
        
        # Initialize execution tracking
        self.active_tasks[execution_id] = {
            "task": task,
            "start_time": start_time,
            "status": TaskStatus.RUNNING,
            "profile_id": profile_id or task.profile_id
        }
        
        try:
            # Validate task
            validation_result = await self.validator.validate_task(task)
            logger.info(f"Task validation passed: {validation_result}")
            
            # Preprocess task
            processed_task = await self.preprocessor.preprocess_task(task)
            logger.info(f"Task preprocessed: {len(processed_task.actions)} actions")
            
            # Execute task
            result = await self._execute_processed_task(
                processed_task, 
                execution_id,
                profile_id or task.profile_id
            )
            
            # Update stats
            self.execution_stats["total_executed"] += 1
            if result.success:
                self.execution_stats["successful"] += 1
            else:
                self.execution_stats["failed"] += 1
            
            return result
            
        except TaskValidationError as e:
            logger.error(f"Task validation failed: {e}")
            result = TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
                actions_completed=0,
                metadata={"validation_error": True}
            )
            self.execution_stats["failed"] += 1
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            result = TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
                actions_completed=0,
                metadata={"execution_error": True}
            )
            self.execution_stats["failed"] += 1
            return result
            
        finally:
            # Clean up tracking
            if execution_id in self.active_tasks:
                del self.active_tasks[execution_id]
    
    async def _execute_processed_task(
        self, 
        task: AutomationTask, 
        execution_id: str,
        profile_id: str
    ) -> TaskResult:
        """Execute preprocessed task."""
        browser_manager = get_browser_instance_manager()
        captcha_solver = get_captcha_solver()
        behavioral_engine = get_behavioral_engine()
        
        browser_instance = None
        actions_completed = 0
        action_results = []
        
        try:
            # Get browser instance with profile
            browser_instance = await browser_manager.get_instance(profile_id)
            page = browser_instance.page
            
            logger.info(f"Browser instance acquired for profile {profile_id}")
            
            # Execute actions with behavioral integration
            for i, action in enumerate(task.actions):
                # Check for cancellation
                if self.active_tasks[execution_id]["status"] == TaskStatus.CANCELLED:
                    logger.info(f"Task {execution_id} cancelled")
                    break
                
                # Handle CAPTCHA checks
                if action.metadata and action.metadata.get("captcha_check"):
                    captcha_handled = await self._handle_captcha_check(page, captcha_solver)
                    if captcha_handled:
                        logger.info("CAPTCHA detected and handled")
                
                # Execute action with retries
                action_result = await self._execute_action_with_retry(
                    page, action, behavioral_engine
                )
                
                action_results.append(action_result)
                actions_completed += 1
                
                # Stop on critical failure
                if not action_result["success"] and not action.metadata.get("continue_on_error"):
                    logger.error(f"Critical action failure: {action_result.get('error')}")
                    break
                
                # Update progress
                progress = (i + 1) / len(task.actions)
                self.active_tasks[execution_id]["progress"] = progress
            
            # Determine overall success
            success = actions_completed == len(task.actions) and all(
                r["success"] for r in action_results
            )
            
            # Create result
            result = TaskResult(
                task_id=task.id,
                success=success,
                execution_time=time.time() - self.active_tasks[execution_id]["start_time"],
                actions_completed=actions_completed,
                action_results=action_results,
                metadata={
                    "profile_id": profile_id,
                    "execution_id": execution_id,
                    "browser_fingerprint": browser_instance.fingerprint_id if browser_instance else None
                }
            )
            
            # Log result to database
            await self._log_task_result(result)
            
            return result
            
        finally:
            # Clean up browser instance
            if browser_instance:
                await browser_manager.release_instance(browser_instance.id)
    
    async def _execute_action_with_retry(
        self, 
        page, 
        action: AutomationAction,
        behavioral_engine
    ) -> Dict[str, Any]:
        """Execute action with retry logic."""
        max_retries = action.metadata.get("max_retries", 1) if action.metadata else 1
        retry_delay = action.metadata.get("retry_delay", 1.0) if action.metadata else 1.0
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Execute single action
                from braf.core.browser.automation_utils import AutomationExecutor
                executor = AutomationExecutor(page)
                result = await executor.execute_action(action)
                
                if result["success"]:
                    if attempt > 0:
                        logger.info(f"Action succeeded on attempt {attempt + 1}")
                    return result
                else:
                    last_error = result.get("error", "Unknown error")
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Action attempt {attempt + 1} failed: {e}")
            
            # Wait before retry (except on last attempt)
            if attempt < max_retries - 1:
                await behavioral_engine.wait_with_human_delay(
                    "retry", 
                    {"delay": retry_delay, "attempt": attempt}
                )
        
        # All retries failed
        return {
            "success": False,
            "action_type": action.type.value,
            "error": f"Failed after {max_retries} attempts: {last_error}",
            "attempts": max_retries
        }
    
    async def _handle_captcha_check(self, page, captcha_solver) -> bool:
        """Check for and handle CAPTCHAs."""
        if not captcha_solver:
            return False
        
        try:
            # Detect CAPTCHA type
            captcha_type = await captcha_solver.detect_captcha_type(page)
            
            if not captcha_type:
                return False
            
            logger.info(f"CAPTCHA detected: {captcha_type}")
            
            if captcha_type == "recaptcha_v2":
                # Get site key and solve
                site_key = await page.evaluate("""
                    () => {
                        const element = document.querySelector('[data-sitekey]');
                        return element ? element.getAttribute('data-sitekey') : null;
                    }
                """)
                
                if site_key:
                    solution = await captcha_solver.solve_recaptcha_v2(site_key, page.url)
                    if solution:
                        return await captcha_solver.inject_recaptcha_solution(page, solution)
            
            elif captcha_type == "image_captcha":
                # Extract image and solve
                image_data = await page.evaluate("""
                    () => {
                        const img = document.querySelector('img[src*="captcha"], img[src*="challenge"]');
                        if (img) {
                            const canvas = document.createElement('canvas');
                            const ctx = canvas.getContext('2d');
                            canvas.width = img.naturalWidth;
                            canvas.height = img.naturalHeight;
                            ctx.drawImage(img, 0, 0);
                            return canvas.toDataURL().split(',')[1];
                        }
                        return null;
                    }
                """)
                
                if image_data:
                    import base64
                    image_bytes = base64.b64decode(image_data)
                    solution = await captcha_solver.solve_image_captcha(image_bytes)
                    
                    if solution:
                        # Find input field and enter solution
                        await page.fill('input[name*="captcha"], input[id*="captcha"]', solution)
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"CAPTCHA handling failed: {e}")
            return False
    
    async def _log_task_result(self, result: TaskResult):
        """Log task result to database."""
        try:
            db = get_database()
            if db:
                await db.log_task_execution(result)
        except Exception as e:
            logger.error(f"Failed to log task result: {e}")
    
    async def cancel_task(self, execution_id: str) -> bool:
        """
        Cancel running task.
        
        Args:
            execution_id: Execution ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        if execution_id in self.active_tasks:
            self.active_tasks[execution_id]["status"] = TaskStatus.CANCELLED
            logger.info(f"Task {execution_id} marked for cancellation")
            self.execution_stats["cancelled"] += 1
            return True
        
        return False
    
    def get_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active tasks."""
        return self.active_tasks.copy()
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self.execution_stats.copy()


# Global task executor instance
_task_executor: Optional[TaskExecutor] = None


def get_task_executor() -> Optional[TaskExecutor]:
    """
    Get global task executor instance.
    
    Returns:
        Task executor instance or None if not initialized
    """
    return _task_executor


def init_task_executor() -> TaskExecutor:
    """
    Initialize global task executor.
    
    Returns:
        Initialized task executor
    """
    global _task_executor
    
    _task_executor = TaskExecutor()
    
    return _task_executor