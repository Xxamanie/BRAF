"""
Task Management System for BRAF Monetization
Handles automation task execution, scheduling, and monitoring
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from database.service import DatabaseService
from templates.survey_automation import SurveyAutomation
from templates.video_monetization import VideoPlatformAutomation
from compliance.automation_checker import ComplianceChecker
from security.authentication import SecurityManager
from worker import (
    process_survey_automation, 
    process_video_automation,
    compliance_check,
    security_monitoring
)

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    SURVEY_AUTOMATION = "survey_automation"
    VIDEO_AUTOMATION = "video_automation"
    COMPLIANCE_CHECK = "compliance_check"
    SECURITY_MONITORING = "security_monitoring"


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    result: Optional[Dict] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    earnings: float = 0.0


class TaskManager:
    """Manages automation tasks and their execution"""
    
    def __init__(self):
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, TaskResult] = {}
        self.compliance_checker = ComplianceChecker()
        self.security_manager = SecurityManager()
    
    async def create_automation_task(self, enterprise_id: str, task_config: Dict) -> str:
        """Create and queue an automation task"""
        task_id = f"{enterprise_id}_{task_config['type']}_{datetime.utcnow().timestamp()}"
        
        # Validate enterprise and subscription
        with DatabaseService() as db:
            enterprise = db.get_enterprise(enterprise_id)
            if not enterprise:
                raise ValueError(f"Enterprise {enterprise_id} not found")
            
            subscription = db.get_active_subscription(enterprise_id)
            if not subscription:
                raise ValueError(f"No active subscription for enterprise {enterprise_id}")
        
        # Check compliance before starting
        compliance_result = self.compliance_checker.check_automation_compliance(
            template_type=task_config['type'],
            automation_config=task_config
        )
        
        if not compliance_result['compliant']:
            raise ValueError(f"Task failed compliance check: {compliance_result['violations']}")
        
        # Create task result tracker
        task_result = TaskResult(
            task_id=task_id,
            status=TaskStatus.PENDING,
            started_at=datetime.utcnow()
        )
        self.task_results[task_id] = task_result
        
        # Queue the task based on type
        if task_config['type'] == TaskType.SURVEY_AUTOMATION.value:
            task = asyncio.create_task(self._run_survey_automation(task_id, enterprise_id, task_config))
        elif task_config['type'] == TaskType.VIDEO_AUTOMATION.value:
            task = asyncio.create_task(self._run_video_automation(task_id, enterprise_id, task_config))
        else:
            raise ValueError(f"Unknown task type: {task_config['type']}")
        
        self.running_tasks[task_id] = task
        
        logger.info(f"Created automation task {task_id} for enterprise {enterprise_id}")
        return task_id
    
    async def _run_survey_automation(self, task_id: str, enterprise_id: str, config: Dict):
        """Run survey automation task"""
        try:
            self.task_results[task_id].status = TaskStatus.RUNNING
            
            # Execute survey automation
            survey_automation = SurveyAutomation()
            accounts = config.get('accounts', [])
            
            # Run automation
            total_earnings = 0.0
            for account in accounts:
                await survey_automation.initialize_driver()
                
                for platform in survey_automation.platforms.keys():
                    if platform in account.get("platforms", []):
                        try:
                            await survey_automation.complete_survey(platform, account)
                            earnings = survey_automation.platforms[platform]["earning_rate"]
                            total_earnings += earnings
                            
                            # Record earning in database
                            with DatabaseService() as db:
                                # Get or create automation record
                                automations = db.get_automations(enterprise_id)
                                automation = next((a for a in automations if a.template_type == "survey" and a.platform == platform), None)
                                
                                if not automation:
                                    automation = db.create_automation({
                                        "enterprise_id": enterprise_id,
                                        "template_type": "survey",
                                        "platform": platform,
                                        "config": config
                                    })
                                
                                # Record earning
                                db.record_earning({
                                    "automation_id": automation.id,
                                    "amount": earnings,
                                    "platform": platform,
                                    "task_type": "survey_completion",
                                    "task_details": {"account": account.get("username", "unknown")}
                                })
                        
                        except Exception as e:
                            logger.error(f"Survey automation error for {platform}: {e}")
                            continue
                
                survey_automation.driver.quit()
            
            # Update task result
            self.task_results[task_id].status = TaskStatus.COMPLETED
            self.task_results[task_id].completed_at = datetime.utcnow()
            self.task_results[task_id].earnings = total_earnings
            self.task_results[task_id].result = {
                "total_earnings": total_earnings,
                "platforms_processed": len(accounts),
                "completed_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Survey automation task {task_id} completed with earnings: ${total_earnings}")
            
        except Exception as e:
            logger.error(f"Survey automation task {task_id} failed: {e}")
            self.task_results[task_id].status = TaskStatus.FAILED
            self.task_results[task_id].error = str(e)
            self.task_results[task_id].completed_at = datetime.utcnow()
        
        finally:
            # Clean up
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    async def _run_video_automation(self, task_id: str, enterprise_id: str, config: Dict):
        """Run video automation task"""
        try:
            self.task_results[task_id].status = TaskStatus.RUNNING
            
            # Execute video automation
            video_automation = VideoPlatformAutomation()
            platform = config.get('platform', 'youtube')
            device_type = config.get('device_type', 'desktop')
            video_count = config.get('video_count', 50)
            
            # Run automation
            earnings = await video_automation.watch_videos(platform, device_type, video_count)
            
            # Record earning in database
            with DatabaseService() as db:
                # Get or create automation record
                automations = db.get_automations(enterprise_id)
                automation = next((a for a in automations if a.template_type == "video" and a.platform == platform), None)
                
                if not automation:
                    automation = db.create_automation({
                        "enterprise_id": enterprise_id,
                        "template_type": "video",
                        "platform": platform,
                        "config": config
                    })
                
                # Record earning
                db.record_earning({
                    "automation_id": automation.id,
                    "amount": earnings,
                    "platform": platform,
                    "task_type": "video_viewing",
                    "task_details": {"videos_watched": video_count, "device_type": device_type}
                })
            
            # Update task result
            self.task_results[task_id].status = TaskStatus.COMPLETED
            self.task_results[task_id].completed_at = datetime.utcnow()
            self.task_results[task_id].earnings = earnings
            self.task_results[task_id].result = {
                "earnings": earnings,
                "videos_watched": video_count,
                "platform": platform,
                "completed_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Video automation task {task_id} completed with earnings: ${earnings}")
            
        except Exception as e:
            logger.error(f"Video automation task {task_id} failed: {e}")
            self.task_results[task_id].status = TaskStatus.FAILED
            self.task_results[task_id].error = str(e)
            self.task_results[task_id].completed_at = datetime.utcnow()
        
        finally:
            # Clean up
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get task status and result"""
        return self.task_results.get(task_id)
    
    def get_running_tasks(self, enterprise_id: str = None) -> List[str]:
        """Get list of running task IDs"""
        if enterprise_id:
            return [task_id for task_id in self.running_tasks.keys() if task_id.startswith(enterprise_id)]
        return list(self.running_tasks.keys())
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.cancel()
            
            # Update task result
            if task_id in self.task_results:
                self.task_results[task_id].status = TaskStatus.CANCELLED
                self.task_results[task_id].completed_at = datetime.utcnow()
            
            del self.running_tasks[task_id]
            logger.info(f"Cancelled task {task_id}")
            return True
        return False
    
    def get_task_history(self, enterprise_id: str, limit: int = 50) -> List[TaskResult]:
        """Get task history for enterprise"""
        enterprise_tasks = [
            result for task_id, result in self.task_results.items()
            if task_id.startswith(enterprise_id)
        ]
        
        # Sort by start time, most recent first
        enterprise_tasks.sort(key=lambda x: x.started_at or datetime.min, reverse=True)
        
        return enterprise_tasks[:limit]
    
    async def schedule_compliance_check(self, enterprise_id: str) -> str:
        """Schedule a compliance check"""
        task_id = f"{enterprise_id}_compliance_{datetime.utcnow().timestamp()}"
        
        # Run compliance check asynchronously
        compliance_check.delay(enterprise_id, {
            "template_type": "general",
            "config": {},
            "activities": []
        })
        
        logger.info(f"Scheduled compliance check {task_id} for enterprise {enterprise_id}")
        return task_id
    
    async def schedule_security_monitoring(self, enterprise_id: str, activity: Dict) -> str:
        """Schedule security monitoring"""
        task_id = f"{enterprise_id}_security_{datetime.utcnow().timestamp()}"
        
        # Run security monitoring asynchronously
        security_monitoring.delay(enterprise_id, activity)
        
        logger.info(f"Scheduled security monitoring {task_id} for enterprise {enterprise_id}")
        return task_id


# Global task manager instance
task_manager = TaskManager()
