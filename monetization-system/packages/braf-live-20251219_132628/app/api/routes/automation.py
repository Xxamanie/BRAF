"""
Automation API endpoints for BRAF Monetization System
Handles automation task creation, monitoring, and management
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from core.task_manager import task_manager, TaskType, TaskStatus
from database.service import DatabaseService

router = APIRouter(prefix="/api/v1/automation", tags=["Automation"])


class AutomationTaskRequest(BaseModel):
    enterprise_id: str
    task_type: str  # survey_automation, video_automation
    config: Dict[str, Any]


class SurveyAutomationConfig(BaseModel):
    accounts: List[Dict[str, Any]]
    platforms: List[str]
    max_surveys_per_session: int = 5
    daily_limit: float = 10.0


class VideoAutomationConfig(BaseModel):
    platform: str = "youtube"
    device_type: str = "desktop"
    video_count: int = 50
    watch_time_optimization: bool = True


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    earnings: float
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    result: Optional[Dict[str, Any]]
    error: Optional[str]


@router.post("/create/{enterprise_id}")
async def create_automation(enterprise_id: str, request: dict):
    """Create a new automation"""
    try:
        with DatabaseService() as db:
            automation_data = {
                "enterprise_id": enterprise_id,
                "template_type": request["template_type"],
                "platform": request["platform"],
                "config": request["config"]
            }
            
            automation = db.create_automation(automation_data)
            
            return {
                "success": True,
                "automation_id": automation.id,
                "status": automation.status,
                "message": "Automation created successfully"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tasks/create", response_model=Dict[str, str])
async def create_automation_task(request: AutomationTaskRequest):
    """Create a new automation task"""
    try:
        # Validate task type
        if request.task_type not in [t.value for t in TaskType]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task type. Must be one of: {[t.value for t in TaskType]}"
            )
        
        # Validate enterprise exists
        with DatabaseService() as db:
            enterprise = db.get_enterprise(request.enterprise_id)
            if not enterprise:
                raise HTTPException(status_code=404, detail="Enterprise not found")
            
            # Check subscription limits
            subscription = db.get_active_subscription(request.enterprise_id)
            if not subscription:
                raise HTTPException(status_code=403, detail="No active subscription")
            
            # Check running task limits based on subscription tier
            running_tasks = task_manager.get_running_tasks(request.enterprise_id)
            max_concurrent = {
                "basic": 2,
                "pro": 5,
                "enterprise": 20
            }.get(subscription.tier, 1)
            
            if len(running_tasks) >= max_concurrent:
                raise HTTPException(
                    status_code=429,
                    detail=f"Maximum concurrent tasks ({max_concurrent}) reached for {subscription.tier} tier"
                )
        
        # Create the task
        task_id = await task_manager.create_automation_task(
            enterprise_id=request.enterprise_id,
            task_config=request.config
        )
        
        return {
            "task_id": task_id,
            "status": "created",
            "message": "Automation task created successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")


@router.get("/tasks/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get status of a specific task"""
    task_result = task_manager.get_task_status(task_id)
    
    if not task_result:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskStatusResponse(
        task_id=task_result.task_id,
        status=task_result.status.value,
        earnings=task_result.earnings,
        started_at=task_result.started_at,
        completed_at=task_result.completed_at,
        result=task_result.result,
        error=task_result.error
    )


@router.get("/list/{enterprise_id}")
async def list_automations(enterprise_id: str):
    """Get list of automations for enterprise"""
    try:
        with DatabaseService() as db:
            automations = db.get_automations(enterprise_id)
            
            return {
                "automations": [
                    {
                        "id": automation.id,
                        "template_type": automation.template_type,
                        "platform": automation.platform,
                        "status": automation.status,
                        "earnings_today": automation.earnings_today,
                        "earnings_total": automation.earnings_total,
                        "success_rate": automation.success_rate,
                        "last_run": automation.last_run.isoformat() if automation.last_run else None,
                        "created_at": automation.created_at.isoformat()
                    } for automation in automations
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/running", response_model=List[str])
async def get_running_tasks(enterprise_id: Optional[str] = None):
    """Get list of running tasks"""
    return task_manager.get_running_tasks(enterprise_id)


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a running task"""
    success = await task_manager.cancel_task(task_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or not running")
    
    return {"message": "Task cancelled successfully"}


@router.get("/tasks/history", response_model=List[TaskStatusResponse])
async def get_task_history(enterprise_id: str, limit: int = 50):
    """Get task history for enterprise"""
    history = task_manager.get_task_history(enterprise_id, limit)
    
    return [
        TaskStatusResponse(
            task_id=task.task_id,
            status=task.status.value,
            earnings=task.earnings,
            started_at=task.started_at,
            completed_at=task.completed_at,
            result=task.result,
            error=task.error
        ) for task in history
    ]


@router.post("/survey/start")
async def start_survey_automation(
    enterprise_id: str,
    config: SurveyAutomationConfig,
    background_tasks: BackgroundTasks
):
    """Start survey automation with specific configuration"""
    try:
        task_config = {
            "type": TaskType.SURVEY_AUTOMATION.value,
            "accounts": config.accounts,
            "platforms": config.platforms,
            "max_surveys_per_session": config.max_surveys_per_session,
            "daily_limit": config.daily_limit
        }
        
        task_id = await task_manager.create_automation_task(enterprise_id, task_config)
        
        return {
            "task_id": task_id,
            "message": "Survey automation started",
            "config": config.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video/start")
async def start_video_automation(
    enterprise_id: str,
    config: VideoAutomationConfig,
    background_tasks: BackgroundTasks
):
    """Start video automation with specific configuration"""
    try:
        task_config = {
            "type": TaskType.VIDEO_AUTOMATION.value,
            "platform": config.platform,
            "device_type": config.device_type,
            "video_count": config.video_count,
            "watch_time_optimization": config.watch_time_optimization
        }
        
        task_id = await task_manager.create_automation_task(enterprise_id, task_config)
        
        return {
            "task_id": task_id,
            "message": "Video automation started",
            "config": config.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/automations", response_model=List[Dict[str, Any]])
async def get_automations(enterprise_id: str):
    """Get all automations for enterprise"""
    with DatabaseService() as db:
        automations = db.get_automations(enterprise_id)
        
        return [
            {
                "id": automation.id,
                "template_type": automation.template_type,
                "platform": automation.platform,
                "status": automation.status,
                "earnings_today": automation.earnings_today,
                "earnings_total": automation.earnings_total,
                "success_rate": automation.success_rate,
                "last_run": automation.last_run.isoformat() if automation.last_run else None,
                "created_at": automation.created_at.isoformat()
            } for automation in automations
        ]


@router.get("/earnings", response_model=Dict[str, Any])
async def get_earnings_summary(enterprise_id: str, days: int = 30):
    """Get earnings summary for enterprise"""
    with DatabaseService() as db:
        earnings = db.get_earnings(enterprise_id, days)
        
        # Calculate summary
        total_earnings = sum(e.amount for e in earnings)
        earnings_by_platform = {}
        earnings_by_day = {}
        
        for earning in earnings:
            # By platform
            if earning.platform not in earnings_by_platform:
                earnings_by_platform[earning.platform] = 0
            earnings_by_platform[earning.platform] += earning.amount
            
            # By day
            day = earning.earned_at.date().isoformat()
            if day not in earnings_by_day:
                earnings_by_day[day] = 0
            earnings_by_day[day] += earning.amount
        
        return {
            "total_earnings": total_earnings,
            "earnings_count": len(earnings),
            "average_earning": total_earnings / len(earnings) if earnings else 0,
            "earnings_by_platform": earnings_by_platform,
            "earnings_by_day": earnings_by_day,
            "recent_earnings": [
                {
                    "amount": e.amount,
                    "platform": e.platform,
                    "task_type": e.task_type,
                    "earned_at": e.earned_at.isoformat()
                } for e in earnings[:20]
            ]
        }


@router.post("/compliance/check")
async def trigger_compliance_check(enterprise_id: str):
    """Trigger compliance check for enterprise"""
    try:
        task_id = await task_manager.schedule_compliance_check(enterprise_id)
        
        return {
            "task_id": task_id,
            "message": "Compliance check scheduled"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))