"""
Intelligence API Routes
Provides endpoints for accessing the BRAF intelligence system
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# Import intelligence components
from intelligence.braf_intelligence_integration import braf_intelligence
from intelligence.platform_intelligence_engine import platform_intelligence
from intelligence.behavior_profile_manager import behavior_profile_manager
from intelligence.earning_optimizer import earning_optimizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/intelligence", tags=["Intelligence System"])

# Request/Response Models
class AutomationRequest(BaseModel):
    platform_name: str
    task_config: Dict[str, Any]
    intelligence_config: Optional[Dict[str, Any]] = None

class PlatformAnalysisRequest(BaseModel):
    platform_url: str

class BehaviorProfileRequest(BaseModel):
    platform_name: str
    risk_level: float = 0.5

class CustomProfileRequest(BaseModel):
    platform_name: str
    base_profile_name: Optional[str] = None
    adjustments: Optional[Dict[str, Any]] = None

# Intelligence System Management
@router.get("/status")
async def get_intelligence_status():
    """Get intelligence system status and statistics"""
    try:
        status = braf_intelligence.get_intelligence_status()
        
        return {
            "success": True,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get intelligence status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/enable")
async def enable_intelligence():
    """Enable intelligence system"""
    try:
        braf_intelligence.enable_intelligence()
        
        return {
            "success": True,
            "message": "Intelligence system enabled",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to enable intelligence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/disable")
async def disable_intelligence():
    """Disable intelligence system"""
    try:
        braf_intelligence.disable_intelligence()
        
        return {
            "success": True,
            "message": "Intelligence system disabled - using standard BRAF",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to disable intelligence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Platform Intelligence
@router.get("/platforms")
async def get_supported_platforms():
    """Get list of supported platforms"""
    try:
        platforms = platform_intelligence.get_all_platforms()
        top_platforms = platform_intelligence.get_top_earning_platforms(10)
        
        return {
            "success": True,
            "supported_platforms": platforms,
            "top_earning_platforms": top_platforms,
            "total_platforms": len(platforms)
        }
        
    except Exception as e:
        logger.error(f"Failed to get supported platforms: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/platforms/analyze")
async def analyze_platform(request: PlatformAnalysisRequest):
    """Analyze a platform for intelligence data"""
    try:
        analysis = await platform_intelligence.analyze_platform(request.platform_url)
        
        return {
            "success": True,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze platform: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/platforms/{platform_name}/recommendations")
async def get_platform_recommendations(platform_name: str):
    """Get intelligence recommendations for a specific platform"""
    try:
        recommendations = await braf_intelligence.get_platform_recommendations(platform_name)
        
        return {
            "success": True,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get platform recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/platforms/{platform_name}/profile")
async def get_platform_profile(platform_name: str):
    """Get detailed platform profile"""
    try:
        profile = platform_intelligence.get_platform_profile(platform_name)
        
        if not profile:
            raise HTTPException(status_code=404, detail="Platform profile not found")
        
        return {
            "success": True,
            "profile": {
                "name": profile.name,
                "platform_type": profile.platform_type.value,
                "base_url": profile.base_url,
                "earning_rate_usd_per_hour": profile.earning_rate_usd_per_hour,
                "reliability_score": profile.reliability_score,
                "payment_threshold": profile.payment_threshold,
                "payment_methods": profile.payment_methods,
                "session_timeout_minutes": profile.session_timeout_minutes,
                "detection_vectors": profile.detection_vectors,
                "bypass_methods": profile.bypass_methods,
                "best_time_of_day": profile.best_time_of_day,
                "best_day_of_week": profile.best_day_of_week,
                "geographic_preferences": profile.geographic_preferences,
                "demographic_bonuses": profile.demographic_bonuses
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get platform profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Behavior Profiles
@router.get("/behavior/profiles")
async def get_behavior_profiles():
    """Get all available behavior profiles"""
    try:
        profiles = behavior_profile_manager.get_all_profiles()
        
        return {
            "success": True,
            "profiles": profiles,
            "total_platforms": len(profiles)
        }
        
    except Exception as e:
        logger.error(f"Failed to get behavior profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/behavior/profiles/optimal")
async def get_optimal_behavior_profile(request: BehaviorProfileRequest):
    """Get optimal behavior profile for platform"""
    try:
        profile = await behavior_profile_manager.get_optimal_profile(
            request.platform_name, 
            request.risk_level
        )
        
        return {
            "success": True,
            "profile": profile,
            "platform": request.platform_name,
            "risk_level": request.risk_level
        }
        
    except Exception as e:
        logger.error(f"Failed to get optimal behavior profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/behavior/profiles/create")
async def create_custom_behavior_profile(request: CustomProfileRequest):
    """Create custom behavior profile"""
    try:
        profile_name = behavior_profile_manager.create_custom_profile(
            request.platform_name,
            request.base_profile_name,
            request.adjustments
        )
        
        return {
            "success": True,
            "message": "Custom behavior profile created",
            "profile_name": profile_name,
            "platform": request.platform_name
        }
        
    except Exception as e:
        logger.error(f"Failed to create custom behavior profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/behavior/profiles/{platform_name}/performance")
async def get_behavior_profile_performance(platform_name: str):
    """Get behavior profile performance statistics"""
    try:
        performance = behavior_profile_manager.get_profile_performance(platform_name)
        
        return {
            "success": True,
            "performance": performance,
            "platform": platform_name
        }
        
    except Exception as e:
        logger.error(f"Failed to get behavior profile performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Earning Optimization
@router.get("/optimization/stats")
async def get_optimization_stats():
    """Get earning optimization statistics"""
    try:
        stats = earning_optimizer.get_optimization_stats()
        
        return {
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get optimization stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/optimization/{platform_name}/timing")
async def get_optimal_timing(platform_name: str):
    """Get optimal execution timing for platform"""
    try:
        timing = earning_optimizer.get_optimal_execution_time(platform_name)
        
        return {
            "success": True,
            "timing": timing,
            "platform": platform_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get optimal timing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/optimization/{platform_name}/forecast")
async def get_earning_forecast(platform_name: str, hours_ahead: int = 24):
    """Get earning forecast for platform"""
    try:
        if hours_ahead < 1 or hours_ahead > 168:  # Max 1 week
            raise HTTPException(status_code=400, detail="hours_ahead must be between 1 and 168")
        
        forecast = earning_optimizer.get_earning_forecast(platform_name, hours_ahead)
        
        return {
            "success": True,
            "forecast": forecast,
            "platform": platform_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get earning forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Intelligent Automation Execution
@router.post("/automation/execute")
async def execute_intelligent_automation(request: AutomationRequest, background_tasks: BackgroundTasks):
    """Execute automation with intelligence integration"""
    try:
        # Validate request
        if not request.platform_name:
            raise HTTPException(status_code=400, detail="platform_name is required")
        
        if not request.task_config:
            raise HTTPException(status_code=400, detail="task_config is required")
        
        # Execute automation
        result = await braf_intelligence.execute_intelligent_automation(
            request.platform_name,
            request.task_config,
            request.intelligence_config
        )
        
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute intelligent automation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/automation/execute/async")
async def execute_intelligent_automation_async(request: AutomationRequest, background_tasks: BackgroundTasks):
    """Execute automation asynchronously with intelligence integration"""
    try:
        # Generate task ID
        task_id = f"intel_task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Add to background tasks
        background_tasks.add_task(
            _execute_automation_background,
            task_id,
            request.platform_name,
            request.task_config,
            request.intelligence_config
        )
        
        return {
            "success": True,
            "message": "Automation task queued for execution",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to queue intelligent automation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Learning and Analytics
@router.get("/learning/platforms")
async def get_learning_data():
    """Get learning data for all platforms"""
    try:
        learning_stats = {}
        
        if hasattr(earning_optimizer, 'learning_data'):
            for platform, data in earning_optimizer.learning_data.items():
                learning_stats[platform] = {
                    'total_executions': len(data),
                    'recent_success_rate': sum(1 for d in data[-10:] if d.get('success', False)) / min(len(data), 10),
                    'avg_earning': sum(d.get('earning_amount', 0) for d in data) / len(data) if data else 0,
                    'last_execution': data[-1]['timestamp'].isoformat() if data else None
                }
        
        return {
            "success": True,
            "learning_stats": learning_stats,
            "total_platforms": len(learning_stats)
        }
        
    except Exception as e:
        logger.error(f"Failed to get learning data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/performance")
async def get_performance_analytics():
    """Get performance analytics across all platforms"""
    try:
        analytics = {}
        
        if hasattr(earning_optimizer, 'platform_performance'):
            for platform, metrics in earning_optimizer.platform_performance.items():
                analytics[platform] = {
                    'total_executions': metrics.get('total_executions', 0),
                    'success_rate': metrics.get('avg_success_rate', 0),
                    'earning_rate_per_hour': metrics.get('avg_earning_rate', 0),
                    'total_earnings': metrics.get('total_earnings', 0),
                    'total_time_hours': metrics.get('total_time', 0) / 3600,
                    'last_updated': metrics.get('last_updated').isoformat() if metrics.get('last_updated') else None
                }
        
        # Calculate overall statistics
        total_earnings = sum(a.get('total_earnings', 0) for a in analytics.values())
        total_executions = sum(a.get('total_executions', 0) for a in analytics.values())
        avg_success_rate = sum(a.get('success_rate', 0) for a in analytics.values()) / len(analytics) if analytics else 0
        
        return {
            "success": True,
            "platform_analytics": analytics,
            "overall_stats": {
                "total_platforms": len(analytics),
                "total_executions": total_executions,
                "total_earnings": total_earnings,
                "average_success_rate": avg_success_rate
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System Configuration
@router.get("/config")
async def get_intelligence_config():
    """Get intelligence system configuration"""
    try:
        config = {
            "intelligence_enabled": braf_intelligence.intelligence_enabled,
            "supported_platforms": platform_intelligence.get_all_platforms(),
            "default_risk_tolerance": 0.5,
            "default_optimization_level": 3,
            "learning_enabled": True,
            "adaptive_behavior": True,
            "stealth_mode": True,
            "version": "2.0"
        }
        
        return {
            "success": True,
            "config": config
        }
        
    except Exception as e:
        logger.error(f"Failed to get intelligence config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task function
async def _execute_automation_background(task_id: str, platform_name: str, 
                                       task_config: Dict[str, Any], 
                                       intelligence_config: Optional[Dict[str, Any]]):
    """Execute automation in background"""
    try:
        logger.info(f"Starting background automation task: {task_id}")
        
        result = await braf_intelligence.execute_intelligent_automation(
            platform_name,
            task_config,
            intelligence_config
        )
        
        # Store result (in production, would use database or cache)
        # For now, just log the result
        logger.info(f"Background task {task_id} completed: {result.get('success', False)}")
        
    except Exception as e:
        logger.error(f"Background automation task {task_id} failed: {e}")

# Example task configurations for testing
@router.get("/examples/task-configs")
async def get_example_task_configs():
    """Get example task configurations for testing"""
    try:
        examples = {
            "swagbucks_survey": {
                "task_id": "swagbucks_survey_example",
                "name": "Swagbucks Survey Completion",
                "description": "Complete a survey on Swagbucks",
                "actions": [
                    {
                        "type": "navigate",
                        "data": "https://www.swagbucks.com/surveys",
                        "timeout": 10
                    },
                    {
                        "type": "click",
                        "selector": ".survey-card:first-child .start-button",
                        "timeout": 5
                    },
                    {
                        "type": "wait",
                        "data": "2",
                        "timeout": 5
                    },
                    {
                        "type": "click",
                        "selector": "input[type='radio']:first-child",
                        "timeout": 5,
                        "optional": True
                    }
                ],
                "priority": 1,
                "timeout": 300
            },
            "survey_junkie_profile": {
                "task_id": "survey_junkie_profile_example",
                "name": "Survey Junkie Profile Update",
                "description": "Update profile information on Survey Junkie",
                "actions": [
                    {
                        "type": "navigate",
                        "data": "https://www.surveyjunkie.com/profile",
                        "timeout": 10
                    },
                    {
                        "type": "click",
                        "selector": ".edit-profile-button",
                        "timeout": 5
                    },
                    {
                        "type": "type",
                        "selector": "input[name='age']",
                        "data": "30",
                        "timeout": 5
                    },
                    {
                        "type": "click",
                        "selector": ".save-button",
                        "timeout": 5
                    }
                ],
                "priority": 2,
                "timeout": 180
            }
        }
        
        intelligence_config_example = {
            "risk_tolerance": 0.6,
            "optimization_level": 4,
            "stealth_mode": True,
            "learning_enabled": True,
            "adaptive_behavior": True
        }
        
        return {
            "success": True,
            "task_examples": examples,
            "intelligence_config_example": intelligence_config_example,
            "usage_instructions": [
                "Use POST /api/v1/intelligence/automation/execute with one of these examples",
                "Adjust intelligence_config parameters based on your risk tolerance",
                "Monitor execution through the analytics endpoints"
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get example configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
