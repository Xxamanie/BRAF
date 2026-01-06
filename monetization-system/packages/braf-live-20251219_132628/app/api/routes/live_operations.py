"""
Live Operations API Routes
Provides endpoints for managing real money operations
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# Import live integration components
from live_integration_orchestrator import live_orchestrator
from earnings.swagbucks_integration import swagbucks_client
from earnings.youtube_integration import youtube_client
from payments.opay_integration import opay_client
from payments.palmpay_integration import palmpay_client
from automation.browser_automation import browser_automation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/live", tags=["Live Operations"])

# Request/Response Models
class StartOperationsRequest(BaseModel):
    auto_withdrawal: bool = True
    max_daily_earnings: float = 500.0
    min_withdrawal_amount: float = 10.0

class WithdrawalRequest(BaseModel):
    amount_usd: float
    method: str  # 'opay' or 'palmpay'
    phone_number: str
    enterprise_id: Optional[int] = None

class SurveyTaskRequest(BaseModel):
    survey_id: str
    answers: Dict[str, Any] = {}

class VideoTaskRequest(BaseModel):
    video_url: str
    watch_duration: Optional[int] = None

# Live Operations Management
@router.post("/start")
async def start_live_operations(request: StartOperationsRequest):
    """Start live money-making operations"""
    try:
        if live_orchestrator.is_running:
            return {
                "success": False,
                "message": "Live operations are already running",
                "stats": live_orchestrator.get_live_stats()
            }
        
        # Update configuration
        import os
        os.environ['AUTO_WITHDRAWAL_ENABLED'] = str(request.auto_withdrawal).lower()
        os.environ['MAX_DAILY_EARNINGS_USD'] = str(request.max_daily_earnings)
        os.environ['MIN_WITHDRAWAL_USD'] = str(request.min_withdrawal_amount)
        
        # Start operations
        result = live_orchestrator.start_live_operations()
        
        return {
            "success": result['status'] == 'started',
            "message": result['message'],
            "configuration": result.get('configuration', {}),
            "start_time": result.get('start_time')
        }
        
    except Exception as e:
        logger.error(f"Failed to start live operations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_live_operations():
    """Stop live money-making operations"""
    try:
        if not live_orchestrator.is_running:
            return {
                "success": False,
                "message": "Live operations are not running"
            }
        
        result = live_orchestrator.stop_live_operations()
        
        return {
            "success": result['status'] == 'stopped',
            "message": result['message'],
            "uptime": result.get('uptime'),
            "final_stats": result.get('final_stats', {})
        }
        
    except Exception as e:
        logger.error(f"Failed to stop live operations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_live_stats():
    """Get current live operations statistics"""
    try:
        stats = live_orchestrator.get_live_stats()
        
        # Add additional system information
        stats.update({
            "system_info": {
                "demo_mode": browser_automation.demo_mode,
                "payment_providers": {
                    "opay_configured": not opay_client.demo_mode,
                    "palmpay_configured": not palmpay_client.demo_mode
                },
                "earning_platforms": {
                    "swagbucks_configured": not swagbucks_client.demo_mode,
                    "youtube_configured": not youtube_client.demo_mode
                }
            }
        })
        
        return {
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get live stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Earning Operations
@router.get("/earnings/estimate")
async def get_earnings_estimate():
    """Get estimated daily earnings potential"""
    try:
        # Get estimates from different sources
        swagbucks_estimate = swagbucks_client.estimate_daily_earnings()
        youtube_estimate = youtube_client.get_estimated_earnings(1)  # 1 day
        
        total_estimate = {
            "min_daily_usd": swagbucks_estimate['min_usd'] + youtube_estimate.get('daily_average_usd', 0) * 0.5,
            "max_daily_usd": swagbucks_estimate['max_usd'] + youtube_estimate.get('daily_average_usd', 0) * 1.5,
            "avg_daily_usd": swagbucks_estimate['avg_usd'] + youtube_estimate.get('daily_average_usd', 0),
            "sources": {
                "swagbucks": swagbucks_estimate,
                "youtube": youtube_estimate
            }
        }
        
        return {
            "success": True,
            "estimate": total_estimate,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get earnings estimate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/earnings/survey/complete")
async def complete_survey_task(request: SurveyTaskRequest, background_tasks: BackgroundTasks):
    """Complete a survey task"""
    try:
        # Add task to background processing
        background_tasks.add_task(
            _process_survey_task,
            request.survey_id,
            request.answers
        )
        
        return {
            "success": True,
            "message": f"Survey task {request.survey_id} queued for processing",
            "task_id": f"survey_{request.survey_id}_{int(datetime.now().timestamp())}"
        }
        
    except Exception as e:
        logger.error(f"Failed to queue survey task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/earnings/video/watch")
async def watch_video_task(request: VideoTaskRequest, background_tasks: BackgroundTasks):
    """Watch a video for earnings"""
    try:
        # Add task to background processing
        background_tasks.add_task(
            _process_video_task,
            request.video_url,
            request.watch_duration
        )
        
        return {
            "success": True,
            "message": f"Video watch task queued for processing",
            "task_id": f"video_{int(datetime.now().timestamp())}"
        }
        
    except Exception as e:
        logger.error(f"Failed to queue video task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Payment Operations
@router.post("/withdrawal/process")
async def process_live_withdrawal(request: WithdrawalRequest):
    """Process a live withdrawal request"""
    try:
        # Validate withdrawal amount
        if request.amount_usd < 1.0:
            raise HTTPException(status_code=400, detail="Minimum withdrawal amount is $1.00")
        
        if request.amount_usd > 10000.0:
            raise HTTPException(status_code=400, detail="Maximum withdrawal amount is $10,000.00")
        
        # Validate method
        if request.method not in ['opay', 'palmpay']:
            raise HTTPException(status_code=400, detail="Supported methods: opay, palmpay")
        
        # Validate phone number
        if request.method == 'opay' and not opay_client.validate_phone_number(request.phone_number):
            raise HTTPException(status_code=400, detail="Invalid phone number for OPay")
        
        if request.method == 'palmpay' and not palmpay_client.validate_phone_number(request.phone_number):
            raise HTTPException(status_code=400, detail="Invalid phone number for PalmPay")
        
        # Process withdrawal
        result = live_orchestrator.process_withdrawal(
            amount_usd=request.amount_usd,
            method=request.method,
            account_details={'phone_number': request.phone_number}
        )
        
        if result['success']:
            return {
                "success": True,
                "message": "Withdrawal processed successfully",
                "transaction": result
            }
        else:
            raise HTTPException(status_code=400, detail=result['error'])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process withdrawal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/withdrawal/balance")
async def get_withdrawal_balance():
    """Get available balance for withdrawal"""
    try:
        stats = live_orchestrator.get_live_stats()
        
        return {
            "success": True,
            "balance": {
                "available_usd": stats['available_balance_usd'],
                "total_earned_usd": stats['total_earned_usd'],
                "total_withdrawn_usd": stats['total_withdrawn_usd'],
                "daily_earnings_usd": stats['daily_earnings_usd']
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get withdrawal balance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System Information
@router.get("/system/status")
async def get_system_status():
    """Get system status and health information"""
    try:
        # Check payment provider status
        opay_balance = opay_client.check_balance()
        palmpay_balance = palmpay_client.check_balance()
        
        # Check earning platform status
        swagbucks_balance = swagbucks_client.get_account_balance()
        youtube_analytics = youtube_client.get_channel_analytics(1)
        
        # Get automation stats
        automation_stats = browser_automation.get_automation_stats()
        
        return {
            "success": True,
            "system_status": {
                "live_operations_running": live_orchestrator.is_running,
                "payment_providers": {
                    "opay": {
                        "configured": not opay_client.demo_mode,
                        "status": "connected" if opay_balance.get('code') == '00000' else "error",
                        "demo_mode": opay_client.demo_mode
                    },
                    "palmpay": {
                        "configured": not palmpay_client.demo_mode,
                        "status": "connected" if palmpay_balance.get('responseCode') == '00' else "error",
                        "demo_mode": palmpay_client.demo_mode
                    }
                },
                "earning_platforms": {
                    "swagbucks": {
                        "configured": not swagbucks_client.demo_mode,
                        "status": "connected" if swagbucks_balance.get('status') == 'success' else "error",
                        "demo_mode": swagbucks_client.demo_mode
                    },
                    "youtube": {
                        "configured": not youtube_client.demo_mode,
                        "status": "connected" if youtube_analytics.get('kind') else "error",
                        "demo_mode": youtube_client.demo_mode
                    }
                },
                "browser_automation": automation_stats
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def _process_survey_task(survey_id: str, answers: Dict[str, Any]):
    """Background task to process survey completion"""
    try:
        result = swagbucks_client.complete_survey(survey_id, answers)
        
        if result.get('status') == 'success':
            earned_usd = result.get('data', {}).get('usdValue', 0)
            
            # Update orchestrator stats
            live_orchestrator.stats['total_earned_usd'] += earned_usd
            live_orchestrator.stats['successful_surveys'] += 1
            
            logger.info(f"Survey {survey_id} completed: ${earned_usd:.2f} earned")
        else:
            live_orchestrator.stats['failed_operations'] += 1
            logger.error(f"Survey {survey_id} failed: {result.get('message')}")
            
    except Exception as e:
        logger.error(f"Survey task processing error: {e}")
        live_orchestrator.stats['failed_operations'] += 1

async def _process_video_task(video_url: str, watch_duration: Optional[int]):
    """Background task to process video watching"""
    try:
        result = browser_automation.watch_video_automation(video_url, watch_duration)
        
        if result.get('success'):
            # Estimate earnings (would be more sophisticated in production)
            watch_time_minutes = result.get('actual_watch_time', 0) / 60
            estimated_earnings = watch_time_minutes * 0.02  # $0.02 per minute estimate
            
            # Update orchestrator stats
            live_orchestrator.stats['total_earned_usd'] += estimated_earnings
            live_orchestrator.stats['successful_videos'] += 1
            
            logger.info(f"Video watched: {watch_time_minutes:.1f} minutes, ${estimated_earnings:.2f} earned")
        else:
            live_orchestrator.stats['failed_operations'] += 1
            logger.error(f"Video task failed: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Video task processing error: {e}")
        live_orchestrator.stats['failed_operations'] += 1
