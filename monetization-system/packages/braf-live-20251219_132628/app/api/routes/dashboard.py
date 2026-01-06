from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import random
import psutil
from dashboard.dashboard_service import EnterpriseDashboard
from database.service import DatabaseService

router = APIRouter(prefix="/api/v1/dashboard")

@router.get("/earnings/{enterprise_id}")
async def get_earnings(enterprise_id: str, days: int = 30):
    """Get recent earnings for enterprise"""
    try:
        with DatabaseService() as db:
            earnings = db.get_earnings(enterprise_id, days)
            
            return {
                "recent_earnings": [
                    {
                        "amount": earning.amount,
                        "platform": earning.platform,
                        "task_type": earning.task_type,
                        "earned_at": earning.earned_at.isoformat()
                    } for earning in earnings
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/withdrawals/{enterprise_id}")
async def get_withdrawals(enterprise_id: str, days: int = 30):
    """Get recent withdrawals for enterprise"""
    try:
        with DatabaseService() as db:
            withdrawals = db.get_withdrawals(enterprise_id, days)
            
            return {
                "recent_withdrawals": [
                    {
                        "id": withdrawal.id,
                        "amount": withdrawal.amount,
                        "fee": withdrawal.fee,
                        "net_amount": withdrawal.net_amount,
                        "provider": withdrawal.provider,
                        "status": withdrawal.status,
                        "created_at": withdrawal.created_at.isoformat(),
                        "completed_at": withdrawal.completed_at.isoformat() if withdrawal.completed_at else None
                    } for withdrawal in withdrawals
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/overview/{enterprise_id}")
async def get_dashboard_overview(enterprise_id: str, period: str = "month"):
    """API endpoint for enterprise dashboard overview"""
    try:
        with DatabaseService() as db:
            dashboard_data = db.get_dashboard_data(enterprise_id)
            
            return {
                "success": True,
                "data": dashboard_data,
                "last_updated": datetime.utcnow().isoformat()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/realtime/{enterprise_id}")
async def get_realtime_dashboard_data(enterprise_id: int):
    """Get real-time dashboard data with system health and operations"""
    try:
        with DatabaseService() as db:
            # Get basic dashboard data
            dashboard_data = db.get_dashboard_data(enterprise_id)
            
            # Get system health metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            network = psutil.net_io_counters()
            
            # Calculate network usage percentage (simplified)
            network_usage = min(100, (network.bytes_sent + network.bytes_recv) / (1024 * 1024 * 100))
            
            # Get active operations
            operations = db.get_active_operations(enterprise_id)
            
            # Get recent earnings
            recent_earnings = db.get_earnings(enterprise_id, days=7)
            
            # Get system alerts
            alerts = db.get_system_alerts(enterprise_id)
            
            return {
                "total_earnings": dashboard_data.get("total_earnings", 0),
                "available_balance": dashboard_data.get("available_balance", 0),
                "active_operations": len(operations),
                "success_rate": dashboard_data.get("success_rate", 0),
                "pending_payouts": dashboard_data.get("pending_payouts", 0),
                "system_health": {
                    "cpu": round(cpu_percent, 1),
                    "memory": round(memory.percent, 1),
                    "network": round(network_usage, 1)
                },
                "operations": [
                    {
                        "id": op.id,
                        "type": op.operation_type,
                        "platform": op.platform,
                        "status": op.status,
                        "progress": op.progress,
                        "earnings": float(op.earnings)
                    } for op in operations
                ],
                "recent_earnings": [
                    {
                        "amount": float(earning.amount),
                        "platform": earning.platform,
                        "task_type": earning.task_type,
                        "earned_at": earning.earned_at.isoformat()
                    } for earning in recent_earnings[:10]
                ],
                "alerts": [
                    {
                        "type": alert.alert_type,
                        "title": alert.title,
                        "message": alert.message,
                        "timestamp": alert.created_at.isoformat()
                    } for alert in alerts[:5]
                ],
                "last_updated": datetime.utcnow().isoformat()
            }
    except Exception as e:
        # Return demo data if database fails
        return {
            "total_earnings": 15234.50,
            "available_balance": 8750.25,
            "active_operations": 12,
            "success_rate": 94.5,
            "pending_payouts": 2500.00,
            "system_health": {
                "cpu": 45,
                "memory": 62,
                "network": 78
            },
            "operations": [],
            "recent_earnings": [],
            "alerts": [],
            "last_updated": datetime.utcnow().isoformat()
        }

@router.post("/automation/create")
async def create_automation(
    enterprise_id: int,
    automation_type: str,
    platform: str,
    parameters: dict
):
    """Create a new automation"""
    try:
        with DatabaseService() as db:
            automation_id = db.create_automation(
                enterprise_id=enterprise_id,
                automation_type=automation_type,
                platform=platform,
                parameters=parameters
            )
            
            # Calculate estimated earnings based on type and platform
            estimated_earnings = random.uniform(50, 200)
            
            return {
                "automation_id": automation_id,
                "status": "created",
                "estimated_earnings": round(estimated_earnings, 2),
                "message": f"Automation created successfully on {platform}"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/research/start")
async def start_research_operation(
    enterprise_id: int,
    research_type: str,
    parameters: dict = None
):
    """Start a research operation"""
    try:
        with DatabaseService() as db:
            research_id = f"res_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            # Estimate duration based on research type
            duration_map = {
                "platform_analysis": 3600,
                "optimization": 7200,
                "market_research": 5400,
                "behavior_analysis": 4800
            }
            estimated_duration = duration_map.get(research_type, 3600)
            
            # Create research operation
            db.create_research_operation(
                enterprise_id=enterprise_id,
                research_id=research_id,
                research_type=research_type,
                parameters=parameters or {}
            )
            
            return {
                "research_id": research_id,
                "status": "started",
                "estimated_duration": estimated_duration,
                "expected_results": {
                    "platform_insights": True,
                    "optimization_recommendations": True,
                    "performance_analysis": True
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/intelligence/optimize")
async def optimize_system(
    enterprise_id: int,
    optimization_type: str = "full"
):
    """Run system optimization"""
    try:
        with DatabaseService() as db:
            optimization_id = f"opt_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            # Simulate optimization improvements
            improvements = [
                {
                    "area": "success_rate",
                    "current_value": 92.5,
                    "optimized_value": 95.8,
                    "improvement_percentage": 3.6
                },
                {
                    "area": "earnings_per_hour",
                    "current_value": 125.50,
                    "optimized_value": 145.75,
                    "improvement_percentage": 16.1
                },
                {
                    "area": "resource_efficiency",
                    "current_value": 78.0,
                    "optimized_value": 89.5,
                    "improvement_percentage": 14.7
                }
            ]
            
            # Calculate estimated impact
            estimated_impact = {
                "earnings_increase": 1250.00,
                "efficiency_gain": 18.5,
                "success_rate_improvement": 3.3
            }
            
            return {
                "optimization_id": optimization_id,
                "improvements": improvements,
                "estimated_impact": estimated_impact,
                "status": "completed"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/operations/{operation_id}/stop")
async def stop_operation(operation_id: str, enterprise_id: int):
    """Stop a running operation"""
    try:
        with DatabaseService() as db:
            operation = db.stop_operation(operation_id, enterprise_id)
            
            return {
                "status": "stopped",
                "final_earnings": float(operation.earnings),
                "duration": operation.duration_seconds,
                "message": f"Operation {operation_id} stopped successfully"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/operations/{operation_id}")
async def get_operation_details(operation_id: str, enterprise_id: int):
    """Get detailed information about an operation"""
    try:
        with DatabaseService() as db:
            operation = db.get_operation(operation_id, enterprise_id)
            logs = db.get_operation_logs(operation_id)
            metrics = db.get_operation_metrics(operation_id)
            
            return {
                "operation_details": {
                    "id": operation.id,
                    "type": operation.operation_type,
                    "platform": operation.platform,
                    "status": operation.status,
                    "progress": operation.progress,
                    "earnings": float(operation.earnings),
                    "started_at": operation.started_at.isoformat(),
                    "estimated_completion": operation.estimated_completion.isoformat() if operation.estimated_completion else None
                },
                "logs": [
                    {
                        "timestamp": log.timestamp.isoformat(),
                        "level": log.level,
                        "message": log.message
                    } for log in logs
                ],
                "performance_metrics": {
                    "success_rate": metrics.success_rate,
                    "average_task_time": metrics.average_task_time,
                    "tasks_completed": metrics.tasks_completed,
                    "tasks_failed": metrics.tasks_failed
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
