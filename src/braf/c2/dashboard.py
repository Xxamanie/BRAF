"""
C2 Dashboard API for BRAF using FastAPI.

This module provides a web interface for system management,
worker monitoring, analytics, and task submission.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from braf.core.models import (
    AutomationTask, AutomationAction, TaskStatus, TaskPriority,
    WorkerStatus, QueueMetrics, AnalyticsReport, ComplianceViolation
)
from braf.core.job_scheduler import get_job_scheduler
from braf.core.compliance_logger import get_compliance_logger
from braf.core.database import get_database

logger = logging.getLogger(__name__)


# Request/Response Models
class TaskSubmissionRequest(BaseModel):
    """Request model for task submission."""
    
    profile_id: str
    actions: List[Dict[str, Any]]
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskSubmissionResponse(BaseModel):
    """Response model for task submission."""
    
    job_id: str
    task_id: str
    status: str
    message: str


class WorkerStatusResponse(BaseModel):
    """Response model for worker status."""
    
    workers: List[WorkerStatus]
    total_workers: int
    active_workers: int
    total_capacity: int
    current_load: int


class SystemHealthResponse(BaseModel):
    """Response model for system health."""
    
    status: str
    components: Dict[str, str]
    uptime: float
    version: str
    timestamp: datetime


class AnalyticsRequest(BaseModel):
    """Request model for analytics."""
    
    start_date: datetime
    end_date: datetime
    metrics: Optional[List[str]] = None


# WebSocket Connection Manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected WebSockets."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)


# Create FastAPI app
app = FastAPI(
    title="BRAF C2 Dashboard",
    description="Command and Control Dashboard for Browser Automation Framework",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket manager
manager = ConnectionManager()

# System startup time
startup_time = datetime.now(timezone.utc)


# Dependency functions
async def get_scheduler():
    """Get job scheduler dependency."""
    scheduler = get_job_scheduler()
    if not scheduler:
        raise HTTPException(status_code=503, detail="Job scheduler not available")
    return scheduler


async def get_compliance():
    """Get compliance logger dependency."""
    compliance = get_compliance_logger()
    if not compliance:
        raise HTTPException(status_code=503, detail="Compliance logger not available")
    return compliance


# API Routes

@app.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Serve dashboard home page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BRAF C2 Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
            .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .status-good { color: green; }
            .status-warning { color: orange; }
            .status-error { color: red; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>BRAF Command & Control Dashboard</h1>
            <p>Browser Automation Framework Management Interface</p>
        </div>
        
        <div class="section">
            <h2>Quick Links</h2>
            <ul>
                <li><a href="/api/docs">API Documentation</a></li>
                <li><a href="/api/health">System Health</a></li>
                <li><a href="/api/workers/status">Worker Status</a></li>
                <li><a href="/api/metrics/queue">Queue Metrics</a></li>
            </ul>
        </div>
        
        <div class="section">
            <h2>Real-time Status</h2>
            <div id="status">Loading...</div>
        </div>
        
        <script>
            // Simple status update
            async function updateStatus() {
                try {
                    const response = await fetch('/api/health');
                    const health = await response.json();
                    document.getElementById('status').innerHTML = 
                        `<span class="status-${health.status === 'healthy' ? 'good' : 'error'}">
                            System Status: ${health.status}
                        </span><br>
                        Uptime: ${Math.round(health.uptime)} seconds<br>
                        Version: ${health.version}`;
                } catch (error) {
                    document.getElementById('status').innerHTML = 
                        '<span class="status-error">Error loading status</span>';
                }
            }
            
            updateStatus();
            setInterval(updateStatus, 5000);
        </script>
    </body>
    </html>
    """


@app.get("/api/health", response_model=SystemHealthResponse)
async def get_system_health():
    """Get system health status."""
    scheduler = get_job_scheduler()
    compliance = get_compliance_logger()
    
    components = {
        "job_scheduler": "healthy" if scheduler else "unavailable",
        "compliance_logger": "healthy" if compliance else "unavailable",
        "database": "healthy",  # Would check actual DB connection
        "redis": "healthy"      # Would check actual Redis connection
    }
    
    # Determine overall status
    if all(status == "healthy" for status in components.values()):
        overall_status = "healthy"
    elif any(status == "unavailable" for status in components.values()):
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    uptime = (datetime.now(timezone.utc) - startup_time).total_seconds()
    
    return SystemHealthResponse(
        status=overall_status,
        components=components,
        uptime=uptime,
        version="0.1.0",
        timestamp=datetime.now(timezone.utc)
    )


@app.get("/api/workers/status", response_model=WorkerStatusResponse)
async def get_workers_status(scheduler = Depends(get_scheduler)):
    """Get status of all workers."""
    workers = scheduler.get_worker_status()
    
    active_workers = len([w for w in workers if w.status == "online"])
    total_capacity = sum(w.max_tasks for w in workers)
    current_load = sum(w.current_tasks for w in workers)
    
    return WorkerStatusResponse(
        workers=workers,
        total_workers=len(workers),
        active_workers=active_workers,
        total_capacity=total_capacity,
        current_load=current_load
    )


@app.get("/api/metrics/queue", response_model=QueueMetrics)
async def get_queue_metrics(scheduler = Depends(get_scheduler)):
    """Get current queue metrics."""
    return await scheduler.get_queue_metrics()


@app.post("/api/tasks/submit", response_model=TaskSubmissionResponse)
async def submit_task(
    request: TaskSubmissionRequest,
    background_tasks: BackgroundTasks,
    scheduler = Depends(get_scheduler)
):
    """Submit new automation task."""
    try:
        # Create AutomationTask from request
        actions = [AutomationAction(**action_data) for action_data in request.actions]
        
        task = AutomationTask(
            profile_id=request.profile_id,
            actions=actions,
            priority=request.priority,
            timeout=request.timeout,
            metadata=request.metadata
        )
        
        # Submit to scheduler
        job_id = await scheduler.submit_task(task, request.priority)
        
        # Broadcast update to WebSocket clients
        background_tasks.add_task(
            manager.broadcast,
            f"New task submitted: {task.id} (Job: {job_id})"
        )
        
        return TaskSubmissionResponse(
            job_id=job_id,
            task_id=task.id,
            status="submitted",
            message="Task submitted successfully"
        )
    
    except Exception as e:
        logger.error(f"Task submission failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/tasks/{job_id}/status")
async def get_task_status(job_id: str, scheduler = Depends(get_scheduler)):
    """Get status of submitted task."""
    try:
        status = await scheduler.get_job_status(job_id)
        return status
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=404, detail="Task not found")


@app.delete("/api/tasks/{job_id}")
async def cancel_task(
    job_id: str,
    background_tasks: BackgroundTasks,
    scheduler = Depends(get_scheduler)
):
    """Cancel submitted task."""
    try:
        success = await scheduler.cancel_job(job_id)
        
        if success:
            background_tasks.add_task(
                manager.broadcast,
                f"Task cancelled: {job_id}"
            )
            return {"message": "Task cancelled successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to cancel task")
    
    except Exception as e:
        logger.error(f"Error cancelling task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/compliance/status")
async def get_compliance_status(compliance = Depends(get_compliance)):
    """Get compliance system status."""
    metrics = compliance.get_compliance_metrics()
    lockdown_status = await compliance.check_lockdown_status()
    
    return {
        "metrics": metrics,
        "lockdown": lockdown_status,
        "recent_violations": compliance.get_recent_violations(hours=24)
    }


@app.get("/api/compliance/violations")
async def get_compliance_violations(
    hours: int = 24,
    compliance = Depends(get_compliance)
):
    """Get recent compliance violations."""
    violations = compliance.get_recent_violations(hours=hours)
    return {"violations": violations, "count": len(violations)}


@app.post("/api/compliance/lockdown/release")
async def release_compliance_lockdown(
    admin_override: bool = False,
    compliance = Depends(get_compliance)
):
    """Release compliance lockdown."""
    try:
        success = await compliance.release_lockdown(admin_override=admin_override)
        
        if success:
            return {"message": "Lockdown released successfully"}
        else:
            raise HTTPException(status_code=400, detail="Cannot release lockdown at this time")
    
    except Exception as e:
        logger.error(f"Error releasing lockdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analytics/generate", response_model=AnalyticsReport)
async def generate_analytics(request: AnalyticsRequest):
    """Generate analytics report."""
    try:
        # Mock analytics generation
        # In real implementation, this would query the database
        
        duration = request.end_date - request.start_date
        
        report = AnalyticsReport(
            time_range_start=request.start_date,
            time_range_end=request.end_date,
            total_tasks=100,  # Mock data
            success_rate=0.85,
            average_execution_time=45.2,
            detection_rate=0.02,
            top_domains=[("example.com", 25), ("test.org", 15)],
            worker_performance={"worker_1": {"success_rate": 0.9, "avg_time": 42.1}},
            compliance_violations=3
        )
        
        return report
    
    except Exception as e:
        logger.error(f"Error generating analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        # Send initial status
        await manager.send_personal_message(
            "Connected to BRAF C2 Dashboard",
            websocket
        )
        
        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            
            # Echo back for now (could handle commands)
            await manager.send_personal_message(f"Echo: {data}", websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.get("/api/workers/{worker_id}/health")
async def check_worker_health(worker_id: str, scheduler = Depends(get_scheduler)):
    """Check health of specific worker."""
    workers = scheduler.get_worker_status()
    
    worker = next((w for w in workers if w.worker_id == worker_id), None)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    
    # Perform health check
    try:
        # This would send a health check task to the specific worker
        health_result = {
            "worker_id": worker_id,
            "status": worker.status,
            "last_heartbeat": worker.last_heartbeat,
            "current_tasks": worker.current_tasks,
            "health_check_time": datetime.now(timezone.utc)
        }
        
        return health_result
    
    except Exception as e:
        logger.error(f"Health check failed for worker {worker_id}: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/api/system/scaling")
async def trigger_scaling(
    action: str,  # "up" or "down"
    count: int = 1,
    scheduler = Depends(get_scheduler)
):
    """Trigger manual scaling of workers."""
    if action not in ["up", "down"]:
        raise HTTPException(status_code=400, detail="Action must be 'up' or 'down'")
    
    try:
        if action == "up":
            # Trigger scale up
            for _ in range(count):
                await scheduler._trigger_scale_up()
            message = f"Triggered scale up by {count} workers"
        else:
            # Trigger scale down
            for _ in range(count):
                await scheduler._trigger_scale_down()
            message = f"Triggered scale down by {count} workers"
        
        return {"message": message}
    
    except Exception as e:
        logger.error(f"Scaling operation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background tasks
async def broadcast_periodic_updates():
    """Send periodic updates to WebSocket clients."""
    while True:
        try:
            scheduler = get_job_scheduler()
            if scheduler:
                metrics = await scheduler.get_queue_metrics()
                
                update_message = f"Queue Update: {metrics.pending_tasks} pending, {metrics.running_tasks} running"
                await manager.broadcast(update_message)
            
            await asyncio.sleep(30)  # Update every 30 seconds
        
        except Exception as e:
            logger.error(f"Error in periodic updates: {e}")
            await asyncio.sleep(10)


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("BRAF C2 Dashboard starting up")
    
    # Start background tasks
    asyncio.create_task(broadcast_periodic_updates())
    
    logger.info("C2 Dashboard ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("BRAF C2 Dashboard shutting down")


# Development server function
def run_dashboard(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """Run the dashboard server."""
    uvicorn.run(
        "braf.c2.dashboard:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_dashboard(reload=True)