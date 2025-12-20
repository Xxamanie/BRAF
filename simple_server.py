#!/usr/bin/env python3
"""
Simple BRAF Server - Basic FastAPI server for BRAF without complex dependencies
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="BRAF - Browser Automation Framework",
    description="Distributed browser automation with ethical constraints",
    version="0.1.0"
)

# In-memory storage for demo
tasks_db: Dict[str, Dict] = {}
system_stats = {
    "tasks_submitted": 0,
    "tasks_completed": 0,
    "workers_active": 0,
    "uptime_start": datetime.now()
}

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page."""
    uptime = datetime.now() - system_stats["uptime_start"]
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BRAF Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }}
            .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .stat-value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
            .stat-label {{ color: #7f8c8d; margin-top: 5px; }}
            .section {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
            .status-good {{ color: #27ae60; }}
            .status-warning {{ color: #f39c12; }}
            .api-link {{ color: #3498db; text-decoration: none; }}
            .api-link:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 BRAF - Browser Automation Framework</h1>
                <p>Distributed browser automation with ethical constraints and human-like behavior</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{system_stats['tasks_submitted']}</div>
                    <div class="stat-label">Tasks Submitted</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{system_stats['tasks_completed']}</div>
                    <div class="stat-label">Tasks Completed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{system_stats['workers_active']}</div>
                    <div class="stat-label">Active Workers</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{str(uptime).split('.')[0]}</div>
                    <div class="stat-label">Uptime</div>
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 System Status</h2>
                <p><span class="status-good">✅ BRAF Core:</span> Running</p>
                <p><span class="status-good">✅ Task Executor:</span> Ready</p>
                <p><span class="status-good">✅ API Server:</span> Online</p>
                <p><span class="status-warning">⚠️ Database:</span> In-Memory Mode (Demo)</p>
                <p><span class="status-warning">⚠️ Workers:</span> Not Connected (Demo Mode)</p>
            </div>
            
            <div class="section">
                <h2>📚 API Documentation</h2>
                <p>Explore the BRAF API:</p>
                <ul>
                    <li><a href="/docs" class="api-link">📖 Interactive API Docs (Swagger)</a></li>
                    <li><a href="/redoc" class="api-link">📋 ReDoc Documentation</a></li>
                    <li><a href="/health" class="api-link">🏥 Health Check</a></li>
                    <li><a href="/tasks" class="api-link">📋 Tasks API</a></li>
                </ul>
            </div>
            
            <div class="section">
                <h2>🚀 Quick Start</h2>
                <p>Submit a task via API:</p>
                <pre style="background: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto;">
curl -X POST "http://localhost:8000/tasks" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "id": "demo_task",
       "profile_id": "demo_profile", 
       "actions": [
         {{"type": "navigate", "url": "https://httpbin.org/html", "timeout": 30}},
         {{"type": "extract", "selector": "h1", "timeout": 10}}
       ]
     }}'</pre>
            </div>
            
            <div class="section">
                <h2>📊 Features</h2>
                <ul>
                    <li>🤖 <strong>Human-like Automation:</strong> Realistic mouse movements and typing patterns</li>
                    <li>🔒 <strong>Ethical Constraints:</strong> Built-in rate limiting and compliance monitoring</li>
                    <li>🎭 <strong>Fingerprint Management:</strong> Advanced browser fingerprinting and rotation</li>
                    <li>🌐 <strong>Proxy Integration:</strong> Residential proxy rotation with health monitoring</li>
                    <li>🧩 <strong>CAPTCHA Solving:</strong> Multi-tier CAPTCHA resolution with fallbacks</li>
                    <li>📈 <strong>Monitoring:</strong> Comprehensive metrics and observability</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
        "components": {
            "api_server": "online",
            "task_executor": "ready",
            "database": "in_memory_demo",
            "workers": "demo_mode"
        }
    }

@app.get("/tasks")
async def list_tasks():
    """List all tasks."""
    return {
        "tasks": list(tasks_db.values()),
        "total": len(tasks_db),
        "stats": system_stats
    }

@app.post("/tasks")
async def submit_task(task_data: dict):
    """Submit a new automation task."""
    try:
        # Basic validation
        required_fields = ["id", "actions"]
        for field in required_fields:
            if field not in task_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        task_id = task_data["id"]
        
        # Store task
        task_record = {
            "id": task_id,
            "status": "submitted",
            "submitted_at": datetime.now().isoformat(),
            "data": task_data
        }
        
        tasks_db[task_id] = task_record
        system_stats["tasks_submitted"] += 1
        
        logger.info(f"📋 Task submitted: {task_id}")
        
        return {
            "message": "Task submitted successfully",
            "task_id": task_id,
            "status": "submitted"
        }
        
    except Exception as e:
        logger.error(f"❌ Task submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task details."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return tasks_db[task_id]

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    uptime = datetime.now() - system_stats["uptime_start"]
    
    return {
        **system_stats,
        "uptime_seconds": uptime.total_seconds(),
        "uptime_formatted": str(uptime).split('.')[0],
        "tasks_in_queue": len(tasks_db)
    }

async def main():
    """Run the server."""
    logger.info("🚀 Starting BRAF Simple Server...")
    logger.info("📊 Dashboard: http://localhost:8000")
    logger.info("📚 API Docs: http://localhost:8000/docs")
    logger.info("🏥 Health: http://localhost:8000/health")
    
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())