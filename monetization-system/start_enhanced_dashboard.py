#!/usr/bin/env python3
"""
Enhanced Dashboard Server for BRAF System
Serves the enhanced dashboard with real-time functionality
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import API routes
from api.routes.dashboard import router as dashboard_router
from api.routes.enterprise import router as enterprise_router
from api.routes.automation import router as automation_router
from api.routes.withdrawal import router as withdrawal_router

app = FastAPI(
    title="BRAF Enhanced Dashboard",
    description="Advanced Browser Automation Revenue Framework - Enhanced Dashboard",
    version="2.0.0"
)

# Mount static files
static_dir = project_root / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include API routers
app.include_router(dashboard_router)
app.include_router(enterprise_router)
app.include_router(automation_router)
app.include_router(withdrawal_router)

@app.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Serve the enhanced dashboard"""
    dashboard_file = project_root / "templates" / "enhanced_dashboard.html"
    
    if dashboard_file.exists():
        with open(dashboard_file, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <html>
            <head><title>BRAF Enhanced Dashboard</title></head>
            <body>
                <h1>BRAF Enhanced Dashboard</h1>
                <p>Dashboard file not found. Please ensure enhanced_dashboard.html exists in templates/</p>
                <p>Available endpoints:</p>
                <ul>
                    <li><a href="/docs">API Documentation</a></li>
                    <li><a href="/api/v1/dashboard/realtime/1">Real-time Dashboard Data</a></li>
                </ul>
            </body>
        </html>
        """)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the enhanced dashboard"""
    return await dashboard_home()

@app.get("/enhanced", response_class=HTMLResponse)
async def enhanced_dashboard():
    """Serve the enhanced dashboard"""
    return await dashboard_home()

@app.get("/create-automation", response_class=HTMLResponse)
async def create_automation_page():
    """Serve automation creation page"""
    automation_file = project_root / "templates" / "create_automation.html"
    
    if automation_file.exists():
        with open(automation_file, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <html>
            <head><title>Create Automation</title></head>
            <body>
                <h1>Create Automation</h1>
                <p>Automation creation page coming soon...</p>
                <a href="/">Back to Dashboard</a>
            </body>
        </html>
        """)

@app.get("/request-withdrawal", response_class=HTMLResponse)
async def request_withdrawal_page():
    """Serve withdrawal request page"""
    withdrawal_file = project_root / "templates" / "request_withdrawal.html"
    
    if withdrawal_file.exists():
        with open(withdrawal_file, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <html>
            <head><title>Request Withdrawal</title></head>
            <body>
                <h1>Request Withdrawal</h1>
                <p>Withdrawal request page coming soon...</p>
                <a href="/">Back to Dashboard</a>
            </body>
        </html>
        """)

@app.get("/enhanced-withdrawal", response_class=HTMLResponse)
async def enhanced_withdrawal_page():
    """Serve enhanced withdrawal page"""
    withdrawal_file = project_root / "templates" / "enhanced_withdrawal.html"
    
    if withdrawal_file.exists():
        with open(withdrawal_file, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <html>
            <head><title>Enhanced Withdrawal</title></head>
            <body>
                <h1>Enhanced Withdrawal System</h1>
                <p>Enhanced withdrawal page not found.</p>
                <a href="/">Back to Dashboard</a>
            </body>
        </html>
        """)

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page():
    """Serve analytics page"""
    return HTMLResponse(content="""
    <html>
        <head>
            <title>BRAF Analytics</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .metric { background: #f5f5f5; padding: 20px; margin: 10px 0; border-radius: 8px; }
            </style>
        </head>
        <body>
            <h1>BRAF Advanced Analytics</h1>
            
            <div class="metric">
                <h3>Revenue Analytics</h3>
                <p>Total Revenue: $15,234.50</p>
                <p>Monthly Growth: +23.5%</p>
                <p>Best Performing Platform: Swagbucks (45% of revenue)</p>
            </div>
            
            <div class="metric">
                <h3>Performance Metrics</h3>
                <p>Success Rate: 94.5%</p>
                <p>Average Task Time: 45.2 seconds</p>
                <p>Operations Completed: 1,247</p>
            </div>
            
            <div class="metric">
                <h3>System Health</h3>
                <p>Uptime: 99.8%</p>
                <p>CPU Usage: 45%</p>
                <p>Memory Usage: 62%</p>
            </div>
            
            <div class="metric">
                <h3>Research Insights</h3>
                <p>NEXUS7 Operations: 156 completed</p>
                <p>Optimization Improvements: +18.5% efficiency</p>
                <p>Intelligence Gathering: 2.3TB data processed</p>
            </div>
            
            <a href="/">Back to Dashboard</a>
        </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "BRAF Enhanced Dashboard",
        "version": "2.0.0",
        "timestamp": "2024-12-18T10:30:00Z"
    }

if __name__ == "__main__":
    print("🚀 Starting BRAF Enhanced Dashboard Server...")
    print("📊 Dashboard URL: http://127.0.0.1:8004")
    print("📋 API Docs: http://127.0.0.1:8004/docs")
    print("🔧 Health Check: http://127.0.0.1:8004/health")
    print("\n✨ Enhanced Features:")
    print("   • Real-time dashboard updates")
    print("   • Interactive charts and analytics")
    print("   • Actionable operation controls")
    print("   • System health monitoring")
    print("   • Research operation management")
    print("   • Intelligence system integration")
    
    try:
        uvicorn.run(
            "start_enhanced_dashboard:app",
            host="127.0.0.1",
            port=8004,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Dashboard server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)