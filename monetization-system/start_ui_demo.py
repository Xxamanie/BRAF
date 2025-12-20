#!/usr/bin/env python3
"""
Start UI Demo Server
Simple server startup without complex dependencies
"""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from datetime import datetime

# Create FastAPI app
app = FastAPI(
    title="NEXUS7 Monetization System",
    description="Advanced Browser Automation and Monetization Platform",
    version="1.0.0"
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to dashboard"""
    return RedirectResponse(url="/dashboard")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard"""
    context = {
        "request": request,
        "title": "NEXUS7 Dashboard",
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system_status": "🟢 ONLINE",
        "total_earnings": "$12,847.50",
        "active_automations": 15,
        "success_rate": "94.2%",
        "recent_withdrawals": [
            {"amount": "$500.00", "method": "OPay", "status": "Completed", "date": "2025-12-17"},
            {"amount": "$750.00", "method": "PalmPay", "status": "Processing", "date": "2025-12-17"},
            {"amount": "$300.00", "method": "TON Crypto", "status": "Completed", "date": "2025-12-16"},
        ],
        "active_campaigns": [
            {"name": "Swagbucks Survey Campaign", "status": "Running", "earnings": "$2,340.00"},
            {"name": "YouTube Monetization", "status": "Running", "earnings": "$1,890.00"},
            {"name": "Cashback Optimization", "status": "Paused", "earnings": "$980.00"},
        ]
    }
    return templates.TemplateResponse("dashboard.html", context)

@app.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    """Login page"""
    context = {
        "request": request,
        "title": "Login - NEXUS7"
    }
    return templates.TemplateResponse("login.html", context)

@app.get("/register", response_class=HTMLResponse)
async def register(request: Request):
    """Registration page"""
    context = {
        "request": request,
        "title": "Register - NEXUS7"
    }
    return templates.TemplateResponse("register.html", context)

@app.get("/automations", response_class=HTMLResponse)
async def automations(request: Request):
    """Automations page"""
    context = {
        "request": request,
        "title": "Automations - NEXUS7",
        "automations": [
            {
                "id": 1,
                "name": "Swagbucks Survey Bot",
                "platform": "Swagbucks",
                "status": "Running",
                "earnings_today": "$145.50",
                "success_rate": "92%",
                "last_run": "2 minutes ago"
            },
            {
                "id": 2,
                "name": "YouTube Watch Time",
                "platform": "YouTube",
                "status": "Running", 
                "earnings_today": "$89.20",
                "success_rate": "96%",
                "last_run": "5 minutes ago"
            },
            {
                "id": 3,
                "name": "Cashback Optimizer",
                "platform": "Multiple",
                "status": "Paused",
                "earnings_today": "$67.80",
                "success_rate": "88%",
                "last_run": "1 hour ago"
            }
        ]
    }
    return templates.TemplateResponse("automations.html", context)

@app.get("/withdrawals", response_class=HTMLResponse)
async def withdrawals(request: Request):
    """Withdrawals page"""
    context = {
        "request": request,
        "title": "Withdrawals - NEXUS7",
        "balance": "$2,847.50",
        "pending_withdrawals": [
            {"id": 1, "amount": "$750.00", "method": "PalmPay", "account": "8161129466", "status": "Processing", "date": "2025-12-17"},
            {"id": 2, "amount": "$200.00", "method": "TON", "account": "UQBmMxSNU5PLmtib4xKsBH9zAg08681Tec0rcOHYB6F4vST7", "status": "Pending", "date": "2025-12-17"}
        ],
        "completed_withdrawals": [
            {"id": 3, "amount": "$500.00", "method": "OPay", "account": "8161129466", "status": "Completed", "date": "2025-12-17"},
            {"id": 4, "amount": "$300.00", "method": "TON", "account": "UQBmMxSNU5PLmtib4xKsBH9zAg08681Tec0rcOHYB6F4vST7", "status": "Completed", "date": "2025-12-16"},
            {"id": 5, "amount": "$1,200.00", "method": "PalmPay", "account": "8161129466", "status": "Completed", "date": "2025-12-15"}
        ]
    }
    return templates.TemplateResponse("withdrawals.html", context)

@app.get("/create-automation", response_class=HTMLResponse)
async def create_automation(request: Request):
    """Create automation page"""
    context = {
        "request": request,
        "title": "Create Automation - NEXUS7"
    }
    return templates.TemplateResponse("create_automation.html", context)

@app.get("/request-withdrawal", response_class=HTMLResponse)
async def request_withdrawal(request: Request):
    """Request withdrawal page"""
    context = {
        "request": request,
        "title": "Request Withdrawal - NEXUS7",
        "balance": "$2,847.50",
        "exchange_rate": "1,452.12"
    }
    return templates.TemplateResponse("request_withdrawal.html", context)

@app.get("/profile", response_class=HTMLResponse)
async def profile(request: Request):
    """Profile page"""
    context = {
        "request": request,
        "title": "Profile - NEXUS7",
        "user": {
            "name": "Demo User",
            "email": "demo@nexus7.com",
            "member_since": "2025-12-01",
            "total_earnings": "$12,847.50",
            "successful_operations": 1247,
            "success_rate": "94.2%"
        }
    }
    return templates.TemplateResponse("profile.html", context)

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {
        "status": "online",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "nexus7_active": True,
        "components": {
            "web_server": "online",
            "database": "online", 
            "automation_engine": "online",
            "payment_system": "demo_mode",
            "intelligence_system": "online"
        }
    }

@app.get("/api/v1/dashboard/stats")
async def dashboard_stats():
    """Dashboard statistics"""
    return {
        "total_earnings": 12847.50,
        "active_automations": 15,
        "success_rate": 94.2,
        "daily_earnings": 456.80,
        "weekly_earnings": 2847.50,
        "monthly_earnings": 12847.50
    }

if __name__ == "__main__":
    print("🚀 NEXUS7 UI Demo Server")
    print("=" * 40)
    print("🌐 Dashboard: http://localhost:8003/dashboard")
    print("🔐 Login: http://localhost:8003/login")
    print("📝 Register: http://localhost:8003/register")
    print("🤖 Automations: http://localhost:8003/automations")
    print("💰 Withdrawals: http://localhost:8003/withdrawals")
    print("👤 Profile: http://localhost:8003/profile")
    print("📊 API Status: http://localhost:8003/api/status")
    print("=" * 40)
    print("Press Ctrl+C to stop the server")
    print()
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8003,
        reload=False,
        log_level="info"
    )