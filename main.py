#!/usr/bin/env python3
"""
FastAPI main entry point for BRAF application
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Add the monetization-system directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'monetization-system'))

# Create FastAPI app
app = FastAPI(
    title="BRAF - Browser Automation Framework",
    description="Modern async browser automation and monetization system",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "BRAF"}

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "BRAF - Browser Automation Framework",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

# Try to import and mount additional apps
try:
    # Import maxelpay webhook FastAPI app
    from maxel_webhook_fastapi import app as webhook_app
    app.mount("/webhook", webhook_app)
except ImportError:
    pass

try:
    # Import monetization system if available
    from monetization_system.main import app as monetization_app
    app.mount("/monetization", monetization_app)
except ImportError:
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
