#!/usr/bin/env python3
"""
FastAPI-based Maxel webhook server with async support
Modern replacement for the Flask version
"""

from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
import logging
from datetime import datetime
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Maxel Webhook Server",
    description="Modern async webhook server for Maxel payment notifications",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security
security = HTTPBearer(auto_error=False)

# Configuration
MAXEL_SECRET = os.environ.get("MAXEL_SECRET", "default_secret")

# Pydantic models for request validation
class WebhookPayload(BaseModel):
    event_type: str = Field(..., description="Type of webhook event")
    payment_id: Optional[str] = Field(None, description="Payment identifier")
    withdrawal_id: Optional[str] = Field(None, description="Withdrawal identifier")
    amount: Optional[float] = Field(None, description="Transaction amount")
    currency: Optional[str] = Field(None, description="Currency code")
    user_id: Optional[str] = Field(None, description="User identifier")
    reason: Optional[str] = Field(None, description="Failure reason")
    destination: Optional[str] = Field(None, description="Withdrawal destination")
    timestamp: Optional[str] = Field(None, description="Event timestamp")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data")

class WebhookResponse(BaseModel):
    status: str
    timestamp: str
    message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

class InfoResponse(BaseModel):
    service: str
    status: str
    version: str
    endpoints: Dict[str, str]

# Authentication dependency
async def verify_maxel_secret(request: Request) -> bool:
    """Verify Maxel webhook secret from headers"""
    secret = request.headers.get("X-Maxel-Secret")
    if secret != MAXEL_SECRET:
        logger.warning(f"Unauthorized webhook attempt from {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Maxel-Secret header"
        )
    return True

# Webhook handlers
async def handle_payment_received(data: WebhookPayload) -> None:
    """Process payment received notification"""
    logger.info(f"Payment received: ID={data.payment_id}, Amount={data.amount} {data.currency}")
    
    # Add your async payment processing logic here
    # For example: await update_database(), await send_notification(), etc.

async def handle_payment_failed(data: WebhookPayload) -> None:
    """Process payment failed notification"""
    logger.warning(f"Payment failed: ID={data.payment_id}, Reason={data.reason}")
    
    # Add your async failure handling logic here
    # For example: await notify_user(), await retry_payment(), etc.

async def handle_withdrawal_completed(data: WebhookPayload) -> None:
    """Process withdrawal completed notification"""
    logger.info(f"Withdrawal completed: ID={data.withdrawal_id}, Amount={data.amount} {data.currency}")
    
    # Add your async withdrawal processing logic here
    # For example: await update_balance(), await send_confirmation(), etc.

# API Routes
@app.post("/webhook", response_model=WebhookResponse)
async def webhook(
    payload: WebhookPayload,
    request: Request,
    authenticated: bool = Depends(verify_maxel_secret)
):
    """Handle Maxel webhook notifications with async processing"""
    try:
        timestamp = datetime.now().isoformat()
        
        # Log webhook payload
        logger.info(f"Webhook received at {timestamp}: {payload.dict()}")
        
        # Handle different webhook event types asynchronously
        if payload.event_type == "payment_received":
            await handle_payment_received(payload)
        elif payload.event_type == "payment_failed":
            await handle_payment_failed(payload)
        elif payload.event_type == "withdrawal_completed":
            await handle_withdrawal_completed(payload)
        else:
            logger.info(f"Unknown event type: {payload.event_type}")
        
        return WebhookResponse(
            status="success",
            timestamp=timestamp,
            message=f"Processed {payload.event_type} event"
        )
        
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0"
    )

@app.get("/", response_model=InfoResponse)
async def root():
    """API information endpoint"""
    return InfoResponse(
        service="Maxel Webhook Server (FastAPI)",
        status="running",
        version="2.0.0",
        endpoints={
            "webhook": "/webhook (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)",
            "redoc": "/redoc (GET)"
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("🚀 Maxel FastAPI webhook server starting up...")
    logger.info(f"Secret configured: {'✅' if MAXEL_SECRET != 'default_secret' else '⚠️  Using default secret!'}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("🛑 Maxel FastAPI webhook server shutting down...")

if __name__ == "__main__":
    # Configuration from environment
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting Maxel FastAPI webhook server on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"API docs available at: http://{host}:{port}/docs")
    
    uvicorn.run(
        "maxel_webhook_fastapi:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )