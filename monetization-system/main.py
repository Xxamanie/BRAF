from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import uvicorn
import logging
import os

from config import Config
from api.routes import subscription, withdrawal, dashboard
from templates.survey_automation import SurveyAutomation
from templates.video_monetization import VideoPlatformAutomation
from payments.mobile_money import MobileMoneyWithdrawal
from payments.crypto_withdrawal import CryptoWithdrawal
from compliance.automation_checker import ComplianceChecker
from security.authentication import SecurityManager
from database import get_db, engine
from database.models import Base

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="BRAF Monetization System",
    version="1.0.0",
    description="Enterprise Browser Automation Revenue Framework with Monetization",
    docs_url="/docs" if Config.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if Config.ENVIRONMENT == "development" else None
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if Config.ENVIRONMENT == "development" else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Create database tables
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    try:
        # Create database tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Initialize services
        global survey_automation, video_automation, mobile_money, crypto_withdrawal, compliance, security_manager
        survey_automation = SurveyAutomation()
        video_automation = VideoPlatformAutomation()
        mobile_money = MobileMoneyWithdrawal()
        crypto_withdrawal = CryptoWithdrawal()
        compliance = ComplianceChecker()
        security_manager = SecurityManager()
        
        logger.info("BRAF Monetization System started successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

# Include routers
from api.routes import automation, enterprise

app.include_router(subscription.router, tags=["Subscriptions"])
app.include_router(withdrawal.router, tags=["Withdrawals"])
app.include_router(dashboard.router, tags=["Dashboard"])
app.include_router(automation.router, tags=["Automation"])
app.include_router(enterprise.router, tags=["Enterprise"])

# Include live operations router
try:
    from api.routes import live_operations
    app.include_router(live_operations.router, tags=["Live Operations"])
    logger.info("Live operations routes loaded successfully")
except ImportError as e:
    logger.warning(f"Live operations routes not available: {e}")

# Include intelligence router
try:
    from api.routes import intelligence
    app.include_router(intelligence.router, tags=["Intelligence System"])
    logger.info("Intelligence system routes loaded successfully")
except ImportError as e:
    logger.warning(f"Intelligence system routes not available: {e}")

# Static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass  # Static directory might not exist

# Web interface routes
@app.get("/", response_class=HTMLResponse, tags=["Web"])
async def root():
    """Root endpoint - redirect to register page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BRAF Monetization System</title>
        <meta http-equiv="refresh" content="0; url=/register">
    </head>
    <body>
        <p>Redirecting to registration page...</p>
        <p><a href="/register">Click here if not redirected automatically</a></p>
    </body>
    </html>
    """

@app.get("/register", response_class=HTMLResponse, tags=["Web"])
async def register_page():
    """Registration page"""
    try:
        with open("templates/register.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Registration page not found</h1>", status_code=404)

@app.get("/login", response_class=HTMLResponse, tags=["Web"])
async def login_page():
    """Login page"""
    try:
        with open("templates/login.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Login page not found</h1>", status_code=404)

@app.get("/dashboard", response_class=HTMLResponse, tags=["Web"])
async def dashboard_page():
    """Dashboard page"""
    try:
        with open("templates/dashboard.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Dashboard page not found</h1>", status_code=404)

@app.get("/create-automation", response_class=HTMLResponse, tags=["Web"])
async def create_automation_page():
    """Create automation page"""
    try:
        with open("templates/create_automation.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Create automation page not found</h1>", status_code=404)

@app.get("/request-withdrawal", response_class=HTMLResponse, tags=["Web"])
async def request_withdrawal_page():
    """Request withdrawal page"""
    try:
        with open("templates/request_withdrawal.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>Request withdrawal page not found</h1>", status_code=404)

# API status endpoint
@app.get("/api/status", tags=["System"])
async def api_status():
    """API status endpoint with system information"""
    return {
        "message": "BRAF Monetization System API",
        "status": "operational",
        "version": "1.0.0",
        "environment": Config.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat()
    }

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check database connection
        db = next(get_db())
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_status = "unhealthy"
    
    # Check Redis connection (if available)
    redis_status = "healthy"  # Implement Redis health check if needed
    
    health_status = {
        "status": "healthy" if db_status == "healthy" else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": db_status,
            "redis": redis_status,
            "subscription": "active",
            "withdrawal": "active",
            "automation": "active",
            "compliance": "active"
        },
        "version": "1.0.0",
        "environment": Config.ENVIRONMENT
    }
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)

# Metrics endpoint for Prometheus
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    # Implement Prometheus metrics collection
    return {"message": "Metrics endpoint - implement Prometheus metrics here"}

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Initialize global services
survey_automation = None
video_automation = None
mobile_money = None
crypto_withdrawal = None
compliance = None
security_manager = None

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 4))
    
    # SSL configuration for production
    ssl_keyfile = os.getenv("SSL_KEY_PATH")
    ssl_certfile = os.getenv("SSL_CERT_PATH")
    
    uvicorn_config = {
        "app": "main:app",
        "host": host,
        "port": port,
        "workers": workers if Config.ENVIRONMENT == "production" else 1,
        "reload": Config.ENVIRONMENT == "development",
        "log_level": "info",
        "access_log": True
    }
    
    # Add SSL configuration if certificates are available
    if ssl_keyfile and ssl_certfile and os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile):
        uvicorn_config.update({
            "ssl_keyfile": ssl_keyfile,
            "ssl_certfile": ssl_certfile
        })
        logger.info("SSL enabled")
    
    logger.info(f"Starting BRAF Monetization System on {host}:{port}")
    uvicorn.run(**uvicorn_config)
