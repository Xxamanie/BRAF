#!/usr/bin/env python3
"""
BRAF C2 Server Main Entry Point
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from braf.c2.simple_dashboard import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for C2 server."""
    logger.info("🚀 Starting BRAF C2 Server...")
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    # Create FastAPI app
    app = create_app()
    
    logger.info(f"📊 C2 Server starting on {host}:{port}")
    logger.info(f"📚 API Documentation: http://{host}:{port}/docs")
    
    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True
    )

if __name__ == "__main__":
    main()
