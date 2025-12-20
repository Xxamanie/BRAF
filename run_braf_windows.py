#!/usr/bin/env python3
"""
BRAF Windows Runner - Run BRAF without Docker on Windows
This starts the C2 server and demonstrates the framework functionality.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_c2_server():
    """Run the BRAF C2 server directly."""
    try:
        from braf.c2.dashboard import create_app
        import uvicorn
        
        logger.info("🚀 Starting BRAF C2 Server...")
        
        # Create FastAPI app
        app = create_app()
        
        logger.info("📊 C2 Dashboard will be available at: http://localhost:8000")
        logger.info("📚 API Documentation at: http://localhost:8000/docs")
        
        # Run the server
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("💡 Make sure all dependencies are installed")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")

def run_simple_example():
    """Run a simple BRAF example without external dependencies."""
    try:
        from braf.core.models import AutomationTask, AutomationAction, ActionType, TaskPriority
        from braf.core.task_executor import init_task_executor
        
        logger.info("🎯 Running Simple BRAF Example")
        
        # Create task executor
        executor = init_task_executor()
        
        # Create a simple task
        task = AutomationTask(
            id="windows_demo_task",
            profile_id="windows_profile",
            actions=[
                AutomationAction(
                    type=ActionType.NAVIGATE,
                    url="https://httpbin.org/html",
                    timeout=30
                ),
                AutomationAction(
                    type=ActionType.EXTRACT,
                    selector="h1",
                    timeout=10,
                    metadata={"attribute": "text"}
                )
            ],
            priority=TaskPriority.NORMAL,
            timeout=300
        )
        
        logger.info(f"✅ Task created: {task.id}")
        logger.info(f"📋 Actions: {len(task.actions)}")
        logger.info(f"🎯 Target URL: {task.actions[0].url}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        return False

async def main():
    """Main entry point."""
    logger.info("🎯 BRAF Windows Runner")
    logger.info("=" * 50)
    
    # Check if we should run the server or just examples
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        logger.info("🌐 Starting C2 Server Mode...")
        await run_c2_server()
    else:
        logger.info("📋 Running Example Mode...")
        
        # Run simple example
        success = run_simple_example()
        
        if success:
            logger.info("=" * 50)
            logger.info("🎉 BRAF is working on Windows!")
            logger.info("")
            logger.info("🚀 Next Steps:")
            logger.info("   1. Install Docker Desktop for full deployment")
            logger.info("   2. Run: python run_braf_windows.py --server")
            logger.info("   3. Or use the deployment script: ./scripts/deploy-dev.sh")
            logger.info("")
            logger.info("📚 Usage Examples:")
            logger.info("   • Check USAGE_GUIDE.md for complete examples")
            logger.info("   • Run the examples from the usage guide")
            logger.info("   • Use the REST API once server is running")
        else:
            logger.error("❌ BRAF example failed. Check the error messages above.")

if __name__ == "__main__":
    asyncio.run(main())