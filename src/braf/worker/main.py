#!/usr/bin/env python3
"""
BRAF Worker Node Main Entry Point
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from braf.worker.worker_node import WorkerNode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main entry point for worker node."""
    logger.info("🤖 Starting BRAF Worker Node...")
    
    # Get configuration from environment
    worker_id = os.getenv("WORKER_ID", f"worker_{int(time.time())}")
    c2_endpoint = os.getenv("C2_ENDPOINT", "http://localhost:8000")
    max_tasks = int(os.getenv("MAX_CONCURRENT_TASKS", "2"))
    
    logger.info(f"👤 Worker ID: {worker_id}")
    logger.info(f"🌐 C2 Endpoint: {c2_endpoint}")
    logger.info(f"📊 Max concurrent tasks: {max_tasks}")
    
    # Create and start worker node
    try:
        worker = WorkerNode(
            worker_id=worker_id,
            c2_endpoint=c2_endpoint,
            max_concurrent_tasks=max_tasks
        )
        
        logger.info("✅ Worker node initialized")
        
        # Start the worker (this will run indefinitely)
        await worker.start()
        
    except Exception as e:
        logger.error(f"❌ Worker node failed to start: {e}")
        logger.exception("Full error details:")
        sys.exit(1)

def run_main():
    """Synchronous wrapper for main."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Worker node stopped by user")
    except Exception as e:
        logger.error(f"❌ Worker node crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_main()
