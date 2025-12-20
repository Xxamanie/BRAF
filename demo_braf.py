#!/usr/bin/env python3
"""
BRAF Demo Script - Run BRAF without Docker dependencies
This demonstrates the core BRAF functionality using in-memory storage.
"""

import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def demo_task_execution():
    """Demonstrate BRAF task execution capabilities."""
    try:
        from braf.core.models import AutomationTask, AutomationAction, ActionType, TaskPriority
        from braf.core.task_executor import init_task_executor
        
        logger.info("🚀 Starting BRAF Demo")
        
        # Initialize components
        logger.info("📦 Initializing BRAF components...")
        task_executor = init_task_executor()
        
        # Create a simple web scraping task
        logger.info("📋 Creating automation task...")
        task = AutomationTask(
            id="demo_task_001",
            profile_id="demo_profile",
            actions=[
                # Navigate to a test page
                AutomationAction(
                    type=ActionType.NAVIGATE,
                    url="https://httpbin.org/html",
                    timeout=30
                ),
                # Wait for page load
                AutomationAction(
                    type=ActionType.WAIT,
                    data="2.0",
                    timeout=10
                ),
                # Extract page title
                AutomationAction(
                    type=ActionType.EXTRACT,
                    selector="h1",
                    timeout=10,
                    metadata={"attribute": "text"}
                ),
                # Take a screenshot
                AutomationAction(
                    type=ActionType.SCREENSHOT,
                    data="demo_screenshot.png",
                    timeout=10
                )
            ],
            priority=TaskPriority.NORMAL,
            timeout=300
        )
        
        # Execute the task
        logger.info("⚡ Executing automation task...")
        result = await task_executor.execute_task(task)
        
        # Display results
        if result.success:
            logger.info("✅ Task completed successfully!")
            logger.info(f"⏱️  Execution time: {result.execution_time:.2f} seconds")
            logger.info(f"📊 Actions completed: {result.actions_completed}")
            if result.extracted_data:
                logger.info(f"📄 Extracted data: {result.extracted_data}")
        else:
            logger.error(f"❌ Task failed: {result.error}")
            
        logger.info("🧹 Task execution completed")
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("💡 Make sure BRAF is properly installed: pip install -e .")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        logger.exception("Full error details:")

async def demo_behavioral_engine():
    """Demonstrate BRAF behavioral simulation."""
    try:
        from braf.core.behavioral.mouse_movement import BezierMouseMovement
        from braf.core.behavioral.typing_simulation import HumanTyper
        from braf.core.behavioral.timing_delays import TimingDelays
        
        logger.info("🎭 Demonstrating behavioral simulation...")
        
        # Mouse movement simulation
        mouse = BezierMouseMovement()
        path = mouse.generate_bezier_path((100, 100), (500, 300))
        logger.info(f"🖱️  Generated mouse path with {len(path)} points")
        
        # Typing simulation
        typer = HumanTyper()
        typing_events = typer.generate_typing_events("Hello, BRAF!")
        logger.info(f"⌨️  Generated {len(typing_events)} typing events")
        
        # Timing delays
        timing = TimingDelays()
        delay = timing.human_delay()
        logger.info(f"⏰ Generated human-like delay: {delay:.3f} seconds")
        
        logger.info("✅ Behavioral simulation demo completed")
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
    except Exception as e:
        logger.error(f"❌ Behavioral demo failed: {e}")

async def demo_fingerprint_management():
    """Demonstrate BRAF fingerprint management."""
    try:
        from braf.core.fingerprint_store import FingerprintStore
        from braf.core.models import BrowserFingerprint
        
        logger.info("🔐 Demonstrating fingerprint management...")
        
        # Initialize fingerprint store
        fingerprint_store = FingerprintStore()
        
        # Generate a fingerprint
        fingerprint = fingerprint_store.generate_fingerprint()
        logger.info(f"🌐 User agent: {fingerprint.user_agent[:50]}...")
        logger.info(f"📱 Screen resolution: {fingerprint.screen_resolution}")
        logger.info(f"🌍 Timezone: {fingerprint.timezone}")
        logger.info(f"🎨 Canvas fingerprint: {fingerprint.canvas_fingerprint[:20]}...")
        
        logger.info("✅ Fingerprint management demo completed")
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
    except Exception as e:
        logger.error(f"❌ Fingerprint demo failed: {e}")

async def main():
    """Run all BRAF demos."""
    logger.info("🎯 BRAF Framework Demo")
    logger.info("=" * 50)
    
    # Run demos
    await demo_behavioral_engine()
    print()
    
    await demo_fingerprint_management()
    print()
    
    await demo_task_execution()
    
    logger.info("=" * 50)
    logger.info("🎉 BRAF Demo Complete!")
    logger.info("📚 Check USAGE_GUIDE.md for more examples")
    logger.info("🐳 For full deployment, install Docker and run: ./scripts/deploy-dev.sh")

if __name__ == "__main__":
    asyncio.run(main())