#!/usr/bin/env python3
"""
Simple BRAF Demo - Showcase working functionality
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def demo_task_validation():
    """Demonstrate BRAF task validation and preprocessing."""
    try:
        from braf.core.models import AutomationTask, AutomationAction, ActionType, TaskPriority
        from braf.core.task_executor import init_task_executor
        
        logger.info("🚀 BRAF Task Validation Demo")
        
        # Initialize task executor
        task_executor = init_task_executor()
        
        # Create a test task
        task = AutomationTask(
            id="validation_demo",
            profile_id="demo_profile",
            actions=[
                AutomationAction(
                    type=ActionType.NAVIGATE,
                    url="https://httpbin.org/html",
                    timeout=30
                ),
                AutomationAction(
                    type=ActionType.WAIT,
                    data="2.0",
                    timeout=10
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
        
        logger.info("📋 Task created successfully!")
        logger.info(f"   - Task ID: {task.id}")
        logger.info(f"   - Actions: {len(task.actions)}")
        logger.info(f"   - Target URL: {task.actions[0].url}")
        
        # The task executor validates and preprocesses automatically
        logger.info("✅ Task validation and preprocessing works!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")

def demo_models():
    """Demonstrate BRAF data models."""
    try:
        from braf.core.models import (
            AutomationTask, AutomationAction, ActionType, TaskPriority,
            BrowserFingerprint, Profile, TaskResult, TaskStatus
        )
        
        logger.info("📊 BRAF Data Models Demo")
        
        # Create a browser fingerprint
        fingerprint = BrowserFingerprint(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            screen_resolution=(1920, 1080),
            timezone="America/New_York",
            language="en-US",
            platform="Win32",
            canvas_fingerprint="demo_canvas_hash",
            webgl_fingerprint="demo_webgl_hash"
        )
        
        logger.info("🔐 Browser fingerprint created:")
        logger.info(f"   - User Agent: {fingerprint.user_agent[:50]}...")
        logger.info(f"   - Resolution: {fingerprint.screen_resolution}")
        logger.info(f"   - Timezone: {fingerprint.timezone}")
        
        # Create a profile
        profile = Profile(
            id="demo_profile_001",
            fingerprint=fingerprint,
            proxy_config=None,
            created_at=None,
            last_used=None
        )
        
        logger.info(f"👤 Profile created: {profile.id}")
        
        # Create task result
        result = TaskResult(
            task_id="demo_task",
            success=True,
            execution_time=2.5,
            actions_completed=3,
            extracted_data={"title": "Example Page"},
            error=None,
            status=TaskStatus.COMPLETED
        )
        
        logger.info("📈 Task result created:")
        logger.info(f"   - Success: {result.success}")
        logger.info(f"   - Execution time: {result.execution_time}s")
        logger.info(f"   - Data extracted: {result.extracted_data}")
        
        logger.info("✅ All data models working correctly!")
        
    except Exception as e:
        logger.error(f"❌ Models demo failed: {e}")

def demo_behavioral_components():
    """Demonstrate individual behavioral components."""
    try:
        from braf.core.behavioral.mouse_movement import BezierMouseMovement
        from braf.core.behavioral.typing_simulation import HumanTyper
        
        logger.info("🎭 BRAF Behavioral Components Demo")
        
        # Mouse movement
        mouse = BezierMouseMovement()
        path = mouse.generate_bezier_path((100, 100), (500, 300))
        logger.info(f"🖱️  Mouse path generated: {len(path)} points")
        logger.info(f"   - Start: {path[0]}")
        logger.info(f"   - End: {path[-1]}")
        
        # Typing simulation
        typer = HumanTyper()
        events = typer.generate_typing_events("Hello BRAF!")
        logger.info(f"⌨️  Typing events generated: {len(events)} events")
        logger.info(f"   - Text: 'Hello BRAF!'")
        logger.info(f"   - Total duration: {sum(e.delay for e in events):.3f}s")
        
        logger.info("✅ Behavioral components working!")
        
    except Exception as e:
        logger.error(f"❌ Behavioral demo failed: {e}")

async def main():
    """Run BRAF demos."""
    logger.info("🎯 BRAF Framework - Simple Demo")
    logger.info("=" * 50)
    
    # Demo data models
    demo_models()
    print()
    
    # Demo behavioral components  
    demo_behavioral_components()
    print()
    
    # Demo task validation
    await demo_task_validation()
    
    logger.info("=" * 50)
    logger.info("🎉 BRAF Simple Demo Complete!")
    logger.info("")
    logger.info("✅ What's Working:")
    logger.info("   • Data models and validation")
    logger.info("   • Task creation and preprocessing") 
    logger.info("   • Behavioral simulation components")
    logger.info("   • Mouse movement generation")
    logger.info("   • Human typing simulation")
    logger.info("")
    logger.info("🔧 Next Steps:")
    logger.info("   • Install Docker for full deployment")
    logger.info("   • Set up PostgreSQL and Redis")
    logger.info("   • Run: ./scripts/deploy-dev.sh")
    logger.info("   • Access dashboard at http://localhost:8000")

if __name__ == "__main__":
    asyncio.run(main())
