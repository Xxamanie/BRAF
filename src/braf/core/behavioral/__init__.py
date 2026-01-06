"""
Behavioral simulation module for BRAF.

This module provides human-like behavioral patterns including mouse movements,
typing simulation, and timing delays for realistic web automation.
"""

from braf.core.behavioral.behavioral_engine import (
    BehavioralEngine,
    BehavioralPatternAnalyzer,
    get_behavioral_engine,
    init_behavioral_engine
)
from braf.core.behavioral.mouse_movement import (
    BezierMouseMovement,
    MouseMovementOptimizer,
    generate_human_mouse_movement,
    calculate_movement_metrics
)
from braf.core.behavioral.typing_simulation import (
    HumanTyper,
    TypingSession,
    simulate_human_typing,
    calculate_typing_metrics
)
from braf.core.behavioral.timing_delays import (
    DelayGenerator,
    ActivityScheduler,
    BehavioralTimingManager,
    get_timing_manager,
    get_human_delay
)

__all__ = [
    # Main engine
    "BehavioralEngine",
    "BehavioralPatternAnalyzer",
    "get_behavioral_engine",
    "init_behavioral_engine",
    
    # Mouse movement
    "BezierMouseMovement",
    "MouseMovementOptimizer", 
    "generate_human_mouse_movement",
    "calculate_movement_metrics",
    
    # Typing simulation
    "HumanTyper",
    "TypingSession",
    "simulate_human_typing",
    "calculate_typing_metrics",
    
    # Timing and delays
    "DelayGenerator",
    "ActivityScheduler",
    "BehavioralTimingManager",
    "get_timing_manager",
    "get_human_delay"
]
