"""
Main Behavioral Engine for BRAF human-like automation.

This module coordinates mouse movements, typing simulation, and timing delays
to create realistic human behavior patterns for web automation.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from braf.core.behavioral.mouse_movement import (
    BezierMouseMovement, 
    generate_human_mouse_movement,
    calculate_movement_metrics
)
from braf.core.behavioral.typing_simulation import (
    TypingSession,
    simulate_human_typing,
    calculate_typing_metrics
)
from braf.core.behavioral.timing_delays import (
    BehavioralTimingManager,
    DelayGenerator,
    ActivityScheduler,
    get_timing_manager
)
from braf.core.models import BehavioralConfig

logger = logging.getLogger(__name__)

# Type aliases
Point = Tuple[float, float]
TimedPoint = Tuple[float, float, float]


class BehavioralEngine:
    """
    Main engine for coordinating human-like behavioral patterns.
    
    This engine combines mouse movements, typing simulation, and timing delays
    to create realistic automation behavior that mimics human interactions.
    """
    
    def __init__(self, config: Optional[BehavioralConfig] = None):
        """
        Initialize behavioral engine.
        
        Args:
            config: Optional behavioral configuration
        """
        self.config = config or BehavioralConfig()
        
        # Initialize components
        self.mouse_generator = BezierMouseMovement(
            noise_factor=0.5,
            min_velocity=0.5,
            max_velocity=1.5
        )
        
        self.typing_session = TypingSession(config)
        self.timing_manager = get_timing_manager()
        
        # Behavioral state
        self.current_mouse_position = (0, 0)
        self.session_stats = {
            "actions_performed": 0,
            "mouse_movements": 0,
            "text_typed": 0,
            "total_delays": 0.0,
            "session_start": datetime.now()
        }
    
    async def move_mouse(
        self, 
        target: Point, 
        duration_hint: Optional[float] = None
    ) -> List[TimedPoint]:
        """
        Generate and execute human-like mouse movement.
        
        Args:
            target: Target coordinate (x, y)
            duration_hint: Optional hint for movement duration
            
        Returns:
            List of (x, y, timestamp) points for the movement path
        """
        start_pos = self.current_mouse_position
        
        # Generate movement path
        movement_path = generate_human_mouse_movement(
            start=start_pos,
            end=target,
            noise_factor=0.5,
            num_points=100
        )
        
        # Update current position
        self.current_mouse_position = target
        
        # Update stats
        self.session_stats["mouse_movements"] += 1
        self.session_stats["actions_performed"] += 1
        
        logger.debug(f"Generated mouse movement from {start_pos} to {target} "
                    f"with {len(movement_path)} points")
        
        return movement_path
    
    async def type_text(self, text: str) -> List[Tuple[str, float]]:
        """
        Generate human-like typing sequence for text.
        
        Args:
            text: Text to type
            
        Returns:
            List of (character/action, delay) tuples
        """
        typing_sequence = simulate_human_typing(text, self.config)
        
        # Update stats
        self.session_stats["text_typed"] += len(text)
        self.session_stats["actions_performed"] += len(typing_sequence)
        
        logger.debug(f"Generated typing sequence for '{text[:50]}...' "
                    f"with {len(typing_sequence)} keystrokes")
        
        return typing_sequence
    
    async def wait_with_human_delay(
        self, 
        action_type: str, 
        context: Optional[Dict] = None
    ) -> float:
        """
        Wait with human-like delay for specified action type.
        
        Args:
            action_type: Type of action causing the delay
            context: Optional context information
            
        Returns:
            Actual delay duration in seconds
        """
        delay = self.timing_manager.get_next_action_delay(action_type, context)
        
        await asyncio.sleep(delay)
        
        # Update stats
        self.session_stats["total_delays"] += delay
        
        logger.debug(f"Applied {delay:.3f}s delay for {action_type}")
        
        return delay
    
    async def simulate_reading(self, text_length: int, complexity: str = 'medium') -> float:
        """
        Simulate human reading behavior with appropriate delay.
        
        Args:
            text_length: Number of characters to read
            complexity: Text complexity ('simple', 'medium', 'complex')
            
        Returns:
            Reading delay in seconds
        """
        delay_generator = DelayGenerator()
        reading_delay = delay_generator.get_reading_delay(text_length, complexity)
        
        await asyncio.sleep(reading_delay)
        
        logger.debug(f"Simulated reading {text_length} characters "
                    f"({complexity} complexity) for {reading_delay:.2f}s")
        
        return reading_delay
    
    async def simulate_decision_making(
        self, 
        options_count: int, 
        complexity: str = 'medium'
    ) -> float:
        """
        Simulate human decision-making delay.
        
        Args:
            options_count: Number of options to choose from
            complexity: Decision complexity ('simple', 'medium', 'complex')
            
        Returns:
            Decision delay in seconds
        """
        delay_generator = DelayGenerator()
        decision_delay = delay_generator.get_decision_delay(options_count, complexity)
        
        await asyncio.sleep(decision_delay)
        
        logger.debug(f"Simulated decision making with {options_count} options "
                    f"({complexity} complexity) for {decision_delay:.2f}s")
        
        return decision_delay
    
    async def check_break_needed(self) -> Tuple[bool, float]:
        """
        Check if a behavioral break is needed.
        
        Returns:
            Tuple of (break_needed, break_duration_seconds)
        """
        return self.timing_manager.should_take_break()
    
    async def take_behavioral_break(self, duration: float) -> None:
        """
        Take a behavioral break to simulate human rest patterns.
        
        Args:
            duration: Break duration in seconds
        """
        logger.info(f"Taking behavioral break for {duration:.1f} seconds")
        
        await asyncio.sleep(duration)
        
        logger.info("Behavioral break completed")
    
    def update_behavioral_state(self, state_updates: Dict) -> None:
        """
        Update behavioral state based on external factors.
        
        Args:
            state_updates: Dictionary of state updates
        """
        if 'mouse_speed_factor' in state_updates:
            factor = state_updates['mouse_speed_factor']
            self.mouse_generator.min_velocity *= factor
            self.mouse_generator.max_velocity *= factor
        
        if 'typing_speed_wpm' in state_updates:
            wpm = state_updates['typing_speed_wpm']
            self.typing_session.typer.wpm_range = (wpm - 20, wpm + 20)
        
        if 'error_rate' in state_updates:
            self.typing_session.typer.error_rate = state_updates['error_rate']
        
        logger.debug(f"Updated behavioral state: {state_updates}")
    
    def get_behavioral_metrics(self) -> Dict:
        """
        Get comprehensive behavioral metrics for analysis.
        
        Returns:
            Dictionary of behavioral metrics
        """
        session_duration = (datetime.now() - self.session_stats["session_start"]).total_seconds()
        
        metrics = {
            "session_duration_seconds": session_duration,
            "actions_performed": self.session_stats["actions_performed"],
            "mouse_movements": self.session_stats["mouse_movements"],
            "text_typed": self.session_stats["text_typed"],
            "total_delays": self.session_stats["total_delays"],
            "actions_per_minute": self.session_stats["actions_performed"] / max(1, session_duration / 60),
            "average_delay": self.session_stats["total_delays"] / max(1, self.session_stats["actions_performed"]),
            "current_mouse_position": self.current_mouse_position,
            "timing_stats": self.timing_manager.get_timing_stats(),
            "typing_stats": self.typing_session.get_session_stats()
        }
        
        return metrics
    
    def reset_session(self) -> None:
        """Reset behavioral engine for new session."""
        self.timing_manager.reset_session()
        self.typing_session = TypingSession(self.config)
        
        self.session_stats = {
            "actions_performed": 0,
            "mouse_movements": 0,
            "text_typed": 0,
            "total_delays": 0.0,
            "session_start": datetime.now()
        }
        
        logger.info("Behavioral engine session reset")


class BehavioralPatternAnalyzer:
    """Analyzer for behavioral patterns to detect and prevent bot-like behavior."""
    
    def __init__(self):
        """Initialize behavioral pattern analyzer."""
        self.action_history: List[Dict] = []
        self.max_history_size = 1000
    
    def record_action(
        self, 
        action_type: str, 
        timestamp: datetime, 
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Record an action for pattern analysis.
        
        Args:
            action_type: Type of action performed
            timestamp: When the action occurred
            metadata: Optional action metadata
        """
        action_record = {
            "action_type": action_type,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }
        
        self.action_history.append(action_record)
        
        # Maintain history size limit
        if len(self.action_history) > self.max_history_size:
            self.action_history.pop(0)
    
    def analyze_patterns(self) -> Dict:
        """
        Analyze recorded actions for bot-like patterns.
        
        Returns:
            Analysis results with risk scores and recommendations
        """
        if len(self.action_history) < 10:
            return {"risk_score": 0.0, "patterns": [], "recommendations": []}
        
        patterns = []
        risk_factors = []
        
        # Check for overly regular timing
        timing_regularity = self._analyze_timing_regularity()
        if timing_regularity > 0.8:
            patterns.append("Highly regular timing detected")
            risk_factors.append(0.3)
        
        # Check for identical sequences
        sequence_repetition = self._analyze_sequence_repetition()
        if sequence_repetition > 0.7:
            patterns.append("Repetitive action sequences detected")
            risk_factors.append(0.4)
        
        # Check for inhuman speeds
        speed_analysis = self._analyze_action_speeds()
        if speed_analysis["too_fast_ratio"] > 0.2:
            patterns.append("Suspiciously fast actions detected")
            risk_factors.append(0.5)
        
        # Check for lack of variation
        variation_score = self._analyze_behavioral_variation()
        if variation_score < 0.3:
            patterns.append("Insufficient behavioral variation")
            risk_factors.append(0.3)
        
        # Calculate overall risk score
        risk_score = min(1.0, sum(risk_factors))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(patterns, risk_score)
        
        return {
            "risk_score": risk_score,
            "patterns": patterns,
            "recommendations": recommendations,
            "timing_regularity": timing_regularity,
            "sequence_repetition": sequence_repetition,
            "speed_analysis": speed_analysis,
            "variation_score": variation_score
        }
    
    def _analyze_timing_regularity(self) -> float:
        """Analyze timing regularity between actions."""
        if len(self.action_history) < 5:
            return 0.0
        
        intervals = []
        for i in range(1, len(self.action_history)):
            interval = (self.action_history[i]["timestamp"] - 
                       self.action_history[i-1]["timestamp"]).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return 0.0
        
        # Calculate coefficient of variation (lower = more regular)
        mean_interval = sum(intervals) / len(intervals)
        if mean_interval == 0:
            return 1.0
        
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = variance ** 0.5
        cv = std_dev / mean_interval
        
        # Convert to regularity score (higher = more regular)
        regularity = max(0.0, 1.0 - cv)
        return regularity
    
    def _analyze_sequence_repetition(self) -> float:
        """Analyze repetition in action sequences."""
        if len(self.action_history) < 10:
            return 0.0
        
        # Look for repeated sequences of 3-5 actions
        sequences = {}
        sequence_length = 3
        
        for i in range(len(self.action_history) - sequence_length + 1):
            sequence = tuple(
                action["action_type"] 
                for action in self.action_history[i:i + sequence_length]
            )
            sequences[sequence] = sequences.get(sequence, 0) + 1
        
        if not sequences:
            return 0.0
        
        # Calculate repetition ratio
        total_sequences = len(sequences)
        repeated_sequences = sum(1 for count in sequences.values() if count > 1)
        
        return repeated_sequences / total_sequences if total_sequences > 0 else 0.0
    
    def _analyze_action_speeds(self) -> Dict:
        """Analyze action speeds for inhuman patterns."""
        if len(self.action_history) < 5:
            return {"too_fast_ratio": 0.0, "too_slow_ratio": 0.0}
        
        # Define reasonable speed thresholds for different actions
        speed_thresholds = {
            "click": {"min": 0.1, "max": 2.0},
            "type": {"min": 0.05, "max": 0.5},
            "navigate": {"min": 0.5, "max": 10.0},
            "scroll": {"min": 0.2, "max": 3.0}
        }
        
        too_fast_count = 0
        too_slow_count = 0
        total_analyzed = 0
        
        for i in range(1, len(self.action_history)):
            current_action = self.action_history[i]
            prev_action = self.action_history[i-1]
            
            action_type = current_action["action_type"]
            if action_type not in speed_thresholds:
                continue
            
            interval = (current_action["timestamp"] - prev_action["timestamp"]).total_seconds()
            thresholds = speed_thresholds[action_type]
            
            if interval < thresholds["min"]:
                too_fast_count += 1
            elif interval > thresholds["max"]:
                too_slow_count += 1
            
            total_analyzed += 1
        
        if total_analyzed == 0:
            return {"too_fast_ratio": 0.0, "too_slow_ratio": 0.0}
        
        return {
            "too_fast_ratio": too_fast_count / total_analyzed,
            "too_slow_ratio": too_slow_count / total_analyzed
        }
    
    def _analyze_behavioral_variation(self) -> float:
        """Analyze overall behavioral variation."""
        if len(self.action_history) < 20:
            return 1.0  # Assume good variation for small samples
        
        # Check variation in different aspects
        action_types = set(action["action_type"] for action in self.action_history)
        type_variation = len(action_types) / 10.0  # Normalize to common action types
        
        # Check timing variation (already calculated in regularity)
        timing_variation = 1.0 - self._analyze_timing_regularity()
        
        # Combine variation scores
        overall_variation = (type_variation + timing_variation) / 2.0
        return min(1.0, overall_variation)
    
    def _generate_recommendations(self, patterns: List[str], risk_score: float) -> List[str]:
        """Generate recommendations based on detected patterns."""
        recommendations = []
        
        if risk_score > 0.7:
            recommendations.append("High bot detection risk - consider increasing behavioral variation")
        
        if "regular timing" in str(patterns).lower():
            recommendations.append("Add more randomness to action timing")
        
        if "repetitive" in str(patterns).lower():
            recommendations.append("Vary action sequences to appear more natural")
        
        if "fast actions" in str(patterns).lower():
            recommendations.append("Slow down action execution to human speeds")
        
        if "variation" in str(patterns).lower():
            recommendations.append("Increase diversity in behavioral patterns")
        
        if not recommendations:
            recommendations.append("Behavioral patterns appear natural")
        
        return recommendations


# Global behavioral engine instance
_behavioral_engine: Optional[BehavioralEngine] = None


def get_behavioral_engine() -> BehavioralEngine:
    """
    Get global behavioral engine instance.
    
    Returns:
        Behavioral engine instance
    """
    global _behavioral_engine
    
    if _behavioral_engine is None:
        _behavioral_engine = BehavioralEngine()
    
    return _behavioral_engine


def init_behavioral_engine(config: Optional[BehavioralConfig] = None) -> BehavioralEngine:
    """
    Initialize global behavioral engine.
    
    Args:
        config: Optional behavioral configuration
        
    Returns:
        Initialized behavioral engine
    """
    global _behavioral_engine
    
    _behavioral_engine = BehavioralEngine(config)
    return _behavioral_engine