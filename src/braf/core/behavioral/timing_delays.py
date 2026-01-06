"""
Human-like timing delays and activity scheduling for BRAF.

This module provides realistic delay generation using log-normal distributions
and activity scheduling within human-like time windows.
"""

import math
import random
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np


class DelayGenerator:
    """Generator for human-like delays using statistical distributions."""
    
    def __init__(self, base_delay_range: Tuple[float, float] = (0.1, 0.3)):
        """
        Initialize delay generator.
        
        Args:
            base_delay_range: Base delay range in seconds (min, max)
        """
        self.base_delay_range = base_delay_range
    
    def get_action_delay(self, action_type: str, context: Optional[Dict] = None) -> float:
        """
        Generate realistic delay for specific action type.
        
        Args:
            action_type: Type of action (click, type, navigate, etc.)
            context: Optional context information
            
        Returns:
            Delay in seconds
        """
        # Base delays for different action types
        base_delays = {
            'click': (0.1, 0.4),
            'double_click': (0.05, 0.2),
            'type': (0.05, 0.15),
            'navigate': (0.5, 2.0),
            'scroll': (0.2, 0.8),
            'hover': (0.3, 1.0),
            'drag': (0.2, 0.6),
            'key_press': (0.05, 0.2),
            'wait': (1.0, 3.0),
            'page_load': (2.0, 8.0),
            'form_fill': (0.3, 1.2),
            'menu_select': (0.4, 1.0),
            'tab_switch': (0.2, 0.6)
        }
        
        delay_range = base_delays.get(action_type, self.base_delay_range)
        
        # Generate log-normal delay (more realistic than uniform)
        mean_delay = (delay_range[0] + delay_range[1]) / 2
        sigma = 0.3  # Standard deviation for log-normal
        
        # Log-normal distribution parameters
        mu = math.log(mean_delay) - (sigma ** 2) / 2
        
        delay = np.random.lognormal(mu, sigma)
        
        # Clamp to reasonable bounds
        min_delay, max_delay = delay_range
        delay = max(min_delay, min(max_delay * 2, delay))
        
        # Apply context modifiers
        if context:
            delay = self._apply_context_modifiers(delay, action_type, context)
        
        return delay
    
    def _apply_context_modifiers(self, base_delay: float, action_type: str, context: Dict) -> float:
        """
        Apply context-based modifiers to delay.
        
        Args:
            base_delay: Base delay value
            action_type: Type of action
            context: Context information
            
        Returns:
            Modified delay
        """
        delay = base_delay
        
        # Page complexity modifier
        if 'page_complexity' in context:
            complexity = context['page_complexity']  # 'simple', 'medium', 'complex'
            if complexity == 'complex':
                delay *= random.uniform(1.2, 1.8)
            elif complexity == 'simple':
                delay *= random.uniform(0.7, 0.9)
        
        # User experience modifier
        if 'user_experience' in context:
            experience = context['user_experience']  # 'novice', 'intermediate', 'expert'
            if experience == 'novice':
                delay *= random.uniform(1.5, 2.5)
            elif experience == 'expert':
                delay *= random.uniform(0.6, 0.8)
        
        # Task urgency modifier
        if 'urgency' in context:
            urgency = context['urgency']  # 'low', 'medium', 'high'
            if urgency == 'high':
                delay *= random.uniform(0.5, 0.7)
            elif urgency == 'low':
                delay *= random.uniform(1.2, 1.6)
        
        # Fatigue modifier (increases over time)
        if 'session_duration' in context:
            duration_minutes = context['session_duration']
            if duration_minutes > 30:
                fatigue_factor = 1 + (duration_minutes - 30) * 0.01
                delay *= min(2.0, fatigue_factor)
        
        # Error recovery modifier
        if 'recent_error' in context and context['recent_error']:
            delay *= random.uniform(1.3, 2.0)  # Slower after errors
        
        return delay
    
    def get_reading_delay(self, text_length: int, complexity: str = 'medium') -> float:
        """
        Generate realistic reading delay based on text length and complexity.
        
        Args:
            text_length: Number of characters to read
            complexity: Text complexity ('simple', 'medium', 'complex')
            
        Returns:
            Reading delay in seconds
        """
        # Average reading speeds (words per minute)
        reading_speeds = {
            'simple': random.uniform(250, 350),    # Easy text
            'medium': random.uniform(200, 280),    # Normal text
            'complex': random.uniform(150, 220)    # Technical/difficult text
        }
        
        wpm = reading_speeds[complexity]
        
        # Estimate word count (average 5 characters per word)
        word_count = text_length / 5
        
        # Calculate reading time
        reading_time = (word_count / wpm) * 60
        
        # Add some variation and minimum time
        variation = random.uniform(0.7, 1.4)
        reading_time *= variation
        
        # Minimum reading time (even for short text)
        min_time = max(0.5, text_length * 0.02)
        
        return max(min_time, reading_time)
    
    def get_decision_delay(self, options_count: int, complexity: str = 'medium') -> float:
        """
        Generate realistic decision-making delay.
        
        Args:
            options_count: Number of options to choose from
            complexity: Decision complexity ('simple', 'medium', 'complex')
            
        Returns:
            Decision delay in seconds
        """
        # Base decision time increases with options (Hick's Law)
        base_time = math.log2(options_count + 1) * 0.5
        
        # Complexity multipliers
        complexity_multipliers = {
            'simple': random.uniform(0.5, 0.8),
            'medium': random.uniform(0.8, 1.2),
            'complex': random.uniform(1.5, 2.5)
        }
        
        multiplier = complexity_multipliers[complexity]
        decision_time = base_time * multiplier
        
        # Add random variation
        decision_time *= random.uniform(0.6, 1.8)
        
        # Reasonable bounds
        return max(0.3, min(10.0, decision_time))


class ActivityScheduler:
    """Scheduler for human-like activity patterns and timing."""
    
    def __init__(self):
        """Initialize activity scheduler."""
        # Human activity patterns (hours of day)
        self.activity_patterns = {
            'weekday': {
                'high': [(9, 12), (14, 17)],      # Peak work hours
                'medium': [(8, 9), (12, 14), (17, 19)],  # Moderate activity
                'low': [(19, 22)],                # Evening activity
                'minimal': [(22, 24), (0, 8)]    # Night/early morning
            },
            'weekend': {
                'high': [(10, 13), (15, 18)],    # Weekend peak
                'medium': [(13, 15), (18, 21)],  # Moderate weekend
                'low': [(9, 10), (21, 23)],      # Low weekend
                'minimal': [(23, 24), (0, 9)]    # Weekend night
            }
        }
    
    def get_activity_level(self, timestamp: datetime) -> str:
        """
        Get activity level for given timestamp.
        
        Args:
            timestamp: Time to check
            
        Returns:
            Activity level ('high', 'medium', 'low', 'minimal')
        """
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        
        pattern_key = 'weekend' if is_weekend else 'weekday'
        patterns = self.activity_patterns[pattern_key]
        
        for level, time_ranges in patterns.items():
            for start_hour, end_hour in time_ranges:
                if start_hour <= hour < end_hour:
                    return level
        
        return 'minimal'
    
    def get_activity_multiplier(self, timestamp: datetime) -> float:
        """
        Get activity speed multiplier for timestamp.
        
        Args:
            timestamp: Time to check
            
        Returns:
            Speed multiplier (higher = faster actions)
        """
        activity_level = self.get_activity_level(timestamp)
        
        multipliers = {
            'high': random.uniform(0.8, 1.2),     # Normal to fast
            'medium': random.uniform(0.9, 1.1),   # Normal
            'low': random.uniform(1.1, 1.4),      # Slower
            'minimal': random.uniform(1.3, 2.0)   # Much slower
        }
        
        return multipliers[activity_level]
    
    def schedule_next_action(self, current_time: datetime, action_type: str) -> datetime:
        """
        Schedule next action based on human activity patterns.
        
        Args:
            current_time: Current timestamp
            action_type: Type of action to schedule
            
        Returns:
            Scheduled time for next action
        """
        # Get base delay for action type
        delay_generator = DelayGenerator()
        base_delay = delay_generator.get_action_delay(action_type)
        
        # Apply activity multiplier
        activity_multiplier = self.get_activity_multiplier(current_time)
        adjusted_delay = base_delay * activity_multiplier
        
        # Add some random jitter
        jitter = random.uniform(-0.1, 0.1) * adjusted_delay
        final_delay = max(0.05, adjusted_delay + jitter)
        
        return current_time + timedelta(seconds=final_delay)
    
    def is_appropriate_time(self, timestamp: datetime, min_activity_level: str = 'low') -> bool:
        """
        Check if timestamp is appropriate for automation activity.
        
        Args:
            timestamp: Time to check
            min_activity_level: Minimum required activity level
            
        Returns:
            True if time is appropriate for activity
        """
        current_level = self.get_activity_level(timestamp)
        
        level_hierarchy = ['minimal', 'low', 'medium', 'high']
        current_index = level_hierarchy.index(current_level)
        min_index = level_hierarchy.index(min_activity_level)
        
        return current_index >= min_index
    
    def get_break_duration(self, session_duration: float) -> float:
        """
        Calculate appropriate break duration based on session length.
        
        Args:
            session_duration: Current session duration in minutes
            
        Returns:
            Break duration in seconds
        """
        if session_duration < 15:
            # Short sessions, minimal break
            return random.uniform(5, 15)
        elif session_duration < 45:
            # Medium sessions, short break
            return random.uniform(30, 120)
        elif session_duration < 90:
            # Long sessions, medium break
            return random.uniform(300, 900)  # 5-15 minutes
        else:
            # Very long sessions, long break
            return random.uniform(900, 1800)  # 15-30 minutes


class BehavioralTimingManager:
    """Manager for coordinating all behavioral timing aspects."""
    
    def __init__(self):
        """Initialize behavioral timing manager."""
        self.delay_generator = DelayGenerator()
        self.activity_scheduler = ActivityScheduler()
        self.session_start_time = datetime.now()
        self.last_action_time = datetime.now()
        self.action_count = 0
    
    def get_next_action_delay(
        self, 
        action_type: str, 
        context: Optional[Dict] = None
    ) -> float:
        """
        Get delay for next action considering all behavioral factors.
        
        Args:
            action_type: Type of action
            context: Optional context information
            
        Returns:
            Delay in seconds
        """
        # Build comprehensive context
        full_context = context or {}
        
        # Add session information
        session_duration = (datetime.now() - self.session_start_time).total_seconds() / 60
        full_context['session_duration'] = session_duration
        
        # Add activity level
        activity_level = self.activity_scheduler.get_activity_level(datetime.now())
        full_context['activity_level'] = activity_level
        
        # Get base delay
        delay = self.delay_generator.get_action_delay(action_type, full_context)
        
        # Apply activity multiplier
        activity_multiplier = self.activity_scheduler.get_activity_multiplier(datetime.now())
        delay *= activity_multiplier
        
        # Track action
        self.action_count += 1
        self.last_action_time = datetime.now()
        
        return delay
    
    def should_take_break(self) -> Tuple[bool, float]:
        """
        Determine if a break should be taken.
        
        Returns:
            Tuple of (should_break, break_duration_seconds)
        """
        session_duration = (datetime.now() - self.session_start_time).total_seconds() / 60
        
        # Break probability increases with session duration
        if session_duration > 60:
            break_probability = 0.3
        elif session_duration > 30:
            break_probability = 0.1
        elif session_duration > 15:
            break_probability = 0.05
        else:
            break_probability = 0.01
        
        # Also consider action count
        if self.action_count > 100:
            break_probability *= 1.5
        
        should_break = random.random() < break_probability
        
        if should_break:
            break_duration = self.activity_scheduler.get_break_duration(session_duration)
            return True, break_duration
        
        return False, 0.0
    
    def reset_session(self):
        """Reset session timing for new session."""
        self.session_start_time = datetime.now()
        self.last_action_time = datetime.now()
        self.action_count = 0
    
    def get_timing_stats(self) -> Dict:
        """
        Get timing statistics for current session.
        
        Returns:
            Dictionary of timing statistics
        """
        session_duration = (datetime.now() - self.session_start_time).total_seconds()
        
        return {
            "session_duration_seconds": session_duration,
            "session_duration_minutes": session_duration / 60,
            "total_actions": self.action_count,
            "actions_per_minute": self.action_count / max(1, session_duration / 60),
            "last_action_time": self.last_action_time,
            "current_activity_level": self.activity_scheduler.get_activity_level(datetime.now())
        }


# Global timing manager instance
_timing_manager: Optional[BehavioralTimingManager] = None


def get_timing_manager() -> BehavioralTimingManager:
    """
    Get global behavioral timing manager.
    
    Returns:
        Timing manager instance
    """
    global _timing_manager
    
    if _timing_manager is None:
        _timing_manager = BehavioralTimingManager()
    
    return _timing_manager


def get_human_delay(action_type: str, context: Optional[Dict] = None) -> float:
    """
    Convenience function to get human-like delay for action.
    
    Args:
        action_type: Type of action
        context: Optional context information
        
    Returns:
        Delay in seconds
    """
    manager = get_timing_manager()
    return manager.get_next_action_delay(action_type, context)
