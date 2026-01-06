"""
Behavior Profile Manager
Manages and optimizes human-like behavior patterns for different platforms
"""

import json
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import asyncio
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class BehaviorProfile:
    """Human behavior profile for automation"""
    name: str
    platform: str
    
    # Mouse behavior
    mouse_speed_range: Dict[str, float] = field(default_factory=lambda: {"min": 100, "max": 300})
    click_variance: float = 0.2
    double_click_chance: float = 0.05
    right_click_chance: float = 0.02
    
    # Typing behavior
    typing_speed: Dict[str, float] = field(default_factory=lambda: {"min": 40, "max": 65})
    typing_errors: float = 0.02  # 2% chance of typos
    backspace_delay: float = 0.3
    
    # Scrolling behavior
    scroll_patterns: List[Dict] = field(default_factory=list)
    scroll_variance: float = 0.3
    
    # Timing behavior
    action_delays: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])
    timing_variance: float = 0.25
    pause_frequency: float = 0.1  # 10% chance of random pauses
    
    # Session behavior
    session_duration_range: Dict[str, int] = field(default_factory=lambda: {"min": 15, "max": 45})
    break_frequency: float = 0.15  # 15% chance of taking breaks
    break_duration_range: Dict[str, int] = field(default_factory=lambda: {"min": 30, "max": 180})
    
    # Browser behavior
    user_agent: str = ""
    viewport: Dict[str, int] = field(default_factory=lambda: {"width": 1366, "height": 768})
    locale: str = "en-US"
    timezone: str = "America/New_York"
    
    # Learning data
    success_rate: float = 0.0
    usage_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

class BehaviorProfileManager:
    """Manages behavior profiles for different platforms and scenarios"""
    
    def __init__(self):
        self.profiles: Dict[str, List[BehaviorProfile]] = {}
        self.active_profiles: Dict[str, BehaviorProfile] = {}
        self.performance_data: Dict[str, List[Dict]] = {}
        
        # Initialize default profiles
        self._create_default_profiles()
        
        logger.info("Behavior Profile Manager initialized")
    
    def _create_default_profiles(self):
        """Create default behavior profiles for major platforms"""
        
        # Swagbucks profiles
        self.profiles["swagbucks"] = [
            # Conservative profile
            BehaviorProfile(
                name="swagbucks_conservative",
                platform="swagbucks",
                mouse_speed_range={"min": 80, "max": 150},
                click_variance=0.15,
                typing_speed={"min": 35, "max": 50},
                typing_errors=0.01,
                action_delays=[1.0, 1.5, 2.0],
                timing_variance=0.3,
                pause_frequency=0.15,
                session_duration_range={"min": 20, "max": 35},
                break_frequency=0.2
            ),
            
            # Balanced profile
            BehaviorProfile(
                name="swagbucks_balanced",
                platform="swagbucks",
                mouse_speed_range={"min": 120, "max": 200},
                click_variance=0.2,
                typing_speed={"min": 45, "max": 65},
                typing_errors=0.02,
                action_delays=[0.8, 1.2, 1.8],
                timing_variance=0.25,
                pause_frequency=0.12,
                session_duration_range={"min": 15, "max": 30},
                break_frequency=0.15
            ),
            
            # Aggressive profile
            BehaviorProfile(
                name="swagbucks_aggressive",
                platform="swagbucks",
                mouse_speed_range={"min": 150, "max": 250},
                click_variance=0.25,
                typing_speed={"min": 55, "max": 75},
                typing_errors=0.025,
                action_delays=[0.5, 0.8, 1.2],
                timing_variance=0.2,
                pause_frequency=0.08,
                session_duration_range={"min": 10, "max": 25},
                break_frequency=0.1
            )
        ]
        
        # Survey Junkie profiles
        self.profiles["survey_junkie"] = [
            BehaviorProfile(
                name="survey_junkie_careful",
                platform="survey_junkie",
                mouse_speed_range={"min": 90, "max": 160},
                click_variance=0.18,
                typing_speed={"min": 40, "max": 55},
                typing_errors=0.015,
                action_delays=[1.2, 1.8, 2.5],
                timing_variance=0.35,
                pause_frequency=0.18,
                session_duration_range={"min": 25, "max": 40}
            ),
            
            BehaviorProfile(
                name="survey_junkie_normal",
                platform="survey_junkie",
                mouse_speed_range={"min": 110, "max": 180},
                click_variance=0.22,
                typing_speed={"min": 50, "max": 70},
                typing_errors=0.02,
                action_delays=[0.9, 1.4, 2.0],
                timing_variance=0.28,
                pause_frequency=0.14,
                session_duration_range={"min": 18, "max": 35}
            )
        ]
        
        # PrizeRebel profiles
        self.profiles["prizerebel"] = [
            BehaviorProfile(
                name="prizerebel_standard",
                platform="prizerebel",
                mouse_speed_range={"min": 100, "max": 170},
                click_variance=0.2,
                typing_speed={"min": 45, "max": 60},
                typing_errors=0.018,
                action_delays=[0.7, 1.1, 1.6],
                timing_variance=0.25,
                pause_frequency=0.12,
                session_duration_range={"min": 15, "max": 30}
            )
        ]
        
        # Generic profiles for unknown platforms
        self.profiles["generic"] = [
            BehaviorProfile(
                name="generic_safe",
                platform="generic",
                mouse_speed_range={"min": 100, "max": 180},
                click_variance=0.2,
                typing_speed={"min": 40, "max": 60},
                typing_errors=0.02,
                action_delays=[1.0, 1.5, 2.0],
                timing_variance=0.3,
                pause_frequency=0.15,
                session_duration_range={"min": 20, "max": 35}
            )
        ]
        
        # Add scroll patterns to all profiles
        for platform_profiles in self.profiles.values():
            for profile in platform_profiles:
                profile.scroll_patterns = [
                    {"speed": 120, "duration": 2.0, "pause": 1.0, "direction": "down"},
                    {"speed": 80, "duration": 1.5, "pause": 0.8, "direction": "up"},
                    {"speed": 150, "duration": 1.2, "pause": 0.5, "direction": "down"}
                ]
    
    async def get_optimal_profile(self, platform: str, risk_level: float = 0.5) -> Dict[str, Any]:
        """Get optimal behavior profile for platform"""
        
        platform_lower = platform.lower()
        
        # Get platform profiles or fallback to generic
        available_profiles = self.profiles.get(platform_lower, self.profiles["generic"])
        
        if not available_profiles:
            return self._create_fallback_profile(platform)
        
        # Select profile based on risk level and performance
        selected_profile = self._select_best_profile(available_profiles, risk_level)
        
        # Convert to dictionary format
        profile_dict = self._profile_to_dict(selected_profile)
        
        # Apply dynamic adjustments
        profile_dict = self._apply_dynamic_adjustments(profile_dict, platform, risk_level)
        
        # Update usage statistics
        selected_profile.usage_count += 1
        selected_profile.last_updated = datetime.now()
        
        # Store as active profile
        self.active_profiles[platform] = selected_profile
        
        return profile_dict
    
    def _select_best_profile(self, profiles: List[BehaviorProfile], risk_level: float) -> BehaviorProfile:
        """Select best profile based on risk level and performance"""
        
        # Score profiles based on success rate and risk appropriateness
        scored_profiles = []
        
        for profile in profiles:
            # Base score from success rate
            score = profile.success_rate if profile.usage_count > 0 else 0.5
            
            # Adjust based on risk level match
            if "conservative" in profile.name and risk_level < 0.3:
                score += 0.2
            elif "balanced" in profile.name and 0.3 <= risk_level <= 0.7:
                score += 0.2
            elif "aggressive" in profile.name and risk_level > 0.7:
                score += 0.2
            
            # Prefer profiles with more usage data
            if profile.usage_count > 10:
                score += 0.1
            elif profile.usage_count > 5:
                score += 0.05
            
            scored_profiles.append((score, profile))
        
        # Sort by score and return best
        scored_profiles.sort(key=lambda x: x[0], reverse=True)
        return scored_profiles[0][1]
    
    def _profile_to_dict(self, profile: BehaviorProfile) -> Dict[str, Any]:
        """Convert BehaviorProfile to dictionary"""
        
        return {
            "name": profile.name,
            "platform": profile.platform,
            
            # Mouse behavior
            "mouse_speed_range": profile.mouse_speed_range,
            "click_variance": profile.click_variance,
            "double_click_chance": profile.double_click_chance,
            "right_click_chance": profile.right_click_chance,
            
            # Typing behavior
            "typing_speed": profile.typing_speed,
            "typing_errors": profile.typing_errors,
            "backspace_delay": profile.backspace_delay,
            
            # Scrolling behavior
            "scroll_patterns": profile.scroll_patterns,
            "scroll_variance": profile.scroll_variance,
            
            # Timing behavior
            "action_delays": profile.action_delays,
            "timing_variance": profile.timing_variance,
            "pause_frequency": profile.pause_frequency,
            
            # Session behavior
            "session_duration_range": profile.session_duration_range,
            "break_frequency": profile.break_frequency,
            "break_duration_range": profile.break_duration_range,
            
            # Browser behavior
            "user_agent": profile.user_agent or self._generate_user_agent(),
            "viewport": profile.viewport,
            "locale": profile.locale,
            "timezone": profile.timezone,
            
            # Performance data
            "success_rate": profile.success_rate,
            "usage_count": profile.usage_count
        }
    
    def _apply_dynamic_adjustments(self, profile_dict: Dict[str, Any], 
                                 platform: str, risk_level: float) -> Dict[str, Any]:
        """Apply dynamic adjustments based on current conditions"""
        
        # Time-based adjustments
        current_hour = datetime.now().hour
        
        # Slower during peak hours (more competition/detection)
        if 9 <= current_hour <= 17:  # Business hours
            profile_dict["action_delays"] = [d * 1.2 for d in profile_dict["action_delays"]]
            profile_dict["timing_variance"] *= 1.1
        
        # Faster during off-peak hours
        elif current_hour < 6 or current_hour > 22:  # Late night/early morning
            profile_dict["action_delays"] = [d * 0.8 for d in profile_dict["action_delays"]]
        
        # Risk-based adjustments
        if risk_level > 0.7:  # High risk - be more careful
            profile_dict["action_delays"] = [d * 1.3 for d in profile_dict["action_delays"]]
            profile_dict["timing_variance"] *= 1.2
            profile_dict["pause_frequency"] *= 1.5
        elif risk_level < 0.3:  # Low risk - can be more aggressive
            profile_dict["action_delays"] = [d * 0.7 for d in profile_dict["action_delays"]]
            profile_dict["timing_variance"] *= 0.9
        
        # Platform-specific adjustments
        if platform.lower() == "swagbucks":
            # Swagbucks is more sensitive to timing
            profile_dict["timing_variance"] *= 1.1
        elif platform.lower() == "survey_junkie":
            # Survey Junkie prefers consistent behavior
            profile_dict["timing_variance"] *= 0.9
        
        return profile_dict
    
    def _generate_user_agent(self) -> str:
        """Generate realistic user agent string"""
        
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15"
        ]
        
        return random.choice(user_agents)
    
    def _create_fallback_profile(self, platform: str) -> Dict[str, Any]:
        """Create fallback profile for unknown platforms"""
        
        return {
            "name": f"{platform}_fallback",
            "platform": platform,
            "mouse_speed_range": {"min": 100, "max": 180},
            "click_variance": 0.2,
            "typing_speed": {"min": 40, "max": 60},
            "typing_errors": 0.02,
            "action_delays": [1.0, 1.5, 2.0],
            "timing_variance": 0.3,
            "pause_frequency": 0.15,
            "scroll_patterns": [
                {"speed": 120, "duration": 2.0, "pause": 1.0, "direction": "down"}
            ],
            "session_duration_range": {"min": 20, "max": 35},
            "user_agent": self._generate_user_agent(),
            "viewport": {"width": 1366, "height": 768},
            "locale": "en-US",
            "timezone": "America/New_York"
        }
    
    async def update_from_execution(self, platform: str, execution_result: Dict[str, Any]):
        """Update profile performance based on execution results"""
        
        if platform not in self.active_profiles:
            return
        
        profile = self.active_profiles[platform]
        success = execution_result.get('success', False)
        
        # Update success rate using exponential moving average
        if profile.usage_count == 0:
            profile.success_rate = 1.0 if success else 0.0
        else:
            alpha = 0.1  # Learning rate
            profile.success_rate = (1 - alpha) * profile.success_rate + alpha * (1.0 if success else 0.0)
        
        # Store performance data
        if platform not in self.performance_data:
            self.performance_data[platform] = []
        
        self.performance_data[platform].append({
            'timestamp': datetime.now(),
            'profile_name': profile.name,
            'success': success,
            'actions_completed': execution_result.get('actions_completed', 0),
            'execution_time': execution_result.get('execution_time', 0)
        })
        
        # Keep only recent performance data
        if len(self.performance_data[platform]) > 100:
            self.performance_data[platform] = self.performance_data[platform][-100:]
        
        # Adaptive profile optimization
        await self._optimize_profile_from_performance(platform, profile)
    
    async def _optimize_profile_from_performance(self, platform: str, profile: BehaviorProfile):
        """Optimize profile based on performance data"""
        
        if platform not in self.performance_data or len(self.performance_data[platform]) < 5:
            return
        
        recent_data = self.performance_data[platform][-10:]  # Last 10 executions
        success_rate = sum(1 for d in recent_data if d['success']) / len(recent_data)
        
        # If success rate is low, make profile more conservative
        if success_rate < 0.6:
            profile.action_delays = [d * 1.1 for d in profile.action_delays]
            profile.timing_variance *= 1.05
            profile.pause_frequency *= 1.1
            
            logger.info(f"Made profile {profile.name} more conservative due to low success rate: {success_rate:.2f}")
        
        # If success rate is very high, can be slightly more aggressive
        elif success_rate > 0.9 and profile.usage_count > 20:
            profile.action_delays = [max(0.1, d * 0.98) for d in profile.action_delays]
            profile.timing_variance *= 0.99
            
            logger.info(f"Made profile {profile.name} slightly more aggressive due to high success rate: {success_rate:.2f}")
    
    def create_custom_profile(self, platform: str, base_profile_name: str = None, 
                            adjustments: Dict[str, Any] = None) -> str:
        """Create custom behavior profile"""
        
        # Get base profile
        if base_profile_name and platform in self.profiles:
            base_profile = next((p for p in self.profiles[platform] if p.name == base_profile_name), None)
        else:
            base_profile = self.profiles.get(platform, self.profiles["generic"])[0]
        
        if not base_profile:
            base_profile = self.profiles["generic"][0]
        
        # Create new profile with adjustments
        custom_name = f"{platform}_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        custom_profile = BehaviorProfile(
            name=custom_name,
            platform=platform,
            mouse_speed_range=base_profile.mouse_speed_range.copy(),
            click_variance=base_profile.click_variance,
            typing_speed=base_profile.typing_speed.copy(),
            typing_errors=base_profile.typing_errors,
            action_delays=base_profile.action_delays.copy(),
            timing_variance=base_profile.timing_variance,
            pause_frequency=base_profile.pause_frequency,
            scroll_patterns=base_profile.scroll_patterns.copy(),
            session_duration_range=base_profile.session_duration_range.copy(),
            user_agent=base_profile.user_agent,
            viewport=base_profile.viewport.copy(),
            locale=base_profile.locale,
            timezone=base_profile.timezone
        )
        
        # Apply adjustments
        if adjustments:
            for key, value in adjustments.items():
                if hasattr(custom_profile, key):
                    setattr(custom_profile, key, value)
        
        # Add to profiles
        if platform not in self.profiles:
            self.profiles[platform] = []
        
        self.profiles[platform].append(custom_profile)
        
        logger.info(f"Created custom profile: {custom_name}")
        return custom_name
    
    def get_profile_performance(self, platform: str) -> Dict[str, Any]:
        """Get performance statistics for platform profiles"""
        
        if platform not in self.performance_data:
            return {"message": "No performance data available"}
        
        data = self.performance_data[platform]
        
        # Calculate statistics
        total_executions = len(data)
        successful_executions = sum(1 for d in data if d['success'])
        success_rate = successful_executions / total_executions if total_executions > 0 else 0
        
        # Profile breakdown
        profile_stats = {}
        for d in data:
            profile_name = d['profile_name']
            if profile_name not in profile_stats:
                profile_stats[profile_name] = {'total': 0, 'successful': 0}
            
            profile_stats[profile_name]['total'] += 1
            if d['success']:
                profile_stats[profile_name]['successful'] += 1
        
        # Calculate success rates for each profile
        for profile_name, stats in profile_stats.items():
            stats['success_rate'] = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
        
        return {
            'platform': platform,
            'total_executions': total_executions,
            'overall_success_rate': success_rate,
            'profile_performance': profile_stats,
            'recent_performance': data[-10:] if len(data) >= 10 else data
        }
    
    def get_all_profiles(self) -> Dict[str, List[str]]:
        """Get list of all available profiles by platform"""
        
        result = {}
        for platform, profiles in self.profiles.items():
            result[platform] = [p.name for p in profiles]
        
        return result
    
    def export_profiles(self) -> Dict[str, Any]:
        """Export all profiles for backup/sharing"""
        
        exported = {}
        for platform, profiles in self.profiles.items():
            exported[platform] = []
            for profile in profiles:
                exported[platform].append({
                    'name': profile.name,
                    'platform': profile.platform,
                    'mouse_speed_range': profile.mouse_speed_range,
                    'click_variance': profile.click_variance,
                    'typing_speed': profile.typing_speed,
                    'typing_errors': profile.typing_errors,
                    'action_delays': profile.action_delays,
                    'timing_variance': profile.timing_variance,
                    'pause_frequency': profile.pause_frequency,
                    'scroll_patterns': profile.scroll_patterns,
                    'session_duration_range': profile.session_duration_range,
                    'success_rate': profile.success_rate,
                    'usage_count': profile.usage_count
                })
        
        return exported
    
    def import_profiles(self, profile_data: Dict[str, Any]):
        """Import profiles from exported data"""
        
        for platform, profiles in profile_data.items():
            if platform not in self.profiles:
                self.profiles[platform] = []
            
            for profile_dict in profiles:
                profile = BehaviorProfile(
                    name=profile_dict['name'],
                    platform=profile_dict['platform'],
                    mouse_speed_range=profile_dict.get('mouse_speed_range', {"min": 100, "max": 180}),
                    click_variance=profile_dict.get('click_variance', 0.2),
                    typing_speed=profile_dict.get('typing_speed', {"min": 40, "max": 60}),
                    typing_errors=profile_dict.get('typing_errors', 0.02),
                    action_delays=profile_dict.get('action_delays', [1.0, 1.5, 2.0]),
                    timing_variance=profile_dict.get('timing_variance', 0.3),
                    pause_frequency=profile_dict.get('pause_frequency', 0.15),
                    scroll_patterns=profile_dict.get('scroll_patterns', []),
                    session_duration_range=profile_dict.get('session_duration_range', {"min": 20, "max": 35}),
                    success_rate=profile_dict.get('success_rate', 0.0),
                    usage_count=profile_dict.get('usage_count', 0)
                )
                
                self.profiles[platform].append(profile)
        
        logger.info(f"Imported profiles for {len(profile_data)} platforms")

# Global instance
behavior_profile_manager = BehaviorProfileManager()
