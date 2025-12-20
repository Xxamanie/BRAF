"""
BRAF Intelligence Integration
Integrates the intelligence layer with existing BRAF TaskExecutor and components
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add BRAF to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import BRAF components
from src.braf.core.task_executor import TaskExecutor, AutomationTask, TaskResult
from src.braf.core.behavioral.behavioral_engine import BehavioralEngine
from src.braf.core.browser.browser_instance import BrowserInstance
from src.braf.core.captcha.captcha_solver import CaptchaSolver
from src.braf.core.proxy_rotator import ProxyRotator
from src.braf.core.profile_manager import ProfileManager

# Import intelligence components
from .platform_intelligence_engine import platform_intelligence
from .network_traffic_analyzer import network_analyzer
from .behavior_profile_manager import behavior_profile_manager
from .earning_optimizer import earning_optimizer
from .intelligent_task_executor import intelligent_executor, IntelligentTaskConfig

logger = logging.getLogger(__name__)

class BRAFIntelligenceIntegration:
    """Integration layer between BRAF framework and intelligence system"""
    
    def __init__(self):
        # BRAF components
        self.task_executor = TaskExecutor()
        self.behavioral_engine = BehavioralEngine()
        self.captcha_solver = CaptchaSolver()
        self.proxy_rotator = ProxyRotator()
        self.profile_manager = ProfileManager()
        
        # Intelligence components
        self.platform_intelligence = platform_intelligence
        self.network_analyzer = network_analyzer
        self.behavior_profiles = behavior_profile_manager
        self.earning_optimizer = earning_optimizer
        self.intelligent_executor = intelligent_executor
        
        # Integration state
        self.active_sessions = {}
        self.intelligence_enabled = True
        
        logger.info("BRAF Intelligence Integration initialized")
    
    async def execute_intelligent_automation(self, 
                                           platform_name: str,
                                           task_config: Dict[str, Any],
                                           intelligence_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute automation with full intelligence integration"""
        
        try:
            # Create intelligent task configuration
            intel_config = IntelligentTaskConfig(
                platform_name=platform_name,
                risk_tolerance=intelligence_config.get('risk_tolerance', 0.5) if intelligence_config else 0.5,
                optimization_level=intelligence_config.get('optimization_level', 3) if intelligence_config else 3,
                stealth_mode=intelligence_config.get('stealth_mode', True) if intelligence_config else True,
                learning_enabled=intelligence_config.get('learning_enabled', True) if intelligence_config else True,
                adaptive_behavior=intelligence_config.get('adaptive_behavior', True) if intelligence_config else True
            )
            
            # Create automation task from config
            task = self._create_task_from_config(task_config)
            
            # Check if we should use intelligence or fallback to standard execution
            if self.intelligence_enabled and self._should_use_intelligence(platform_name):
                logger.info(f"Using intelligent execution for {platform_name}")
                result = await self.intelligent_executor.execute_intelligent_survey_task(task, intel_config)
            else:
                logger.info(f"Using standard execution for {platform_name}")
                result = await self._execute_standard_task(task, intel_config)
            
            # Post-process results
            enhanced_result = await self._enhance_execution_result(result, platform_name, intel_config)
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Intelligent automation execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'platform': platform_name,
                'intelligence_used': False
            }
    
    def _create_task_from_config(self, task_config: Dict[str, Any]) -> AutomationTask:
        """Create AutomationTask from configuration"""
        
        # Import action types
        from src.braf.core.task_executor import ActionType
        
        # Create actions from config
        actions = []
        for action_config in task_config.get('actions', []):
            action = self._create_action_from_config(action_config)
            if action:
                actions.append(action)
        
        # Create task
        task = AutomationTask(
            task_id=task_config.get('task_id', f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            name=task_config.get('name', 'Intelligent Automation Task'),
            description=task_config.get('description', 'Task with intelligence integration'),
            actions=actions,
            priority=task_config.get('priority', 1),
            timeout=task_config.get('timeout', 300)
        )
        
        return task
    
    def _create_action_from_config(self, action_config: Dict[str, Any]):
        """Create action from configuration"""
        
        from src.braf.core.task_executor import ActionType, AutomationAction
        
        # Map action type
        action_type_map = {
            'click': ActionType.CLICK,
            'type': ActionType.TYPE,
            'wait': ActionType.WAIT,
            'navigate': ActionType.NAVIGATE,
            'scroll': ActionType.SCROLL,
            'hover': ActionType.HOVER
        }
        
        action_type_str = action_config.get('type', '').lower()
        action_type = action_type_map.get(action_type_str)
        
        if not action_type:
            logger.warning(f"Unknown action type: {action_type_str}")
            return None
        
        # Create action
        action = AutomationAction(
            type=action_type,
            selector=action_config.get('selector', ''),
            data=action_config.get('data', ''),
            timeout=action_config.get('timeout', 10),
            optional=action_config.get('optional', False)
        )
        
        # Add metadata
        if 'metadata' in action_config:
            action.metadata = action_config['metadata']
        
        return action
    
    def _should_use_intelligence(self, platform_name: str) -> bool:
        """Determine if intelligence should be used for platform"""
        
        # Check if platform is supported by intelligence system
        supported_platforms = self.platform_intelligence.get_all_platforms()
        
        if platform_name.lower() in supported_platforms:
            return True
        
        # Check if we have learning data for this platform
        if hasattr(self.earning_optimizer, 'learning_data'):
            if platform_name in self.earning_optimizer.learning_data:
                return len(self.earning_optimizer.learning_data[platform_name]) > 0
        
        # Default to intelligence for unknown platforms (adaptive learning)
        return True
    
    async def _execute_standard_task(self, task: AutomationTask, 
                                   config: IntelligentTaskConfig) -> TaskResult:
        """Execute task using standard BRAF components with some intelligence enhancements"""
        
        try:
            # Get enhanced browser profile
            browser_profile = await self.behavior_profiles.get_optimal_profile(
                config.platform_name, 
                1.0 - config.risk_tolerance
            )
            
            # Create browser instance with profile
            browser_config = {
                'headless': False,
                'user_agent': browser_profile.get('user_agent'),
                'viewport': browser_profile.get('viewport'),
                'locale': browser_profile.get('locale', 'en-US')
            }
            
            browser = BrowserInstance(browser_config)
            await browser.start()
            
            try:
                # Apply behavioral enhancements
                await self._apply_behavioral_enhancements(browser, browser_profile)
                
                # Execute task with enhanced behavior
                result = await self._execute_task_with_enhancements(
                    browser, task, browser_profile, config
                )
                
                return result
                
            finally:
                await browser.close()
                
        except Exception as e:
            logger.error(f"Standard task execution failed: {e}")
            return TaskResult(
                task_id=task.task_id,
                success=False,
                message=f"Standard execution failed: {str(e)}",
                data={'error': str(e)},
                execution_time=0,
                actions_completed=0
            )
    
    async def _apply_behavioral_enhancements(self, browser: BrowserInstance, 
                                           browser_profile: Dict[str, Any]):
        """Apply behavioral enhancements to browser"""
        
        page = browser.page
        
        # Apply anti-detection measures
        await page.evaluate("""
            // Remove webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            
            // Override plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            
            // Add chrome object
            window.chrome = { runtime: {} };
        """)
        
        # Set behavioral parameters
        await page.evaluate(f"""
            window.behaviorProfile = {browser_profile};
            window.humanBehavior = {{
                typingSpeed: {browser_profile.get('typing_speed', {}).get('min', 40)},
                mouseSpeed: {browser_profile.get('mouse_speed_range', {}).get('min', 100)},
                clickVariance: {browser_profile.get('click_variance', 0.2)}
            }};
        """)
    
    async def _execute_task_with_enhancements(self, browser: BrowserInstance,
                                            task: AutomationTask,
                                            browser_profile: Dict[str, Any],
                                            config: IntelligentTaskConfig) -> TaskResult:
        """Execute task with behavioral enhancements"""
        
        page = browser.page
        actions_completed = 0
        start_time = datetime.now()
        
        try:
            for i, action in enumerate(task.actions):
                # Apply pre-action behavior
                await self._apply_pre_action_behavior(page, action, browser_profile)
                
                # Execute action with enhancements
                action_success = await self._execute_enhanced_action(
                    page, action, browser_profile
                )
                
                if action_success:
                    actions_completed += 1
                elif not action.optional:
                    break
                
                # Apply post-action behavior
                await self._apply_post_action_behavior(page, action, browser_profile)
            
            # Calculate results
            execution_time = (datetime.now() - start_time).total_seconds()
            success_rate = actions_completed / len(task.actions) if task.actions else 0
            overall_success = success_rate >= 0.8
            
            return TaskResult(
                task_id=task.task_id,
                success=overall_success,
                message=f"Completed {actions_completed}/{len(task.actions)} actions",
                data={
                    'actions_completed': actions_completed,
                    'success_rate': success_rate,
                    'enhanced_execution': True,
                    'behavior_profile': browser_profile.get('name', 'unknown')
                },
                execution_time=execution_time,
                actions_completed=actions_completed
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TaskResult(
                task_id=task.task_id,
                success=False,
                message=f"Enhanced execution failed: {str(e)}",
                data={'error': str(e), 'actions_completed': actions_completed},
                execution_time=execution_time,
                actions_completed=actions_completed
            )
    
    async def _apply_pre_action_behavior(self, page, action, browser_profile: Dict[str, Any]):
        """Apply human-like behavior before action"""
        
        # Variable delay based on profile
        action_delays = browser_profile.get('action_delays', [1.0])
        base_delay = action_delays[0] if action_delays else 1.0
        
        # Add variance
        variance = browser_profile.get('timing_variance', 0.2)
        actual_delay = base_delay * (1 + (random.random() - 0.5) * variance * 2)
        
        await asyncio.sleep(max(0.1, actual_delay))
        
        # Random human actions
        if random.random() < browser_profile.get('pause_frequency', 0.1):
            await self._perform_random_human_action(page, browser_profile)
    
    async def _execute_enhanced_action(self, page, action, browser_profile: Dict[str, Any]) -> bool:
        """Execute action with behavioral enhancements"""
        
        try:
            from src.braf.core.task_executor import ActionType
            
            if action.type == ActionType.CLICK:
                return await self._enhanced_click(page, action, browser_profile)
            elif action.type == ActionType.TYPE:
                return await self._enhanced_type(page, action, browser_profile)
            elif action.type == ActionType.WAIT:
                await asyncio.sleep(float(action.data) if action.data else 1.0)
                return True
            elif action.type == ActionType.NAVIGATE:
                await page.goto(action.data)
                return True
            else:
                # Fallback to standard execution
                return await self._standard_action_execution(page, action)
                
        except Exception as e:
            logger.error(f"Enhanced action execution failed: {e}")
            return False
    
    async def _enhanced_click(self, page, action, browser_profile: Dict[str, Any]) -> bool:
        """Enhanced click with human-like behavior"""
        
        try:
            # Wait for element
            element = await page.wait_for_selector(action.selector, timeout=action.timeout * 1000)
            
            if not element:
                return False
            
            # Human-like mouse movement
            box = await element.bounding_box()
            if box:
                # Add click variance
                variance = browser_profile.get('click_variance', 0.2)
                offset_x = (random.random() - 0.5) * box['width'] * variance
                offset_y = (random.random() - 0.5) * box['height'] * variance
                
                click_x = box['x'] + box['width'] / 2 + offset_x
                click_y = box['y'] + box['height'] / 2 + offset_y
                
                # Move mouse first
                await page.mouse.move(click_x, click_y)
                await asyncio.sleep(random.uniform(0.05, 0.15))
                
                # Click
                await page.mouse.click(click_x, click_y)
            else:
                await element.click()
            
            return True
            
        except Exception as e:
            logger.error(f"Enhanced click failed: {e}")
            return False
    
    async def _enhanced_type(self, page, action, browser_profile: Dict[str, Any]) -> bool:
        """Enhanced typing with human-like patterns"""
        
        try:
            element = await page.wait_for_selector(action.selector, timeout=action.timeout * 1000)
            
            if not element:
                return False
            
            # Clear field
            await element.click()
            await page.keyboard.press('Control+A')
            
            # Type with human-like speed
            typing_speed = browser_profile.get('typing_speed', {'min': 40, 'max': 60})
            wpm = random.uniform(typing_speed['min'], typing_speed['max'])
            char_delay = 60 / (wpm * 5)  # Average 5 chars per word
            
            text = action.data
            for char in text:
                # Variable typing speed
                actual_delay = char_delay * random.uniform(0.5, 1.5)
                
                # Occasional typos
                if random.random() < browser_profile.get('typing_errors', 0.02):
                    # Type wrong character then correct
                    wrong_char = random.choice('abcdefghijklmnopqrstuvwxyz')
                    await page.keyboard.type(wrong_char)
                    await asyncio.sleep(actual_delay * 0.5)
                    await page.keyboard.press('Backspace')
                    await asyncio.sleep(browser_profile.get('backspace_delay', 0.3))
                
                await page.keyboard.type(char)
                await asyncio.sleep(actual_delay)
            
            return True
            
        except Exception as e:
            logger.error(f"Enhanced typing failed: {e}")
            return False
    
    async def _perform_random_human_action(self, page, browser_profile: Dict[str, Any]):
        """Perform random human-like action"""
        
        actions = ['scroll', 'mouse_move', 'pause']
        action = random.choice(actions)
        
        if action == 'scroll':
            scroll_patterns = browser_profile.get('scroll_patterns', [])
            if scroll_patterns:
                pattern = random.choice(scroll_patterns)
                distance = pattern.get('speed', 100)
                await page.mouse.wheel(0, distance)
        
        elif action == 'mouse_move':
            viewport = await page.viewport_size()
            x = random.randint(0, viewport['width'])
            y = random.randint(0, viewport['height'])
            await page.mouse.move(x, y)
        
        elif action == 'pause':
            await asyncio.sleep(random.uniform(0.5, 2.0))
    
    async def _apply_post_action_behavior(self, page, action, browser_profile: Dict[str, Any]):
        """Apply behavior after action execution"""
        
        # Short delay after action
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Occasional additional human behavior
        if random.random() < 0.05:  # 5% chance
            await self._perform_random_human_action(page, browser_profile)
    
    async def _standard_action_execution(self, page, action) -> bool:
        """Fallback to standard action execution"""
        
        try:
            # Use the base task executor for standard execution
            result = await self.task_executor._execute_action(page, action)
            return result.get('success', False)
        except Exception as e:
            logger.error(f"Standard action execution failed: {e}")
            return False
    
    async def _enhance_execution_result(self, result: TaskResult, 
                                      platform_name: str,
                                      config: IntelligentTaskConfig) -> Dict[str, Any]:
        """Enhance execution result with intelligence data"""
        
        # Convert TaskResult to dictionary
        enhanced_result = {
            'success': result.success,
            'message': result.message,
            'data': result.data,
            'execution_time': result.execution_time,
            'actions_completed': result.actions_completed,
            'task_id': result.task_id,
            
            # Intelligence enhancements
            'platform': platform_name,
            'intelligence_used': self.intelligence_enabled,
            'optimization_level': config.optimization_level,
            'risk_tolerance': config.risk_tolerance,
            'stealth_mode': config.stealth_mode,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add platform analysis if available
        if platform_name in self.platform_intelligence.platform_analyses:
            enhanced_result['platform_analysis'] = self.platform_intelligence.platform_analyses[platform_name]
        
        # Add earning optimization data
        if hasattr(self.earning_optimizer, 'platform_performance'):
            if platform_name in self.earning_optimizer.platform_performance:
                enhanced_result['performance_metrics'] = self.earning_optimizer.platform_performance[platform_name]
        
        # Learn from execution if enabled
        if config.learning_enabled:
            await self.earning_optimizer.learn_from_execution(platform_name, enhanced_result)
            await self.behavior_profiles.update_from_execution(platform_name, enhanced_result)
        
        return enhanced_result
    
    async def get_platform_recommendations(self, platform_name: str) -> Dict[str, Any]:
        """Get intelligence recommendations for platform"""
        
        try:
            # Get platform analysis
            platform_profile = self.platform_intelligence.get_platform_profile(platform_name)
            
            if not platform_profile:
                return {
                    'platform': platform_name,
                    'supported': False,
                    'message': 'Platform not in intelligence database'
                }
            
            # Get optimal execution time
            optimal_time = self.earning_optimizer.get_optimal_execution_time(platform_name)
            
            # Get earning forecast
            forecast = self.earning_optimizer.get_earning_forecast(platform_name, 24)
            
            # Get behavior recommendations
            behavior_profile = await self.behavior_profiles.get_optimal_profile(platform_name)
            
            return {
                'platform': platform_name,
                'supported': True,
                'platform_profile': {
                    'name': platform_profile.name,
                    'earning_rate_usd_per_hour': platform_profile.earning_rate_usd_per_hour,
                    'reliability_score': platform_profile.reliability_score,
                    'payment_threshold': platform_profile.payment_threshold,
                    'best_time_of_day': platform_profile.best_time_of_day
                },
                'optimal_timing': optimal_time,
                'earning_forecast': forecast,
                'recommended_behavior': {
                    'profile_name': behavior_profile.get('name'),
                    'risk_level': 'low' if behavior_profile.get('timing_variance', 0.3) > 0.25 else 'medium',
                    'estimated_success_rate': behavior_profile.get('success_rate', 0.8)
                },
                'recommendations': [
                    f"Best earning rate: ${platform_profile.earning_rate_usd_per_hour:.2f}/hour",
                    f"Optimal times: {', '.join(platform_profile.best_time_of_day)}",
                    f"Payment threshold: ${platform_profile.payment_threshold}",
                    f"Reliability score: {platform_profile.reliability_score}%"
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get platform recommendations: {e}")
            return {
                'platform': platform_name,
                'supported': False,
                'error': str(e)
            }
    
    def get_intelligence_status(self) -> Dict[str, Any]:
        """Get status of intelligence system"""
        
        return {
            'intelligence_enabled': self.intelligence_enabled,
            'supported_platforms': self.platform_intelligence.get_all_platforms(),
            'active_sessions': len(self.active_sessions),
            'platform_intelligence_stats': {
                'platforms_analyzed': len(self.platform_intelligence.platform_analyses),
                'cached_analyses': len(self.platform_intelligence.analysis_cache)
            },
            'behavior_profile_stats': {
                'total_profiles': sum(len(profiles) for profiles in self.behavior_profiles.profiles.values()),
                'platforms_with_profiles': len(self.behavior_profiles.profiles)
            },
            'earning_optimizer_stats': self.earning_optimizer.get_optimization_stats(),
            'integration_version': '2.0'
        }
    
    def enable_intelligence(self):
        """Enable intelligence system"""
        self.intelligence_enabled = True
        logger.info("Intelligence system enabled")
    
    def disable_intelligence(self):
        """Disable intelligence system (fallback to standard BRAF)"""
        self.intelligence_enabled = False
        logger.info("Intelligence system disabled - using standard BRAF")

# Global instance
braf_intelligence = BRAFIntelligenceIntegration()

# Import random for behavioral variance
import random