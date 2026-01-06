"""
Earning Optimizer
Machine learning-based optimization for maximum earnings
"""

import json
import logging
import random
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict, deque
import math

logger = logging.getLogger(__name__)

@dataclass
class EarningPattern:
    """Pattern for earning optimization"""
    platform: str
    time_of_day: int
    day_of_week: int
    task_type: str
    success_rate: float
    avg_earning: float
    completion_time: float
    sample_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class OptimizationStrategy:
    """Optimization strategy for a platform"""
    platform: str
    priority_tasks: List[str]
    optimal_times: List[int]  # Hours of day
    risk_tolerance: float
    speed_multiplier: float
    success_threshold: float
    earning_target: float

class EarningOptimizer:
    """Advanced earning optimization using machine learning techniques"""
    
    def __init__(self):
        self.earning_patterns: Dict[str, List[EarningPattern]] = defaultdict(list)
        self.optimization_strategies: Dict[str, OptimizationStrategy] = {}
        self.execution_history: deque = deque(maxlen=1000)
        self.learning_data: Dict[str, List[Dict]] = defaultdict(list)
        
        # ML parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.15  # 15% exploration vs exploitation
        self.confidence_threshold = 0.7
        
        # Performance tracking
        self.platform_performance: Dict[str, Dict] = defaultdict(dict)
        self.optimization_metrics: Dict[str, Any] = {}
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
        logger.info("Earning Optimizer initialized with ML capabilities")
    
    def _initialize_default_strategies(self):
        """Initialize default optimization strategies for known platforms"""
        
        # Swagbucks strategy
        self.optimization_strategies["swagbucks"] = OptimizationStrategy(
            platform="swagbucks",
            priority_tasks=["gold_surveys", "daily_poll", "search", "watch"],
            optimal_times=[9, 10, 14, 15, 20, 21],  # Peak hours
            risk_tolerance=0.6,
            speed_multiplier=1.0,
            success_threshold=0.8,
            earning_target=25.0  # Daily target in USD
        )
        
        # Survey Junkie strategy
        self.optimization_strategies["survey_junkie"] = OptimizationStrategy(
            platform="survey_junkie",
            priority_tasks=["surveys", "pulse_surveys", "product_testing"],
            optimal_times=[10, 11, 15, 16, 19, 20],
            risk_tolerance=0.5,
            speed_multiplier=0.9,
            success_threshold=0.75,
            earning_target=20.0
        )
        
        # PrizeRebel strategy
        self.optimization_strategies["prizerebel"] = OptimizationStrategy(
            platform="prizerebel",
            priority_tasks=["surveys", "offers", "tasks", "referrals"],
            optimal_times=[11, 12, 16, 17, 21, 22],
            risk_tolerance=0.7,
            speed_multiplier=1.1,
            success_threshold=0.7,
            earning_target=18.0
        )
    
    async def optimize_task_sequence(self, task, platform_analysis: Dict[str, Any]) -> List[Any]:
        """Optimize task action sequence for maximum earnings"""
        
        platform_name = platform_analysis.get('platform_identification', {}).get('platform_name', 'unknown')
        
        # Get current optimization strategy
        strategy = self.optimization_strategies.get(platform_name)
        if not strategy:
            strategy = self._create_adaptive_strategy(platform_name, platform_analysis)
        
        # Analyze task for optimization opportunities
        optimized_actions = await self._optimize_actions(task.actions, strategy, platform_analysis)
        
        # Apply ML-based sequence optimization
        optimized_actions = await self._apply_ml_optimization(optimized_actions, platform_name)
        
        # Add earning-focused enhancements
        optimized_actions = await self._add_earning_enhancements(optimized_actions, strategy)
        
        return optimized_actions
    
    def _create_adaptive_strategy(self, platform_name: str, platform_analysis: Dict[str, Any]) -> OptimizationStrategy:
        """Create adaptive strategy for unknown platforms"""
        
        # Extract platform characteristics
        earning_potential = platform_analysis.get('earning_potential', {})
        risk_assessment = platform_analysis.get('risk_assessment', {})
        
        daily_potential = earning_potential.get('daily_potential', 15.0)
        risk_level = risk_assessment.get('overall_risk', 0.5)
        
        # Create adaptive strategy
        strategy = OptimizationStrategy(
            platform=platform_name,
            priority_tasks=["surveys", "offers", "tasks"],
            optimal_times=earning_potential.get('peak_hours', [9, 14, 20]),
            risk_tolerance=1.0 - risk_level,
            speed_multiplier=1.0 - (risk_level * 0.3),
            success_threshold=0.8 - (risk_level * 0.1),
            earning_target=daily_potential * 0.8  # Conservative target
        )
        
        self.optimization_strategies[platform_name] = strategy
        return strategy
    
    async def _optimize_actions(self, actions: List[Any], strategy: OptimizationStrategy, 
                              platform_analysis: Dict[str, Any]) -> List[Any]:
        """Optimize individual actions based on strategy"""
        
        optimized_actions = []
        
        for action in actions:
            # Create optimized version of action
            optimized_action = self._create_optimized_action(action, strategy)
            
            # Add pre-action optimizations
            pre_actions = self._generate_pre_action_optimizations(action, strategy)
            optimized_actions.extend(pre_actions)
            
            # Add the main action
            optimized_actions.append(optimized_action)
            
            # Add post-action optimizations
            post_actions = self._generate_post_action_optimizations(action, strategy)
            optimized_actions.extend(post_actions)
        
        return optimized_actions
    
    def _create_optimized_action(self, action, strategy: OptimizationStrategy):
        """Create optimized version of an action"""
        
        # Clone the action (simplified - in real implementation would properly clone)
        optimized_action = action
        
        # Apply speed multiplier to timing
        if hasattr(action, 'delay'):
            optimized_action.delay = action.delay / strategy.speed_multiplier
        
        # Add optimization metadata
        if hasattr(action, 'metadata'):
            action.metadata = getattr(action, 'metadata', {})
            action.metadata.update({
                'optimized': True,
                'strategy': strategy.platform,
                'speed_multiplier': strategy.speed_multiplier
            })
        
        return optimized_action
    
    def _generate_pre_action_optimizations(self, action, strategy: OptimizationStrategy) -> List[Any]:
        """Generate pre-action optimization steps"""
        
        pre_actions = []
        
        # Add intelligent waiting based on platform patterns
        if hasattr(action, 'type') and str(action.type) == 'CLICK':
            # Add pre-click analysis action
            pre_actions.append(self._create_analysis_action('pre_click_analysis'))
        
        return pre_actions
    
    def _generate_post_action_optimizations(self, action, strategy: OptimizationStrategy) -> List[Any]:
        """Generate post-action optimization steps"""
        
        post_actions = []
        
        # Add verification steps for critical actions
        if hasattr(action, 'type') and str(action.type) in ['CLICK', 'TYPE']:
            post_actions.append(self._create_verification_action(action))
        
        return post_actions
    
    def _create_analysis_action(self, analysis_type: str):
        """Create analysis action for optimization"""
        
        # Simplified action creation - in real implementation would use proper action classes
        class AnalysisAction:
            def __init__(self, analysis_type):
                self.type = 'ANALYSIS'
                self.analysis_type = analysis_type
                self.delay = 0.1
        
        return AnalysisAction(analysis_type)
    
    def _create_verification_action(self, original_action):
        """Create verification action"""
        
        class VerificationAction:
            def __init__(self, original_action):
                self.type = 'VERIFY'
                self.original_action = original_action
                self.delay = 0.2
        
        return VerificationAction(original_action)
    
    async def _apply_ml_optimization(self, actions: List[Any], platform_name: str) -> List[Any]:
        """Apply machine learning-based optimization"""
        
        # Get historical performance data
        performance_data = self.platform_performance.get(platform_name, {})
        
        if not performance_data or len(self.learning_data[platform_name]) < 10:
            # Not enough data for ML optimization
            return actions
        
        # Apply learned optimizations
        optimized_actions = []
        
        for i, action in enumerate(actions):
            # Predict optimal timing for this action
            optimal_delay = self._predict_optimal_delay(action, platform_name, i)
            
            # Apply predicted optimization
            if hasattr(action, 'delay'):
                action.delay = optimal_delay
            
            # Predict success probability
            success_prob = self._predict_action_success(action, platform_name)
            
            # Add retry logic for low-probability actions
            if success_prob < self.confidence_threshold:
                action = self._add_retry_logic(action)
            
            optimized_actions.append(action)
        
        return optimized_actions
    
    def _predict_optimal_delay(self, action, platform_name: str, position: int) -> float:
        """Predict optimal delay for action using ML"""
        
        # Get historical timing data
        timing_data = []
        for execution in self.learning_data[platform_name]:
            if len(execution.get('action_timings', [])) > position:
                timing_data.append(execution['action_timings'][position])
        
        if not timing_data:
            return 1.0  # Default delay
        
        # Calculate optimal timing based on success correlation
        successful_timings = []
        failed_timings = []
        
        for execution in self.learning_data[platform_name]:
            if execution.get('success', False) and len(execution.get('action_timings', [])) > position:
                successful_timings.append(execution['action_timings'][position])
            elif not execution.get('success', False) and len(execution.get('action_timings', [])) > position:
                failed_timings.append(execution['action_timings'][position])
        
        if successful_timings:
            # Use average of successful timings
            optimal_delay = sum(successful_timings) / len(successful_timings)
            
            # Add some variance to avoid detection
            variance = 0.2
            optimal_delay *= (1 + random.uniform(-variance, variance))
            
            return max(0.1, optimal_delay)
        
        return 1.0
    
    def _predict_action_success(self, action, platform_name: str) -> float:
        """Predict action success probability"""
        
        # Simplified ML prediction based on historical data
        action_type = str(getattr(action, 'type', 'UNKNOWN'))
        
        success_rates = []
        for execution in self.learning_data[platform_name]:
            for action_result in execution.get('action_results', []):
                if action_result.get('type') == action_type:
                    success_rates.append(1.0 if action_result.get('success', False) else 0.0)
        
        if success_rates:
            return sum(success_rates) / len(success_rates)
        
        return 0.8  # Default confidence
    
    def _add_retry_logic(self, action):
        """Add retry logic to action"""
        
        if hasattr(action, 'metadata'):
            action.metadata = getattr(action, 'metadata', {})
        else:
            action.metadata = {}
        
        action.metadata.update({
            'retry_enabled': True,
            'max_retries': 2,
            'retry_delay': 1.0
        })
        
        return action
    
    async def _add_earning_enhancements(self, actions: List[Any], strategy: OptimizationStrategy) -> List[Any]:
        """Add earning-focused enhancements"""
        
        enhanced_actions = []
        
        # Add earning opportunity detection
        enhanced_actions.append(self._create_earning_scan_action())
        
        # Process original actions with enhancements
        for action in actions:
            enhanced_actions.append(action)
            
            # Add earning verification after key actions
            if self._is_earning_critical_action(action):
                enhanced_actions.append(self._create_earning_verification_action())
        
        # Add final earning collection
        enhanced_actions.append(self._create_earning_collection_action())
        
        return enhanced_actions
    
    def _create_earning_scan_action(self):
        """Create action to scan for earning opportunities"""
        
        class EarningScanAction:
            def __init__(self):
                self.type = 'EARNING_SCAN'
                self.delay = 0.5
                self.metadata = {'optimization': 'earning_scan'}
        
        return EarningScanAction()
    
    def _create_earning_verification_action(self):
        """Create action to verify earnings"""
        
        class EarningVerificationAction:
            def __init__(self):
                self.type = 'EARNING_VERIFY'
                self.delay = 0.3
                self.metadata = {'optimization': 'earning_verification'}
        
        return EarningVerificationAction()
    
    def _create_earning_collection_action(self):
        """Create action to collect earnings"""
        
        class EarningCollectionAction:
            def __init__(self):
                self.type = 'EARNING_COLLECT'
                self.delay = 1.0
                self.metadata = {'optimization': 'earning_collection'}
        
        return EarningCollectionAction()
    
    def _is_earning_critical_action(self, action) -> bool:
        """Check if action is critical for earnings"""
        
        action_type = str(getattr(action, 'type', ''))
        critical_types = ['CLICK', 'TYPE', 'SUBMIT']
        
        return action_type in critical_types
    
    async def learn_from_execution(self, platform_name: str, execution_result: Dict[str, Any]):
        """Learn from execution results to improve optimization"""
        
        # Extract learning data
        learning_entry = {
            'timestamp': datetime.now(),
            'platform': platform_name,
            'success': execution_result.get('success', False),
            'earning_amount': execution_result.get('data', {}).get('earning_amount', 0),
            'execution_time': execution_result.get('execution_time', 0),
            'actions_completed': execution_result.get('actions_completed', 0),
            'action_results': execution_result.get('data', {}).get('action_results', []),
            'action_timings': self._extract_action_timings(execution_result)
        }
        
        # Store learning data
        self.learning_data[platform_name].append(learning_entry)
        
        # Keep only recent data (last 100 executions per platform)
        if len(self.learning_data[platform_name]) > 100:
            self.learning_data[platform_name] = self.learning_data[platform_name][-100:]
        
        # Update performance metrics
        await self._update_performance_metrics(platform_name, learning_entry)
        
        # Update optimization strategy
        await self._update_optimization_strategy(platform_name)
        
        # Update earning patterns
        await self._update_earning_patterns(platform_name, learning_entry)
    
    def _extract_action_timings(self, execution_result: Dict[str, Any]) -> List[float]:
        """Extract action timing data from execution result"""
        
        action_results = execution_result.get('data', {}).get('action_results', [])
        timings = []
        
        for result in action_results:
            timing = result.get('execution_time', 1.0)
            timings.append(timing)
        
        return timings
    
    async def _update_performance_metrics(self, platform_name: str, learning_entry: Dict[str, Any]):
        """Update performance metrics for platform"""
        
        if platform_name not in self.platform_performance:
            self.platform_performance[platform_name] = {
                'total_executions': 0,
                'successful_executions': 0,
                'total_earnings': 0.0,
                'total_time': 0.0,
                'avg_success_rate': 0.0,
                'avg_earning_rate': 0.0,
                'last_updated': datetime.now()
            }
        
        metrics = self.platform_performance[platform_name]
        
        # Update counters
        metrics['total_executions'] += 1
        if learning_entry['success']:
            metrics['successful_executions'] += 1
        
        metrics['total_earnings'] += learning_entry['earning_amount']
        metrics['total_time'] += learning_entry['execution_time']
        
        # Calculate rates
        metrics['avg_success_rate'] = metrics['successful_executions'] / metrics['total_executions']
        metrics['avg_earning_rate'] = metrics['total_earnings'] / max(metrics['total_time'] / 3600, 0.01)  # Per hour
        metrics['last_updated'] = datetime.now()
    
    async def _update_optimization_strategy(self, platform_name: str):
        """Update optimization strategy based on performance"""
        
        if platform_name not in self.optimization_strategies:
            return
        
        strategy = self.optimization_strategies[platform_name]
        metrics = self.platform_performance.get(platform_name, {})
        
        success_rate = metrics.get('avg_success_rate', 0.5)
        
        # Adaptive strategy adjustment
        if success_rate < 0.6:  # Low success rate - be more conservative
            strategy.speed_multiplier *= 0.95
            strategy.risk_tolerance *= 0.9
            strategy.success_threshold = min(0.9, strategy.success_threshold + 0.05)
        elif success_rate > 0.85:  # High success rate - can be more aggressive
            strategy.speed_multiplier *= 1.02
            strategy.risk_tolerance = min(1.0, strategy.risk_tolerance * 1.05)
            strategy.success_threshold = max(0.6, strategy.success_threshold - 0.02)
        
        logger.info(f"Updated strategy for {platform_name}: speed={strategy.speed_multiplier:.2f}, risk={strategy.risk_tolerance:.2f}")
    
    async def _update_earning_patterns(self, platform_name: str, learning_entry: Dict[str, Any]):
        """Update earning patterns for time-based optimization"""
        
        timestamp = learning_entry['timestamp']
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Find or create pattern
        pattern = None
        for p in self.earning_patterns[platform_name]:
            if p.time_of_day == hour and p.day_of_week == day_of_week:
                pattern = p
                break
        
        if not pattern:
            pattern = EarningPattern(
                platform=platform_name,
                time_of_day=hour,
                day_of_week=day_of_week,
                task_type='general',
                success_rate=0.0,
                avg_earning=0.0,
                completion_time=0.0
            )
            self.earning_patterns[platform_name].append(pattern)
        
        # Update pattern with exponential moving average
        alpha = 0.1  # Learning rate
        
        pattern.success_rate = (1 - alpha) * pattern.success_rate + alpha * (1.0 if learning_entry['success'] else 0.0)
        pattern.avg_earning = (1 - alpha) * pattern.avg_earning + alpha * learning_entry['earning_amount']
        pattern.completion_time = (1 - alpha) * pattern.completion_time + alpha * learning_entry['execution_time']
        pattern.sample_count += 1
        pattern.last_updated = datetime.now()
    
    def get_optimal_execution_time(self, platform_name: str) -> Dict[str, Any]:
        """Get optimal execution time based on learned patterns"""
        
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        
        # Get patterns for platform
        patterns = self.earning_patterns.get(platform_name, [])
        
        if not patterns:
            return {
                'optimal_now': True,
                'reason': 'No historical data available',
                'confidence': 0.5
            }
        
        # Find current time pattern
        current_pattern = None
        for pattern in patterns:
            if pattern.time_of_day == current_hour and pattern.day_of_week == current_day:
                current_pattern = pattern
                break
        
        # Calculate scores for all time slots
        time_scores = []
        for pattern in patterns:
            if pattern.sample_count >= 3:  # Minimum samples for reliability
                # Score based on success rate and earning potential
                score = (pattern.success_rate * 0.6 + 
                        (pattern.avg_earning / max(pattern.completion_time, 1)) * 0.4)
                
                time_scores.append({
                    'hour': pattern.time_of_day,
                    'day': pattern.day_of_week,
                    'score': score,
                    'success_rate': pattern.success_rate,
                    'avg_earning': pattern.avg_earning
                })
        
        if not time_scores:
            return {
                'optimal_now': True,
                'reason': 'Insufficient data for optimization',
                'confidence': 0.5
            }
        
        # Sort by score
        time_scores.sort(key=lambda x: x['score'], reverse=True)
        best_time = time_scores[0]
        
        # Check if current time is optimal
        current_score = 0.5  # Default score
        if current_pattern and current_pattern.sample_count >= 3:
            current_score = (current_pattern.success_rate * 0.6 + 
                           (current_pattern.avg_earning / max(current_pattern.completion_time, 1)) * 0.4)
        
        is_optimal_now = current_score >= best_time['score'] * 0.9  # Within 90% of best
        
        return {
            'optimal_now': is_optimal_now,
            'current_score': current_score,
            'best_score': best_time['score'],
            'best_time': {
                'hour': best_time['hour'],
                'day': best_time['day']
            },
            'confidence': min(current_pattern.sample_count / 10, 1.0) if current_pattern else 0.1,
            'recommendation': 'Execute now' if is_optimal_now else f"Wait until {best_time['hour']}:00"
        }
    
    def get_earning_forecast(self, platform_name: str, hours_ahead: int = 24) -> Dict[str, Any]:
        """Get earning forecast for next N hours"""
        
        patterns = self.earning_patterns.get(platform_name, [])
        
        if not patterns:
            return {
                'forecast': [],
                'total_potential': 0.0,
                'confidence': 0.0
            }
        
        forecast = []
        total_potential = 0.0
        
        current_time = datetime.now()
        
        for i in range(hours_ahead):
            future_time = current_time + timedelta(hours=i)
            hour = future_time.hour
            day = future_time.weekday()
            
            # Find matching pattern
            matching_pattern = None
            for pattern in patterns:
                if pattern.time_of_day == hour and pattern.day_of_week == day:
                    matching_pattern = pattern
                    break
            
            if matching_pattern and matching_pattern.sample_count >= 2:
                hourly_potential = matching_pattern.avg_earning * matching_pattern.success_rate
                confidence = min(matching_pattern.sample_count / 10, 1.0)
            else:
                # Use average from all patterns
                avg_earning = sum(p.avg_earning for p in patterns) / len(patterns) if patterns else 0
                avg_success = sum(p.success_rate for p in patterns) / len(patterns) if patterns else 0.5
                hourly_potential = avg_earning * avg_success * 0.5  # Reduced confidence
                confidence = 0.3
            
            forecast.append({
                'hour': hour,
                'day': day,
                'datetime': future_time.isoformat(),
                'potential_earning': hourly_potential,
                'confidence': confidence
            })
            
            total_potential += hourly_potential
        
        # Calculate overall confidence
        overall_confidence = sum(f['confidence'] for f in forecast) / len(forecast) if forecast else 0
        
        return {
            'forecast': forecast,
            'total_potential': total_potential,
            'confidence': overall_confidence,
            'platform': platform_name,
            'forecast_period_hours': hours_ahead
        }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization system statistics"""
        
        return {
            'platforms_optimized': len(self.optimization_strategies),
            'total_executions': len(self.execution_history),
            'learning_data_points': sum(len(data) for data in self.learning_data.values()),
            'earning_patterns': sum(len(patterns) for patterns in self.earning_patterns.values()),
            'platform_performance': dict(self.platform_performance),
            'optimization_version': '2.0'
        }

# Global instance
earning_optimizer = EarningOptimizer()
