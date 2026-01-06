"""
ACADEMIC TASK SCHEDULER
Intelligent scheduling for academic research tasks
"""

import asyncio
import heapq
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random


class AcademicSchedulingAlgorithm(Enum):
    """Academic scheduling algorithms"""
    PRIORITY_QUEUE = "priority_based_academic"
    DEADLINE_DRIVEN = "deadline_driven_academic"
    RESOURCE_AWARE = "resource_aware_academic"
    ML_OPTIMIZED = "ml_optimized_academic"
    HYBRID_APPROACH = "hybrid_academic_scheduling"


@dataclass(order=True)
class ScheduledAcademicTask:
    """Scheduled academic task"""
    priority: int
    estimated_completion: datetime
    task_id: str = field(compare=False)
    task_data: Dict = field(compare=False)
    scheduling_algorithm: AcademicSchedulingAlgorithm = field(compare=False)


class AcademicTaskScheduler:
    """Intelligent academic task scheduler"""
    
    def __init__(self):
        self.task_queue = []
        self.scheduled_tasks: Dict[str, ScheduledAcademicTask] = {}
        self.scheduling_history = []
        self.performance_metrics = {
            "tasks_scheduled": 0,
            "tasks_completed": 0,
            "average_scheduling_delay": timedelta(),
            "algorithm_effectiveness": {}
        }

    async def initialize_scheduler(self) -> Dict:
        """Initialize academic task scheduler"""
        print("⏰ Initializing Academic Task Scheduler")
        
        # Load scheduling algorithms
        algorithms = self._load_scheduling_algorithms()
        
        # Initialize performance tracking
        await self._initialize_performance_tracking()
        
        # Start background optimization
        asyncio.create_task(self._optimize_scheduling_strategies())
        
        return {
            "scheduler_initialized": True,
            "available_algorithms": [alg.value for alg in algorithms],
            "initial_queue_size": len(self.task_queue),
            "scheduler_capacity": "unlimited_academic"
        }

    def _load_scheduling_algorithms(self) -> List[AcademicSchedulingAlgorithm]:
        """Load scheduling algorithms"""
        return [
            AcademicSchedulingAlgorithm.PRIORITY_QUEUE,
            AcademicSchedulingAlgorithm.DEADLINE_DRIVEN,
            AcademicSchedulingAlgorithm.RESOURCE_AWARE,
            AcademicSchedulingAlgorithm.ML_OPTIMIZED,
            AcademicSchedulingAlgorithm.HYBRID_APPROACH
        ]

    async def _initialize_performance_tracking(self):
        """Initialize performance tracking"""
        # Setup metrics collection
        print("📊 Initializing Academic Performance Tracking")
        
        self.performance_metrics = {
            "tasks_scheduled": 0,
            "tasks_completed": 0,
            "average_scheduling_delay": timedelta(),
            "algorithm_effectiveness": {
                alg.value: {
                    "success_rate": 0.0, 
                    "average_completion_time": timedelta()
                } for alg in AcademicSchedulingAlgorithm
            },
            "resource_utilization": {
                "cpu_efficiency": 0.0,
                "memory_efficiency": 0.0,
                "network_efficiency": 0.0
            }
        }

    async def schedule_academic_task(self, task_data: Dict) -> str:
        """Schedule an academic task"""
        print(f"📅 Scheduling Academic Task: {task_data.get('task_type', 'unknown')}")
        
        # Generate task ID
        task_id = f"academic_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        # Select scheduling algorithm
        algorithm = self._select_scheduling_algorithm(task_data)
        
        # Calculate priority
        priority = self._calculate_task_priority(task_data, algorithm)
        
        # Estimate completion time
        estimated_completion = self._estimate_completion_time(task_data, algorithm)
        
        # Create scheduled task
        scheduled_task = ScheduledAcademicTask(
            priority=priority,
            estimated_completion=estimated_completion,
            task_id=task_id,
            task_data=task_data,
            scheduling_algorithm=algorithm
        )
        
        # Add to queue
        heapq.heappush(self.task_queue, scheduled_task)
        self.scheduled_tasks[task_id] = scheduled_task
        
        # Update metrics
        self.performance_metrics["tasks_scheduled"] += 1
        self.scheduling_history.append({
            "task_id": task_id,
            "scheduled_at": datetime.now(),
            "algorithm": algorithm.value,
            "priority": priority
        })
        
        return task_id

    def _select_scheduling_algorithm(self, task_data: Dict) -> AcademicSchedulingAlgorithm:
        """Select appropriate scheduling algorithm"""
        task_type = task_data.get("type", "general_research")
        
        # Algorithm selection logic
        algorithm_map = {
            "urgent_research": AcademicSchedulingAlgorithm.PRIORITY_QUEUE,
            "deadline_driven": AcademicSchedulingAlgorithm.DEADLINE_DRIVEN,
            "resource_intensive": AcademicSchedulingAlgorithm.RESOURCE_AWARE,
            "complex_analysis": AcademicSchedulingAlgorithm.ML_OPTIMIZED,
            "long_term_study": AcademicSchedulingAlgorithm.HYBRID_APPROACH
        }
        
        return algorithm_map.get(task_type, AcademicSchedulingAlgorithm.HYBRID_APPROACH)

    def _calculate_task_priority(self, task_data: Dict, algorithm: AcademicSchedulingAlgorithm) -> int:
        """Calculate task priority"""
        base_priority = 0
        
        # Priority factors
        factors = {
            "urgency": task_data.get("urgency", 1) * 10,
            "importance": task_data.get("importance", 1) * 8,
            "deadline_proximity": self._calculate_deadline_proximity(task_data) * 6,
            "resource_requirements": (1 / max(task_data.get("resource_intensity", 1), 1)) * 4,
            "academic_impact": task_data.get("academic_impact", 1) * 12
        }
        
        # Weight based on algorithm
        weights = {
            AcademicSchedulingAlgorithm.PRIORITY_QUEUE: {
                "academic_impact": 2.0, "urgency": 1.5
            },
            AcademicSchedulingAlgorithm.DEADLINE_DRIVEN: {
                "deadline_proximity": 2.0
            },
            AcademicSchedulingAlgorithm.RESOURCE_AWARE: {
                "resource_requirements": 2.0
            },
            AcademicSchedulingAlgorithm.ML_OPTIMIZED: {
                "academic_impact": 1.8, "importance": 1.2
            },
            AcademicSchedulingAlgorithm.HYBRID_APPROACH: {
                factor: 1.0 for factor in factors
            }
        }
        
        # Calculate weighted priority
        for factor, value in factors.items():
            weight = weights[algorithm].get(factor, 1.0)
            base_priority += int(value * weight)
        
        # Invert for heapq (lower number = higher priority)
        return -base_priority

    def _calculate_deadline_proximity(self, task_data: Dict) -> int:
        """Calculate deadline proximity score"""
        deadline_str = task_data.get("deadline")
        if not deadline_str:
            return 1
        
        try:
            deadline = datetime.fromisoformat(deadline_str)
            days_until_deadline = (deadline - datetime.now()).days
            
            if days_until_deadline <= 0:
                return 10  # Overdue
            elif days_until_deadline <= 1:
                return 9   # Due tomorrow
            elif days_until_deadline <= 7:
                return 7   # Due this week
            elif days_until_deadline <= 30:
                return 5   # Due this month
            else:
                return 3   # Long-term
        except:
            return 1

    def _estimate_completion_time(self, task_data: Dict, algorithm: AcademicSchedulingAlgorithm) -> datetime:
        """Estimate completion time"""
        base_duration_hours = task_data.get("estimated_duration_hours", 1)
        
        # Adjust based on algorithm
        algorithm_multipliers = {
            AcademicSchedulingAlgorithm.PRIORITY_QUEUE: 0.8,  # Faster for high priority
            AcademicSchedulingAlgorithm.DEADLINE_DRIVEN: 0.9,
            AcademicSchedulingAlgorithm.RESOURCE_AWARE: 1.0,
            AcademicSchedulingAlgorithm.ML_OPTIMIZED: 0.7,  # ML optimized
            AcademicSchedulingAlgorithm.HYBRID_APPROACH: 0.85
        }
        
        adjusted_hours = base_duration_hours * algorithm_multipliers.get(algorithm, 1.0)
        
        # Add scheduling delay estimate
        current_queue_size = len(self.task_queue)
        queue_delay_hours = current_queue_size * 0.1  # 6 minutes per queued task
        
        total_hours = adjusted_hours + queue_delay_hours
        return datetime.now() + timedelta(hours=total_hours)

    async def get_next_academic_task(self) -> Optional[Dict]:
        """Get next academic task for execution"""
        if not self.task_queue:
            return None
        
        # Get highest priority task
        scheduled_task = heapq.heappop(self.task_queue)
        
        # Remove from scheduled tasks
        if scheduled_task.task_id in self.scheduled_tasks:
            del self.scheduled_tasks[scheduled_task.task_id]
        
        # Prepare task for execution
        task_data = scheduled_task.task_data.copy()
        task_data["scheduled_task_id"] = scheduled_task.task_id
        task_data["scheduled_algorithm"] = scheduled_task.scheduling_algorithm.value
        task_data["estimated_completion"] = scheduled_task.estimated_completion.isoformat()
        
        return task_data

    async def mark_task_completed(self, task_id: str, completion_data: Dict) -> bool:
        """Mark academic task as completed"""
        print(f"✅ Marking Academic Task Completed: {task_id}")
        
        # Update performance metrics
        self.performance_metrics["tasks_completed"] += 1
        
        # Calculate actual completion time
        scheduled_time = None
        for history in self.scheduling_history:
            if history["task_id"] == task_id:
                scheduled_time = history["scheduled_at"]
                break
        
        if scheduled_time:
            actual_delay = datetime.now() - scheduled_time
            self.performance_metrics["average_scheduling_delay"] = (
                self.performance_metrics["average_scheduling_delay"] + actual_delay
            ) / 2
        
        # Update algorithm effectiveness
        algorithm = completion_data.get("scheduling_algorithm")
        if algorithm:
            if algorithm not in self.performance_metrics["algorithm_effectiveness"]:
                self.performance_metrics["algorithm_effectiveness"][algorithm] = {
                    "success_rate": 0.0,
                    "average_completion_time": timedelta()
                }
            
            # Update success rate
            current_stats = self.performance_metrics["algorithm_effectiveness"][algorithm]
            current_success_rate = current_stats["success_rate"]
            new_success_rate = (current_success_rate + 1.0) / 2  # Moving average
            current_stats["success_rate"] = new_success_rate
        
        return True

    async def get_scheduling_metrics(self) -> Dict:
        """Get scheduling performance metrics"""
        metrics = self.performance_metrics.copy()
        
        # Calculate current efficiency
        if metrics["tasks_scheduled"] > 0:
            completion_rate = metrics["tasks_completed"] / metrics["tasks_scheduled"]
        else:
            completion_rate = 0.0
        
        # Add derived metrics
        metrics["derived_metrics"] = {
            "completion_rate": completion_rate,
            "scheduling_efficiency": min(1.0, completion_rate * 1.2),  # Normalized
            "queue_health": len(self.task_queue) / max(metrics["tasks_scheduled"], 1),
            "algorithm_performance": self._calculate_algorithm_performance()
        }
        
        return metrics

    def _calculate_algorithm_performance(self) -> Dict:
        """Calculate algorithm performance"""
        performance = {}
        
        for algorithm, stats in self.performance_metrics["algorithm_effectiveness"].items():
            if stats["success_rate"] > 0:
                performance_score = stats["success_rate"] * 100
                
                if stats["average_completion_time"]:
                    # Factor in completion time (lower is better)
                    time_score = max(0, 100 - (stats["average_completion_time"].total_seconds() / 3600))
                    performance_score = (performance_score + time_score) / 2
                
                performance[algorithm] = {
                    "performance_score": performance_score,
                    "recommendation": (
                        "highly_recommended" if performance_score > 80 else
                        "recommended" if performance_score > 60 else
                        "needs_improvement"
                    )
                }
        
        return performance

    async def _optimize_scheduling_strategies(self):
        """Continuously optimize scheduling strategies"""
        while True:
            try:
                # Analyze current performance
                metrics = await self.get_scheduling_metrics()
                
                # Identify optimization opportunities
                optimizations = self._identify_optimizations(metrics)
                
                # Apply optimizations if significant improvement expected
                for optimization in optimizations:
                    if optimization["expected_improvement"] > 0.1:  # 10% improvement threshold
                        await self._apply_scheduling_optimization(optimization)
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                print(f"⚠️ Scheduling Optimization Error: {e}")
                await asyncio.sleep(600)

    def _identify_optimizations(self, metrics: Dict) -> List[Dict]:
        """Identify scheduling optimizations"""
        optimizations = []
        
        # Check algorithm performance
        algorithm_performance = metrics.get("derived_metrics", {}).get("algorithm_performance", {})
        for algorithm, perf in algorithm_performance.items():
            if perf.get("performance_score", 0) < 60:
                optimizations.append({
                    "type": "algorithm_adjustment",
                    "target": algorithm,
                    "expected_improvement": 0.15,
                    "action": f"adjust_weighting_for_{algorithm}"
                })
        
        # Check queue health
        queue_health = metrics.get("derived_metrics", {}).get("queue_health", 0)
        if queue_health > 0.8:  # Queue is too full
            optimizations.append({
                "type": "queue_management",
                "target": "task_queue",
                "expected_improvement": 0.25,
                "action": "implement_dynamic_batch_sizing"
            })
        
        return optimizations

    async def _apply_scheduling_optimization(self, optimization: Dict):
        """Apply scheduling optimization"""
        print(f"⚡ Applying Scheduling Optimization: {optimization['action']}")
        
        # Apply optimization logic based on type
        if optimization["type"] == "algorithm_adjustment":
            await self._adjust_scheduling_algorithm(optimization["target"])
        elif optimization["type"] == "queue_management":
            await self._optimize_queue_management()

    async def _adjust_scheduling_algorithm(self, algorithm_name: str):
        """Adjust scheduling algorithm parameters"""
        # Adjust algorithm weights or parameters
        print(f"🔄 Adjusting Algorithm: {algorithm_name}")
        
        # This would adjust the internal parameters of the algorithm
        # For now, just log the adjustment
        return {
            "algorithm_adjusted": algorithm_name,
            "adjustment_timestamp": datetime.now().isoformat(),
            "adjustment_type": "parameter_optimization"
        }

    async def _optimize_queue_management(self):
        """Optimize queue management"""
        print("🔄 Optimizing Queue Management")
        
        # Implement dynamic batch sizing or priority adjustments
        current_size = len(self.task_queue)
        if current_size > 100:
            # Implement batch processing
            print("📦 Implementing batch processing for large queue")
        
        return {
            "queue_optimized": True,
            "previous_size": current_size,
            "optimization_strategy": "dynamic_batch_processing"
        }

    async def reschedule_academic_tasks(self, reschedule_criteria: Dict) -> Dict:
        """Reschedule academic tasks based on criteria"""
        print("🔄 Rescheduling Academic Tasks")
        
        rescheduled_count = 0
        
        # Create new temporary queue
        new_queue = []
        
        for scheduled_task in self.task_queue:
            task_data = scheduled_task.task_data
            
            # Check if task meets reschedule criteria
            should_reschedule = self._should_reschedule_task(task_data, reschedule_criteria)
            
            if should_reschedule:
                # Recalculate priority
                new_priority = self._calculate_task_priority(task_data, scheduled_task.scheduling_algorithm)
                new_completion = self._estimate_completion_time(task_data, scheduled_task.scheduling_algorithm)
                
                # Create new scheduled task
                new_scheduled_task = ScheduledAcademicTask(
                    priority=new_priority,
                    estimated_completion=new_completion,
                    task_id=scheduled_task.task_id,
                    task_data=task_data,
                    scheduling_algorithm=scheduled_task.scheduling_algorithm
                )
                
                heapq.heappush(new_queue, new_scheduled_task)
                rescheduled_count += 1
            else:
                # Keep original scheduling
                heapq.heappush(new_queue, scheduled_task)
        
        # Replace queue
        self.task_queue = new_queue
        
        return {
            "tasks_rescheduled": rescheduled_count,
            "total_tasks": len(self.task_queue),
            "reschedule_criteria": reschedule_criteria,
            "reschedule_timestamp": datetime.now().isoformat()
        }

    def _should_reschedule_task(self, task_data: Dict, criteria: Dict) -> bool:
        """Determine if task should be rescheduled"""
        # Check various criteria
        if criteria.get("reprioritize_urgent_tasks", False):
            urgency = task_data.get("urgency", 1)
            if urgency >= criteria.get("urgency_threshold", 8):
                return True
        
        if criteria.get("adjust_for_resource_availability", False):
            resource_intensity = task_data.get("resource_intensity", 1)
            if resource_intensity > criteria.get("resource_threshold", 5):
                return True
        
        if criteria.get("deadline_approaching", False):
            deadline_proximity = self._calculate_deadline_proximity(task_data)
            if deadline_proximity >= criteria.get("deadline_threshold", 8):
                return True
        
        return False
