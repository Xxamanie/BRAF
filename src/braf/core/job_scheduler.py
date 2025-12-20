"""
Job Scheduler for BRAF using Celery and Redis.

This module provides distributed task scheduling and execution
with load balancing, worker failure detection, and dynamic scaling.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum

import redis
from celery import Celery, Task
from celery.result import AsyncResult
from celery.signals import worker_ready, worker_shutdown, task_failure, task_success
from kombu import Queue

from braf.core.models import (
    AutomationTask, TaskStatus, TaskPriority, TaskResult,
    WorkerStatus, QueueMetrics
)
from braf.core.database import get_database
from braf.core.compliance_logger import get_compliance_logger

logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Job priority levels for Celery routing."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class WorkerNode:
    """Represents a worker node in the cluster."""
    
    id: str
    hostname: str
    status: str
    current_tasks: int = 0
    max_tasks: int = 5
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_heartbeat: datetime = None
    capabilities: List[str] = None
    load_score: float = 0.0
    
    def __post_init__(self):
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.now(timezone.utc)
        if self.capabilities is None:
            self.capabilities = []
    
    def calculate_load_score(self) -> float:
        """Calculate load score for load balancing."""
        # Higher score = more loaded (worse for assignment)
        task_load = self.current_tasks / max(self.max_tasks, 1)
        cpu_load = self.cpu_usage / 100.0
        memory_load = self.memory_usage / 100.0
        
        # Weighted average
        self.load_score = (task_load * 0.5) + (cpu_load * 0.3) + (memory_load * 0.2)
        return self.load_score
    
    def is_healthy(self) -> bool:
        """Check if worker is healthy and available."""
        if self.status != "online":
            return False
        
        # Check heartbeat (consider dead if no heartbeat for 60 seconds)
        if self.last_heartbeat:
            age = datetime.now(timezone.utc) - self.last_heartbeat
            if age > timedelta(seconds=60):
                return False
        
        # Check if overloaded
        if self.current_tasks >= self.max_tasks:
            return False
        
        return True


class JobScheduler:
    """Main job scheduler using Celery and Redis."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        broker_url: str = None,
        result_backend: str = None
    ):
        """
        Initialize job scheduler.
        
        Args:
            redis_url: Redis connection URL
            broker_url: Celery broker URL (defaults to redis_url)
            result_backend: Celery result backend URL (defaults to redis_url)
        """
        self.redis_url = redis_url
        self.broker_url = broker_url or redis_url
        self.result_backend = result_backend or redis_url
        
        # Initialize Redis connection
        self.redis_client = redis.from_url(redis_url)
        
        # Initialize Celery app
        self.celery_app = self._create_celery_app()
        
        # Worker tracking
        self.workers: Dict[str, WorkerNode] = {}
        self.worker_lock = asyncio.Lock()
        
        # Metrics
        self.metrics = QueueMetrics()
        self.last_metrics_update = datetime.now(timezone.utc)
        
        # Scaling configuration
        self.min_workers = 1
        self.max_workers = 10
        self.scale_up_threshold = 0.8  # Scale up when 80% loaded
        self.scale_down_threshold = 0.3  # Scale down when 30% loaded
        
        # Setup Celery signal handlers
        self._setup_signal_handlers()
    
    def _create_celery_app(self) -> Celery:
        """Create and configure Celery application."""
        app = Celery(
            'braf_scheduler',
            broker=self.broker_url,
            backend=self.result_backend,
            include=['braf.core.job_scheduler']
        )
        
        # Configure Celery
        app.conf.update(
            # Task routing
            task_routes={
                'braf.execute_automation_task': {
                    'queue': 'automation_tasks',
                    'routing_key': 'automation'
                },
                'braf.health_check': {
                    'queue': 'health_checks',
                    'routing_key': 'health'
                }
            },
            
            # Queue configuration
            task_default_queue='default',
            task_queues=(
                Queue('default', routing_key='default'),
                Queue('automation_tasks', routing_key='automation'),
                Queue('high_priority', routing_key='high_priority'),
                Queue('health_checks', routing_key='health'),
            ),
            
            # Task execution settings
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,
            
            # Worker settings
            worker_prefetch_multiplier=1,
            task_acks_late=True,
            worker_max_tasks_per_child=1000,
            
            # Result settings
            result_expires=3600,  # 1 hour
            
            # Monitoring
            worker_send_task_events=True,
            task_send_sent_event=True,
        )
        
        return app
    
    def _setup_signal_handlers(self):
        """Setup Celery signal handlers for worker monitoring."""
        
        @worker_ready.connect
        def worker_ready_handler(sender=None, **kwargs):
            """Handle worker ready signal."""
            worker_id = sender
            logger.info(f"Worker {worker_id} is ready")
            asyncio.create_task(self._register_worker(worker_id))
        
        @worker_shutdown.connect
        def worker_shutdown_handler(sender=None, **kwargs):
            """Handle worker shutdown signal."""
            worker_id = sender
            logger.info(f"Worker {worker_id} is shutting down")
            asyncio.create_task(self._unregister_worker(worker_id))
        
        @task_success.connect
        def task_success_handler(sender=None, result=None, **kwargs):
            """Handle successful task completion."""
            task_id = kwargs.get('task_id')
            logger.debug(f"Task {task_id} completed successfully")
            asyncio.create_task(self._handle_task_completion(task_id, True, result))
        
        @task_failure.connect
        def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs):
            """Handle task failure."""
            logger.error(f"Task {task_id} failed: {exception}")
            asyncio.create_task(self._handle_task_completion(task_id, False, str(exception)))
    
    async def _register_worker(self, worker_id: str):
        """Register a new worker."""
        async with self.worker_lock:
            # Get worker info from Celery
            inspect = self.celery_app.control.inspect()
            stats = inspect.stats()
            
            if worker_id in stats:
                worker_stats = stats[worker_id]
                
                worker = WorkerNode(
                    id=worker_id,
                    hostname=worker_stats.get('hostname', 'unknown'),
                    status='online',
                    max_tasks=worker_stats.get('pool', {}).get('max-concurrency', 5),
                    capabilities=['automation', 'captcha_solving']
                )
                
                self.workers[worker_id] = worker
                logger.info(f"Registered worker {worker_id}")
    
    async def _unregister_worker(self, worker_id: str):
        """Unregister a worker."""
        async with self.worker_lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                logger.info(f"Unregistered worker {worker_id}")
    
    async def _handle_task_completion(self, task_id: str, success: bool, result: Any):
        """Handle task completion."""
        # Update worker task count
        async with self.worker_lock:
            for worker in self.workers.values():
                if worker.current_tasks > 0:
                    worker.current_tasks -= 1
        
        # Update metrics
        if success:
            self.metrics.completed_tasks += 1
        else:
            self.metrics.failed_tasks += 1
        
        # Log to compliance system
        compliance_logger = get_compliance_logger()
        if compliance_logger:
            await compliance_logger.log_action_execution(
                "task_completed" if success else "task_failed",
                "system",
                task_id,
                {"success": success, "result": str(result)}
            )
    
    async def submit_task(
        self,
        task: AutomationTask,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """
        Submit automation task for execution.
        
        Args:
            task: Automation task to execute
            priority: Task priority
            
        Returns:
            Job ID for tracking
        """
        # Validate task
        if not task.actions:
            raise ValueError("Task must have at least one action")
        
        # Check compliance constraints
        compliance_logger = get_compliance_logger()
        if compliance_logger:
            violations = await compliance_logger.log_task_start(task, task.profile_id)
            if violations:
                raise ValueError(f"Compliance violations prevent task execution: {violations}")
        
        # Determine queue based on priority
        queue_name = self._get_queue_for_priority(priority)
        
        # Select best worker
        worker_id = await self._select_worker(task)
        
        # Submit to Celery
        job = self.celery_app.send_task(
            'braf.execute_automation_task',
            args=[task.dict()],
            queue=queue_name,
            routing_key=queue_name,
            priority=self._priority_to_int(priority),
            task_id=str(uuid.uuid4())
        )
        
        # Update metrics
        self.metrics.pending_tasks += 1
        
        # Update worker task count
        if worker_id:
            async with self.worker_lock:
                if worker_id in self.workers:
                    self.workers[worker_id].current_tasks += 1
        
        logger.info(f"Submitted task {task.id} as job {job.id} to worker {worker_id}")
        return job.id
    
    def _get_queue_for_priority(self, priority: TaskPriority) -> str:
        """Get queue name for task priority."""
        if priority == TaskPriority.CRITICAL:
            return 'high_priority'
        elif priority == TaskPriority.HIGH:
            return 'high_priority'
        else:
            return 'automation_tasks'
    
    def _priority_to_int(self, priority: TaskPriority) -> int:
        """Convert TaskPriority to integer for Celery."""
        mapping = {
            TaskPriority.LOW: 0,
            TaskPriority.NORMAL: 1,
            TaskPriority.HIGH: 2,
            TaskPriority.CRITICAL: 3
        }
        return mapping.get(priority, 1)
    
    async def _select_worker(self, task: AutomationTask) -> Optional[str]:
        """
        Select best worker for task using load balancing.
        
        Args:
            task: Task to be executed
            
        Returns:
            Worker ID or None if no suitable worker
        """
        async with self.worker_lock:
            if not self.workers:
                logger.warning("No workers available")
                return None
            
            # Filter healthy workers
            healthy_workers = [
                worker for worker in self.workers.values()
                if worker.is_healthy()
            ]
            
            if not healthy_workers:
                logger.warning("No healthy workers available")
                return None
            
            # Calculate load scores
            for worker in healthy_workers:
                worker.calculate_load_score()
            
            # Sort by load score (ascending - lower is better)
            healthy_workers.sort(key=lambda w: w.load_score)
            
            # Select worker with lowest load
            selected_worker = healthy_workers[0]
            
            logger.debug(f"Selected worker {selected_worker.id} with load score {selected_worker.load_score:.2f}")
            return selected_worker.id
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of submitted job.
        
        Args:
            job_id: Job ID returned from submit_task
            
        Returns:
            Job status information
        """
        result = AsyncResult(job_id, app=self.celery_app)
        
        return {
            "job_id": job_id,
            "status": result.status,
            "result": result.result if result.ready() else None,
            "traceback": result.traceback if result.failed() else None,
            "date_done": result.date_done,
            "task_id": result.task_id
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a submitted job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if cancellation was successful
        """
        try:
            self.celery_app.control.revoke(job_id, terminate=True)
            logger.info(f"Cancelled job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    async def update_worker_metrics(self):
        """Update worker metrics from Celery."""
        try:
            inspect = self.celery_app.control.inspect()
            
            # Get active tasks
            active_tasks = inspect.active()
            
            # Get worker stats
            stats = inspect.stats()
            
            async with self.worker_lock:
                for worker_id, worker in self.workers.items():
                    # Update task count
                    if active_tasks and worker_id in active_tasks:
                        worker.current_tasks = len(active_tasks[worker_id])
                    else:
                        worker.current_tasks = 0
                    
                    # Update stats if available
                    if stats and worker_id in stats:
                        worker_stats = stats[worker_id]
                        # Note: Celery doesn't provide CPU/memory by default
                        # This would need additional monitoring setup
                        worker.last_heartbeat = datetime.now(timezone.utc)
                        worker.status = 'online'
                    else:
                        # Worker might be offline
                        age = datetime.now(timezone.utc) - worker.last_heartbeat
                        if age > timedelta(seconds=60):
                            worker.status = 'offline'
        
        except Exception as e:
            logger.error(f"Failed to update worker metrics: {e}")
    
    async def get_queue_metrics(self) -> QueueMetrics:
        """Get current queue metrics."""
        try:
            inspect = self.celery_app.control.inspect()
            
            # Get queue lengths
            active_queues = inspect.active_queues()
            reserved_tasks = inspect.reserved()
            
            # Update metrics
            self.metrics.active_workers = len([w for w in self.workers.values() if w.status == 'online'])
            self.metrics.worker_count = len(self.workers)
            
            # Calculate queue length
            total_reserved = 0
            if reserved_tasks:
                for worker_tasks in reserved_tasks.values():
                    total_reserved += len(worker_tasks)
            
            self.metrics.queue_length = total_reserved
            self.metrics.running_tasks = sum(w.current_tasks for w in self.workers.values())
            
            self.last_metrics_update = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Failed to get queue metrics: {e}")
        
        return self.metrics
    
    async def check_worker_failures(self):
        """Check for worker failures and handle reassignment."""
        failed_workers = []
        
        async with self.worker_lock:
            for worker_id, worker in self.workers.items():
                if not worker.is_healthy() and worker.status != 'offline':
                    logger.warning(f"Worker {worker_id} appears to have failed")
                    worker.status = 'failed'
                    failed_workers.append(worker_id)
        
        # Handle failed workers
        for worker_id in failed_workers:
            await self._handle_worker_failure(worker_id)
    
    async def _handle_worker_failure(self, worker_id: str):
        """Handle worker failure by reassigning tasks."""
        logger.info(f"Handling failure of worker {worker_id}")
        
        try:
            # Get active tasks for failed worker
            inspect = self.celery_app.control.inspect()
            active_tasks = inspect.active()
            
            if active_tasks and worker_id in active_tasks:
                failed_tasks = active_tasks[worker_id]
                
                for task_info in failed_tasks:
                    task_id = task_info['id']
                    
                    # Revoke the task
                    self.celery_app.control.revoke(task_id, terminate=True)
                    
                    # Log the failure
                    logger.warning(f"Task {task_id} failed due to worker failure")
                    
                    # Note: Task reassignment would require additional logic
                    # to retrieve and resubmit the original task
        
        except Exception as e:
            logger.error(f"Error handling worker failure: {e}")
    
    async def check_scaling_needs(self):
        """Check if cluster needs scaling up or down."""
        metrics = await self.get_queue_metrics()
        
        if not self.workers:
            return
        
        # Calculate average load
        total_load = sum(w.calculate_load_score() for w in self.workers.values() if w.is_healthy())
        healthy_workers = len([w for w in self.workers.values() if w.is_healthy()])
        
        if healthy_workers == 0:
            return
        
        avg_load = total_load / healthy_workers
        
        # Check if scaling is needed
        if avg_load > self.scale_up_threshold and len(self.workers) < self.max_workers:
            logger.info(f"High load detected ({avg_load:.2f}), scaling up recommended")
            await self._trigger_scale_up()
        
        elif avg_load < self.scale_down_threshold and len(self.workers) > self.min_workers:
            logger.info(f"Low load detected ({avg_load:.2f}), scaling down recommended")
            await self._trigger_scale_down()
    
    async def _trigger_scale_up(self):
        """Trigger scaling up (add more workers)."""
        # This would integrate with container orchestration (Docker, Kubernetes)
        # For now, just log the recommendation
        logger.info("Scale up triggered - additional workers needed")
        
        # In a real implementation, this might:
        # 1. Start new Docker containers
        # 2. Request new instances from cloud provider
        # 3. Update Kubernetes deployment replicas
    
    async def _trigger_scale_down(self):
        """Trigger scaling down (remove workers)."""
        # Find least loaded worker to remove
        async with self.worker_lock:
            healthy_workers = [w for w in self.workers.values() if w.is_healthy()]
            
            if len(healthy_workers) <= self.min_workers:
                return
            
            # Sort by load (ascending)
            healthy_workers.sort(key=lambda w: w.calculate_load_score())
            
            # Select worker with lowest load for removal
            worker_to_remove = healthy_workers[0]
            
            if worker_to_remove.current_tasks == 0:
                logger.info(f"Scale down triggered - removing worker {worker_to_remove.id}")
                # Signal worker to shutdown gracefully
                # In real implementation, this would stop the container/instance
    
    async def start_monitoring(self):
        """Start background monitoring tasks."""
        logger.info("Starting job scheduler monitoring")
        
        async def monitoring_loop():
            while True:
                try:
                    await self.update_worker_metrics()
                    await self.check_worker_failures()
                    await self.check_scaling_needs()
                    
                    # Sleep for monitoring interval
                    await asyncio.sleep(30)  # 30 seconds
                
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(10)  # Shorter sleep on error
        
        # Start monitoring task
        asyncio.create_task(monitoring_loop())
    
    def get_worker_status(self) -> List[WorkerStatus]:
        """Get status of all workers."""
        worker_statuses = []
        
        for worker in self.workers.values():
            status = WorkerStatus(
                worker_id=worker.id,
                status=worker.status,
                current_tasks=worker.current_tasks,
                max_tasks=worker.max_tasks,
                cpu_usage=worker.cpu_usage,
                memory_usage=worker.memory_usage,
                last_heartbeat=worker.last_heartbeat,
                capabilities=worker.capabilities
            )
            worker_statuses.append(status)
        
        return worker_statuses


# Celery task definitions
@celery_app.task(bind=True, name='braf.execute_automation_task')
def execute_automation_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task to execute automation task.
    
    Args:
        task_data: Serialized AutomationTask data
        
    Returns:
        Task execution result
    """
    try:
        # Reconstruct task from data
        task = AutomationTask(**task_data)
        
        # Import here to avoid circular imports
        from braf.core.task_executor import get_task_executor
        
        # Get task executor
        executor = get_task_executor()
        if not executor:
            raise RuntimeError("Task executor not initialized")
        
        # Execute task (this would be async in real implementation)
        # For now, return a mock result
        result = TaskResult(
            task_id=task.id,
            success=True,
            execution_time=10.0,
            actions_completed=len(task.actions),
            metadata={"worker_id": self.request.hostname}
        )
        
        return asdict(result)
    
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        raise


@celery_app.task(name='braf.health_check')
def health_check() -> Dict[str, Any]:
    """Health check task for worker monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "worker_id": health_check.request.hostname
    }


# Global job scheduler instance
_job_scheduler: Optional[JobScheduler] = None


def get_job_scheduler() -> Optional[JobScheduler]:
    """
    Get global job scheduler instance.
    
    Returns:
        Job scheduler instance or None if not initialized
    """
    return _job_scheduler


def init_job_scheduler(
    redis_url: str = "redis://localhost:6379/0",
    broker_url: str = None,
    result_backend: str = None
) -> JobScheduler:
    """
    Initialize global job scheduler.
    
    Args:
        redis_url: Redis connection URL
        broker_url: Celery broker URL
        result_backend: Celery result backend URL
        
    Returns:
        Initialized job scheduler
    """
    global _job_scheduler
    
    _job_scheduler = JobScheduler(
        redis_url=redis_url,
        broker_url=broker_url,
        result_backend=result_backend
    )
    
    return _job_scheduler


# Make celery_app available at module level for Celery worker
celery_app = None

def get_celery_app() -> Celery:
    """Get Celery app instance."""
    global celery_app
    if celery_app is None:
        scheduler = get_job_scheduler()
        if scheduler:
            celery_app = scheduler.celery_app
        else:
            # Create minimal app for worker startup
            celery_app = Celery('braf_scheduler')
    return celery_app