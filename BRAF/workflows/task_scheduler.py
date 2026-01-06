#!/usr/bin/env python3
"""
BRAF Task Scheduler
Schedule and manage automation workflows
"""
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
import logging

logger = logging.getLogger(__name__)

class TaskScheduler:
    """Schedule and execute automation tasks"""
    
    def __init__(self):
        self.scheduled_tasks = {}
        self.running_tasks = {}
        self.task_history = []
        self.is_running = False
        self.scheduler_thread = None
    
    def schedule_task(self, task_id: str, task_config: Dict, 
                     schedule_type: str = 'once', **schedule_params) -> bool:
        """Schedule a task for execution"""
        try:
            task = {
                'id': task_id,
                'config': task_config,
                'schedule_type': schedule_type,
                'schedule_params': schedule_params,
                'created_at': datetime.now().isoformat(),
                'last_run': None,
                'next_run': self._calculate_next_run(schedule_type, **schedule_params),
                'run_count': 0,
                'status': 'scheduled'
            }
            
            self.scheduled_tasks[task_id] = task
            logger.info(f"Task scheduled: {task_id} ({schedule_type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule task {task_id}: {e}")
            return False
    
    def _calculate_next_run(self, schedule_type: str, **params) -> datetime:
        """Calculate next run time based on schedule type"""
        now = datetime.now()
        
        if schedule_type == 'once':
            # Run immediately or at specified time
            run_at = params.get('run_at')
            if run_at:
                if isinstance(run_at, str):
                    return datetime.fromisoformat(run_at)
                return run_at
            return now
        
        elif schedule_type == 'interval':
            # Run at regular intervals
            minutes = params.get('minutes', 0)
            hours = params.get('hours', 0)
            days = params.get('days', 0)
            
            delta = timedelta(minutes=minutes, hours=hours, days=days)
            return now + delta
        
        elif schedule_type == 'daily':
            # Run daily at specified time
            hour = params.get('hour', 0)
            minute = params.get('minute', 0)
            
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            
            return next_run
        
        elif schedule_type == 'weekly':
            # Run weekly on specified day and time
            weekday = params.get('weekday', 0)  # 0 = Monday
            hour = params.get('hour', 0)
            minute = params.get('minute', 0)
            
            days_ahead = weekday - now.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            
            next_run = now + timedelta(days=days_ahead)
            next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            return next_run
        
        else:
            return now
    
    def start_scheduler(self):
        """Start the task scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Task scheduler started")
    
    def stop_scheduler(self):
        """Stop the task scheduler"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Task scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Check for tasks ready to run
                for task_id, task in list(self.scheduled_tasks.items()):
                    if task['next_run'] <= current_time and task['status'] == 'scheduled':
                        self._execute_task(task_id, task)
                
                # Sleep for 30 seconds before next check
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _execute_task(self, task_id: str, task: Dict):
        """Execute a scheduled task"""
        try:
            logger.info(f"Executing task: {task_id}")
            
            # Mark task as running
            task['status'] = 'running'
            task['last_run'] = datetime.now().isoformat()
            task['run_count'] += 1
            
            self.running_tasks[task_id] = task
            
            # Execute task in separate thread
            execution_thread = threading.Thread(
                target=self._run_task_worker,
                args=(task_id, task),
                daemon=True
            )
            execution_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to execute task {task_id}: {e}")
            task['status'] = 'error'
    
    def _run_task_worker(self, task_id: str, task: Dict):
        """Worker thread for task execution"""
        start_time = datetime.now()
        
        try:
            # Get task configuration
            config = task['config']
            task_type = config.get('type', 'unknown')
            
            # Execute based on task type
            result = self._execute_task_type(task_type, config)
            
            # Record execution result
            execution_time = (datetime.now() - start_time).total_seconds()
            
            task_record = {
                'task_id': task_id,
                'execution_time': execution_time,
                'result': result,
                'timestamp': start_time.isoformat(),
                'success': result.get('success', False)
            }
            
            self.task_history.append(task_record)
            
            # Update task status
            if result.get('success', False):
                task['status'] = 'completed'
                logger.info(f"Task {task_id} completed successfully")
            else:
                task['status'] = 'failed'
                logger.error(f"Task {task_id} failed: {result.get('error', 'Unknown error')}")
            
            # Schedule next run if recurring
            if task['schedule_type'] != 'once':
                task['next_run'] = self._calculate_next_run(
                    task['schedule_type'], 
                    **task['schedule_params']
                )
                task['status'] = 'scheduled'
            
        except Exception as e:
            logger.error(f"Task worker error for {task_id}: {e}")
            task['status'] = 'error'
        
        finally:
            # Remove from running tasks
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    def _execute_task_type(self, task_type: str, config: Dict) -> Dict:
        """Execute specific task type"""
        try:
            if task_type == 'scraping':
                return self._execute_scraping_task(config)
            elif task_type == 'automation':
                return self._execute_automation_task(config)
            elif task_type == 'monetization':
                return self._execute_monetization_task(config)
            else:
                return {
                    'success': False,
                    'error': f'Unknown task type: {task_type}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _execute_scraping_task(self, config: Dict) -> Dict:
        """Execute scraping task"""
        try:
            from core.runner import run_targets
            
            targets = config.get('targets', [])
            if not targets:
                return {'success': False, 'error': 'No targets specified'}
            
            results = run_targets(targets)
            successful = sum(1 for r in results if r.get('success', False))
            
            return {
                'success': True,
                'type': 'scraping',
                'targets_processed': len(targets),
                'successful': successful,
                'results': results
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _execute_automation_task(self, config: Dict) -> Dict:
        """Execute automation task"""
        try:
            from automation.browser_automation import run_automation_task
            
            result = run_automation_task(config)
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _execute_monetization_task(self, config: Dict) -> Dict:
        """Execute monetization task"""
        try:
            from monetization.earnings_tracker import MonetizationManager
            
            manager = MonetizationManager()
            platform = config.get('platform', 'unknown')
            task_type = config.get('task_type', 'unknown')
            
            result = manager.run_earning_task(platform, task_type, config)
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_task_status(self, task_id: str = None) -> Dict:
        """Get status of tasks"""
        if task_id:
            return self.scheduled_tasks.get(task_id, {})
        
        return {
            'scheduled_tasks': len(self.scheduled_tasks),
            'running_tasks': len(self.running_tasks),
            'total_executions': len(self.task_history),
            'scheduler_running': self.is_running,
            'tasks': list(self.scheduled_tasks.keys())
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task"""
        try:
            if task_id in self.scheduled_tasks:
                del self.scheduled_tasks[task_id]
                logger.info(f"Task cancelled: {task_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    def get_task_history(self, limit: int = 50) -> List[Dict]:
        """Get task execution history"""
        return self.task_history[-limit:] if limit else self.task_history

def main():
    """Test task scheduler"""
    print("⏰ Testing BRAF Task Scheduler")
    
    scheduler = TaskScheduler()
    
    # Schedule a scraping task
    scraping_task = {
        'type': 'scraping',
        'targets': [
            {'url': 'https://example.com', 'requires_js': False},
            {'url': 'https://httpbin.org/html', 'requires_js': False}
        ]
    }
    
    scheduler.schedule_task(
        task_id='test_scraping',
        task_config=scraping_task,
        schedule_type='interval',
        minutes=2
    )
    
    # Schedule an automation task
    automation_task = {
        'type': 'automation',
        'url': 'https://quotes.toscrape.com/js/',
        'headless': True,
        'actions': [
            {'type': 'wait', 'name': 'page_load', 'duration': 2},
            {'type': 'extract', 'name': 'quotes', 'selectors': {'title': 'title'}}
        ]
    }
    
    scheduler.schedule_task(
        task_id='test_automation',
        task_config=automation_task,
        schedule_type='once'
    )
    
    # Start scheduler
    scheduler.start_scheduler()
    
    print("✅ Scheduler started. Tasks will execute automatically.")
    print("📊 Task status:", scheduler.get_task_status())
    
    # Run for a short time for testing
    time.sleep(10)
    
    print("📋 Task history:", scheduler.get_task_history())
    
    # Stop scheduler
    scheduler.stop_scheduler()
    print("🛑 Scheduler stopped")

if __name__ == "__main__":
    main()
