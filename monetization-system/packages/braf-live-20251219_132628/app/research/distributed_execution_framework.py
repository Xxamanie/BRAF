"""
DISTRIBUTED EXECUTION FRAMEWORK
"""

from __future__ import annotations

import asyncio
import random
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


class DistributedExecutionFramework:
    """Distributes execution across multiple nodes/processes"""

    def __init__(self) -> None:
        self.node_registry: Dict[str, Dict[str, Any]] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.result_store: Dict[str, Dict[str, Any]] = {}
        self.node_health: Dict[str, Dict[str, Any]] = {}
        self.load_balancer = RoundRobinBalancer()
        self.fault_tolerance = FaultToleranceManager()

    async def register_node(self, node_id: str, node_info: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new execution node"""
        self.node_registry[node_id] = {
            **node_info,
            "registered_at": datetime.now().isoformat(),
            "last_heartbeat": datetime.now().isoformat(),
            "active_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "capacity": node_info.get("capacity", 10),
            "capabilities": node_info.get("capabilities", []),
        }
        print(f"Registered node: {node_id}")
        return {"success": True, "node_id": node_id}

    async def distribute_task(self, task_type: str, task_data: Any, priority: int = 1, redundancy: int = 1) -> str:
        """Distribute task to available nodes"""
        task_id = self._generate_task_id()
        # Create task definition
        task_definition: Dict[str, Any] = {
            "task_id": task_id,
            "type": task_type,
            "data": task_data,
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "status": "pending",
            "redundancy_level": redundancy,
            "execution_nodes": [],
            "results": [],
        }
        # Add to task queue
        await self.task_queue.put(task_definition)
        # Immediately attempt to assign to nodes
        await self._assign_tasks_to_nodes()
        return task_id

    def _generate_task_id(self) -> str:
        """Generate unique task identifier"""
        return hashlib.sha256(f"task_{datetime.now().isoformat()}_{random.random()}".encode()).hexdigest()[:16]

    async def _assign_tasks_to_nodes(self) -> None:
        """Assign pending tasks to available nodes"""
        while not self.task_queue.empty():
            try:
                task: Dict[str, Any] = await self.task_queue.get()
                # Find suitable nodes
                suitable_nodes = await self._find_suitable_nodes(task)
                if not suitable_nodes:
                    # No suitable nodes available, requeue
                    await self.task_queue.put(task)
                    await asyncio.sleep(1)
                    continue
                # Assign to nodes based on redundancy
                for _ in range(task["redundancy_level"]):
                    if suitable_nodes:
                        node_id = self.load_balancer.select_node(suitable_nodes)
                        await self._assign_to_node(task, node_id)
                # Update task status
                task["status"] = "assigned"
                self.result_store[task["task_id"]] = task
            except Exception as e:
                print(f"Task assignment error: {e}")

    async def _find_suitable_nodes(self, task: Dict[str, Any]) -> List[str]:
        """Find nodes suitable for the task"""
        suitable_nodes: List[str] = []
        for node_id, node_info in self.node_registry.items():
            # Check if node is healthy
            if not await self._is_node_healthy(node_id):
                continue
            # Check if node has capacity
            if node_info["active_tasks"] >= node_info["capacity"]:
                continue
            # Check if node has required capabilities
            if not await self._node_has_capabilities(node_id, task):
                continue
            suitable_nodes.append(node_id)
        return suitable_nodes

    async def _is_node_healthy(self, node_id: str) -> bool:
        """Check if node is healthy"""
        if node_id not in self.node_health:
            return True  # Assume healthy if no health data
        health_data = self.node_health[node_id]
        # Check last heartbeat
        last_heartbeat = datetime.fromisoformat(health_data.get("last_heartbeat", "2000-01-01"))
        if (datetime.now() - last_heartbeat).total_seconds() > 60:
            return False
        # Check failure rate
        node_info = self.node_registry.get(node_id, {})
        total_tasks = node_info.get("completed_tasks", 0) + node_info.get("failed_tasks", 0)
        if total_tasks > 0:
            failure_rate = node_info.get("failed_tasks", 0) / total_tasks
            if failure_rate > 0.5:  # 50% failure rate
                return False
        return True

    async def _node_has_capabilities(self, node_id: str, task: Dict[str, Any]) -> bool:
        """Check if node has required capabilities for task"""
        node_info = self.node_registry.get(node_id, {})
        node_capabilities = set(node_info.get("capabilities", []))
        # Define task-specific capability requirements
        task_requirements: Dict[str, set] = {
            "profile_generation": {"python", "faker", "async"},
            "data_processing": {"python", "encryption", "async"},
            "research_task": {"selenium", "requests", "async"},
            "heavy_computation": {"numpy", "multiprocessing"},
        }
        required = task_requirements.get(task["type"], set())
        return required.issubset(node_capabilities)

    async def _assign_to_node(self, task: Dict[str, Any], node_id: str) -> None:
        """Assign task to specific node"""
        node_info = self.node_registry[node_id]
        # Update node stats
        node_info["active_tasks"] += 1
        # Add to task's execution nodes
        task["execution_nodes"].append({
            "node_id": node_id,
            "assigned_at": datetime.now().isoformat(),
            "status": "executing",
        })
        # Actually execute the task on the node
        asyncio.create_task(self._execute_on_node(task, node_id))

    async def _execute_on_node(self, task: Dict[str, Any], node_id: str) -> None:
        """Execute task on specific node"""
        try:
            print(f"Executing task {task['task_id']} on node {node_id}")
            # Simulated execution
            await asyncio.sleep(random.uniform(0.5, 2.0))
            # Generate result
            result = await self._execute_task_locally(task)
            # Update task with result
            task["results"].append(
                {
                    "node_id": node_id,
                    "result": result,
                    "completed_at": datetime.now().isoformat(),
                    "success": True,
                }
            )
            # Update node stats
            node_info = self.node_registry[node_id]
            node_info["active_tasks"] -= 1
            node_info["completed_tasks"] += 1
            # Update task status if enough successful results
            successful_results = [r for r in task["results"] if r["success"]]
            if len(successful_results) >= task["redundancy_level"]:
                task["status"] = "completed"
                task["final_result"] = self._aggregate_results(task["results"])
        except Exception as e:
            print(f"Task execution failed on node {node_id}: {e}")
            # Update task with failure
            task["results"].append(
                {
                    "node_id": node_id,
                    "error": str(e),
                    "completed_at": datetime.now().isoformat(),
                    "success": False,
                }
            )
            # Update node stats
            node_info = self.node_registry[node_id]
            node_info["active_tasks"] -= 1
            node_info["failed_tasks"] += 1
            # Handle failure
            await self.fault_tolerance.handle_task_failure(task, node_id, str(e))

    async def _execute_task_locally(self, task: Dict[str, Any]) -> Any:
        """Execute task locally (simulated or actual)"""
        task_type = task["type"]
        task_data = task["data"]
        if task_type == "profile_generation":
            # Simulate profile generation
            return {"profiles_generated": random.randint(1, 10), "quality_score": random.uniform(0.7, 1.0)}
        elif task_type == "data_processing":
            # Simulate data processing
            return {"data_processed": len(str(task_data)), "processing_time": random.uniform(0.1, 1.0)}
        elif task_type == "research_task":
            # Simulate research task
            return {"completed": True, "data_collected": random.uniform(0.5, 5.0)}
        else:
            return {"status": "completed", "task_type": task_type}

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Any:
        """Aggregate results from multiple redundant executions"""
        successful_results = [r["result"] for r in results if r["success"]]
        if not successful_results:
            return {"error": "All executions failed"}
        # Simple aggregation: return first successful result
        return successful_results[0]

    async def start_heartbeat_monitoring(self, interval: int = 30) -> None:
        """Start heartbeat monitoring for nodes"""
        while True:
            try:
                await self._check_node_heartbeats()
                await asyncio.sleep(interval)
            except Exception as e:
                print(f"Heartbeat monitoring error: {e}")
                await asyncio.sleep(5)

    async def _check_node_heartbeats(self) -> None:
        """Check and update node heartbeats"""
        current_time = datetime.now()
        for node_id in list(self.node_registry.keys()):
            node_info = self.node_registry[node_id]
            last_heartbeat = datetime.fromisoformat(node_info["last_heartbeat"])
            if (current_time - last_heartbeat).total_seconds() > 120:  # 2 minutes
                print(f"Node {node_id} appears offline")
                await self._handle_node_failure(node_id)

    async def _handle_node_failure(self, node_id: str) -> None:
        """Handle node failure"""
        print(f"Handling failure for node {node_id}")
        # Mark node as failed
        if node_id in self.node_registry:
            self.node_registry[node_id]["status"] = "failed"
        # Reassign any active tasks from this node
        await self._reassign_node_tasks(node_id)
        # Attempt recovery
        await self.fault_tolerance.attempt_node_recovery(node_id)

    async def _reassign_node_tasks(self, node_id: str) -> None:
        """Reassign tasks from failed node"""
        for _task_id, task in self.result_store.items():
            if task["status"] in ["assigned", "executing"]:
                # Check if this node was executing the task
                for execution in task["execution_nodes"]:
                    if execution["node_id"] == node_id and execution["status"] == "executing":
                        # Reassign to another node
                        await self._reassign_task(task)

    async def _reassign_task(self, task: Dict[str, Any]) -> None:
        """Reassign task to different node"""
        print(f"Reassigning task {task['task_id']}")
        # Remove old execution records for unhealthy nodes still marked executing
        new_exec: List[Dict[str, Any]] = []
        for e in task["execution_nodes"]:
            if e["status"] != "executing":
                new_exec.append(e)
            else:
                if await self._is_node_healthy(e["node_id"]):
                    new_exec.append(e)
        task["execution_nodes"] = new_exec
        # Put back in queue for reassignment
        await self.task_queue.put(task)

    async def parallel_execute(self, tasks: List[Dict[str, Any]], max_concurrent: int = 10) -> Dict[str, Any]:
        """Execute multiple tasks in parallel"""
        print(f"Executing {len(tasks)} tasks in parallel")
        # Create task IDs
        task_ids: List[str] = []
        for task in tasks:
            task_id = await self.distribute_task(
                task_type=task["type"],
                task_data=task["data"],
                priority=task.get("priority", 1),
                redundancy=task.get("redundancy", 1),
            )
            task_ids.append(task_id)
        # Wait for completion
        results = await self._wait_for_completion(task_ids)
        return {
            "total_tasks": len(tasks),
            "completed_tasks": sum(1 for r in results.values() if r["status"] == "completed"),
            "failed_tasks": sum(1 for r in results.values() if r["status"] == "failed"),
            "results": results,
        }

    async def _wait_for_completion(self, task_ids: List[str], timeout: int = 300) -> Dict[str, Dict[str, Any]]:
        """Wait for tasks to complete"""
        start_time = datetime.now()
        results: Dict[str, Dict[str, Any]] = {}
        pending = set(task_ids)
        while pending and (datetime.now() - start_time).total_seconds() < timeout:
            completed_tasks: List[str] = []
            for task_id in list(pending):
                task = self.result_store.get(task_id)
                if task and task.get("status") in ["completed", "failed", "timeout"]:
                    results[task_id] = task
                    completed_tasks.append(task_id)
            for task_id in completed_tasks:
                pending.discard(task_id)
            if pending:
                await asyncio.sleep(1)
        # Mark any remaining tasks as timed out
        for task_id in pending:
            results[task_id] = {
                "task_id": task_id,
                "status": "timeout",
                "error": "Execution timeout",
            }
            # also store in result_store
            self.result_store[task_id] = results[task_id]
        return results

    async def scale_resources(self, target_capacity: int) -> None:
        """Scale resources up or down"""
        current_capacity = self._calculate_total_capacity()
        if target_capacity > current_capacity:
            await self._scale_up(target_capacity - current_capacity)
        elif target_capacity < current_capacity:
            await self._scale_down(current_capacity - target_capacity)

    def _calculate_total_capacity(self) -> int:
        """Calculate total system capacity"""
        total = 0
        for node_info in self.node_registry.values():
            if node_info.get("status") != "failed":
                total += node_info.get("capacity", 0)
        return total

    async def _scale_up(self, additional_capacity: int) -> None:
        """Scale up by adding more nodes"""
        print(f"Scaling up by {additional_capacity} capacity")
        # In a real system, this would spawn new nodes/containers
        num_nodes = max(1, additional_capacity // 10)
        for _ in range(num_nodes):
            node_id = f"auto_node_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"
            await self.register_node(
                node_id,
                {
                    "capacity": 10,
                    "capabilities": ["python", "async", "basic_operations"],
                    "type": "auto_scaled",
                    "auto_scaled": True,
                },
            )

    async def _scale_down(self, reduce_capacity: int) -> None:
        """Scale down by removing nodes"""
        print(f"Scaling down by {reduce_capacity} capacity")
        # Find auto-scaled nodes to remove
        auto_nodes = [
            node_id
            for node_id, info in self.node_registry.items()
            if info.get("auto_scaled", False) and info.get("active_tasks", 0) == 0
        ]
        num_remove = max(1, reduce_capacity // 10)
        for node_id in auto_nodes[:num_remove]:
            await self._remove_node(node_id)

    async def _remove_node(self, node_id: str) -> None:
        """Remove a node from the system"""
        if node_id in self.node_registry:
            print(f"Removing node {node_id}")
            del self.node_registry[node_id]

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        total_nodes = len(self.node_registry)
        active_nodes = sum(1 for n in self.node_registry.values() if n.get("status") != "failed")
        total_capacity = self._calculate_total_capacity()
        active_tasks = sum(n.get("active_tasks", 0) for n in self.node_registry.values())
        return {
            "timestamp": datetime.now().isoformat(),
            "total_nodes": total_nodes,
            "active_nodes": active_nodes,
            "total_capacity": total_capacity,
            "active_tasks": active_tasks,
            "utilization": active_tasks / total_capacity if total_capacity > 0 else 0,
            "pending_tasks": self.task_queue.qsize(),
            "completed_tasks": sum(n.get("completed_tasks", 0) for n in self.node_registry.values()),
            "failed_tasks": sum(n.get("failed_tasks", 0) for n in self.node_registry.values()),
        }

    async def run_distributed_framework(self) -> None:
        """Run the complete distributed framework"""
        print("Starting distributed execution framework...")
        # Start monitoring
        heartbeat_task = asyncio.create_task(self.start_heartbeat_monitoring())
        # Start task assignment loop
        assignment_task = asyncio.create_task(self._continuous_task_assignment())
        # Start resource management
        management_task = asyncio.create_task(self._resource_management_loop())
        # Wait for tasks
        await asyncio.gather(heartbeat_task, assignment_task, management_task)

    async def _continuous_task_assignment(self) -> None:
        """Continuously assign tasks to nodes"""
        while True:
            try:
                await self._assign_tasks_to_nodes()
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Continuous assignment error: {e}")
                await asyncio.sleep(5)

    async def _resource_management_loop(self, interval: int = 60) -> None:
        """Manage resources based on load"""
        while True:
            try:
                status = self.get_system_status()
                utilization = status["utilization"]
                if utilization > 0.8:  # 80% utilization
                    await self.scale_resources(int(status["total_capacity"] * 1.5))
                elif utilization < 0.3:  # 30% utilization
                    await self.scale_resources(int(status["total_capacity"] * 0.7))
                await asyncio.sleep(interval)
            except Exception as e:
                print(f"Resource management error: {e}")
                await asyncio.sleep(5)


class RoundRobinBalancer:
    """Round-robin load balancer"""

    def __init__(self) -> None:
        self.last_used: Dict[tuple, int] = {}

    def select_node(self, node_ids: List[str]) -> str:
        """Select node using round-robin"""
        if not node_ids:
            raise ValueError("No nodes available")
        # Get or initialize counter for this node set
        key = tuple(sorted(node_ids))
        if key not in self.last_used:
            self.last_used[key] = -1
        # Select next node
        self.last_used[key] = (self.last_used[key] + 1) % len(node_ids)
        return node_ids[self.last_used[key]]


class FaultToleranceManager:
    """Manages fault tolerance for distributed execution"""

    def __init__(self) -> None:
        self.failure_history: Dict[str, List[Dict[str, Any]]] = {}
        self.recovery_attempts: Dict[str, int] = {}

    async def handle_task_failure(self, task: Dict[str, Any], node_id: str, error: str) -> None:
        """Handle task execution failure"""
        task_id = task["task_id"]
        # Record failure
        if task_id not in self.failure_history:
            self.failure_history[task_id] = []
        self.failure_history[task_id].append({"node_id": node_id, "error": error, "timestamp": datetime.now().isoformat()})
        # Check if this task is consistently failing
        failures = self.failure_history[task_id]
        if len(failures) > 3:
            print(f"Task {task_id} has failed {len(failures)} times")
            # Analyze failure pattern
            if await self._is_systematic_failure(failures):
                print(f"Task {task_id} appears to have systematic issues")

    async def _is_systematic_failure(self, failures: List[Dict[str, Any]]) -> bool:
        """Check if failures are systematic"""
        # Check if failures are from different nodes with similar errors
        error_patterns: Dict[str, int] = {}
        for failure in failures:
            err = failure["error"]
            error_patterns[err] = error_patterns.get(err, 0) + 1
        # If same error occurs multiple times, likely systematic
        for _error, count in error_patterns.items():
            if count >= 3:
                return True
        return False

    async def attempt_node_recovery(self, node_id: str) -> bool:
        """Attempt to recover a failed node"""
        if node_id not in self.recovery_attempts:
            self.recovery_attempts[node_id] = 0
        attempts = self.recovery_attempts[node_id]
        if attempts < 3:  # Maximum 3 recovery attempts
            print(f"Attempting recovery for node {node_id} (attempt {attempts + 1})")
            # Simulate recovery attempt
            success = random.random() > 0.3  # 70% success rate
            if success:
                print(f"Node {node_id} recovered successfully")
                self.recovery_attempts[node_id] = 0
                return True
            else:
                print(f"Node {node_id} recovery failed")
                self.recovery_attempts[node_id] += 1
                return False
        print(f"Node {node_id} marked as permanently failed")
        return False
