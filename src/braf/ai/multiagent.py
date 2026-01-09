#!/usr/bin/env python3
"""
BRAF Multi-Agent Coordination System
Swarm intelligence and coordinated decision making across distributed agents
"""

import asyncio
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
from collections import defaultdict
import random

logger = logging.getLogger(__name__)

@dataclass
class AgentState:
    """State representation for an agent"""
    agent_id: str
    position: Tuple[float, float] = (0.0, 0.0)  # Abstract position in task space
    velocity: Tuple[float, float] = (0.0, 0.0)
    task_type: str = ""
    status: str = "idle"  # idle, working, coordinating, failed
    performance_score: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    neighbors: Set[str] = field(default_factory=set)
    consensus_values: Dict[str, Any] = field(default_factory=dict)
    local_model_params: Optional[torch.Tensor] = None

@dataclass
class CoordinationMessage:
    """Message for inter-agent communication"""
    sender_id: str
    receiver_id: str
    message_type: str  # consensus, task_allocation, status_update, etc.
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1-5, higher is more urgent

class ConsensusAlgorithm:
    """Consensus algorithms for distributed decision making"""

    def __init__(self, consensus_type: str = 'average'):
        self.consensus_type = consensus_type
        self.iteration_count = 0
        self.convergence_threshold = 0.01

    def update_consensus(self, agent_states: Dict[str, AgentState],
                        consensus_key: str) -> Dict[str, Any]:
        """Update consensus values across agents"""
        self.iteration_count += 1

        if self.consensus_type == 'average':
            return self._average_consensus(agent_states, consensus_key)
        elif self.consensus_type == 'max':
            return self._max_consensus(agent_states, consensus_key)
        elif self.consensus_type == 'weighted_average':
            return self._weighted_average_consensus(agent_states, consensus_key)
        else:
            return {}

    def _average_consensus(self, agent_states: Dict[str, AgentState],
                          consensus_key: str) -> Dict[str, Any]:
        """Simple average consensus"""
        values = []
        weights = []

        for agent in agent_states.values():
            if consensus_key in agent.consensus_values:
                value = agent.consensus_values[consensus_key]
                if isinstance(value, (int, float)):
                    values.append(value)
                    weights.append(agent.performance_score + 0.1)  # Minimum weight

        if not values:
            return {'consensus_value': 0.0, 'converged': False}

        # Weighted average
        total_weight = sum(weights)
        consensus_value = sum(v * w for v, w in zip(values, weights)) / total_weight

        # Check convergence
        converged = self._check_convergence(values, consensus_value)

        return {
            'consensus_value': consensus_value,
            'converged': converged,
            'participating_agents': len(values),
            'variance': np.var(values) if values else 0
        }

    def _max_consensus(self, agent_states: Dict[str, AgentState],
                      consensus_key: str) -> Dict[str, Any]:
        """Maximum value consensus (for selecting best option)"""
        max_value = float('-inf')
        best_agent = None

        for agent_id, agent in agent_states.items():
            if consensus_key in agent.consensus_values:
                value = agent.consensus_values[consensus_key]
                if isinstance(value, (int, float)) and value > max_value:
                    max_value = value
                    best_agent = agent_id

        return {
            'consensus_value': max_value,
            'best_agent': best_agent,
            'converged': True,  # Max consensus converges immediately
            'participating_agents': len([a for a in agent_states.values()
                                       if consensus_key in a.consensus_values])
        }

    def _weighted_average_consensus(self, agent_states: Dict[str, AgentState],
                                   consensus_key: str) -> Dict[str, Any]:
        """Weighted average based on agent performance"""
        values = []
        weights = []

        for agent in agent_states.values():
            if consensus_key in agent.consensus_values:
                value = agent.consensus_values[consensus_key]
                if isinstance(value, (int, float)):
                    values.append(value)
                    # Weight by performance score and recency
                    time_weight = max(0.1, 1.0 - (datetime.now() - agent.last_update).seconds / 3600)
                    weight = agent.performance_score * time_weight
                    weights.append(weight)

        if not values:
            return {'consensus_value': 0.0, 'converged': False}

        total_weight = sum(weights)
        consensus_value = sum(v * w for v, w in zip(values, weights)) / total_weight

        converged = self._check_convergence(values, consensus_value)

        return {
            'consensus_value': consensus_value,
            'converged': converged,
            'participating_agents': len(values),
            'total_weight': total_weight
        }

    def _check_convergence(self, values: List[float], consensus_value: float) -> bool:
        """Check if consensus has converged"""
        if not values:
            return True

        variance = np.var(values)
        distance_from_consensus = abs(np.mean(values) - consensus_value)

        return variance < self.convergence_threshold and distance_from_consensus < self.convergence_threshold

class SwarmIntelligence:
    """Swarm intelligence algorithms for agent coordination"""

    def __init__(self):
        self.separation_weight = 1.0
        self.alignment_weight = 1.0
        self.cohesion_weight = 1.0
        self.max_speed = 2.0
        self.neighborhood_radius = 5.0

    def update_swarm_behavior(self, agent_states: Dict[str, AgentState]) -> Dict[str, Tuple[float, float]]:
        """Update agent velocities using swarm intelligence"""
        new_velocities = {}

        for agent_id, agent in agent_states.items():
            if agent.status == 'coordinating':
                # Calculate swarm forces
                separation = self._calculate_separation(agent, agent_states)
                alignment = self._calculate_alignment(agent, agent_states)
                cohesion = self._calculate_cohesion(agent, agent_states)

                # Combine forces
                total_force = (
                    self.separation_weight * np.array(separation) +
                    self.alignment_weight * np.array(alignment) +
                    self.cohesion_weight * np.array(cohesion)
                )

                # Limit speed
                speed = np.linalg.norm(total_force)
                if speed > self.max_speed:
                    total_force = total_force * (self.max_speed / speed)

                new_velocities[agent_id] = tuple(total_force)

        return new_velocities

    def _calculate_separation(self, agent: AgentState,
                            all_agents: Dict[str, AgentState]) -> Tuple[float, float]:
        """Calculate separation force (avoid crowding)"""
        force = np.array([0.0, 0.0])
        count = 0

        for other_id, other_agent in all_agents.items():
            if other_id == agent.agent_id:
                continue

            distance = np.linalg.norm(np.array(agent.position) - np.array(other_agent.position))

            if 0 < distance < self.neighborhood_radius:
                # Repel from close agents
                repel_vector = np.array(agent.position) - np.array(other_agent.position)
                repel_vector = repel_vector / distance  # Normalize
                force += repel_vector / distance  # Stronger when closer
                count += 1

        if count > 0:
            force = force / count

        return tuple(force)

    def _calculate_alignment(self, agent: AgentState,
                           all_agents: Dict[str, AgentState]) -> Tuple[float, float]:
        """Calculate alignment force (match velocity with neighbors)"""
        avg_velocity = np.array([0.0, 0.0])
        count = 0

        for other_id, other_agent in all_agents.items():
            if other_id == agent.agent_id:
                continue

            distance = np.linalg.norm(np.array(agent.position) - np.array(other_agent.position))

            if distance < self.neighborhood_radius:
                avg_velocity += np.array(other_agent.velocity)
                count += 1

        if count > 0:
            avg_velocity = avg_velocity / count

        return tuple(avg_velocity - np.array(agent.velocity))

    def _calculate_cohesion(self, agent: AgentState,
                          all_agents: Dict[str, AgentState]) -> Tuple[float, float]:
        """Calculate cohesion force (move toward group center)"""
        center_of_mass = np.array([0.0, 0.0])
        count = 0

        for other_id, other_agent in all_agents.items():
            if other_id == agent.agent_id:
                continue

            distance = np.linalg.norm(np.array(agent.position) - np.array(other_agent.position))

            if distance < self.neighborhood_radius:
                center_of_mass += np.array(other_agent.position)
                count += 1

        if count > 0:
            center_of_mass = center_of_mass / count
            return tuple(center_of_mass - np.array(agent.position))
        else:
            return (0.0, 0.0)

class TaskAllocationAlgorithm:
    """Distributed task allocation using auction and consensus"""

    def __init__(self):
        self.auction_timeout = 30  # seconds
        self.bid_history = defaultdict(list)

    async def allocate_tasks(self, tasks: List[Dict[str, Any]],
                           agent_states: Dict[str, AgentState]) -> Dict[str, List[str]]:
        """Allocate tasks to agents using auction mechanism"""
        allocation = defaultdict(list)

        # Sort tasks by priority
        sorted_tasks = sorted(tasks, key=lambda x: x.get('priority', 1), reverse=True)

        for task in sorted_tasks:
            # Run auction for this task
            winner = await self._run_auction(task, agent_states)

            if winner:
                allocation[winner].append(task['task_id'])
                # Update agent status
                agent_states[winner].status = 'working'

        return dict(allocation)

    async def _run_auction(self, task: Dict[str, Any],
                          agent_states: Dict[str, AgentState]) -> Optional[str]:
        """Run auction for task allocation"""
        bids = {}

        # Collect bids from capable agents
        for agent_id, agent in agent_states.items():
            if agent.status == 'idle' and self._agent_can_handle_task(agent, task):
                bid = self._calculate_bid(agent, task)
                bids[agent_id] = bid

        if not bids:
            return None

        # Select winner (lowest cost/highest capability)
        winner = min(bids.keys(), key=lambda x: bids[x])

        # Record bid history
        self.bid_history[task['task_id']].append({
            'winner': winner,
            'winning_bid': bids[winner],
            'all_bids': bids,
            'timestamp': datetime.now()
        })

        return winner

    def _agent_can_handle_task(self, agent: AgentState, task: Dict[str, Any]) -> bool:
        """Check if agent can handle the task"""
        required_skills = set(task.get('required_skills', []))
        agent_skills = set(task.get('agent_skills', []))

        return required_skills.issubset(agent_skills)

    def _calculate_bid(self, agent: AgentState, task: Dict[str, Any]) -> float:
        """Calculate agent's bid for task (lower is better)"""
        # Factors: distance, performance, current load, specialization
        distance_penalty = np.linalg.norm(np.array(agent.position) - np.array(task.get('location', (0, 0))))
        performance_factor = 1.0 / (agent.performance_score + 0.1)
        load_factor = len([t for t in agent.consensus_values.get('assigned_tasks', []) if t]) * 0.1

        # Specialization bonus
        specialization_match = 1.0 if agent.task_type == task.get('type') else 1.5

        bid = (distance_penalty + performance_factor + load_factor) * specialization_match

        return bid

class FederatedLearningCoordinator:
    """Federated learning for distributed model training"""

    def __init__(self):
        self.global_model = None
        self.round_number = 0
        self.participation_threshold = 0.5  # Minimum agents to participate

    def initialize_global_model(self, model_architecture: nn.Module):
        """Initialize global model for federated learning"""
        self.global_model = model_architecture
        logger.info("Global model initialized for federated learning")

    async def coordinate_learning_round(self, agent_states: Dict[str, AgentState]) -> Dict[str, Any]:
        """Coordinate one round of federated learning"""
        self.round_number += 1

        # Select participating agents
        participants = self._select_participants(agent_states)

        if len(participants) < len(agent_states) * self.participation_threshold:
            logger.warning("Insufficient participants for federated learning round")
            return {'status': 'insufficient_participants'}

        # Collect model updates
        model_updates = []
        for agent_id in participants:
            agent = agent_states[agent_id]
            if agent.local_model_params is not None:
                model_updates.append(agent.local_model_params)

        if not model_updates:
            return {'status': 'no_updates'}

        # Aggregate updates (FedAvg)
        global_update = self._aggregate_updates(model_updates)

        # Update global model
        self._update_global_model(global_update)

        # Send updated model back to agents
        for agent_id in participants:
            agent_states[agent_id].consensus_values['global_model_update'] = global_update

        return {
            'status': 'completed',
            'round': self.round_number,
            'participants': len(participants),
            'total_agents': len(agent_states)
        }

    def _select_participants(self, agent_states: Dict[str, AgentState]) -> List[str]:
        """Select agents for participation based on criteria"""
        eligible_agents = []

        for agent_id, agent in agent_states.items():
            # Criteria: recent activity, good performance, available resources
            if (agent.status != 'failed' and
                agent.performance_score > 0.3 and
                (datetime.now() - agent.last_update).seconds < 3600):  # Active in last hour
                eligible_agents.append(agent_id)

        # Randomly select subset to prevent always using same agents
        num_to_select = max(1, int(len(eligible_agents) * 0.8))
        return random.sample(eligible_agents, min(num_to_select, len(eligible_agents)))

    def _aggregate_updates(self, model_updates: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate model updates using federated averaging"""
        stacked_updates = torch.stack(model_updates)
        return torch.mean(stacked_updates, dim=0)

    def _update_global_model(self, update: torch.Tensor):
        """Update global model with aggregated update"""
        if self.global_model is None:
            return

        # Simple model update (in practice, would use proper federated learning)
        with torch.no_grad():
            for param in self.global_model.parameters():
                # This is a simplified update - real FL would be more sophisticated
                param.copy_(param + update * 0.01)  # Small learning rate

class MultiAgentCoordinationSystem:
    """Main multi-agent coordination system"""

    def __init__(self):
        self.agent_states: Dict[str, AgentState] = {}
        self.consensus_algorithms = {
            'task_priority': ConsensusAlgorithm('weighted_average'),
            'resource_allocation': ConsensusAlgorithm('average'),
            'strategy_selection': ConsensusAlgorithm('max')
        }
        self.swarm_intelligence = SwarmIntelligence()
        self.task_allocator = TaskAllocationAlgorithm()
        self.federated_learning = FederatedLearningCoordinator()

        # Message queues for inter-agent communication
        self.message_queues: Dict[str, asyncio.Queue] = {}

        logger.info("Multi-Agent Coordination System initialized")

    async def register_agent(self, agent_id: str, initial_state: Dict[str, Any]) -> bool:
        """Register a new agent in the system"""
        if agent_id in self.agent_states:
            return False

        agent_state = AgentState(
            agent_id=agent_id,
            position=initial_state.get('position', (random.uniform(-10, 10), random.uniform(-10, 10))),
            task_type=initial_state.get('task_type', 'general'),
            performance_score=initial_state.get('performance_score', 0.5)
        )

        self.agent_states[agent_id] = agent_state
        self.message_queues[agent_id] = asyncio.Queue()

        logger.info(f"Agent {agent_id} registered in coordination system")
        return True

    async def update_agent_state(self, agent_id: str, updates: Dict[str, Any]):
        """Update agent state"""
        if agent_id not in self.agent_states:
            return

        agent = self.agent_states[agent_id]
        for key, value in updates.items():
            if hasattr(agent, key):
                setattr(agent, key, value)

        agent.last_update = datetime.now()

    async def coordinate_agents(self) -> Dict[str, Any]:
        """Main coordination loop"""
        # Update swarm behavior
        velocity_updates = self.swarm_intelligence.update_swarm_behavior(self.agent_states)

        # Apply velocity updates
        for agent_id, velocity in velocity_updates.items():
            self.agent_states[agent_id].velocity = velocity
            # Update position (simplified physics)
            current_pos = np.array(self.agent_states[agent_id].position)
            new_pos = current_pos + np.array(velocity) * 0.1  # Small time step
            self.agent_states[agent_id].position = tuple(new_pos)

        # Run consensus algorithms
        consensus_results = {}
        for consensus_name, algorithm in self.consensus_algorithms.items():
            result = algorithm.update_consensus(self.agent_states, consensus_name)
            consensus_results[consensus_name] = result

        # Coordinate federated learning
        fl_result = await self.federated_learning.coordinate_learning_round(self.agent_states)

        # Process messages
        messages_processed = await self._process_messages()

        return {
            'velocity_updates': len(velocity_updates),
            'consensus_results': consensus_results,
            'federated_learning': fl_result,
            'messages_processed': messages_processed,
            'active_agents': len([a for a in self.agent_states.values() if a.status != 'failed'])
        }

    async def allocate_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Allocate tasks to agents"""
        return await self.task_allocator.allocate_tasks(tasks, self.agent_states)

    async def send_message(self, message: CoordinationMessage):
        """Send message to agent"""
        if message.receiver_id in self.message_queues:
            await self.message_queues[message.receiver_id].put(message)
            return True
        return False

    async def _process_messages(self) -> int:
        """Process pending messages"""
        messages_processed = 0

        for agent_id, queue in self.message_queues.items():
            while not queue.empty():
                message = await queue.get()
                await self._handle_message(message)
                messages_processed += 1

        return messages_processed

    async def _handle_message(self, message: CoordinationMessage):
        """Handle coordination message"""
        if message.message_type == 'status_update':
            await self.update_agent_state(message.sender_id, message.payload)
        elif message.message_type == 'consensus_update':
            agent = self.agent_states.get(message.sender_id)
            if agent:
                agent.consensus_values.update(message.payload)
        elif message.message_type == 'task_completion':
            # Update agent status and performance
            agent = self.agent_states.get(message.sender_id)
            if agent:
                agent.status = 'idle'
                success = message.payload.get('success', False)
                if success:
                    agent.performance_score = min(1.0, agent.performance_score + 0.05)
                else:
                    agent.performance_score = max(0.0, agent.performance_score - 0.02)

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        total_agents = len(self.agent_states)
        active_agents = len([a for a in self.agent_states.values() if a.status != 'failed'])
        coordinating_agents = len([a for a in self.agent_states.values() if a.status == 'coordinating'])

        avg_performance = np.mean([a.performance_score for a in self.agent_states.values()]) if self.agent_states else 0

        return {
            'total_agents': total_agents,
            'active_agents': active_agents,
            'coordinating_agents': coordinating_agents,
            'average_performance': avg_performance,
            'system_health': 'healthy' if active_agents / total_agents > 0.8 else 'degraded'
        }

    def get_agent_neighborhood(self, agent_id: str, radius: float = 5.0) -> List[str]:
        """Get agents in neighborhood of given agent"""
        if agent_id not in self.agent_states:
            return []

        agent_pos = np.array(self.agent_states[agent_id].position)
        neighbors = []

        for other_id, other_agent in self.agent_states.items():
            if other_id != agent_id:
                distance = np.linalg.norm(agent_pos - np.array(other_agent.position))
                if distance <= radius:
                    neighbors.append(other_id)

        return neighbors

# Global instance
coordination_system = MultiAgentCoordinationSystem()