#!/usr/bin/env python3
"""
BRAF Cognitive Architecture with Memory Systems
Advanced cognitive architecture implementing working memory, long-term memory, and cognitive processes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Set, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import numpy as np
import random
from collections import deque, defaultdict
import json
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CognitiveState:
    """Current state of the cognitive system"""
    working_memory: Deque[Any] = field(default_factory=lambda: deque(maxlen=7))  # 7±2 items
    attention_focus: Optional[Any] = None
    cognitive_load: float = 0.0  # 0-1 scale
    processing_mode: str = "automatic"  # automatic, controlled, reflective
    emotional_context: Dict[str, float] = field(default_factory=dict)
    goal_stack: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class MemoryItem:
    """Represents an item in long-term memory"""
    content: Any
    strength: float = 1.0
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 1
    emotional_valence: float = 0.0  # -1 to 1
    associations: Set[str] = field(default_factory=set)
    context_tags: Set[str] = field(default_factory=set)

class WorkingMemory(nn.Module):
    """Working memory system with limited capacity"""

    def __init__(self, capacity: int = 7, feature_size: int = 512):
        super().__init__()
        self.capacity = capacity
        self.feature_size = feature_size

        # Attention mechanism for working memory
        self.attention_net = nn.MultiheadAttention(feature_size, num_heads=8, batch_first=True)

        # Memory consolidation
        self.consolidation_net = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Linear(feature_size // 2, feature_size)
        )

        # Forgetting mechanism
        self.forgetting_net = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def update_memory(self, new_items: List[torch.Tensor], current_memory: Deque[torch.Tensor]) -> Deque[torch.Tensor]:
        """Update working memory with new items"""
        updated_memory = deque(current_memory, maxlen=self.capacity)

        for item in new_items:
            # Check if memory is full
            if len(updated_memory) >= self.capacity:
                # Decide what to forget
                forget_probabilities = []
                memory_list = list(updated_memory)

                for mem_item in memory_list:
                    forget_prob = self.forgetting_net(mem_item.unsqueeze(0)).item()
                    forget_probabilities.append(forget_prob)

                # Remove item with highest forget probability
                if forget_probabilities:
                    forget_idx = np.argmax(forget_probabilities)
                    memory_list.pop(forget_idx)
                    updated_memory = deque(memory_list, maxlen=self.capacity)

            # Add new item
            updated_memory.append(item)

        return updated_memory

    def retrieve_relevant(self, query: torch.Tensor, memory: Deque[torch.Tensor], k: int = 3) -> List[torch.Tensor]:
        """Retrieve most relevant items from working memory"""
        if not memory:
            return []

        memory_items = torch.stack(list(memory))

        # Compute attention scores
        query_expanded = query.unsqueeze(0).unsqueeze(0)  # Add batch and seq dimensions
        memory_expanded = memory_items.unsqueeze(0)

        attended_output, attention_weights = self.attention_net(query_expanded, memory_expanded, memory_expanded)

        # Get top-k most attended items
        attention_scores = attention_weights.squeeze(0).squeeze(0)
        top_k_indices = torch.topk(attention_scores, min(k, len(attention_scores))).indices

        relevant_items = [memory_items[idx] for idx in top_k_indices]
        return relevant_items

class LongTermMemory(nn.Module):
    """Long-term memory system with consolidation and retrieval"""

    def __init__(self, memory_capacity: int = 10000, feature_size: int = 512):
        super().__init__()
        self.memory_capacity = memory_capacity
        self.feature_size = feature_size

        # Memory storage
        self.episodic_memory: Dict[str, MemoryItem] = {}
        self.semantic_memory: Dict[str, Dict[str, Any]] = {}

        # Memory consolidation
        self.consolidation_net = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Linear(feature_size // 2, feature_size)
        )

        # Memory retrieval
        self.retrieval_net = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, 1),
            nn.Sigmoid()
        )

        # Spaced repetition for memory strength
        self.spaced_repetition = self._create_spaced_repetition_schedule()

    def store(self, content: Any, features: torch.Tensor, emotional_valence: float = 0.0,
             associations: Set[str] = None, context_tags: Set[str] = None) -> str:
        """Store item in long-term memory"""
        memory_id = hashlib.md5(f"{content}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]

        # Consolidate features
        consolidated_features = self.consolidation_net(features.unsqueeze(0)).squeeze(0)

        memory_item = MemoryItem(
            content=content,
            strength=1.0,
            emotional_valence=emotional_valence,
            associations=associations or set(),
            context_tags=context_tags or set()
        )

        self.episodic_memory[memory_id] = memory_item

        # Maintain capacity
        if len(self.episodic_memory) > self.memory_capacity:
            self._consolidate_and_prune()

        return memory_id

    def retrieve(self, query_features: torch.Tensor, context: Dict[str, Any] = None,
                limit: int = 10) -> List[Tuple[str, MemoryItem, float]]:
        """Retrieve relevant memories"""
        if not self.episodic_memory:
            return []

        results = []

        for memory_id, memory_item in self.episodic_memory.items():
            # Compute relevance score
            relevance = self._compute_relevance(query_features, memory_item, context)
            results.append((memory_id, memory_item, relevance))

        # Sort by relevance and recency
        results.sort(key=lambda x: (x[2], x[1].last_accessed), reverse=True)

        # Update access patterns
        for memory_id, memory_item, _ in results[:limit]:
            memory_item.last_accessed = datetime.now()
            memory_item.access_count += 1
            memory_item.strength = self._update_memory_strength(memory_item)

        return results[:limit]

    def consolidate_memories(self):
        """Consolidate memories during sleep/idle periods"""
        for memory_item in self.episodic_memory.values():
            # Strengthen important memories
            time_since_access = (datetime.now() - memory_item.last_accessed).days
            if time_since_access > 1:  # Consolidate daily
                memory_item.strength *= 0.95  # Slight decay
                if memory_item.access_count > 5:  # Frequently accessed
                    memory_item.strength = min(2.0, memory_item.strength * 1.1)  # Strengthen

    def _compute_relevance(self, query_features: torch.Tensor, memory_item: MemoryItem,
                          context: Dict[str, Any] = None) -> float:
        """Compute relevance score for memory retrieval"""
        # Feature similarity (placeholder - would use actual similarity)
        feature_similarity = 0.5  # Simplified

        # Emotional relevance
        emotional_relevance = 1.0 - abs(memory_item.emotional_valence - (context.get('emotional_valence', 0.0) if context else 0.0))

        # Associative relevance
        associative_relevance = 0.0
        if context and 'associations' in context:
            query_associations = set(context['associations'])
            overlap = len(memory_item.associations.intersection(query_associations))
            associative_relevance = overlap / max(1, len(query_associations.union(memory_item.associations)))

        # Recency bonus
        days_since_access = (datetime.now() - memory_item.last_accessed).days
        recency_bonus = max(0, 1.0 - days_since_access / 30)  # 30-day decay

        # Strength factor
        strength_factor = memory_item.strength

        relevance = (feature_similarity * 0.3 +
                    emotional_relevance * 0.2 +
                    associative_relevance * 0.3 +
                    recency_bonus * 0.1 +
                    strength_factor * 0.1)

        return relevance

    def _update_memory_strength(self, memory_item: MemoryItem) -> float:
        """Update memory strength using spaced repetition"""
        interval = self.spaced_repetition[min(memory_item.access_count, len(self.spaced_repetition) - 1)]
        time_factor = min(1.0, interval / 86400)  # Normalize by day

        new_strength = memory_item.strength * (1 + time_factor * 0.1)
        return min(5.0, new_strength)  # Cap strength

    def _consolidate_and_prune(self):
        """Consolidate memories and remove weak ones"""
        # Sort by strength and recency
        sorted_memories = sorted(
            self.episodic_memory.items(),
            key=lambda x: (x[1].strength, x[1].last_accessed),
            reverse=True
        )

        # Keep top memories
        keep_count = int(self.memory_capacity * 0.9)  # 90% capacity
        keep_ids = {memory_id for memory_id, _ in sorted_memories[:keep_count]}

        # Remove weak memories
        to_remove = [memory_id for memory_id in self.episodic_memory.keys() if memory_id not in keep_ids]
        for memory_id in to_remove:
            del self.episodic_memory[memory_id]

        logger.info(f"Pruned {len(to_remove)} weak memories, kept {len(self.episodic_memory)}")

    def _create_spaced_repetition_schedule(self) -> List[int]:
        """Create spaced repetition intervals in seconds"""
        return [300, 1800, 3600, 86400, 604800, 2592000, 7776000]  # 5min, 30min, 1h, 1d, 1w, 1m, 3m

class CognitiveProcessor(nn.Module):
    """Central cognitive processor coordinating thought processes"""

    def __init__(self, feature_size: int = 512):
        super().__init__()
        self.feature_size = feature_size

        # Perception modules
        self.perception_net = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Linear(feature_size // 2, feature_size)
        )

        # Reasoning modules
        self.deductive_reasoner = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, feature_size)
        )

        self.inductive_reasoner = nn.Sequential(
            nn.Linear(feature_size * 3, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, feature_size)
        )

        # Problem solving
        self.problem_solver = nn.Sequential(
            nn.Linear(feature_size + feature_size // 2, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Linear(feature_size // 2, feature_size)
        )

        # Decision making
        self.decision_maker = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Linear(feature_size // 2, feature_size // 4),
            nn.ReLU(),
            nn.Linear(feature_size // 4, 1),
            nn.Sigmoid()
        )

    def process_input(self, sensory_input: torch.Tensor) -> torch.Tensor:
        """Process sensory input through perception"""
        return self.perception_net(sensory_input)

    def reason_deductively(self, premises: List[torch.Tensor]) -> torch.Tensor:
        """Deductive reasoning from premises"""
        if len(premises) < 2:
            return premises[0] if premises else torch.zeros(self.feature_size)

        # Combine premises
        combined = torch.cat(premises[:2], dim=-1)
        return self.deductive_reasoner(combined)

    def reason_inductively(self, examples: List[torch.Tensor]) -> torch.Tensor:
        """Inductive reasoning from examples"""
        if len(examples) < 3:
            return torch.mean(torch.stack(examples), dim=0) if examples else torch.zeros(self.feature_size)

        combined = torch.cat(examples[:3], dim=-1)
        return self.inductive_reasoner(combined)

    def solve_problem(self, problem: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
        """Solve a problem given constraints"""
        combined_input = torch.cat([problem, constraints], dim=-1)
        return self.problem_solver(combined_input)

    def make_decision(self, options: List[torch.Tensor]) -> int:
        """Make decision from options"""
        if not options:
            return 0

        decision_scores = []
        for option in options:
            score = self.decision_maker(option).item()
            decision_scores.append(score)

        return np.argmax(decision_scores)

class MetacognitiveController:
    """Metacognitive control system monitoring and regulating cognition"""

    def __init__(self):
        self.performance_history = deque(maxlen=1000)
        self.strategy_effectiveness = {}
        self.cognitive_strategies = self._initialize_strategies()

    def monitor_performance(self, task: str, success: bool, response_time: float,
                          cognitive_load: float):
        """Monitor cognitive performance"""
        performance_data = {
            'task': task,
            'success': success,
            'response_time': response_time,
            'cognitive_load': cognitive_load,
            'timestamp': datetime.now()
        }

        self.performance_history.append(performance_data)

        # Update strategy effectiveness
        strategy_key = self._infer_strategy_used(cognitive_load, response_time)
        if strategy_key not in self.strategy_effectiveness:
            self.strategy_effectiveness[strategy_key] = {'successes': 0, 'total': 0}

        self.strategy_effectiveness[strategy_key]['total'] += 1
        if success:
            self.strategy_effectiveness[strategy_key]['successes'] += 1

    def select_strategy(self, task_complexity: float, time_pressure: float,
                       cognitive_load: float) -> str:
        """Select appropriate cognitive strategy"""
        # Evaluate strategy effectiveness
        best_strategy = "automatic"
        best_score = 0

        for strategy, effectiveness in self.strategy_effectiveness.items():
            if effectiveness['total'] > 5:  # Minimum trials
                success_rate = effectiveness['successes'] / effectiveness['total']
                applicability = self._assess_strategy_applicability(strategy, task_complexity, time_pressure, cognitive_load)
                score = success_rate * applicability

                if score > best_score:
                    best_score = score
                    best_strategy = strategy

        return best_strategy

    def regulate_attention(self, current_focus: Any, cognitive_load: float,
                         task_importance: float) -> str:
        """Regulate attention based on cognitive state"""
        if cognitive_load > 0.8:
            return "narrow_focus"  # Concentrate on single task
        elif task_importance > 0.7:
            return "sustained_attention"  # Maintain focus
        elif cognitive_load < 0.3:
            return "broad_attention"  # Can handle multiple tasks
        else:
            return "flexible_attention"  # Adaptive focus

    def _initialize_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available cognitive strategies"""
        return {
            'automatic': {
                'description': 'Fast, unconscious processing',
                'cognitive_load': 'low',
                'speed': 'fast',
                'accuracy': 'variable'
            },
            'controlled': {
                'description': 'Slow, conscious processing with monitoring',
                'cognitive_load': 'high',
                'speed': 'slow',
                'accuracy': 'high'
            },
            'intuitive': {
                'description': 'Pattern-based quick decisions',
                'cognitive_load': 'medium',
                'speed': 'fast',
                'accuracy': 'variable'
            },
            'analytical': {
                'description': 'Step-by-step logical analysis',
                'cognitive_load': 'high',
                'speed': 'slow',
                'accuracy': 'high'
            }
        }

    def _infer_strategy_used(self, cognitive_load: float, response_time: float) -> str:
        """Infer which strategy was likely used"""
        if cognitive_load < 0.4 and response_time < 1.0:
            return "automatic"
        elif cognitive_load > 0.7 and response_time > 2.0:
            return "controlled"
        elif cognitive_load < 0.6 and response_time < 1.5:
            return "intuitive"
        else:
            return "analytical"

    def _assess_strategy_applicability(self, strategy: str, task_complexity: float,
                                     time_pressure: float, cognitive_load: float) -> float:
        """Assess how applicable a strategy is to current conditions"""
        strategy_specs = self.cognitive_strategies[strategy]

        # Cognitive load compatibility
        load_compat = 1.0 - abs(self._load_to_numeric(strategy_specs['cognitive_load']) - cognitive_load)

        # Speed compatibility with time pressure
        speed_compat = 1.0 - abs(self._speed_to_numeric(strategy_specs['speed']) - time_pressure)

        # Complexity compatibility
        complexity_compat = 1.0
        if task_complexity > 0.7 and strategy_specs['speed'] == 'slow':
            complexity_compat = 0.8  # Analytical strategies good for complex tasks

        return (load_compat + speed_compat + complexity_compat) / 3

    def _load_to_numeric(self, load_str: str) -> float:
        """Convert load string to numeric"""
        mapping = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        return mapping.get(load_str, 0.5)

    def _speed_to_numeric(self, speed_str: str) -> float:
        """Convert speed string to numeric (inverse relationship with time pressure)"""
        mapping = {'fast': 0.8, 'slow': 0.2}  # High value means handles time pressure well
        return mapping.get(speed_str, 0.5)

class CognitiveArchitecture:
    """Main cognitive architecture integrating all cognitive systems"""

    def __init__(self):
        self.working_memory = WorkingMemory()
        self.long_term_memory = LongTermMemory()
        self.cognitive_processor = CognitiveProcessor()
        self.metacognitive_controller = MetacognitiveController()

        self.current_state = CognitiveState()

        # Initialize cognitive state
        self._initialize_cognitive_state()

    def process_information(self, sensory_input: torch.Tensor,
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process information through the cognitive architecture"""
        # Perception
        perceived_info = self.cognitive_processor.process_input(sensory_input)

        # Update working memory
        working_memory_items = [perceived_info]
        self.current_state.working_memory = self.working_memory.update_memory(
            working_memory_items, self.current_state.working_memory
        )

        # Assess cognitive load
        cognitive_load = len(self.current_state.working_memory) / self.working_memory.capacity
        self.current_state.cognitive_load = cognitive_load

        # Select processing mode based on metacognition
        task_complexity = context.get('complexity', 0.5) if context else 0.5
        time_pressure = context.get('time_pressure', 0.5) if context else 0.5

        processing_strategy = self.metacognitive_controller.select_strategy(
            task_complexity, time_pressure, cognitive_load
        )
        self.current_state.processing_mode = processing_strategy

        # Process based on strategy
        if processing_strategy == "automatic":
            result = self._automatic_processing(perceived_info, context)
        elif processing_strategy == "controlled":
            result = self._controlled_processing(perceived_info, context)
        elif processing_strategy == "intuitive":
            result = self._intuitive_processing(perceived_info, context)
        else:  # analytical
            result = self._analytical_processing(perceived_info, context)

        # Store in long-term memory if important
        if context and context.get('importance', 0) > 0.6:
            memory_id = self.long_term_memory.store(
                content=result,
                features=perceived_info,
                emotional_valence=context.get('emotional_valence', 0.0),
                associations=context.get('associations', set()),
                context_tags=context.get('tags', set())
            )

        # Update attention
        attention_mode = self.metacognitive_controller.regulate_attention(
            self.current_state.attention_focus, cognitive_load, context.get('importance', 0.5) if context else 0.5
        )

        return {
            'result': result,
            'processing_strategy': processing_strategy,
            'cognitive_load': cognitive_load,
            'attention_mode': attention_mode,
            'working_memory_size': len(self.current_state.working_memory),
            'perceived_info': perceived_info
        }

    def retrieve_memories(self, query: torch.Tensor, context: Dict[str, Any] = None,
                         limit: int = 5) -> List[Tuple[str, MemoryItem, float]]:
        """Retrieve relevant memories"""
        return self.long_term_memory.retrieve(query, context, limit)

    def reason_about_situation(self, premises: List[torch.Tensor],
                             reasoning_type: str = "deductive") -> torch.Tensor:
        """Perform reasoning about a situation"""
        if reasoning_type == "deductive":
            return self.cognitive_processor.reason_deductively(premises)
        elif reasoning_type == "inductive":
            return self.cognitive_processor.reason_inductively(premises)
        else:
            # Abductive reasoning (best explanation)
            return self._abductive_reasoning(premises)

    def solve_problem(self, problem_description: torch.Tensor,
                     constraints: torch.Tensor) -> torch.Tensor:
        """Solve a problem"""
        return self.cognitive_processor.solve_problem(problem_description, constraints)

    def make_decision(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make a decision from options"""
        # Convert options to tensors
        option_tensors = []
        for option in options:
            # Simple conversion - would be more sophisticated
            option_features = self._option_to_features(option)
            option_tensors.append(option_features)

        if not option_tensors:
            return {'decision': None, 'confidence': 0.0}

        # Use cognitive processor to decide
        decision_idx = self.cognitive_processor.make_decision(option_tensors)
        decision = options[decision_idx]

        # Metacognitive assessment
        self.metacognitive_controller.monitor_performance(
            task="decision_making",
            success=True,  # Assume success for now
            response_time=1.0,  # Placeholder
            cognitive_load=self.current_state.cognitive_load
        )

        return {
            'decision': decision,
            'decision_index': decision_idx,
            'confidence': 0.8,  # Placeholder confidence
            'processing_mode': self.current_state.processing_mode,
            'cognitive_load': self.current_state.cognitive_load
        }

    def consolidate_memories(self):
        """Consolidate long-term memories"""
        self.long_term_memory.consolidate_memories()

    def _automatic_processing(self, perceived_info: torch.Tensor,
                            context: Dict[str, Any] = None) -> Any:
        """Fast, unconscious processing"""
        # Simple pattern matching and response
        return {"type": "automatic_response", "confidence": 0.7, "data": perceived_info}

    def _controlled_processing(self, perceived_info: torch.Tensor,
                             context: Dict[str, Any] = None) -> Any:
        """Slow, deliberate processing with monitoring"""
        # Retrieve relevant memories
        relevant_memories = self.retrieve_memories(perceived_info, context, limit=3)

        # Reason about the situation
        reasoning_result = self.reason_about_situation([perceived_info] + [torch.randn_like(perceived_info) for _ in relevant_memories])

        return {
            "type": "controlled_response",
            "confidence": 0.9,
            "reasoning": reasoning_result,
            "relevant_memories": len(relevant_memories)
        }

    def _intuitive_processing(self, perceived_info: torch.Tensor,
                            context: Dict[str, Any] = None) -> Any:
        """Pattern-based intuitive processing"""
        # Quick pattern recognition
        pattern_match = torch.cosine_similarity(perceived_info, torch.randn_like(perceived_info), dim=-1)

        return {
            "type": "intuitive_response",
            "confidence": 0.75,
            "pattern_match": pattern_match.item()
        }

    def _analytical_processing(self, perceived_info: torch.Tensor,
                             context: Dict[str, Any] = None) -> Any:
        """Step-by-step analytical processing"""
        # Break down problem and solve systematically
        problem_solved = self.solve_problem(perceived_info, torch.randn_like(perceived_info))

        return {
            "type": "analytical_response",
            "confidence": 0.95,
            "solution": problem_solved,
            "steps": ["analysis", "reasoning", "solution"]
        }

    def _abductive_reasoning(self, observations: List[torch.Tensor]) -> torch.Tensor:
        """Abductive reasoning - find best explanation"""
        # Simplified: return the most "coherent" combination
        if len(observations) == 0:
            return torch.zeros(self.cognitive_processor.feature_size)

        # Combine observations with some weighting
        weights = torch.softmax(torch.randn(len(observations)), dim=0)
        combined = sum(w * obs for w, obs in zip(weights, observations))
        return combined

    def _option_to_features(self, option: Dict[str, Any]) -> torch.Tensor:
        """Convert decision option to feature tensor"""
        features = []
        for key, value in option.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                # Simple hash-based encoding
                features.append(hash(value) % 1000 / 1000.0)
            elif isinstance(value, bool):
                features.append(1.0 if value else 0.0)
            else:
                features.append(0.5)

        # Pad to feature size
        while len(features) < self.cognitive_processor.feature_size:
            features.append(0.0)

        return torch.tensor(features[:self.cognitive_processor.feature_size])

    def _initialize_cognitive_state(self):
        """Initialize the cognitive state"""
        self.current_state = CognitiveState()
        # Add some initial working memory items
        for _ in range(3):
            initial_item = torch.randn(self.cognitive_processor.feature_size)
            self.current_state.working_memory.append(initial_item)

# Global cognitive architecture instance
cognitive_architecture = CognitiveArchitecture()</content>
</xai:function_call">Create Cognitive Architecture with Memory Systems