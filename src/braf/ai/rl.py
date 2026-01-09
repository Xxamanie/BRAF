#!/usr/bin/env python3
"""
BRAF Reinforcement Learning Framework
Advanced RL for adaptive behaviors, strategy optimization, and continuous improvement
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
from pathlib import Path
import gym
from gym import spaces

logger = logging.getLogger(__name__)

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class BRAFEnvironment(gym.Env):
    """Custom Gym environment for BRAF automation tasks"""

    def __init__(self, task_type: str = 'general'):
        super().__init__()

        self.task_type = task_type
        self.max_steps = 100
        self.current_step = 0

        # Define action and observation spaces
        if task_type == 'browser_automation':
            self.action_space = spaces.Discrete(10)  # Different automation actions
            self.observation_space = spaces.Box(low=0, high=1, shape=(50,), dtype=np.float32)
        elif task_type == 'form_filling':
            self.action_space = spaces.Discrete(5)  # Form interaction actions
            self.observation_space = spaces.Box(low=0, high=1, shape=(30,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(8)  # General actions
            self.observation_space = spaces.Box(low=0, high=1, shape=(20,), dtype=np.float32)

        self.state = None
        self.done = False

    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.done = False
        self.state = np.random.rand(self.observation_space.shape[0])
        return self.state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
        self.current_step += 1

        # Simulate environment dynamics
        next_state = self._calculate_next_state(action)
        reward = self._calculate_reward(action)
        done = self._is_done()

        self.state = next_state

        return next_state, reward, done, {}

    def _calculate_next_state(self, action: int) -> np.ndarray:
        """Calculate next state based on action"""
        # Simplified state transition
        noise = np.random.normal(0, 0.1, size=self.state.shape)
        action_effect = np.zeros(self.state.shape)
        action_effect[action % len(action_effect)] = 0.2

        next_state = self.state + action_effect + noise
        return np.clip(next_state, 0, 1)

    def _calculate_reward(self, action: int) -> float:
        """Calculate reward for action"""
        # Task-specific reward calculation
        if self.task_type == 'browser_automation':
            # Reward successful navigation, penalize errors
            base_reward = np.random.normal(0, 1)
            if action in [0, 1, 2]:  # Good actions
                base_reward += 1
            elif action in [7, 8, 9]:  # Bad actions
                base_reward -= 1
        elif self.task_type == 'form_filling':
            # Reward correct form filling
            base_reward = 0.5 if action in [0, 1] else -0.5
        else:
            base_reward = np.random.normal(0, 0.5)

        return base_reward

    def _is_done(self) -> bool:
        """Check if episode is done"""
        return self.current_step >= self.max_steps or np.random.rand() < 0.05

class QNetwork(nn.Module):
    """Q-Network for Q-learning"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.network(x)

class PolicyNetwork(nn.Module):
    """Policy network for policy gradient methods"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

class ValueNetwork(nn.Module):
    """Value network for actor-critic methods"""

    def __init__(self, state_size: int, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer for Q-learning"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

class QLearningAgent:
    """Q-Learning agent for discrete action spaces"""

    def __init__(self, state_size: int, action_size: int, device: str = 'cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer()

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.batch_size = 32
        self.update_target_every = 100
        self.step_count = 0

    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        experiences = self.replay_buffer.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        # Convert to tensors
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # Compute Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class PolicyGradientAgent:
    """Policy Gradient agent for continuous learning"""

    def __init__(self, state_size: int, action_size: int, device: str = 'cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.policy_network = PolicyNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)

        self.saved_log_probs = []
        self.rewards = []
        self.gamma = 0.99

    def select_action(self, state: np.ndarray) -> int:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).to(self.device)
        probs = self.policy_network(state_tensor)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        self.saved_log_probs.append(action_dist.log_prob(action))
        return action.item()

    def train_step(self):
        """Perform policy gradient update"""
        if not self.saved_log_probs:
            return

        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)

        # Normalize rewards
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Calculate loss
        policy_loss = []
        for log_prob, reward in zip(self.saved_log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        # Reset for next episode
        self.saved_log_probs = []
        self.rewards = []

    def store_reward(self, reward: float):
        """Store reward for current episode"""
        self.rewards.append(reward)

class MultiArmedBandit:
    """Multi-armed bandit for strategy selection"""

    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self) -> int:
        """Select arm using epsilon-greedy strategy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_arms - 1)
        else:
            return np.argmax(self.values)

    def update(self, arm: int, reward: float):
        """Update arm value based on reward"""
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward

class AdaptiveBehaviorEngine:
    """Main engine for adaptive behaviors using reinforcement learning"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize different RL agents for different tasks
        self.agents = {
            'browser_automation': QLearningAgent(50, 10, self.device),
            'form_filling': QLearningAgent(30, 5, self.device),
            'strategy_selection': MultiArmedBandit(5),
            'captcha_solving': PolicyGradientAgent(64, 8, self.device)
        }

        # Task environments
        self.environments = {
            'browser_automation': BRAFEnvironment('browser_automation'),
            'form_filling': BRAFEnvironment('form_filling'),
            'captcha_solving': BRAFEnvironment('general')
        }

        # Performance tracking
        self.performance_history = {}
        self.strategy_success_rates = {}

        logger.info("Adaptive Behavior Engine initialized")

    def adapt_behavior(self, task_type: str, current_state: Dict[str, Any],
                      available_actions: List[str]) -> str:
        """Adapt behavior based on current state and learning"""
        if task_type not in self.agents:
            return random.choice(available_actions)

        agent = self.agents[task_type]

        # Convert state to vector
        state_vector = self._state_to_vector(current_state, task_type)

        # Select action using RL agent
        if hasattr(agent, 'select_action'):
            action_idx = agent.select_action(state_vector)
            action = available_actions[action_idx % len(available_actions)]
        else:
            # Multi-armed bandit
            arm_idx = agent.select_arm()
            action = available_actions[arm_idx % len(available_actions)]

        return action

    def learn_from_experience(self, task_type: str, state: Dict[str, Any],
                            action: str, reward: float, next_state: Dict[str, Any],
                            done: bool):
        """Learn from experience using reinforcement learning"""
        if task_type not in self.agents:
            return

        agent = self.agents[task_type]

        # Convert to vectors
        state_vector = self._state_to_vector(state, task_type)
        next_state_vector = self._state_to_vector(next_state, task_type)

        # Create experience
        action_idx = self._action_to_index(action, task_type)
        experience = Experience(state_vector, action_idx, reward, next_state_vector, done)

        # Store experience and train
        if hasattr(agent, 'replay_buffer'):
            agent.replay_buffer.push(experience)
            agent.train_step()
        elif hasattr(agent, 'store_reward'):
            agent.store_reward(reward)
            if done:
                agent.train_step()
        else:
            # Multi-armed bandit update
            arm_idx = self._action_to_index(action, task_type)
            agent.update(arm_idx, reward)

        # Track performance
        self._track_performance(task_type, reward)

    def simulate_learning(self, task_type: str, episodes: int = 100):
        """Simulate learning process for a task"""
        if task_type not in self.environments:
            logger.warning(f"No environment for task: {task_type}")
            return

        env = self.environments[task_type]
        agent = self.agents.get(task_type)

        if not agent:
            return

        logger.info(f"Simulating learning for {task_type}...")

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done and steps < env.max_steps:
                action = agent.select_action(state) if hasattr(agent, 'select_action') else 0

                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                # Learn from experience
                if hasattr(agent, 'replay_buffer'):
                    experience = Experience(state, action, reward, next_state, done)
                    agent.replay_buffer.push(experience)
                    agent.train_step()

                state = next_state
                steps += 1

            if episode % 20 == 0:
                logger.info(f"Episode {episode}: Total Reward = {total_reward:.2f}")

    def get_strategy_performance(self, task_type: str) -> Dict[str, Any]:
        """Get performance metrics for a task"""
        if task_type not in self.performance_history:
            return {'average_reward': 0, 'success_rate': 0, 'total_episodes': 0}

        history = self.performance_history[task_type]
        rewards = [exp[2] for exp in history]  # reward is at index 2

        return {
            'average_reward': np.mean(rewards) if rewards else 0,
            'success_rate': np.mean([1 if r > 0 else 0 for r in rewards]) if rewards else 0,
            'total_episodes': len(history),
            'recent_performance': np.mean(rewards[-10:]) if len(rewards) >= 10 else 0
        }

    def optimize_strategy(self, task_type: str) -> Dict[str, Any]:
        """Optimize strategy using learned knowledge"""
        performance = self.get_strategy_performance(task_type)

        # Generate optimization recommendations
        recommendations = []

        if performance['average_reward'] < 0:
            recommendations.append("Increase exploration rate")
        if performance['success_rate'] < 0.5:
            recommendations.append("Review reward function")
        if performance['total_episodes'] > 1000:
            recommendations.append("Consider strategy reset")

        return {
            'current_performance': performance,
            'recommendations': recommendations,
            'suggested_parameters': self._suggest_parameters(task_type, performance)
        }

    def _state_to_vector(self, state: Dict[str, Any], task_type: str) -> np.ndarray:
        """Convert state dict to vector"""
        if task_type == 'browser_automation':
            vector_size = 50
        elif task_type == 'form_filling':
            vector_size = 30
        else:
            vector_size = 20

        # Simple feature extraction
        features = []
        for key, value in state.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, bool):
                features.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                # Hash string to float
                features.append(hash(value) % 1000 / 1000.0)
            else:
                features.append(0.0)

        # Pad or truncate to vector size
        while len(features) < vector_size:
            features.append(0.0)

        return np.array(features[:vector_size])

    def _action_to_index(self, action: str, task_type: str) -> int:
        """Convert action string to index"""
        # Simple mapping - would be more sophisticated in production
        action_map = {
            'browser_automation': {
                'click': 0, 'wait': 1, 'scroll': 2, 'type': 3, 'navigate': 4,
                'refresh': 5, 'back': 6, 'forward': 7, 'close': 8, 'switch_tab': 9
            },
            'form_filling': {
                'fill_text': 0, 'select_option': 1, 'click_submit': 2, 'clear_field': 3, 'skip_field': 4
            }
        }

        return action_map.get(task_type, {}).get(action, 0)

    def _track_performance(self, task_type: str, reward: float):
        """Track performance for analysis"""
        if task_type not in self.performance_history:
            self.performance_history[task_type] = []

        # Keep last 1000 experiences
        self.performance_history[task_type].append(reward)
        if len(self.performance_history[task_type]) > 1000:
            self.performance_history[task_type] = self.performance_history[task_type][-1000:]

    def _suggest_parameters(self, task_type: str, performance: Dict) -> Dict[str, Any]:
        """Suggest parameter adjustments based on performance"""
        suggestions = {}

        if performance['success_rate'] < 0.3:
            suggestions['epsilon'] = 'increase'  # More exploration
            suggestions['learning_rate'] = 'decrease'  # More stable learning
        elif performance['success_rate'] > 0.8:
            suggestions['epsilon'] = 'decrease'  # Exploit more
            suggestions['gamma'] = 'increase'  # Consider future rewards more

        return suggestions

    def save_models(self, path: str = "models/rl"):
        """Save RL models"""
        path = Path(path)
        path.mkdir(exist_ok=True)

        for task_type, agent in self.agents.items():
            if hasattr(agent, 'q_network'):
                torch.save(agent.q_network.state_dict(), path / f"{task_type}_q_network.pth")
            elif hasattr(agent, 'policy_network'):
                torch.save(agent.policy_network.state_dict(), path / f"{task_type}_policy.pth")

        logger.info("RL models saved")

    def load_models(self, path: str = "models/rl"):
        """Load RL models"""
        path = Path(path)

        for task_type, agent in self.agents.items():
            model_path = path / f"{task_type}_q_network.pth"
            if model_path.exists() and hasattr(agent, 'q_network'):
                agent.q_network.load_state_dict(torch.load(model_path))
                agent.target_network.load_state_dict(agent.q_network.state_dict())
                logger.info(f"Loaded Q-network for {task_type}")

            policy_path = path / f"{task_type}_policy.pth"
            if policy_path.exists() and hasattr(agent, 'policy_network'):
                agent.policy_network.load_state_dict(torch.load(policy_path))
                logger.info(f"Loaded policy network for {task_type}")

# Global instance
adaptive_engine = AdaptiveBehaviorEngine()