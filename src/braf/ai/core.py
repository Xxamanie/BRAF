#!/usr/bin/env python3
"""
BRAF AI Core - Advanced AI/ML Integration Layer
The foundation for BRAF's intelligence capabilities
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class AIModelManager:
    """Central manager for all AI models in BRAF"""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path("src/braf/ai") / model_dir
        self.model_dir.mkdir(exist_ok=True)
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"AI Model Manager initialized on device: {self.device}")

        # Initialize core models
        self._load_core_models()

    def _load_core_models(self):
        """Load or initialize core AI models"""
        try:
            # Decision making model
            self.models['decision_maker'] = self._load_model('decision_maker')

            # Pattern recognition model
            self.models['pattern_recognizer'] = self._load_model('pattern_recognizer')

            # Behavior prediction model
            self.models['behavior_predictor'] = self._load_model('behavior_predictor')

            # Risk assessment model
            self.models['risk_assessor'] = self._load_model('risk_assessor')

            logger.info("Core AI models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load core models: {e}")
            self.models = {}  # Fallback to empty dict

    def _load_model(self, model_name: str) -> Optional[nn.Module]:
        """Load a specific model or create default if not exists"""
        model_path = self.model_dir / f"{model_name}.pth"

        if model_path.exists():
            try:
                # Load existing model
                model = torch.load(model_path, map_location=self.device)
                model.eval()
                return model
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")

        # Create default model if not exists
        return self._create_default_model(model_name)

    def _create_default_model(self, model_name: str) -> nn.Module:
        """Create a default model for the given type"""
        if model_name == 'decision_maker':
            model = DecisionNetwork(input_size=128, hidden_size=256, output_size=2)
        elif model_name == 'pattern_recognizer':
            model = PatternRecognitionNetwork(input_size=512, num_classes=10)
        elif model_name == 'behavior_predictor':
            model = BehaviorPredictionNetwork(input_size=256, sequence_length=50, output_size=32)
        elif model_name == 'risk_assessor':
            model = RiskAssessmentNetwork(input_size=64, output_size=1)
        else:
            # Generic fallback
            model = GenericNetwork(input_size=128, output_size=64)

        model.to(self.device)
        self._save_model(model, model_name)
        return model

    def _save_model(self, model: nn.Module, model_name: str):
        """Save model to disk"""
        model_path = self.model_dir / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)

    def predict(self, model_name: str, input_data: torch.Tensor) -> torch.Tensor:
        """Make prediction using specified model"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return torch.zeros(1)

        model = self.models[model_name]
        with torch.no_grad():
            input_data = input_data.to(self.device)
            output = model(input_data)

        return output

    def train_model(self, model_name: str, train_data: torch.utils.data.DataLoader,
                   epochs: int = 10, learning_rate: float = 0.001):
        """Train a specific model"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return

        model = self.models[model_name]
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss() if len(model(torch.zeros(1, model.input_size))) == 1 else nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in train_data:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_data)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save updated model
        self._save_model(model, model_name)
        model.eval()

    def update_model_from_experience(self, model_name: str, experience_data: Dict[str, Any]):
        """Update model using reinforcement learning experience"""
        # Implementation for online learning
        pass

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model"""
        if model_name not in self.models:
            return {}

        model = self.models[model_name]
        return {
            'name': model_name,
            'parameters': sum(p.numel() for p in model.parameters()),
            'device': str(self.device),
            'last_updated': datetime.now().isoformat()
        }

class AIFeatures:
    """High-level AI features for BRAF operations"""

    def __init__(self, model_manager: AIModelManager):
        self.model_manager = model_manager
        self.feature_cache = {}

    def intelligent_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make intelligent decisions using AI models"""
        # Convert context to tensor
        context_vector = self._context_to_vector(context)

        # Get decision from model
        decision_output = self.model_manager.predict('decision_maker', context_vector)

        # Interpret decision
        decision = 'browser' if decision_output[0] > decision_output[1] else 'http'
        confidence = abs(decision_output[0] - decision_output[1]).item()

        return {
            'decision': decision,
            'confidence': confidence,
            'factors': self._analyze_decision_factors(context, decision_output)
        }

    def predict_behavior_pattern(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict future behavior patterns"""
        # Process historical data
        sequence = self._historical_to_sequence(historical_data)

        # Predict next behavior
        prediction = self.model_manager.predict('behavior_predictor', sequence)

        return {
            'predicted_actions': self._interpret_behavior_prediction(prediction),
            'confidence': 0.85,  # Placeholder
            'risk_level': self._calculate_prediction_risk(prediction)
        }

    def assess_risk(self, scenario: Dict[str, Any]) -> float:
        """Assess risk level of a given scenario"""
        scenario_vector = self._scenario_to_vector(scenario)
        risk_score = self.model_manager.predict('risk_assessor', scenario_vector)

        return risk_score.item()

    def recognize_patterns(self, data: np.ndarray) -> List[str]:
        """Recognize patterns in data using AI"""
        data_tensor = torch.from_numpy(data).float()
        patterns = self.model_manager.predict('pattern_recognizer', data_tensor)

        # Convert to pattern names
        pattern_indices = torch.argmax(patterns, dim=1).tolist()
        pattern_names = [f'pattern_{i}' for i in pattern_indices]

        return pattern_names

    def _context_to_vector(self, context: Dict[str, Any]) -> torch.Tensor:
        """Convert context dict to tensor"""
        # Simple feature extraction - expand in production
        features = []
        features.append(hash(context.get('url', '')) % 1000 / 1000.0)  # URL hash normalized
        features.append(len(context.get('url', '')) / 1000.0)  # URL length
        features.append(1.0 if 'login' in context.get('url', '').lower() else 0.0)  # Login indicator
        features.append(1.0 if 'dashboard' in context.get('url', '').lower() else 0.0)  # Dashboard indicator

        # Pad to input size
        while len(features) < 128:
            features.append(0.0)

        return torch.tensor(features[:128]).float().unsqueeze(0)

    def _scenario_to_vector(self, scenario: Dict[str, Any]) -> torch.Tensor:
        """Convert scenario to vector"""
        features = [0.0] * 64  # Fixed size
        # Add scenario-specific features
        return torch.tensor(features).float().unsqueeze(0)

    def _historical_to_sequence(self, historical: List[Dict[str, Any]]) -> torch.Tensor:
        """Convert historical data to sequence tensor"""
        sequence = []
        for item in historical[-50:]:  # Last 50 items
            features = [0.0] * 256  # Feature vector
            sequence.append(features)

        # Pad sequence
        while len(sequence) < 50:
            sequence.insert(0, [0.0] * 256)

        return torch.tensor(sequence).float().unsqueeze(0)

    def _analyze_decision_factors(self, context: Dict[str, Any], decision_output: torch.Tensor) -> List[str]:
        """Analyze factors contributing to decision"""
        factors = []
        if 'dashboard' in context.get('url', '').lower():
            factors.append('Dashboard URL detected')
        if len(context.get('url', '')) > 100:
            factors.append('Complex URL structure')
        return factors

    def _interpret_behavior_prediction(self, prediction: torch.Tensor) -> List[str]:
        """Interpret behavior prediction"""
        return ['click', 'wait', 'scroll']  # Placeholder

    def _calculate_prediction_risk(self, prediction: torch.Tensor) -> str:
        """Calculate risk level of prediction"""
        return 'low'  # Placeholder

# Neural Network Architectures

class DecisionNetwork(nn.Module):
    """Neural network for decision making"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)

class PatternRecognitionNetwork(nn.Module):
    """CNN-based pattern recognition"""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.input_size = input_size

        # Simple CNN architecture
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * (input_size // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class BehaviorPredictionNetwork(nn.Module):
    """LSTM-based behavior prediction"""

    def __init__(self, input_size: int, sequence_length: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size, 128, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(128, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

class RiskAssessmentNetwork(nn.Module):
    """Network for risk assessment"""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class GenericNetwork(nn.Module):
    """Generic neural network fallback"""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)

# Global instances
ai_model_manager = AIModelManager()
ai_features = AIFeatures(ai_model_manager)