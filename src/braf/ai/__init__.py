# BRAF AI/ML Integration Layer
"""
Advanced artificial intelligence and machine learning integration for BRAF.
Provides the foundation for intelligent automation, decision making, and adaptive behavior.
"""

from .core import AIModelManager, AIFeatures
from .ml_engine import MLEngine
from .neural_networks import NeuralNetworkManager
from .learning_system import ContinuousLearningSystem

__all__ = [
    'AIModelManager',
    'AIFeatures',
    'MLEngine',
    'NeuralNetworkManager',
    'ContinuousLearningSystem'
]