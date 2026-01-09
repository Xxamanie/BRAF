#!/usr/bin/env python3
"""
BRAF Predictive Analytics Engine
Advanced forecasting and predictive modeling for proactive decision making
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)

class TimeSeriesPredictor(nn.Module):
    """LSTM-based time series prediction model"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class AnomalyDetector:
    """Multi-modal anomaly detection"""

    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False

    def fit(self, data: np.ndarray):
        """Train anomaly detector"""
        scaled_data = self.scaler.fit_transform(data)
        self.isolation_forest.fit(scaled_data)
        self.trained = True

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Detect anomalies in data"""
        if not self.trained:
            return np.zeros(len(data))

        scaled_data = self.scaler.transform(data)
        predictions = self.isolation_forest.predict(scaled_data)
        return predictions  # -1 for anomaly, 1 for normal

    def anomaly_score(self, data: np.ndarray) -> np.ndarray:
        """Get anomaly scores"""
        if not self.trained:
            return np.zeros(len(data))

        scaled_data = self.scaler.transform(data)
        return -self.isolation_forest.score_samples(scaled_data)

class TrendAnalyzer:
    """Advanced trend analysis and forecasting"""

    def __init__(self):
        self.models = {}
        self.data_history = deque(maxlen=10000)

    def add_data_point(self, timestamp: datetime, metrics: Dict[str, float]):
        """Add data point for analysis"""
        self.data_history.append({
            'timestamp': timestamp,
            'metrics': metrics
        })

    def analyze_trends(self, metric_name: str, window_days: int = 7) -> Dict[str, Any]:
        """Analyze trends for a specific metric"""
        if len(self.data_history) < 10:
            return {'trend': 'insufficient_data'}

        # Extract data for metric
        data = []
        for point in self.data_history:
            if metric_name in point['metrics']:
                data.append({
                    'ds': point['timestamp'],
                    'y': point['metrics'][metric_name]
                })

        if len(data) < 10:
            return {'trend': 'insufficient_data'}

        df = pd.DataFrame(data)

        try:
            # Fit Prophet model
            model = Prophet(daily_seasonality=True, weekly_seasonality=True)
            model.fit(df)

            # Make future predictions
            future = model.make_future_dataframe(periods=24, freq='H')
            forecast = model.predict(future)

            # Analyze trend
            trend = forecast['trend'].iloc[-1] - forecast['trend'].iloc[-25]

            # Calculate confidence intervals
            yhat_lower = forecast['yhat_lower'].iloc[-1]
            yhat_upper = forecast['yhat_upper'].iloc[-1]
            current_value = df['y'].iloc[-1]

            return {
                'trend_direction': 'increasing' if trend > 0 else 'decreasing',
                'trend_slope': trend,
                'forecast_next_hour': forecast['yhat'].iloc[-1],
                'confidence_interval': [yhat_lower, yhat_upper],
                'anomaly_risk': self._calculate_anomaly_risk(current_value, yhat_lower, yhat_upper),
                'seasonal_pattern': self._detect_seasonal_pattern(df)
            }

        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {'trend': 'analysis_failed', 'error': str(e)}

    def _calculate_anomaly_risk(self, current: float, lower: float, upper: float) -> str:
        """Calculate anomaly risk level"""
        if current < lower or current > upper:
            return 'high'
        elif current < lower * 1.1 or current > upper * 0.9:
            return 'medium'
        else:
            return 'low'

    def _detect_seasonal_pattern(self, df: pd.DataFrame) -> str:
        """Detect seasonal patterns in data"""
        if len(df) < 24:
            return 'insufficient_data'

        # Simple hourly pattern detection
        hourly_avg = df.groupby(df['ds'].dt.hour)['y'].mean()

        if hourly_avg.std() > hourly_avg.mean() * 0.5:
            return 'strong_hourly_pattern'
        elif hourly_avg.std() > hourly_avg.mean() * 0.2:
            return 'moderate_hourly_pattern'
        else:
            return 'weak_pattern'

class ThreatPredictor:
    """Predict future security threats and anti-bot measures"""

    def __init__(self):
        self.threat_history = []
        self.pattern_recognizer = None

    def record_threat(self, threat_type: str, severity: str, timestamp: datetime,
                     context: Dict[str, Any]):
        """Record a detected threat"""
        self.threat_history.append({
            'type': threat_type,
            'severity': severity,
            'timestamp': timestamp,
            'context': context
        })

        # Keep only recent threats (last 1000)
        if len(self.threat_history) > 1000:
            self.threat_history = self.threat_history[-1000:]

    def predict_future_threats(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Predict future threats based on patterns"""
        if len(self.threat_history) < 10:
            return []

        # Analyze threat patterns
        threat_types = {}
        severity_trends = {}

        for threat in self.threat_history[-100:]:  # Last 100 threats
            t_type = threat['type']
            severity = threat['severity']

            if t_type not in threat_types:
                threat_types[t_type] = []
            threat_types[t_type].append(threat['timestamp'])

            if severity not in severity_trends:
                severity_trends[severity] = []
            severity_trends[severity].append(threat['timestamp'])

        predictions = []

        # Predict based on frequency patterns
        for threat_type, timestamps in threat_types.items():
            if len(timestamps) >= 3:
                # Calculate average interval
                intervals = []
                for i in range(1, len(timestamps)):
                    interval = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # hours
                    intervals.append(interval)

                avg_interval = np.mean(intervals)
                next_occurrence = timestamps[-1] + timedelta(hours=avg_interval)

                predictions.append({
                    'threat_type': threat_type,
                    'predicted_time': next_occurrence,
                    'confidence': min(0.9, len(timestamps) / 10),
                    'severity_trend': self._analyze_severity_trend(threat_type)
                })

        return sorted(predictions, key=lambda x: x['predicted_time'])

    def _analyze_severity_trend(self, threat_type: str) -> str:
        """Analyze if threat severity is increasing"""
        recent_threats = [t for t in self.threat_history[-50:]
                         if t['type'] == threat_type]

        if len(recent_threats) < 5:
            return 'stable'

        # Check if severity is increasing
        severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        severities = [severity_map.get(t['severity'], 1) for t in recent_threats]

        if len(severities) >= 2:
            trend = np.polyfit(range(len(severities)), severities, 1)[0]
            if trend > 0.1:
                return 'increasing'
            elif trend < -0.1:
                return 'decreasing'

        return 'stable'

class OpportunityPredictor:
    """Predict future monetization opportunities"""

    def __init__(self):
        self.opportunity_history = []
        self.success_predictor = None

    def record_opportunity(self, platform: str, task_type: str, success: bool,
                          earnings: float, timestamp: datetime):
        """Record monetization opportunity result"""
        self.opportunity_history.append({
            'platform': platform,
            'task_type': task_type,
            'success': success,
            'earnings': earnings,
            'timestamp': timestamp
        })

    def predict_best_opportunities(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Predict most profitable opportunities"""
        if len(self.opportunity_history) < 20:
            return []

        # Analyze success rates by platform and task
        platform_stats = {}
        task_stats = {}

        for opp in self.opportunity_history[-200:]:  # Last 200 opportunities
            platform = opp['platform']
            task = opp['task_type']

            # Platform stats
            if platform not in platform_stats:
                platform_stats[platform] = {'successes': 0, 'total': 0, 'earnings': 0}
            platform_stats[platform]['total'] += 1
            if opp['success']:
                platform_stats[platform]['successes'] += 1
            platform_stats[platform]['earnings'] += opp['earnings']

            # Task stats
            if task not in task_stats:
                task_stats[task] = {'successes': 0, 'total': 0, 'earnings': 0}
            task_stats[task]['total'] += 1
            if opp['success']:
                task_stats[task]['successes'] += 1
            task_stats[task]['earnings'] += opp['earnings']

        # Calculate success rates and profitability
        opportunities = []

        for platform, stats in platform_stats.items():
            success_rate = stats['successes'] / stats['total']
            avg_earnings = stats['earnings'] / stats['total']

            opportunities.append({
                'type': 'platform',
                'name': platform,
                'success_rate': success_rate,
                'avg_earnings': avg_earnings,
                'profitability_score': success_rate * avg_earnings,
                'recommendation': 'high' if success_rate > 0.7 and avg_earnings > 0.5 else 'medium'
            })

        for task, stats in task_stats.items():
            success_rate = stats['successes'] / stats['total']
            avg_earnings = stats['earnings'] / stats['total']

            opportunities.append({
                'type': 'task',
                'name': task,
                'success_rate': success_rate,
                'avg_earnings': avg_earnings,
                'profitability_score': success_rate * avg_earnings,
                'recommendation': 'high' if success_rate > 0.7 and avg_earnings > 0.5 else 'medium'
            })

        # Sort by profitability
        return sorted(opportunities, key=lambda x: x['profitability_score'], reverse=True)

class PredictiveAnalyticsEngine:
    """Main predictive analytics engine for BRAF"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize components
        self.time_series_predictor = TimeSeriesPredictor(10, 64, 2, 1).to(self.device)
        self.anomaly_detector = AnomalyDetector()
        self.trend_analyzer = TrendAnalyzer()
        self.threat_predictor = ThreatPredictor()
        self.opportunity_predictor = OpportunityPredictor()

        # Performance metrics history
        self.metrics_history = {
            'success_rate': [],
            'earnings': [],
            'detection_rate': [],
            'response_time': []
        }

        logger.info("Predictive Analytics Engine initialized")

    def predict_future_performance(self, metric_name: str, hours_ahead: int = 24) -> Dict[str, Any]:
        """Predict future performance for a metric"""
        if metric_name not in self.metrics_history or len(self.metrics_history[metric_name]) < 10:
            return {'prediction': 'insufficient_data'}

        # Prepare time series data
        data = np.array(self.metrics_history[metric_name][-100:])  # Last 100 points

        try:
            # Use ARIMA for forecasting
            model = ARIMA(data, order=(5, 1, 0))
            model_fit = model.fit()

            # Make prediction
            forecast = model_fit.forecast(steps=hours_ahead)

            # Calculate confidence intervals
            pred_ci = model_fit.get_forecast(steps=hours_ahead).conf_int()

            return {
                'predicted_value': forecast[0],
                'confidence_interval': [pred_ci.iloc[0, 0], pred_ci.iloc[0, 1]],
                'trend': 'increasing' if forecast[0] > data[-1] else 'decreasing',
                'volatility': np.std(data[-24:]) if len(data) >= 24 else 0,
                'prediction_quality': self._assess_prediction_quality(data, forecast[0])
            }

        except Exception as e:
            logger.error(f"Performance prediction failed: {e}")
            return {'prediction': 'failed', 'error': str(e)}

    def detect_anomalies(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect anomalies in current metrics"""
        if not self.anomaly_detector.trained:
            # Train on historical data
            if len(self.metrics_history['success_rate']) >= 50:
                historical_data = np.column_stack([
                    self.metrics_history['success_rate'][-50:],
                    self.metrics_history['earnings'][-50:],
                    self.metrics_history['detection_rate'][-50:],
                    self.metrics_history['response_time'][-50:]
                ])
                self.anomaly_detector.fit(historical_data)

        # Prepare current data
        current_data = np.array([[
            current_metrics.get('success_rate', 0),
            current_metrics.get('earnings', 0),
            current_metrics.get('detection_rate', 0),
            current_metrics.get('response_time', 0)
        ]])

        # Detect anomalies
        anomaly_predictions = self.anomaly_detector.predict(current_data)
        anomaly_scores = self.anomaly_detector.anomaly_score(current_data)

        anomaly_detected = anomaly_predictions[0] == -1
        anomaly_score = anomaly_scores[0]

        return {
            'anomaly_detected': anomaly_detected,
            'anomaly_score': anomaly_score,
            'severity': 'high' if anomaly_score > 0.7 else 'medium' if anomaly_score > 0.5 else 'low',
            'affected_metrics': self._identify_affected_metrics(current_metrics, anomaly_score)
        }

    def analyze_market_trends(self) -> Dict[str, Any]:
        """Analyze current market trends and opportunities"""
        # Get trend analysis for key metrics
        success_trend = self.trend_analyzer.analyze_trends('success_rate')
        earnings_trend = self.trend_analyzer.analyze_trends('earnings')
        detection_trend = self.trend_analyzer.analyze_trends('detection_rate')

        # Predict future opportunities
        opportunities = self.opportunity_predictor.predict_best_opportunities()

        # Predict future threats
        threats = self.threat_predictor.predict_future_threats()

        return {
            'performance_trends': {
                'success_rate': success_trend,
                'earnings': earnings_trend,
                'detection_rate': detection_trend
            },
            'recommended_opportunities': opportunities[:5],  # Top 5
            'predicted_threats': threats[:5],  # Next 5 threats
            'market_sentiment': self._calculate_market_sentiment(success_trend, earnings_trend),
            'risk_assessment': self._assess_market_risks(threats, detection_trend)
        }

    def proactive_decision_making(self) -> List[Dict[str, Any]]:
        """Generate proactive decisions based on predictions"""
        decisions = []

        # Analyze current situation
        market_analysis = self.analyze_market_trends()

        # Performance-based decisions
        if market_analysis['performance_trends']['success_rate'].get('trend_direction') == 'decreasing':
            decisions.append({
                'type': 'strategy_adjustment',
                'action': 'increase_exploration',
                'reason': 'Success rate declining - need to explore new strategies',
                'urgency': 'high',
                'expected_impact': 'Improve adaptability'
            })

        # Threat-based decisions
        if market_analysis['predicted_threats']:
            next_threat = market_analysis['predicted_threats'][0]
            if next_threat['confidence'] > 0.7:
                decisions.append({
                    'type': 'security_enhancement',
                    'action': 'prepare_countermeasures',
                    'reason': f"High-confidence threat prediction: {next_threat['threat_type']}",
                    'urgency': 'medium',
                    'expected_impact': 'Reduce future detection risk'
                })

        # Opportunity-based decisions
        if market_analysis['recommended_opportunities']:
            best_opp = market_analysis['recommended_opportunities'][0]
            if best_opp['recommendation'] == 'high':
                decisions.append({
                    'type': 'resource_allocation',
                    'action': 'increase_capacity',
                    'target': best_opp['name'],
                    'reason': f"High-profitability opportunity: {best_opp['name']}",
                    'urgency': 'medium',
                    'expected_impact': f"Increase earnings by {best_opp['profitability_score']:.2f}x"
                })

        return decisions

    def add_performance_data(self, metrics: Dict[str, float], timestamp: datetime = None):
        """Add performance data point"""
        if timestamp is None:
            timestamp = datetime.now()

        for metric_name, value in metrics.items():
            if metric_name in self.metrics_history:
                self.metrics_history[metric_name].append(value)

                # Keep only recent data
                if len(self.metrics_history[metric_name]) > 1000:
                    self.metrics_history[metric_name] = self.metrics_history[metric_name][-1000:]

        # Add to trend analyzer
        self.trend_analyzer.add_data_point(timestamp, metrics)

    def _assess_prediction_quality(self, historical_data: np.ndarray, prediction: float) -> str:
        """Assess quality of prediction"""
        if len(historical_data) < 5:
            return 'unknown'

        mean_val = np.mean(historical_data)
        std_val = np.std(historical_data)

        if abs(prediction - mean_val) > 2 * std_val:
            return 'high_deviation'
        elif abs(prediction - mean_val) > std_val:
            return 'moderate_deviation'
        else:
            return 'within_range'

    def _identify_affected_metrics(self, metrics: Dict[str, float], anomaly_score: float) -> List[str]:
        """Identify which metrics are affected by anomaly"""
        affected = []

        for metric_name, value in metrics.items():
            if metric_name in self.metrics_history and len(self.metrics_history[metric_name]) >= 10:
                historical_mean = np.mean(self.metrics_history[metric_name][-10:])
                historical_std = np.std(self.metrics_history[metric_name][-10:])

                if abs(value - historical_mean) > 2 * historical_std:
                    affected.append(metric_name)

        return affected

    def _calculate_market_sentiment(self, success_trend: Dict, earnings_trend: Dict) -> str:
        """Calculate overall market sentiment"""
        success_direction = success_trend.get('trend_direction', 'stable')
        earnings_direction = earnings_trend.get('trend_direction', 'stable')

        if success_direction == 'increasing' and earnings_direction == 'increasing':
            return 'bullish'
        elif success_direction == 'decreasing' or earnings_direction == 'decreasing':
            return 'bearish'
        else:
            return 'neutral'

    def _assess_market_risks(self, threats: List[Dict], detection_trend: Dict) -> Dict[str, Any]:
        """Assess market risks"""
        threat_count = len(threats)
        detection_direction = detection_trend.get('trend_direction', 'stable')

        risk_level = 'low'
        if threat_count > 3:
            risk_level = 'high'
        elif threat_count > 1:
            risk_level = 'medium'

        if detection_direction == 'decreasing':
            risk_level = 'high'  # Escalating risk

        return {
            'overall_risk': risk_level,
            'threat_count': threat_count,
            'detection_trend': detection_direction,
            'recommendations': self._generate_risk_recommendations(risk_level)
        }

    def _generate_risk_recommendations(self, risk_level: str) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = {
            'high': [
                'Implement emergency protocols',
                'Reduce operation scale temporarily',
                'Activate backup strategies',
                'Increase monitoring frequency'
            ],
            'medium': [
                'Enhance security measures',
                'Monitor key metrics closely',
                'Prepare contingency plans'
            ],
            'low': [
                'Continue normal operations',
                'Regular performance monitoring'
            ]
        }

        return recommendations.get(risk_level, [])

# Global instance
predictive_engine = PredictiveAnalyticsEngine()