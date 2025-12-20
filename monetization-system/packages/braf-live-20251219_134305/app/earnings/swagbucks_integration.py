"""
Swagbucks Live Integration for Real Survey Earnings
Handles actual Swagbucks API calls for survey completion and earnings
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import time
import random

logger = logging.getLogger(__name__)

class SwagbucksIntegration:
    """Real Swagbucks API integration for survey earnings"""
    
    def __init__(self):
        self.api_key = os.getenv('SWAGBUCKS_API_KEY')
        self.user_id = os.getenv('SWAGBUCKS_USER_ID')
        self.base_url = os.getenv('SWAGBUCKS_BASE_URL', 'https://api.swagbucks.com/v1')
        self.partner_id = os.getenv('SWAGBUCKS_PARTNER_ID')
        
        # Validate credentials
        if not all([self.api_key, self.user_id]):
            logger.warning("Swagbucks credentials not configured - running in demo mode")
            self.demo_mode = True
        else:
            self.demo_mode = False
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make authenticated request to Swagbucks API"""
        if self.demo_mode:
            return self._demo_response(endpoint, params)
        
        url = f"{self.base_url}/{endpoint}"
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'BRAF-Automation/1.0'
        }
        
        if params:
            params['userId'] = self.user_id
            params['partnerId'] = self.partner_id
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Swagbucks API request failed: {e}")
            return {
                'status': 'error',
                'message': f'API request failed: {str(e)}',
                'data': None
            }
    
    def _demo_response(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate demo response for testing"""
        if endpoint == 'surveys/available':
            return {
                'status': 'success',
                'data': {
                    'surveys': [
                        {
                            'id': f'SB_SURVEY_{i}',
                            'title': f'Consumer Survey #{i}',
                            'description': f'Share your opinion about products and services',
                            'reward': random.randint(50, 500),  # SB points (1 SB = $0.01)
                            'estimatedTime': random.randint(5, 20),  # minutes
                            'category': random.choice(['Shopping', 'Technology', 'Food', 'Travel']),
                            'difficulty': random.choice(['Easy', 'Medium']),
                            'expiresAt': (datetime.now() + timedelta(hours=24)).isoformat()
                        }
                        for i in range(1, random.randint(3, 8))
                    ]
                }
            }
        elif endpoint == 'surveys/complete':
            survey_id = params.get('surveyId') if params else 'DEMO_SURVEY'
            reward = random.randint(100, 800)
            return {
                'status': 'success',
                'data': {
                    'surveyId': survey_id,
                    'completed': True,
                    'reward': reward,
                    'currency': 'SB',
                    'usdValue': reward * 0.01,  # 1 SB = $0.01
                    'completedAt': datetime.now().isoformat(),
                    'transactionId': f'SB_TXN_{datetime.now().strftime("%Y%m%d%H%M%S%f")}'
                }
            }
        elif endpoint == 'account/balance':
            return {
                'status': 'success',
                'data': {
                    'balance': random.randint(5000, 50000),  # SB points
                    'usdValue': random.randint(50, 500),  # USD equivalent
                    'lifetimeEarnings': random.randint(100000, 1000000),
                    'lastUpdated': datetime.now().isoformat()
                }
            }
        else:
            return {
                'status': 'success',
                'data': {}
            }
    
    def get_available_surveys(self) -> List[Dict[str, Any]]:
        """Get list of available surveys"""
        result = self._make_request('surveys/available')
        
        if result.get('status') == 'success':
            surveys = result.get('data', {}).get('surveys', [])
            logger.info(f"Found {len(surveys)} available surveys")
            return surveys
        else:
            logger.error(f"Failed to get surveys: {result.get('message')}")
            return []
    
    def complete_survey(self, survey_id: str, answers: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete a survey and earn rewards
        
        Args:
            survey_id: Survey identifier
            answers: Survey responses (for demo purposes)
            
        Returns:
            Dict containing completion result and earnings
        """
        # In demo mode, simulate survey completion time
        if self.demo_mode:
            completion_time = random.randint(3, 15)  # 3-15 minutes
            logger.info(f"Simulating survey completion: {completion_time} minutes")
            time.sleep(min(completion_time, 2))  # Cap simulation time for demo
        
        params = {'surveyId': survey_id}
        if answers:
            params['answers'] = json.dumps(answers)
        
        result = self._make_request('surveys/complete', params)
        
        if result.get('status') == 'success':
            data = result.get('data', {})
            earnings = data.get('usdValue', 0)
            logger.info(f"Survey completed successfully: ${earnings:.2f} earned")
        else:
            logger.error(f"Survey completion failed: {result.get('message')}")
        
        return result
    
    def get_account_balance(self) -> Dict[str, Any]:
        """Get current account balance and earnings"""
        return self._make_request('account/balance')
    
    def redeem_points(self, amount_sb: int, method: str = 'paypal') -> Dict[str, Any]:
        """
        Redeem Swagbucks points for cash
        
        Args:
            amount_sb: Amount in SB points to redeem
            method: Redemption method (paypal, gift_card, etc.)
            
        Returns:
            Dict containing redemption result
        """
        if self.demo_mode:
            return {
                'status': 'success',
                'data': {
                    'redemptionId': f'SB_REDEEM_{datetime.now().strftime("%Y%m%d%H%M%S")}',
                    'amount': amount_sb,
                    'usdValue': amount_sb * 0.01,
                    'method': method,
                    'status': 'pending',
                    'estimatedProcessingTime': '3-5 business days',
                    'redeemedAt': datetime.now().isoformat()
                }
            }
        
        params = {
            'amount': amount_sb,
            'method': method
        }
        
        return self._make_request('redeem', params)
    
    def get_earnings_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get earnings history for specified period"""
        params = {
            'startDate': (datetime.now() - timedelta(days=days)).isoformat(),
            'endDate': datetime.now().isoformat()
        }
        
        result = self._make_request('earnings/history', params)
        
        if result.get('status') == 'success':
            return result.get('data', {}).get('earnings', [])
        else:
            return []
    
    def estimate_daily_earnings(self) -> Dict[str, float]:
        """Estimate potential daily earnings based on available surveys"""
        surveys = self.get_available_surveys()
        
        if not surveys:
            return {'min_usd': 0, 'max_usd': 0, 'avg_usd': 0}
        
        # Calculate potential earnings
        total_rewards = sum(survey.get('reward', 0) for survey in surveys)
        min_earnings = total_rewards * 0.01 * 0.3  # Conservative estimate (30% completion)
        max_earnings = total_rewards * 0.01 * 0.8  # Optimistic estimate (80% completion)
        avg_earnings = total_rewards * 0.01 * 0.5  # Average estimate (50% completion)
        
        return {
            'min_usd': round(min_earnings, 2),
            'max_usd': round(max_earnings, 2),
            'avg_usd': round(avg_earnings, 2),
            'available_surveys': len(surveys)
        }

# Global instance
swagbucks_client = SwagbucksIntegration()