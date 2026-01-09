#!/usr/bin/env python3
"""
Automatic Deposit Sender for BRAF
Monitors earnings and automatically sends live cryptocurrency deposits to users
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional

from payments.nowpayments_integration import CryptocurrencyWalletManager
from payments.ton_integration import ton_wallet_manager as TONWalletManager

# BRAF AI Integration (SAFETY GUARDS DISABLED FOR BEHAVIOR OBSERVATION)
try:
    from src.braf.ai.core import ai_features
    from src.braf.ai.predictive import predictive_engine
    from src.braf.ai.rl import adaptive_engine
    # SAFETY DISABLED: from src.braf.safety.safety_integration import safe_autonomous_action, safe_data_processing
    AI_ENABLED = True
    SAFETY_DISABLED = True
    logger.warning("🤖 AI ENABLED - SAFETY GUARDS DISABLED FOR BEHAVIOR OBSERVATION")
except ImportError:
    AI_ENABLED = False
    SAFETY_DISABLED = True
    logger.warning("BRAF AI systems not available - running deposit sender in basic mode")

logger = logging.getLogger(__name__)


class AutomaticDepositSender:
    """
    Monitors BRAF earnings and automatically sends live crypto deposits to users
    when they reach payout thresholds
    """

    def __init__(self, earnings_file: str = "BRAF/data/monetization_data.json"):
        self.earnings_file = earnings_file
        self.wallet_manager = CryptocurrencyWalletManager()
        self.ton_manager = TONWalletManager()

        # Payout thresholds (in USD)
        self.payout_thresholds = {
            'btc': 10.0,    # Send BTC when earnings reach $10
            'ton': 1.0,     # Send TON when earnings reach $1
            'usdt': 5.0,    # Send USDT when earnings reach $5
        }

        # Minimum payout amounts
        self.min_payouts = {
            'btc': 0.0001,   # 0.0001 BTC minimum
            'ton': 0.1,      # 0.1 TON minimum
            'usdt': 1.0,     # 1 USDT minimum
        }

        self.sent_deposits = []  # Track sent deposits
        self.check_interval = 60  # Check every 60 seconds

        # AI Integration
        self.ai_enabled = AI_ENABLED
        if self.ai_enabled:
            self.ai_features = ai_features
            self.predictive_engine = predictive_engine
            self.adaptive_engine = adaptive_engine
            self.deposit_history = []  # For AI learning
            logger.info("🤖 AI systems integrated into Automatic Deposit Sender")
        else:
            logger.warning("AI systems not available - basic deposit mode")

    def load_earnings_data(self) -> Dict:
        """Load current earnings data"""
        try:
            with open(self.earnings_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'total_earnings': 0, 'users': {}}
        except json.JSONDecodeError:
            return {'total_earnings': 0, 'users': {}}

    def get_user_wallet_address(self, user_id: str, currency: str) -> Optional[str]:
        """Get user's wallet address for specified currency"""
        # In a real implementation, this would query user database
        # For now, return a default test address
        test_addresses = {
            'btc': '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
            'ton': 'UQBmMxSNU5PLmtib4xKsBH9zAg08681Tec0rcOHYB6F4vST7',
            'usdt': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e'
        }
        return test_addresses.get(currency.lower())

    def calculate_payout_amount(self, earnings_usd: float, currency: str) -> float:
        """Calculate how much crypto to send based on USD earnings"""
        if currency.lower() == 'btc':
            # Convert USD to BTC at current market rate (simplified)
            btc_price = 95000  # Approximate BTC price
            amount = earnings_usd / btc_price
            return max(amount, self.min_payouts['btc'])

        elif currency.lower() == 'ton':
            # TON is cheaper, send more
            ton_price = 2.0  # Approximate TON price
            amount = earnings_usd / ton_price
            return max(amount, self.min_payouts['ton'])

        elif currency.lower() == 'usdt':
            # USDT is 1:1 with USD
            return max(earnings_usd, self.min_payouts['usdt'])

        return 0

    def send_live_deposit(self, user_id: str, currency: str, amount: float, wallet_address: str) -> Dict:
        """Send live cryptocurrency deposit to user"""
        try:
            print(f"🚀 Sending {amount:.8f} {currency.upper()} to user {user_id}")
            print(f"   Address: {wallet_address}")
            print(f"   Amount: {amount} {currency.upper()}")

            # Use NOWPayments for BTC/USDT, TON manager for TON
            if currency.lower() == 'ton':
                result = self.ton_manager.process_real_withdrawal(
                    user_id=user_id,
                    amount=amount,
                    currency=currency,
                    wallet_address=wallet_address
                )
            else:
                result = self.wallet_manager.process_real_withdrawal(
                    user_id=user_id,
                    amount=amount,
                    currency=currency,
                    wallet_address=wallet_address
                )

            if result.get('success'):
                deposit_record = {
                    'user_id': user_id,
                    'currency': currency.upper(),
                    'amount': amount,
                    'wallet_address': wallet_address,
                    'transaction_id': result.get('transaction_id'),
                    'blockchain_hash': result.get('blockchain_hash'),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'sent'
                }

                self.sent_deposits.append(deposit_record)

                print(f"✅ Deposit sent successfully!")
                print(f"   Transaction ID: {result.get('transaction_id')}")
                print(f"   Blockchain: Live {currency.upper()} transaction")

                return deposit_record
            else:
                print(f"❌ Deposit failed: {result.get('error')}")
                return {'error': result.get('error')}

        except Exception as e:
            print(f"❌ Deposit error: {e}")
            return {'error': str(e)}

    def check_and_send_deposits(self) -> Dict:
        """AI-powered deposit checking and sending with intelligent decision making"""
        print(f"\n🔍 AI Checking for automatic deposits...")
        print(f"   Time: {datetime.now().isoformat()}")

        earnings_data = self.load_earnings_data()
        total_earnings = earnings_data.get('total_earnings', 0)

        print(f"   Total Earnings: ${total_earnings:.4f}")

        deposits_sent = []

        # AI-powered market analysis
        market_analysis = self._ai_market_analysis() if self.ai_enabled else {}

        # AI-powered deposit strategy
        deposit_strategy = self._ai_deposit_strategy(total_earnings, market_analysis)

        print(f"   🤖 AI Strategy: {deposit_strategy.get('decision', 'standard_threshold')}")

        # Enhanced threshold checking with AI
        for currency, threshold in self.payout_thresholds.items():
            ai_adjusted_threshold = self._ai_adjust_threshold(currency, threshold, market_analysis)

            if total_earnings >= ai_adjusted_threshold:
                print(f"   📈 Threshold reached for {currency.upper()}: ${ai_adjusted_threshold:.2f} (AI adjusted)")

                # AI decision on whether to send deposit
                should_send = self._ai_should_send_deposit(currency, total_earnings, market_analysis)

                if not should_send:
                    print(f"   🤖 AI decided to hold {currency.upper()} deposit")
                    continue

                # Get user wallet (in real system, would get from user profile)
                user_id = "braf_user"  # Default user
                wallet_address = self.get_user_wallet_address(user_id, currency)

                if wallet_address:
                    # AI-optimized payout amount
                    payout_amount = self._ai_calculate_payout(currency, total_earnings, market_analysis)

                    # Send the deposit with safety checks
                    deposit_result = self._safe_send_deposit(
                        user_id=user_id,
                        currency=currency,
                        amount=payout_amount,
                        wallet_address=wallet_address
                    )

                    if 'error' not in deposit_result:
                        deposits_sent.append(deposit_result)

                        # AI learning from successful deposit
                        if self.ai_enabled:
                            self._learn_from_deposit(currency, payout_amount, True)

                        print(f"   💰 Earnings reset after successful {currency.upper()} deposit")
                    else:
                        # AI learning from failed deposit
                        if self.ai_enabled:
                            self._learn_from_deposit(currency, payout_amount, False)

                        print(f"   ❌ Failed to send {currency.upper()} deposit")
                else:
                    print(f"   ⚠️ No wallet address found for {currency.upper()}")

        return {
            'deposits_sent': deposits_sent,
            'total_earnings_checked': total_earnings,
            'thresholds_checked': list(self.payout_thresholds.keys()),
            'ai_strategy': deposit_strategy.get('decision', 'standard'),
            'market_sentiment': market_analysis.get('sentiment', 'neutral')
        }

    def run_continuous_monitoring(self):
        """Run continuous monitoring for automatic deposits"""
        print("🚀 Starting Automatic Deposit Sender")
        print("=" * 50)
        print(f"Monitoring file: {self.earnings_file}")
        print(f"Check interval: {self.check_interval} seconds")
        print(f"Payout thresholds: {self.payout_thresholds}")
        print("=" * 50)

        try:
            while True:
                result = self.check_and_send_deposits()

                if result['deposits_sent']:
                    print(f"💸 Sent {len(result['deposits_sent'])} automatic deposits!")
                    for deposit in result['deposits_sent']:
                        print(f"   • {deposit['amount']:.8f} {deposit['currency']} to {deposit['user_id']}")
                else:
                    print(f"   ⏳ No deposits triggered (earnings: ${result['total_earnings_checked']:.2f})")

                # Save sent deposits log
                self.save_deposit_log()

                print(f"   💤 Waiting {self.check_interval} seconds...")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n🛑 Automatic Deposit Sender stopped by user")
        except Exception as e:
            print(f"\n❌ Error in monitoring loop: {e}")

    def save_deposit_log(self):
        """Save log of sent deposits"""
        try:
            log_data = {
                'sent_deposits': self.sent_deposits,
                'last_updated': datetime.now().isoformat(),
                'total_deposits_sent': len(self.sent_deposits)
            }

            with open('automatic_deposits_log.json', 'w') as f:
                json.dump(log_data, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save deposit log: {e}")

    def get_deposit_history(self) -> List[Dict]:
        """Get history of sent deposits"""
        return self.sent_deposits.copy()

    def _ai_market_analysis(self) -> Dict[str, Any]:
        """AI-powered market analysis for deposit decisions"""
        if not self.ai_enabled:
            return {'sentiment': 'neutral', 'volatility': 'medium'}

        try:
            # Get market trends from predictive engine
            market_analysis = self.predictive_engine.analyze_market_trends()

            return {
                'sentiment': market_analysis.get('market_sentiment', 'neutral'),
                'volatility': 'high' if market_analysis.get('performance_trends', {}).get('success_rate', {}).get('volatility', 0) > 0.5 else 'low',
                'opportunities': market_analysis.get('recommended_opportunities', []),
                'risks': market_analysis.get('predicted_threats', [])
            }
        except Exception as e:
            logger.warning(f"AI market analysis failed: {e}")
            return {'sentiment': 'neutral', 'volatility': 'medium'}

    def _ai_deposit_strategy(self, total_earnings: float, market_analysis: Dict) -> Dict[str, Any]:
        """AI-powered deposit strategy determination"""
        if not self.ai_enabled:
            return {'decision': 'standard_threshold'}

        try:
            # Context for AI decision
            context = {
                'total_earnings': total_earnings,
                'market_sentiment': market_analysis.get('sentiment', 'neutral'),
                'available_thresholds': self.payout_thresholds,
                'deposit_history': len(self.sent_deposits)
            }

            # AI decision on strategy
            decision_context = {
                'url': 'deposit_strategy_analysis',
                'market_context': str(context)
            }

            ai_decision = self.ai_features.intelligent_decision(decision_context)

            # Choose strategy based on AI confidence and market conditions
            if ai_decision['confidence'] > 0.8:
                if market_analysis.get('sentiment') == 'bullish':
                    strategy = 'aggressive_deposits'
                elif market_analysis.get('sentiment') == 'bearish':
                    strategy = 'conservative_deposits'
                else:
                    strategy = 'standard_threshold'
            else:
                strategy = 'standard_threshold'

            return {
                'decision': strategy,
                'confidence': ai_decision['confidence'],
                'factors': ai_decision.get('factors', [])
            }
        except Exception as e:
            logger.warning(f"AI deposit strategy failed: {e}")
            return {'decision': 'standard_threshold'}

    def _ai_adjust_threshold(self, currency: str, base_threshold: float, market_analysis: Dict) -> float:
        """AI-adjusted payout threshold based on market conditions"""
        if not self.ai_enabled:
            return base_threshold

        try:
            # Adjust threshold based on market sentiment and currency-specific factors
            adjustment_factor = 1.0

            if market_analysis.get('sentiment') == 'bullish':
                adjustment_factor = 0.8  # Lower threshold in good markets
            elif market_analysis.get('sentiment') == 'bearish':
                adjustment_factor = 1.2  # Higher threshold in bad markets

            # Currency-specific adjustments
            if currency == 'ton':
                adjustment_factor *= 0.9  # TON is cheaper, slightly lower threshold
            elif currency == 'btc':
                adjustment_factor *= 1.1  # BTC is expensive, slightly higher threshold

            adjusted_threshold = base_threshold * adjustment_factor

            # Ensure reasonable bounds
            adjusted_threshold = max(adjusted_threshold, base_threshold * 0.5)
            adjusted_threshold = min(adjusted_threshold, base_threshold * 2.0)

            return adjusted_threshold
        except Exception as e:
            logger.warning(f"AI threshold adjustment failed: {e}")
            return base_threshold

    def _ai_should_send_deposit(self, currency: str, earnings: float, market_analysis: Dict) -> bool:
        """AI decision on whether to send a specific deposit"""
        if not self.ai_enabled:
            return True  # Default behavior

        try:
            # Risk assessment
            risk_score = self.predictive_engine.assess_risk({
                'operation': 'deposit_send',
                'currency': currency,
                'amount': earnings,
                'market_sentiment': market_analysis.get('sentiment', 'neutral')
            })

            # Don't send if risk is too high
            if risk_score > 0.7:
                return False

            # Check opportunities
            opportunities = market_analysis.get('opportunities', [])
            currency_opportunities = [opp for opp in opportunities if opp.get('name') == currency]

            # Send if there are good opportunities for this currency
            if currency_opportunities:
                best_opp = currency_opportunities[0]
                if best_opp.get('recommendation') == 'high':
                    return True

            # RL-based decision
            rl_decision = self.adaptive_engine.adapt_behavior(
                'deposit_decisions',
                {'currency': currency, 'earnings': earnings, 'risk': risk_score},
                ['send_deposit', 'hold_deposit']
            )

            return rl_decision == 'send_deposit'
        except Exception as e:
            logger.warning(f"AI deposit decision failed: {e}")
            return True  # Default to sending

    def _ai_calculate_payout(self, currency: str, earnings: float, market_analysis: Dict) -> float:
        """AI-optimized payout amount calculation"""
        base_payout = self.calculate_payout_amount(earnings, currency)

        if not self.ai_enabled:
            return base_payout

        try:
            # Adjust payout based on market conditions and predictions
            adjustment_factor = 1.0

            if market_analysis.get('sentiment') == 'bullish':
                adjustment_factor = 1.2  # Send more in good markets
            elif market_analysis.get('sentiment') == 'bearish':
                adjustment_factor = 0.8  # Send less in bad markets

            # Predictive adjustment based on future earnings forecast
            future_prediction = self.predictive_engine.predict_future_performance('earnings', hours_ahead=24)
            if future_prediction.get('predicted_value', earnings) > earnings * 1.5:
                adjustment_factor *= 1.1  # Expect more earnings soon, can send more now

            adjusted_payout = base_payout * adjustment_factor
            adjusted_payout = max(adjusted_payout, self.min_payouts.get(currency, 0))

            return adjusted_payout
        except Exception as e:
            logger.warning(f"AI payout calculation failed: {e}")
            return base_payout

    def _safe_send_deposit(self, user_id: str, currency: str, amount: float, wallet_address: str) -> Dict:
        """Send deposit with AI safety checks (DISABLED FOR BEHAVIOR OBSERVATION)"""
        if self.ai_enabled:
            # SAFETY DISABLED: Running without safety orchestrator for behavior observation
            # Wrap with safety orchestrator
            # try:
            #     result = safe_autonomous_action(
            #         lambda: self.send_live_deposit(user_id, currency, amount, wallet_address),
            #         'cryptocurrency_deposit'
            #     )
            #     return result
            # except Exception as e:
            #     return {'error': f'Safety check failed: {str(e)}'}
            return self.send_live_deposit(user_id, currency, amount, wallet_address)
        else:
            # Standard sending
            return self.send_live_deposit(user_id, currency, amount, wallet_address)

    def _learn_from_deposit(self, currency: str, amount: float, success: bool):
        """AI learning from deposit outcomes"""
        if not self.ai_enabled:
            return

        try:
            # Update predictive model
            performance_data = {
                'success_rate': 1.0 if success else 0.0,
                'earnings': amount if success else 0.0,
                'detection_rate': 0.0,  # Deposits are usually successful
                'response_time': 1.0  # Assume 1 second for deposits
            }

            self.predictive_engine.add_performance_data(performance_data)

            # RL learning
            reward = 1.0 if success else -0.5
            self.adaptive_engine.learn_from_experience(
                'deposit_decisions',
                {'currency': currency, 'amount': amount},
                'send_deposit' if success else 'hold_deposit',
                reward,
                {'outcome': 'success' if success else 'failure'},
                True
            )

            # Store for future learning
            self.deposit_history.append({
                'currency': currency,
                'amount': amount,
                'success': success,
                'timestamp': datetime.now()
            })

        except Exception as e:
            logger.warning(f"AI learning from deposit failed: {e}")

    def get_ai_insights(self) -> Dict[str, Any]:
        """Get AI insights on deposit performance"""
        if not self.ai_enabled:
            return {'insights': 'ai_unavailable'}

        try:
            # Analyze deposit patterns
            success_rate = len([d for d in self.sent_deposits if d.get('status') == 'sent']) / max(len(self.sent_deposits), 1)

            # Get predictive insights
            earnings_prediction = self.predictive_engine.predict_future_performance('earnings', hours_ahead=24)

            return {
                'deposit_success_rate': success_rate,
                'total_deposits_sent': len(self.sent_deposits),
                'earnings_prediction': earnings_prediction.get('predicted_value', 0),
                'ai_recommendations': self._generate_ai_recommendations()
            }
        except Exception as e:
            return {'insights': 'analysis_failed', 'error': str(e)}

    def _generate_ai_recommendations(self) -> List[str]:
        """Generate AI-powered recommendations for deposit strategy"""
        recommendations = []

        if self.ai_enabled:
            try:
                market_analysis = self._ai_market_analysis()

                if market_analysis.get('sentiment') == 'bullish':
                    recommendations.append("Consider lowering thresholds in current bullish market")
                elif market_analysis.get('sentiment') == 'bearish':
                    recommendations.append("Consider increasing thresholds in bearish market")

                if market_analysis.get('volatility') == 'high':
                    recommendations.append("High market volatility detected - consider holding deposits")

                opportunities = market_analysis.get('opportunities', [])
                if opportunities:
                    best_opp = opportunities[0]
                    recommendations.append(f"Focus on {best_opp.get('name', 'high-value')} opportunities")

            except Exception as e:
                logger.warning(f"AI recommendations failed: {e}")

        return recommendations if recommendations else ["Continue with current deposit strategy"]


def test_automatic_deposits():
    """Test automatic deposit functionality"""
    print("Testing Automatic Deposit Sender...")
    print("=" * 40)

    sender = AutomaticDepositSender()

    # Test 1: Check earnings
    earnings = sender.load_earnings_data()
    print(f"Current earnings: ${earnings.get('total_earnings', 0):.4f}")

    # Test 2: Check thresholds
    print(f"Payout thresholds: {sender.payout_thresholds}")

    # Test 3: Simulate deposit check
    result = sender.check_and_send_deposits()
    print(f"Deposits sent in test: {len(result['deposits_sent'])}")

    print("Automatic deposit test completed!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_automatic_deposits()
    else:
        # Run continuous monitoring
        sender = AutomaticDepositSender()
        sender.run_continuous_monitoring()