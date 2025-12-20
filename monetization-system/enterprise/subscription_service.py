import stripe
from datetime import datetime, timedelta
from typing import Dict, Optional
from enum import Enum

class SubscriptionTier(Enum):
    BASIC = "basic"  # $99/month - 1 template
    PROFESSIONAL = "pro"  # $299/month - All templates
    ENTERPRISE = "enterprise"  # $999/month - Custom limits

class EnterpriseSubscription:
    def __init__(self, stripe_api_key: str):
        stripe.api_key = stripe_api_key
        self.plans = {
            "basic": {
                "price": 9900,  # $99 in cents
                "templates": ["survey"],
                "max_automations": 5,
                "daily_earnings_limit": 50
            },
            "pro": {
                "price": 29900,
                "templates": ["survey", "video", "content"],
                "max_automations": 20,
                "daily_earnings_limit": 200
            },
            "enterprise": {
                "price": 99900,
                "templates": ["survey", "video", "content"],
                "max_automations": 100,
                "daily_earnings_limit": 1000,
                "custom_integrations": True
            }
        }

    def create_subscription(self, enterprise_id: str, tier: str,
                           payment_method_id: str) -> Dict:
        """Create new subscription for enterprise"""
        try:
            # Create Stripe customer
            customer = stripe.Customer.create(
                email=f"enterprise-{enterprise_id}@braf.com",
                payment_method=payment_method_id,
                invoice_settings={
                    'default_payment_method': payment_method_id
                }
            )
            
            # Create subscription
            subscription = stripe.Subscription.create(
                customer=customer.id,
                items=[{
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': f'{tier.upper()} Tier Subscription',
                        },
                        'unit_amount': self.plans[tier]["price"],
                        'recurring': {
                            'interval': 'month',
                        },
                    },
                }],
                payment_behavior='default_incomplete',
                expand=['latest_invoice.payment_intent']
            )
            
            return {
                "subscription_id": subscription.id,
                "client_secret": subscription.latest_invoice.payment_intent.client_secret,
                "status": "active",
                "tier": tier,
                "features": self.plans[tier]
            }
        except Exception as e:
            raise Exception(f"Subscription creation failed: {str(e)}")