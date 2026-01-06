#!/usr/bin/env python3
"""
Cryptocurrency Webhook Handlers
Processes real payment notifications from NOWPayments and other crypto providers
"""

import os
import json
import hmac
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from payments.nowpayments_integration import NOWPaymentsIntegration
from crypto.real_crypto_infrastructure import RealCryptoInfrastructure
from database.service import DatabaseService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/crypto", tags=["crypto-webhooks"])

# Initialize services
nowpayments = NOWPaymentsIntegration()
crypto_infra = RealCryptoInfrastructure()
db_service = DatabaseService()


def verify_nowpayments_signature(payload: bytes, signature: str) -> bool:
    """Verify NOWPayments webhook signature"""
    webhook_secret = os.getenv('NOWPAYMENTS_WEBHOOK_SECRET', '')
    if not webhook_secret:
        logger.warning("NOWPayments webhook secret not configured")
        return True  # Allow in development
    
    expected_signature = hmac.new(
        webhook_secret.encode('utf-8'),
        payload,
        hashlib.sha512
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)


@router.post("/webhook/nowpayments")
async def nowpayments_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Handle NOWPayments webhook notifications
    Processes real cryptocurrency payment confirmations
    """
    try:
        # Get raw payload and signature
        payload = await request.body()
        signature = request.headers.get('x-nowpayments-sig', '')
        
        # Verify signature
        if not verify_nowpayments_signature(payload, signature):
            logger.error("Invalid NOWPayments webhook signature")
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        # Parse webhook data
        webhook_data = json.loads(payload.decode('utf-8'))
        
        logger.info(f"NOWPayments webhook received: {webhook_data}")
        
        # Process webhook in background
        background_tasks.add_task(process_nowpayments_webhook, webhook_data)
        
        return JSONResponse({"status": "success", "message": "Webhook processed"})
        
    except json.JSONDecodeError:
        logger.error("Invalid JSON in NOWPayments webhook")
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        logger.error(f"NOWPayments webhook error: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")


async def process_nowpayments_webhook(webhook_data: Dict[str, Any]):
    """Process NOWPayments webhook data"""
    try:
        payment_id = webhook_data.get('payment_id')
        payment_status = webhook_data.get('payment_status')
        order_id = webhook_data.get('order_id', '')
        
        logger.info(f"Processing payment {payment_id} with status {payment_status}")
        
        # Extract user ID from order ID (format: deposit_user123_btc_timestamp)
        if order_id.startswith('deposit_'):
            parts = order_id.split('_')
            if len(parts) >= 2:
                user_id = parts[1]
            else:
                logger.error(f"Invalid order ID format: {order_id}")
                return
        else:
            logger.error(f"Unknown order ID format: {order_id}")
            return
        
        # Process based on payment status
        if payment_status == 'finished':
            # Payment confirmed - credit user account
            await process_confirmed_deposit(webhook_data, user_id)
        elif payment_status == 'partially_paid':
            # Partial payment received
            await process_partial_payment(webhook_data, user_id)
        elif payment_status == 'failed':
            # Payment failed
            await process_failed_payment(webhook_data, user_id)
        elif payment_status == 'refunded':
            # Payment refunded
            await process_refunded_payment(webhook_data, user_id)
        else:
            logger.info(f"Payment {payment_id} status: {payment_status} - no action needed")
        
    except Exception as e:
        logger.error(f"Error processing NOWPayments webhook: {e}")


async def process_confirmed_deposit(webhook_data: Dict, user_id: str):
    """Process confirmed cryptocurrency deposit"""
    try:
        payment_id = webhook_data['payment_id']
        pay_amount = float(webhook_data.get('pay_amount', 0))
        pay_currency = webhook_data.get('pay_currency', '').upper()
        outcome_hash = webhook_data.get('outcome_hash')
        
        # Create deposit record
        deposit_data = {
            'user_id': user_id,
            'currency': pay_currency,
            'amount': pay_amount,
            'payment_id': payment_id,
            'blockchain_hash': outcome_hash
        }
        
        # Process deposit through crypto infrastructure
        result = crypto_infra.process_deposit(deposit_data)
        
        if result['success']:
            logger.info(f"Deposit processed: {pay_amount} {pay_currency} for user {user_id}")
            
            # Send notification to user (implement notification service)
            await send_deposit_notification(user_id, pay_amount, pay_currency, outcome_hash)
        else:
            logger.error(f"Failed to process deposit: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Error processing confirmed deposit: {e}")


async def process_partial_payment(webhook_data: Dict, user_id: str):
    """Process partial cryptocurrency payment"""
    try:
        payment_id = webhook_data['payment_id']
        pay_amount = float(webhook_data.get('pay_amount', 0))
        pay_currency = webhook_data.get('pay_currency', '').upper()
        price_amount = float(webhook_data.get('price_amount', 0))
        
        logger.info(f"Partial payment: {pay_amount} {pay_currency} of {price_amount} for user {user_id}")
        
        # Store partial payment record
        partial_payment = {
            'user_id': user_id,
            'payment_id': payment_id,
            'currency': pay_currency,
            'amount_received': pay_amount,
            'amount_expected': price_amount,
            'status': 'partial',
            'created_at': datetime.now().isoformat()
        }
        
        # Store in database (implement storage)
        # await db_service.store_partial_payment(partial_payment)
        
        # Notify user about partial payment
        await send_partial_payment_notification(user_id, pay_amount, pay_currency, price_amount)
        
    except Exception as e:
        logger.error(f"Error processing partial payment: {e}")


async def process_failed_payment(webhook_data: Dict, user_id: str):
    """Process failed cryptocurrency payment"""
    try:
        payment_id = webhook_data['payment_id']
        pay_currency = webhook_data.get('pay_currency', '').upper()
        
        logger.info(f"Payment failed: {payment_id} for user {user_id}")
        
        # Store failed payment record
        failed_payment = {
            'user_id': user_id,
            'payment_id': payment_id,
            'currency': pay_currency,
            'status': 'failed',
            'reason': webhook_data.get('failure_reason', 'Unknown'),
            'created_at': datetime.now().isoformat()
        }
        
        # Store in database
        # await db_service.store_failed_payment(failed_payment)
        
        # Notify user about failed payment
        await send_payment_failure_notification(user_id, payment_id, pay_currency)
        
    except Exception as e:
        logger.error(f"Error processing failed payment: {e}")


async def process_refunded_payment(webhook_data: Dict, user_id: str):
    """Process refunded cryptocurrency payment"""
    try:
        payment_id = webhook_data['payment_id']
        pay_amount = float(webhook_data.get('pay_amount', 0))
        pay_currency = webhook_data.get('pay_currency', '').upper()
        
        logger.info(f"Payment refunded: {pay_amount} {pay_currency} for user {user_id}")
        
        # Store refund record
        refund_record = {
            'user_id': user_id,
            'payment_id': payment_id,
            'currency': pay_currency,
            'amount': pay_amount,
            'status': 'refunded',
            'created_at': datetime.now().isoformat()
        }
        
        # Store in database
        # await db_service.store_refund_record(refund_record)
        
        # Notify user about refund
        await send_refund_notification(user_id, pay_amount, pay_currency)
        
    except Exception as e:
        logger.error(f"Error processing refunded payment: {e}")


# Notification functions (implement with your notification service)
async def send_deposit_notification(user_id: str, amount: float, currency: str, tx_hash: str):
    """Send deposit confirmation notification to user"""
    logger.info(f"Sending deposit notification to user {user_id}: {amount} {currency}")
    # Implement notification logic (email, push notification, etc.)


async def send_partial_payment_notification(user_id: str, received: float, currency: str, expected: float):
    """Send partial payment notification to user"""
    logger.info(f"Sending partial payment notification to user {user_id}: {received}/{expected} {currency}")
    # Implement notification logic


async def send_payment_failure_notification(user_id: str, payment_id: str, currency: str):
    """Send payment failure notification to user"""
    logger.info(f"Sending payment failure notification to user {user_id}: {payment_id}")
    # Implement notification logic


async def send_refund_notification(user_id: str, amount: float, currency: str):
    """Send refund notification to user"""
    logger.info(f"Sending refund notification to user {user_id}: {amount} {currency}")
    # Implement notification logic


@router.get("/webhook/test")
async def test_webhook():
    """Test webhook endpoint"""
    return {
        "status": "success",
        "message": "Crypto webhook endpoint is working",
        "timestamp": datetime.now().isoformat(),
        "nowpayments_configured": bool(os.getenv('NOWPAYMENTS_API_KEY')),
        "webhook_secret_configured": bool(os.getenv('NOWPAYMENTS_WEBHOOK_SECRET'))
    }


@router.post("/webhook/test-payment")
async def test_payment_webhook(test_data: Dict[str, Any]):
    """Test payment webhook with sample data"""
    try:
        logger.info(f"Test webhook data: {test_data}")
        
        # Process test webhook
        await process_nowpayments_webhook(test_data)
        
        return {
            "status": "success",
            "message": "Test webhook processed successfully",
            "data": test_data
        }
        
    except Exception as e:
        logger.error(f"Test webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/payment/{payment_id}/status")
async def get_payment_status(payment_id: str):
    """Get real-time payment status from NOWPayments"""
    try:
        status = nowpayments.get_payment_status(payment_id)
        
        return {
            "payment_id": payment_id,
            "status": status,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting payment status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/balance")
async def get_crypto_balance():
    """Get current cryptocurrency balances"""
    try:
        balance = nowpayments.get_balance()
        
        return {
            "balance": balance,
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting crypto balance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/currencies")
async def get_supported_currencies():
    """Get list of supported cryptocurrencies"""
    try:
        currencies = nowpayments.get_available_currencies()
        currency_details = nowpayments.get_supported_currencies_with_details()
        
        return {
            "currencies": currencies,
            "currency_details": currency_details,
            "total_supported": len(currencies),
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting supported currencies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rates")
async def get_crypto_rates():
    """Get real-time cryptocurrency exchange rates"""
    try:
        rates = crypto_infra.get_real_time_prices()
        
        return {
            "rates": rates,
            "base_currency": "USD",
            "retrieved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting crypto rates: {e}")
        raise HTTPException(status_code=500, detail=str(e))
