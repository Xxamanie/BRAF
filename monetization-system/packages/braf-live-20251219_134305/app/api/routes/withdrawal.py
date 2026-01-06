from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
from payments.mobile_money import MobileMoneyWithdrawal
from payments.crypto_withdrawal import CryptoWithdrawal
from payments.enhanced_crypto_withdrawal import EnhancedCryptoWithdrawal
from payments.currency_converter import currency_converter
from payments.ton_integration import ton_client
from security.authentication import SecurityManager
from database.service import DatabaseService
import uuid

router = APIRouter(prefix="/api/v1/withdrawal")

class WithdrawalRequest(BaseModel):
    enterprise_id: str
    amount: float
    provider: str  # opay, palmpay, crypto
    recipient: str  # phone number or wallet address
    currency: str = "USD"
    network: Optional[str] = None  # For crypto
    crypto_type: Optional[str] = None  # USDT, BTC, ETH

class OPayWithdrawalRequest(BaseModel):
    enterprise_id: str
    amount: float
    phone_number: str
    country: str = "NG"
    pin_code: str  # For 2FA

class CryptoWithdrawalRequest(BaseModel):
    enterprise_id: str
    amount: float
    cryptocurrency: str  # USDT, BTC, ETH
    network: str  # TRC20, ERC20, etc.
    wallet_address: str

class TONWithdrawalRequest(BaseModel):
    enterprise_id: str
    amount: float  # Amount in USD
    ton_address: str  # TON wallet address
    memo: Optional[str] = ""  # Optional memo

@router.post("/create/{enterprise_id}")
async def create_withdrawal(enterprise_id: str, request: dict):
    """Create a withdrawal request with proper currency conversion"""
    try:
        with DatabaseService() as db:
            # Check available balance (always in USD)
            dashboard_data = db.get_dashboard_data(enterprise_id)
            available_balance = dashboard_data["available_balance"]
            
            usd_amount = request["amount"]
            provider = request["provider"]
            recipient = request["recipient"]
            
            if usd_amount > available_balance:
                raise HTTPException(status_code=400, detail="Insufficient balance")
            
            if usd_amount < 10:
                raise HTTPException(status_code=400, detail="Minimum withdrawal amount is $10 USD")
            
            # Handle TON withdrawals differently
            if provider == "ton":
                # Validate TON address
                if not ton_client.validate_ton_address(recipient):
                    raise HTTPException(status_code=400, detail="Invalid TON wallet address")
                
                # Get TON conversion
                conversion_result = ton_client.convert_usd_to_ton(usd_amount)
                if not conversion_result['success']:
                    raise HTTPException(status_code=400, detail="TON conversion failed")
                
                withdrawal_calc = {
                    "is_valid": True,
                    "converted_amount": conversion_result['amount_ton'],
                    "provider_currency": "TON",
                    "exchange_rate": conversion_result['conversion_rate'],
                    "fee_amount": 0.01 * conversion_result['conversion_rate'],  # Network fee in USD
                    "net_amount": conversion_result['amount_ton'] - 0.01  # Net TON amount
                }
            else:
                # Calculate withdrawal amounts with currency conversion for other providers
                withdrawal_calc = currency_converter.calculate_withdrawal_amounts(usd_amount, provider)
                
                if not withdrawal_calc["is_valid"]:
                    min_amount = withdrawal_calc["minimum_amount"]
                    currency = withdrawal_calc["provider_currency"]
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Minimum withdrawal amount is {min_amount} {currency} for {provider}"
                    )
            
            # Create withdrawal record
            withdrawal_data = {
                "enterprise_id": enterprise_id,
                "transaction_id": f"WD_{uuid.uuid4().hex[:8].upper()}",
                "amount": usd_amount,  # Store original USD amount
                "fee": withdrawal_calc["fee_amount"],
                "net_amount": withdrawal_calc["net_amount"],
                "currency": withdrawal_calc["provider_currency"],  # Store provider currency
                "provider": provider,
                "recipient": recipient,
                "status": "pending",
                "estimated_completion": datetime.utcnow() + timedelta(hours=24)
            }
            
            withdrawal = db.create_withdrawal(withdrawal_data)
            
            return {
                "success": True,
                "transaction_id": withdrawal.transaction_id,
                "status": withdrawal.status,
                "original_amount_usd": usd_amount,
                "converted_amount": withdrawal_calc["converted_amount"],
                "currency": withdrawal_calc["provider_currency"],
                "fee": withdrawal_calc["fee_amount"],
                "net_amount": withdrawal_calc["net_amount"],
                "exchange_rate": withdrawal_calc["exchange_rate"],
                "estimated_completion": withdrawal.estimated_completion.isoformat() if withdrawal.estimated_completion else None,
                "message": f"Withdrawal request submitted successfully. You will receive {withdrawal_calc['net_amount']} {withdrawal_calc['provider_currency']}"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/request")
async def request_withdrawal(request: WithdrawalRequest):
    """Request a withdrawal"""
    try:
        with DatabaseService() as db:
            # Check available balance
            dashboard_data = db.get_dashboard_data(request.enterprise_id)
            available_balance = dashboard_data["available_balance"]
            
            if request.amount > available_balance:
                raise HTTPException(status_code=400, detail="Insufficient balance")
            
            if request.amount < 10:
                raise HTTPException(status_code=400, detail="Minimum withdrawal amount is $10")
            
            # Calculate fees
            fee_rate = 0.01 if request.provider == "crypto" else 0.02  # 1% for crypto, 2% for mobile
            fee = request.amount * fee_rate
            net_amount = request.amount - fee
            
            # Create withdrawal record
            withdrawal_data = {
                "enterprise_id": request.enterprise_id,
                "transaction_id": f"WD_{uuid.uuid4().hex[:8].upper()}",
                "amount": request.amount,
                "fee": fee,
                "net_amount": net_amount,
                "currency": request.currency,
                "provider": request.provider,
                "recipient": request.recipient,
                "status": "pending",
                "network": request.network,
                "estimated_completion": datetime.utcnow() + timedelta(hours=24)
            }
            
            withdrawal = db.create_withdrawal(withdrawal_data)
            
            return {
                "success": True,
                "transaction_id": withdrawal.transaction_id,
                "status": withdrawal.status,
                "amount": withdrawal.amount,
                "fee": withdrawal.fee,
                "net_amount": withdrawal.net_amount,
                "estimated_completion": withdrawal.estimated_completion.isoformat() if withdrawal.estimated_completion else None,
                "message": "Withdrawal request submitted successfully"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/withdraw/opay")
async def withdraw_opay(request: OPayWithdrawalRequest):
    """API endpoint for OPay withdrawals"""
    # TODO: Implement 2FA verification
    # auth = SecurityManager()
    # if not auth.verify_2fa(request.enterprise_id, request.pin_code):
    #     raise HTTPException(status_code=401, detail="Invalid 2FA code")

    # TODO: Check withdrawal limits
    # daily_limit = limits.get_daily_withdrawal_limit(request.enterprise_id)
    # if request.amount > daily_limit["remaining"]:
    #     raise HTTPException(
    #         status_code=400, 
    #         detail=f"Daily limit exceeded. Remaining: ${daily_limit['remaining']}"
    #     )

    # Process withdrawal
    try:
        mobile_money = MobileMoneyWithdrawal()
        result = await mobile_money.withdraw_opay(
            amount=request.amount,
            phone_number=request.phone_number,
            country=request.country,
            enterprise_id=request.enterprise_id
        )
        
        # TODO: Update limits
        # limits.update_withdrawal(
        #     enterprise_id=request.enterprise_id,
        #     amount=request.amount,
        #     provider="opay"
        # )
        
        return {
            "success": True,
            "data": result,
            "message": "Withdrawal initiated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/withdraw/palmpay")
async def withdraw_palmpay(request: OPayWithdrawalRequest):
    """API endpoint for PalmPay withdrawals"""
    return {"message": "PalmPay withdrawal endpoint - implementation pending"}

@router.post("/withdraw/crypto")
async def withdraw_crypto(request: CryptoWithdrawalRequest):
    """API endpoint for cryptocurrency withdrawals"""
    # TODO: Verify wallet is whitelisted
    # security = SecurityManager()
    # if not security.is_whitelisted(request.enterprise_id, request.wallet_address):
    #     raise HTTPException(status_code=400, detail="Wallet not whitelisted")

    # Check minimum amount
    if request.amount < 100:
        raise HTTPException(status_code=400, detail="Minimum crypto withdrawal is $100")

    # Process crypto withdrawal
    try:
        crypto = CryptoWithdrawal()
        result = await crypto.process_withdrawal(
            enterprise_id=request.enterprise_id,
            amount=request.amount,
            cryptocurrency=request.cryptocurrency,
            network=request.network,
            wallet_address=request.wallet_address
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Crypto withdrawal initiated"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/withdraw/ton")
async def withdraw_ton(request: TONWithdrawalRequest):
    """API endpoint for TON cryptocurrency withdrawals"""
    try:
        # Validate TON address
        if not ton_client.validate_ton_address(request.ton_address):
            raise HTTPException(status_code=400, detail="Invalid TON wallet address")
        
        # Check minimum amount
        if request.amount < 10:
            raise HTTPException(status_code=400, detail="Minimum TON withdrawal is $10 USD")
        
        # Check available balance
        with DatabaseService() as db:
            dashboard_data = db.get_dashboard_data(request.enterprise_id)
            available_balance = dashboard_data["available_balance"]
            
            if request.amount > available_balance:
                raise HTTPException(status_code=400, detail="Insufficient balance")
        
        # Process TON withdrawal
        result = ton_client.process_withdrawal_to_ton(
            amount_usd=request.amount,
            ton_address=request.ton_address,
            reference=f"Enterprise_{request.enterprise_id}_{request.memo}"
        )
        
        if result['success']:
            # Create withdrawal record in database
            withdrawal_data = {
                "enterprise_id": request.enterprise_id,
                "transaction_id": result['withdrawal_id'],
                "amount": request.amount,
                "fee": result.get('network_fee_ton', 0.01) * result.get('ton_price_usd', 2.45),  # Convert TON fee to USD
                "net_amount": request.amount - (result.get('network_fee_ton', 0.01) * result.get('ton_price_usd', 2.45)),
                "currency": "TON",
                "provider": "ton",
                "recipient": request.ton_address,
                "status": result.get('status', 'pending'),
                "estimated_completion": datetime.utcnow() + timedelta(minutes=30)  # TON is fast
            }
            
            with DatabaseService() as db:
                withdrawal = db.create_withdrawal(withdrawal_data)
            
            return {
                "success": True,
                "transaction_id": result['withdrawal_id'],
                "transaction_hash": result['transaction_hash'],
                "amount_usd": result['amount_usd'],
                "amount_ton": result['amount_ton'],
                "ton_price_usd": result['ton_price_usd'],
                "to_address": result['to_address'],
                "network_fee_ton": result.get('network_fee_ton', 0.01),
                "status": result.get('status', 'pending'),
                "demo_mode": result.get('demo_mode', False),
                "estimated_completion": withdrawal_data['estimated_completion'].isoformat(),
                "message": f"TON withdrawal initiated. You will receive {result['amount_ton']:.6f} TON"
            }
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'TON withdrawal failed'))
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced Withdrawal System
enhanced_crypto = EnhancedCryptoWithdrawal()

class EnhancedWithdrawalRequest(BaseModel):
    enterprise_id: int
    amount: float
    method: str  # btc, eth, usdt, usdc, bnb, ada, xmr, zcash, dash, ton, trx, ltc, sol, opay, palmpay
    network: Optional[str] = None  # For multi-network cryptos
    recipient: str  # Wallet address or phone number
    memo: Optional[str] = ""
    account_name: Optional[str] = ""

@router.get("/supported-cryptos")
async def get_supported_cryptocurrencies():
    """Get list of supported cryptocurrencies with details"""
    try:
        cryptos = enhanced_crypto.get_supported_cryptocurrencies()
        return {
            "success": True,
            "cryptocurrencies": cryptos,
            "total_supported": len(cryptos)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate-address")
async def validate_crypto_address(crypto: str, address: str, network: str = None):
    """Validate cryptocurrency address format"""
    try:
        validation = enhanced_crypto.validate_address(crypto, address, network)
        return {
            "success": True,
            "validation": validation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/calculate-fee")
async def calculate_withdrawal_fee(crypto: str, network: str = None, amount_usd: float = 0):
    """Calculate withdrawal fees for cryptocurrency"""
    try:
        fee_calculation = enhanced_crypto.calculate_withdrawal_fee(crypto, network, amount_usd)
        
        if 'error' in fee_calculation:
            raise HTTPException(status_code=400, detail=fee_calculation['error'])
        
        return {
            "success": True,
            "fee_calculation": fee_calculation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/enhanced-request")
async def create_enhanced_withdrawal(request: EnhancedWithdrawalRequest):
    """Create enhanced withdrawal request with multi-crypto support"""
    try:
        with DatabaseService() as db:
            # Check available balance
            dashboard_data = db.get_dashboard_data(str(request.enterprise_id))
            available_balance = dashboard_data.get("available_balance", 0)
            
            if request.amount > available_balance:
                raise HTTPException(status_code=400, detail="Insufficient balance")
            
            if request.amount < 10:
                raise HTTPException(status_code=400, detail="Minimum withdrawal amount is $10")
            
            # Process withdrawal based on method
            if request.method in ['opay', 'palmpay']:
                # Mobile money withdrawal
                mobile_processor = MobileMoneyWithdrawal()
                
                # Convert USD to NGN
                usd_to_ngn = await currency_converter.get_rate("USD", "NGN")
                ngn_amount = request.amount * usd_to_ngn
                
                withdrawal_data = {
                    "enterprise_id": str(request.enterprise_id),
                    "amount": ngn_amount,
                    "provider": request.method,
                    "recipient": request.recipient,
                    "currency": "NGN",
                    "original_amount_usd": request.amount,
                    "exchange_rate": usd_to_ngn,
                    "account_name": request.account_name
                }
                
                result = mobile_processor.process_withdrawal(withdrawal_data)
                
            else:
                # Cryptocurrency withdrawal
                withdrawal_data = {
                    "enterprise_id": request.enterprise_id,
                    "amount": request.amount,
                    "method": request.method,
                    "network": request.network,
                    "recipient": request.recipient,
                    "memo": request.memo
                }
                
                result = enhanced_crypto.process_withdrawal(withdrawal_data)
            
            if result.get('success'):
                # Record withdrawal in database
                withdrawal_id = db.create_withdrawal(
                    enterprise_id=str(request.enterprise_id),
                    amount=request.amount,
                    provider=request.method,
                    recipient=request.recipient,
                    status="processing",
                    transaction_id=result.get('transaction_id'),
                    network=request.network,
                    currency=result.get('crypto', 'USD')
                )
                
                return {
                    "success": True,
                    "withdrawal_id": withdrawal_id,
                    "transaction_id": result.get('transaction_id'),
                    "status": result.get('status', 'processing'),
                    "method": request.method,
                    "network": request.network,
                    "amount_usd": request.amount,
                    "amount_crypto": result.get('amount_crypto'),
                    "fee_usd": result.get('fee_usd'),
                    "fee_crypto": result.get('fee_crypto'),
                    "net_amount": result.get('amount_crypto', request.amount),
                    "estimated_completion": result.get('estimated_completion'),
                    "blockchain_tx": result.get('blockchain_tx'),
                    "message": f"Withdrawal request submitted successfully to {request.method.upper()}"
                }
            else:
                raise HTTPException(status_code=400, detail=result.get('error', 'Withdrawal failed'))
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@router.get("/status/{transaction_id}")
async def get_withdrawal_status(transaction_id: str):
    """Get withdrawal status and confirmation progress"""
    try:
        # Check if it's a crypto withdrawal
        if any(crypto in transaction_id.upper() for crypto in ['BTC', 'ETH', 'USDT', 'USDC', 'BNB', 'ADA', 'XMR', 'ZEC', 'DASH', 'TON', 'TRX', 'LTC', 'SOL']):
            status = enhanced_crypto.get_withdrawal_status(transaction_id)
        else:
            # Mobile money or other withdrawal
            with DatabaseService() as db:
                withdrawal = db.get_withdrawal_by_transaction_id(transaction_id)
                if not withdrawal:
                    raise HTTPException(status_code=404, detail="Withdrawal not found")
                
                status = {
                    'transaction_id': transaction_id,
                    'status': withdrawal.status,
                    'progress': 100 if withdrawal.status == 'completed' else 50,
                    'estimated_completion': 'Within 24 hours'
                }
        
        return {
            "success": True,
            "status": status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{enterprise_id}")
async def get_withdrawal_history(enterprise_id: int, limit: int = 50):
    """Get withdrawal history for enterprise"""
    try:
        with DatabaseService() as db:
            withdrawals = db.get_withdrawals(str(enterprise_id), limit)
            
            withdrawal_list = []
            for withdrawal in withdrawals:
                withdrawal_list.append({
                    "id": withdrawal.id,
                    "transaction_id": withdrawal.transaction_id,
                    "amount": float(withdrawal.amount),
                    "fee": float(withdrawal.fee) if withdrawal.fee else 0,
                    "net_amount": float(withdrawal.net_amount) if withdrawal.net_amount else float(withdrawal.amount),
                    "provider": withdrawal.provider,
                    "recipient": withdrawal.recipient,
                    "status": withdrawal.status,
                    "currency": withdrawal.currency,
                    "network": withdrawal.network,
                    "created_at": withdrawal.created_at.isoformat(),
                    "completed_at": withdrawal.completed_at.isoformat() if withdrawal.completed_at else None
                })
            
            return {
                "success": True,
                "withdrawals": withdrawal_list,
                "total": len(withdrawal_list)
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/methods")
async def get_withdrawal_methods():
    """Get available withdrawal methods with details"""
    try:
        # Get crypto methods
        cryptos = enhanced_crypto.get_supported_cryptocurrencies()
        
        # Mobile money methods
        mobile_methods = {
            'opay': {
                'name': 'OPay',
                'type': 'mobile_money',
                'currency': 'NGN',
                'fee_percentage': 1.5,
                'min_amount_usd': 10,
                'max_amount_usd': 10000,
                'countries': ['Nigeria'],
                'processing_time': '24-48 hours'
            },
            'palmpay': {
                'name': 'PalmPay',
                'type': 'mobile_money',
                'currency': 'NGN',
                'fee_percentage': 1.5,
                'min_amount_usd': 10,
                'max_amount_usd': 10000,
                'countries': ['Nigeria'],
                'processing_time': '24-48 hours'
            }
        }
        
        # Combine all methods
        all_methods = {}
        
        # Add crypto methods
        for crypto_id, crypto_info in cryptos.items():
            all_methods[crypto_id] = {
                **crypto_info,
                'type': 'cryptocurrency',
                'processing_time': '1-6 hours'
            }
        
        # Add mobile methods
        all_methods.update(mobile_methods)
        
        return {
            "success": True,
            "methods": all_methods,
            "categories": {
                "cryptocurrencies": len(cryptos),
                "mobile_money": len(mobile_methods),
                "total": len(all_methods)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
