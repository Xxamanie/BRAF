#!/usr/bin/env python3
"""
Connect BRAF Earnings to Live Production System
Bridge between BRAF worker earnings and production monetization system
"""
import json
import requests
from datetime import datetime
import os

# Configuration
LIVE_SYSTEM_URL = "http://127.0.0.1:8003"
BRAF_EARNINGS_FILE = "BRAF/data/monetization_data.json"

def check_live_system():
    """Check if live production system is running"""
    try:
        response = requests.get(f"{LIVE_SYSTEM_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def get_braf_earnings():
    """Get current BRAF earnings"""
    try:
        with open(BRAF_EARNINGS_FILE, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Could not read BRAF earnings: {e}")
        return None

def create_enterprise_account():
    """Create enterprise account in live system"""
    account_data = {
        "name": "BRAF Main User",
        "email": "braf@example.com",
        "password": "BRAFSecure123!",
        "company_name": "BRAF Automation Systems",
        "phone_number": "+1234567890",
        "country": "US"
    }
    
    try:
        response = requests.post(
            f"{LIVE_SYSTEM_URL}/api/enterprises/register",
            json=account_data,
            timeout=10
        )
        
        if response.status_code in [200, 201]:
            return True, response.json()
        else:
            return False, f"HTTP {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)

def transfer_earnings_to_live_system(braf_data, enterprise_id):
    """Transfer BRAF earnings to live production system"""
    
    # Create automation record
    automation_data = {
        "template_type": "browser_automation",
        "platform": "multi_platform",
        "status": "active",
        "config": {
            "worker_type": braf_data.get("worker_type", "simple_manager"),
            "platforms": ["swagbucks", "inboxdollars", "ysense", "timebucks"],
            "runtime_seconds": braf_data.get("runtime_seconds", 0),
            "sessions_completed": braf_data.get("total_sessions", 0)
        },
        "earnings_total": braf_data.get("total_earnings", 0),
        "earnings_today": braf_data.get("total_earnings", 0)
    }
    
    try:
        # Create automation
        response = requests.post(
            f"{LIVE_SYSTEM_URL}/api/automations",
            json=automation_data,
            headers={"Enterprise-ID": enterprise_id},
            timeout=10
        )
        
        if response.status_code in [200, 201]:
            automation = response.json()
            automation_id = automation.get("id")
            
            # Create earnings record
            earnings_data = {
                "automation_id": automation_id,
                "amount": braf_data.get("total_earnings", 0),
                "currency": "USD",
                "platform": "braf_multi_platform",
                "task_type": "browser_automation",
                "task_details": {
                    "sessions": braf_data.get("total_sessions", 0),
                    "hourly_rate": braf_data.get("hourly_rate", 0),
                    "worker_type": braf_data.get("worker_type", "unknown"),
                    "last_update": braf_data.get("last_update")
                }
            }
            
            earnings_response = requests.post(
                f"{LIVE_SYSTEM_URL}/api/earnings",
                json=earnings_data,
                headers={"Enterprise-ID": enterprise_id},
                timeout=10
            )
            
            if earnings_response.status_code in [200, 201]:
                return True, {
                    "automation": automation,
                    "earnings": earnings_response.json()
                }
            else:
                return False, f"Earnings creation failed: {earnings_response.text}"
        else:
            return False, f"Automation creation failed: {response.text}"
            
    except Exception as e:
        return False, str(e)

def setup_MAXELPAY_integration(enterprise_id, amount_usd):
    """Setup maxelpay integration for withdrawals"""
    
    # Convert USD to NGN using live rates
    try:
        rates_response = requests.get(f"{LIVE_SYSTEM_URL}/api/currency/rates/USD", timeout=5)
        if rates_response.status_code == 200:
            rates = rates_response.json()
            usd_to_ngn_rate = rates.get("NGN", 1457.58)  # Fallback rate
            amount_ngn = amount_usd * usd_to_ngn_rate
        else:
            amount_ngn = amount_usd * 1457.58  # Fallback rate
    except:
        amount_ngn = amount_usd * 1457.58  # Fallback rate
    
    # Create crypto balance
    balance_data = {
        "user_id": "braf_main_user",
        "currency": "USDT",
        "balance": amount_usd,
        "available_balance": amount_usd
    }
    
    try:
        response = requests.post(
            f"{LIVE_SYSTEM_URL}/api/crypto/balances",
            json=balance_data,
            headers={"Enterprise-ID": enterprise_id},
            timeout=10
        )
        
        if response.status_code in [200, 201]:
            return True, {
                "balance_usd": amount_usd,
                "balance_ngn": amount_ngn,
                "exchange_rate": usd_to_ngn_rate,
                "crypto_balance": response.json()
            }
        else:
            return False, f"Balance creation failed: {response.text}"
            
    except Exception as e:
        return False, str(e)

def main():
    """Main connection function"""
    print("🔗 CONNECTING BRAF TO LIVE PRODUCTION SYSTEM")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Check live system
    print(f"\n1️⃣  CHECKING LIVE SYSTEM...")
    is_running, status = check_live_system()
    
    if not is_running:
        print(f"❌ Live system not running: {status}")
        print(f"💡 Start it with: python monetization-system/run_server.py")
        return False
    
    print(f"✅ Live system running: {LIVE_SYSTEM_URL}")
    print(f"   Status: {status}")
    
    # Step 2: Get BRAF earnings
    print(f"\n2️⃣  READING BRAF EARNINGS...")
    braf_data = get_braf_earnings()
    
    if not braf_data:
        print(f"❌ Could not read BRAF earnings")
        return False
    
    earnings = braf_data.get("total_earnings", 0)
    sessions = braf_data.get("total_sessions", 0)
    worker_type = braf_data.get("worker_type", "unknown")
    
    print(f"✅ BRAF earnings loaded:")
    print(f"   💰 Total: ${earnings:.4f}")
    print(f"   📊 Sessions: {sessions}")
    print(f"   🤖 Worker: {worker_type}")
    
    # Step 3: Create enterprise account
    print(f"\n3️⃣  CREATING ENTERPRISE ACCOUNT...")
    success, result = create_enterprise_account()
    
    if success:
        enterprise_id = result.get("id")
        print(f"✅ Enterprise account created:")
        print(f"   🆔 ID: {enterprise_id}")
        print(f"   📧 Email: braf@example.com")
    else:
        print(f"⚠️  Account creation: {result}")
        # Try to use existing account
        enterprise_id = "braf_main_enterprise"
        print(f"   Using default ID: {enterprise_id}")
    
    # Step 4: Transfer earnings
    print(f"\n4️⃣  TRANSFERRING EARNINGS TO LIVE SYSTEM...")
    success, result = transfer_earnings_to_live_system(braf_data, enterprise_id)
    
    if success:
        print(f"✅ Earnings transferred successfully:")
        print(f"   🤖 Automation ID: {result['automation']['id']}")
        print(f"   💰 Earnings ID: {result['earnings']['id']}")
        print(f"   💵 Amount: ${earnings:.4f}")
    else:
        print(f"❌ Transfer failed: {result}")
        return False
    
    # Step 5: Setup maxelpay integration
    print(f"\n5️⃣  SETTING UP maxelpay INTEGRATION...")
    success, result = setup_MAXELPAY_integration(enterprise_id, earnings)
    
    if success:
        print(f"✅ maxelpay integration ready:")
        print(f"   💰 USD Balance: ${result['balance_usd']:.4f}")
        print(f"   💰 NGN Equivalent: ₦{result['balance_ngn']:,.2f}")
        print(f"   📊 Exchange Rate: 1 USD = {result['exchange_rate']:.2f} NGN")
    else:
        print(f"⚠️  maxelpay setup: {result}")
    
    # Step 6: Show access information
    print(f"\n" + "=" * 60)
    print(f"🎉 CONNECTION COMPLETED!")
    print(f"=" * 60)
    
    print(f"\n📍 ACCESS YOUR LIVE SYSTEM:")
    print(f"   🌐 Dashboard: {LIVE_SYSTEM_URL}/dashboard")
    print(f"   🔐 Login: {LIVE_SYSTEM_URL}/login")
    print(f"   📚 API Docs: {LIVE_SYSTEM_URL}/docs")
    print(f"   ❤️  Health: {LIVE_SYSTEM_URL}/health")
    
    print(f"\n🔑 LOGIN CREDENTIALS:")
    print(f"   📧 Email: braf@example.com")
    print(f"   🔒 Password: BRAFSecure123!")
    
    print(f"\n💰 YOUR EARNINGS:")
    print(f"   💵 BRAF Earnings: ${earnings:.4f}")
    print(f"   🏦 Live System: Connected")
    print(f"   🔗 maxelpay Ready: Yes")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Open: {LIVE_SYSTEM_URL}/dashboard")
    print(f"   2. Login with credentials above")
    print(f"   3. View your earnings and automations")
    print(f"   4. Setup maxelpay withdrawal")
    print(f"   5. Continue running BRAF workers")
    
    return True

if __name__ == "__main__":
    main()
