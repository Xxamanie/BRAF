#!/usr/bin/env python3
"""
Get Your maxelpay Wallet Deposit Address
Use this to deposit real cryptocurrency into your maxelpay wallet
"""
import requests
import json
from datetime import datetime

# maxelpay API Configuration
MAXELPAY_API_KEY = "pk_Eq8N27HLVFDrPFd34j7a7cpIJd6PncsWMAXELPAY_SECRET_KEY"
MAXELPAY_SECRET_KEY = "sk_rI7pJyhIyaiU5js1BCpjYA53y5iS7Ny0"

def get_MAXELPAY_deposit_address(currency='USDT'):
    """
    Get your maxelpay wallet deposit address for a specific cryptocurrency
    """
    print(f"🔑 Getting maxelpay Deposit Address for {currency}")
    print("=" * 60)
    
    headers = {
        'Authorization': f'Bearer {MAXELPAY_API_KEY}',
        'X-Secret-Key': MAXELPAY_SECRET_KEY,
        'Content-Type': 'application/json'
    }
    
    # Try to get deposit address from maxelpay API
    try:
        # Method 1: Create new address
        url = "https://maxelpay.com/api/v1/addresses"
        data = {
            'currency': currency,
            'user_id': 'braf_main_user'
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if 'address' in result:
                print(f"✅ Deposit Address Generated!")
                print(f"\n📍 Your {currency} Deposit Address:")
                print(f"   {result['address']}")
                
                if 'memo' in result and result['memo']:
                    print(f"\n📝 Memo/Tag (IMPORTANT):")
                    print(f"   {result['memo']}")
                
                print(f"\n💡 How to Deposit:")
                print(f"   1. Copy the address above")
                print(f"   2. Go to your crypto exchange (Coinbase, Binance, etc.)")
                print(f"   3. Select 'Send' or 'Withdraw' {currency}")
                print(f"   4. Paste the address")
                print(f"   5. Enter amount to send")
                print(f"   6. Confirm transaction")
                
                print(f"\n⏰ Confirmation Time:")
                if currency == 'BTC':
                    print(f"   Bitcoin: ~10-60 minutes")
                elif currency == 'ETH':
                    print(f"   Ethereum: ~5-15 minutes")
                elif currency in ['USDT', 'USDC']:
                    print(f"   {currency}: ~1-5 minutes (TRC20) or ~5-15 minutes (ERC20)")
                
                return result['address']
        
        # Method 2: Get existing addresses
        url = "https://maxelpay.com/api/v1/account/addresses"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if 'addresses' in result:
                for addr in result['addresses']:
                    if addr.get('currency') == currency:
                        print(f"✅ Existing Deposit Address Found!")
                        print(f"\n📍 Your {currency} Deposit Address:")
                        print(f"   {addr['address']}")
                        return addr['address']
        
    except Exception as e:
        print(f"❌ API Error: {e}")
    
    # Fallback: Manual instructions
    print(f"\n📱 Manual Method:")
    print(f"=" * 60)
    print(f"1. Login to maxelpay: https://maxelpay.com/login")
    print(f"2. Go to 'Wallet' or 'Deposit' section")
    print(f"3. Select {currency}")
    print(f"4. Click 'Deposit' or 'Receive'")
    print(f"5. Copy your deposit address")
    print(f"6. Use that address to receive {currency}")
    
    return None

def show_deposit_instructions():
    """Show detailed deposit instructions"""
    print(f"\n💰 How to Get Real Money into maxelpay Wallet")
    print("=" * 60)
    
    print(f"\n📋 Step-by-Step Guide:")
    print(f"\n1️⃣  Get Cryptocurrency")
    print(f"   Option A: Buy on Exchange")
    print(f"      - Coinbase: https://coinbase.com")
    print(f"      - Binance: https://binance.com")
    print(f"      - Kraken: https://kraken.com")
    print(f"   Option B: Receive Payment")
    print(f"      - Get paid in crypto for work/services")
    print(f"   Option C: Transfer from Another Wallet")
    print(f"      - Move crypto from existing wallet")
    
    print(f"\n2️⃣  Get Your maxelpay Deposit Address")
    print(f"   - Login to maxelpay")
    print(f"   - Go to Wallet → Deposit")
    print(f"   - Select cryptocurrency (USDT recommended)")
    print(f"   - Copy deposit address")
    
    print(f"\n3️⃣  Send Cryptocurrency")
    print(f"   - Go to your exchange/wallet")
    print(f"   - Select 'Send' or 'Withdraw'")
    print(f"   - Paste maxelpay deposit address")
    print(f"   - Enter amount")
    print(f"   - Confirm transaction")
    
    print(f"\n4️⃣  Wait for Confirmation")
    print(f"   - Transaction processes on blockchain")
    print(f"   - Usually 1-60 minutes depending on crypto")
    print(f"   - Check maxelpay wallet for balance update")
    
    print(f"\n✅ Once Confirmed:")
    print(f"   - Money appears in maxelpay wallet")
    print(f"   - You can use it for withdrawals")
    print(f"   - You can send to other addresses")
    print(f"   - You can convert to other cryptocurrencies")

def check_MAXELPAY_balance():
    """Check current maxelpay wallet balance"""
    print(f"\n💼 Checking maxelpay Wallet Balance")
    print("=" * 60)
    
    headers = {
        'Authorization': f'Bearer {MAXELPAY_API_KEY}',
        'X-Secret-Key': MAXELPAY_SECRET_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        url = "https://maxelpay.com/api/v1/account/balance"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            if 'balances' in result:
                print(f"✅ Current Balances:")
                total_usd = 0
                
                for currency, amount in result['balances'].items():
                    if float(amount) > 0:
                        print(f"   {currency}: {amount}")
                        # Rough USD conversion (would need real rates)
                        total_usd += float(amount)
                
                print(f"\n💰 Approximate Total: ${total_usd:.2f} USD")
                
                if total_usd == 0:
                    print(f"\n⚠️  Wallet is empty - no deposits yet")
                    print(f"   Follow deposit instructions above to add funds")
            else:
                print(f"⚠️  No balance data available")
        else:
            print(f"❌ Could not retrieve balance (HTTP {response.status_code})")
            
    except Exception as e:
        print(f"❌ Error checking balance: {e}")
    
    print(f"\n💡 To check balance manually:")
    print(f"   Login to: https://maxelpay.com/dashboard")
    print(f"   View your wallet balances there")

def main():
    """Main function"""
    print(f"🚀 maxelpay Wallet Deposit Tool")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check current balance
    check_MAXELPAY_balance()
    
    # Get deposit addresses for popular cryptocurrencies
    print(f"\n" + "=" * 60)
    currencies = ['USDT', 'BTC', 'ETH', 'USDC']
    
    for currency in currencies:
        print(f"\n")
        get_MAXELPAY_deposit_address(currency)
        print(f"\n" + "-" * 60)
    
    # Show detailed instructions
    show_deposit_instructions()
    
    print(f"\n" + "=" * 60)
    print(f"🎯 Summary:")
    print(f"   1. BRAF earnings are currently simulated (not real money)")
    print(f"   2. To get real money in maxelpay, you need to deposit crypto")
    print(f"   3. Buy crypto on exchange → Send to maxelpay address → Money appears")
    print(f"   4. Or earn crypto from real work → Send to maxelpay address")
    print(f"\n💡 maxelpay Dashboard: https://maxelpay.com/dashboard")

if __name__ == "__main__":
    main()
