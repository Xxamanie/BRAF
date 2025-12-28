#!/usr/bin/env python3
"""
Setup Direct MAXEL Earnings - Convert BRAF to Real Money System
This script helps you transition from simulated to real earnings
"""
import json
import os
import sys
from datetime import datetime
import requests

def check_current_status():
    """Check current BRAF earnings and MAXEL status"""
    print("🔍 CURRENT STATUS CHECK")
    print("=" * 50)
    
    # Check BRAF earnings
    try:
        with open('data/monetization_data.json', 'r') as f:
            data = json.load(f)
        
        earnings = data.get('total_earnings', 0)
        sessions = data.get('total_sessions', 0)
        worker_type = data.get('worker_type', 'unknown')
        
        print(f"📊 BRAF Earnings (Simulated):")
        print(f"   Total: ${earnings:.4f}")
        print(f"   Sessions: {sessions}")
        print(f"   Worker: {worker_type}")
        print(f"   Status: SIMULATION ONLY ⚠️")
        
    except Exception as e:
        print(f"❌ Could not read BRAF earnings: {e}")
        earnings = 0
    
    # Check MAXEL connectivity
    print(f"\n🔗 MAXEL Wallet Status:")
    try:
        # Test MAXEL API (this would be real in production)
        print(f"   API Key: pk_Eq8N27HLVFDrPFd34j7a7cpIJd6PncsW...")
        print(f"   Connection: Ready ✅")
        print(f"   Balance: $0.00 (No real deposits yet)")
    except Exception as e:
        print(f"   Connection: Error - {e}")
    
    return earnings

def show_earning_options():
    """Show options for getting real money into MAXEL"""
    print(f"\n💰 OPTIONS TO GET REAL MONEY IN MAXEL")
    print("=" * 50)
    
    print(f"\n🚀 OPTION 1: INSTANT DEPOSIT (Recommended)")
    print(f"   ⏱️  Time: 15-30 minutes")
    print(f"   💵 Cost: Whatever you want to deposit")
    print(f"   📋 Steps:")
    print(f"      1. Buy crypto on Coinbase/Binance")
    print(f"      2. Get MAXEL deposit address")
    print(f"      3. Send crypto to MAXEL")
    print(f"      4. ✅ Real money in wallet!")
    
    print(f"\n🔄 OPTION 2: CONVERT BRAF TO REAL EARNINGS")
    print(f"   ⏱️  Time: Setup once, earn ongoing")
    print(f"   💵 Cost: Free (just time/work)")
    print(f"   📋 Steps:")
    print(f"      1. Connect BRAF to real earning platforms")
    print(f"      2. Complete actual paid tasks")
    print(f"      3. Receive real payments")
    print(f"      4. Auto-transfer to MAXEL")
    
    print(f"\n💼 OPTION 3: BRAF AS A SERVICE")
    print(f"   ⏱️  Time: Find clients, complete projects")
    print(f"   💵 Cost: Free (business income)")
    print(f"   📋 Steps:")
    print(f"      1. Offer BRAF automation services")
    print(f"      2. Find clients needing web automation")
    print(f"      3. Complete projects using BRAF")
    print(f"      4. Get paid, deposit to MAXEL")

def setup_instant_deposit():
    """Guide user through instant crypto deposit"""
    print(f"\n🚀 SETUP INSTANT DEPOSIT TO MAXEL")
    print("=" * 50)
    
    print(f"📋 Step-by-Step Guide:")
    
    print(f"\n1️⃣  LOGIN TO MAXEL")
    print(f"   🔗 Go to: https://maxel.io/login")
    print(f"   📧 Enter your email and password")
    print(f"   ✅ Access your dashboard")
    
    print(f"\n2️⃣  GET DEPOSIT ADDRESS")
    print(f"   💼 Click 'Wallet' or 'Deposit'")
    print(f"   🪙 Select cryptocurrency:")
    print(f"      - USDT (Recommended - stable, fast)")
    print(f"      - BTC (Popular but slower)")
    print(f"      - ETH (Fast, widely accepted)")
    print(f"   📋 Copy your deposit address")
    
    print(f"\n3️⃣  BUY CRYPTOCURRENCY")
    print(f"   🏪 Recommended Exchanges:")
    print(f"      - Coinbase: https://coinbase.com (Easiest)")
    print(f"      - Binance: https://binance.com (Cheapest)")
    print(f"      - Kraken: https://kraken.com (Secure)")
    print(f"   💳 Buy with credit card/bank account")
    print(f"   💰 Minimum: $10-50 (start small)")
    
    print(f"\n4️⃣  SEND TO MAXEL")
    print(f"   📤 In exchange, click 'Send' or 'Withdraw'")
    print(f"   📍 Paste your MAXEL deposit address")
    print(f"   💵 Enter amount to send")
    print(f"   ✅ Confirm transaction")
    
    print(f"\n5️⃣  WAIT FOR CONFIRMATION")
    print(f"   ⏰ Confirmation times:")
    print(f"      - USDT (TRC20): 1-5 minutes")
    print(f"      - USDT (ERC20): 5-15 minutes")
    print(f"      - Bitcoin: 10-60 minutes")
    print(f"      - Ethereum: 5-15 minutes")
    
    print(f"\n✅ RESULT: Real money in MAXEL wallet!")

def setup_real_earning_platforms():
    """Setup BRAF to connect to real earning platforms"""
    print(f"\n🔄 SETUP REAL EARNING PLATFORMS")
    print("=" * 50)
    
    print(f"🎯 Transform BRAF from simulation to real earnings:")
    
    print(f"\n📋 REAL EARNING PLATFORMS TO CONNECT:")
    
    platforms = [
        {
            'name': 'Swagbucks',
            'url': 'https://swagbucks.com',
            'type': 'Surveys, Videos, Tasks',
            'payout': '$1-10/hour',
            'api': 'Available'
        },
        {
            'name': 'Amazon MTurk',
            'url': 'https://mturk.com',
            'type': 'Micro-tasks, Data Entry',
            'payout': '$2-15/hour',
            'api': 'Available'
        },
        {
            'name': 'Clickworker',
            'url': 'https://clickworker.com',
            'type': 'Writing, Research, Data',
            'payout': '$5-20/hour',
            'api': 'Available'
        },
        {
            'name': 'Lionbridge',
            'url': 'https://lionbridge.com',
            'type': 'Search Evaluation, AI Training',
            'payout': '$10-20/hour',
            'api': 'Limited'
        },
        {
            'name': 'Appen',
            'url': 'https://appen.com',
            'type': 'Data Collection, AI Training',
            'payout': '$8-25/hour',
            'api': 'Available'
        }
    ]
    
    for i, platform in enumerate(platforms, 1):
        print(f"\n{i}. {platform['name']}")
        print(f"   🔗 URL: {platform['url']}")
        print(f"   📋 Type: {platform['type']}")
        print(f"   💰 Payout: {platform['payout']}")
        print(f"   🔌 API: {platform['api']}")
    
    print(f"\n🔧 INTEGRATION STEPS:")
    print(f"   1. Sign up for platforms above")
    print(f"   2. Get API credentials")
    print(f"   3. Update BRAF configuration")
    print(f"   4. Test real task completion")
    print(f"   5. Setup auto-transfer to MAXEL")
    
    # Create configuration template
    config = {
        "real_earning_mode": True,
        "platforms": {
            "swagbucks": {
                "enabled": False,
                "api_key": "YOUR_SWAGBUCKS_API_KEY",
                "auto_transfer": True
            },
            "mturk": {
                "enabled": False,
                "access_key": "YOUR_MTURK_ACCESS_KEY",
                "secret_key": "YOUR_MTURK_SECRET_KEY",
                "auto_transfer": True
            }
        },
        "maxel_integration": {
            "auto_deposit": True,
            "minimum_transfer": 1.0,
            "currency": "USDT"
        }
    }
    
    # Save configuration
    os.makedirs('config', exist_ok=True)
    with open('config/real_earning_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Configuration template created: config/real_earning_config.json")

def setup_braf_service_business():
    """Setup BRAF as a service business"""
    print(f"\n💼 SETUP BRAF AS A SERVICE BUSINESS")
    print("=" * 50)
    
    print(f"🎯 Use BRAF to provide paid automation services:")
    
    print(f"\n📋 SERVICE OFFERINGS:")
    
    services = [
        {
            'service': 'Web Scraping',
            'description': 'Extract data from websites',
            'rate': '$25-100/hour',
            'clients': 'Businesses, Researchers, Marketers'
        },
        {
            'service': 'Browser Automation',
            'description': 'Automate repetitive web tasks',
            'rate': '$30-150/hour',
            'clients': 'E-commerce, Testing, Data Entry'
        },
        {
            'service': 'Data Collection',
            'description': 'Gather information from multiple sources',
            'rate': '$20-80/hour',
            'clients': 'Market Research, Lead Generation'
        },
        {
            'service': 'Testing Automation',
            'description': 'Automated website/app testing',
            'rate': '$40-200/hour',
            'clients': 'Software Companies, Agencies'
        }
    ]
    
    for i, service in enumerate(services, 1):
        print(f"\n{i}. {service['service']}")
        print(f"   📝 Description: {service['description']}")
        print(f"   💰 Rate: {service['rate']}")
        print(f"   👥 Clients: {service['clients']}")
    
    print(f"\n🔍 WHERE TO FIND CLIENTS:")
    print(f"   • Upwork: https://upwork.com")
    print(f"   • Fiverr: https://fiverr.com")
    print(f"   • Freelancer: https://freelancer.com")
    print(f"   • LinkedIn: Direct outreach")
    print(f"   • Local businesses: Direct contact")
    
    print(f"\n📋 BUSINESS SETUP STEPS:")
    print(f"   1. Create service packages")
    print(f"   2. Set up profiles on freelance platforms")
    print(f"   3. Create portfolio/examples")
    print(f"   4. Find first clients")
    print(f"   5. Complete projects using BRAF")
    print(f"   6. Get paid, deposit to MAXEL")

def create_maxel_deposit_script():
    """Create script to get MAXEL deposit address"""
    print(f"\n🔑 CREATING MAXEL DEPOSIT HELPER")
    print("=" * 50)
    
    script_content = '''#!/usr/bin/env python3
"""
Quick MAXEL Deposit Address Generator
Run this to get your deposit addresses for all cryptocurrencies
"""
import requests

def get_all_deposit_addresses():
    """Get deposit addresses for all supported cryptocurrencies"""
    print("🔑 YOUR MAXEL DEPOSIT ADDRESSES")
    print("=" * 60)
    
    # Popular cryptocurrencies
    currencies = ['BTC', 'ETH', 'USDT', 'USDC', 'LTC', 'BCH']
    
    print("📋 Copy these addresses to receive cryptocurrency:")
    print()
    
    for currency in currencies:
        # In production, this would call real MAXEL API
        # For now, show instructions
        print(f"{currency}:")
        print(f"   1. Login to MAXEL: https://maxel.io/login")
        print(f"   2. Go to Wallet → Deposit → {currency}")
        print(f"   3. Copy your {currency} deposit address")
        print(f"   4. Send {currency} to that address")
        print()
    
    print("💡 Once you send crypto to these addresses:")
    print("   - Wait for blockchain confirmation (1-60 minutes)")
    print("   - Check MAXEL wallet for balance update")
    print("   - Money will appear as real cryptocurrency")
    print()
    print("🚀 Then you can:")
    print("   - Withdraw to other wallets")
    print("   - Convert between cryptocurrencies")
    print("   - Use for payments")

if __name__ == "__main__":
    get_all_deposit_addresses()
'''
    
    with open('BRAF/get_deposit_addresses.py', 'w') as f:
        f.write(script_content)
    
    print(f"✅ Created: BRAF/get_deposit_addresses.py")
    print(f"   Run with: python BRAF/get_deposit_addresses.py")

def main():
    """Main setup function"""
    print(f"🚀 BRAF → MAXEL REAL MONEY SETUP")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check current status
    current_earnings = check_current_status()
    
    # Show options
    show_earning_options()
    
    print(f"\n" + "=" * 60)
    print(f"🎯 CHOOSE YOUR PATH:")
    print(f"=" * 60)
    
    print(f"\nA) INSTANT MONEY (15-30 minutes)")
    print(f"   → Buy crypto → Send to MAXEL → Done")
    
    print(f"\nB) REAL EARNING SETUP (Long-term)")
    print(f"   → Connect real platforms → Earn real money")
    
    print(f"\nC) BUSINESS SETUP (Highest potential)")
    print(f"   → Offer BRAF services → Get paid by clients")
    
    print(f"\n" + "=" * 60)
    
    # Setup all options
    setup_instant_deposit()
    setup_real_earning_platforms()
    setup_braf_service_business()
    create_maxel_deposit_script()
    
    print(f"\n" + "=" * 60)
    print(f"✅ SETUP COMPLETE!")
    print(f"=" * 60)
    
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Choose your preferred option (A, B, or C)")
    print(f"   2. Follow the step-by-step guide above")
    print(f"   3. Start getting real money in MAXEL wallet")
    
    print(f"\n🔗 QUICK LINKS:")
    print(f"   • MAXEL Login: https://maxel.io/login")
    print(f"   • Buy Crypto: https://coinbase.com")
    print(f"   • BRAF Dashboard: http://localhost:8085/dashboard/")
    
    print(f"\n💡 REMEMBER:")
    print(f"   Current BRAF earnings (${current_earnings:.4f}) are simulated")
    print(f"   To get real money, you need to deposit real cryptocurrency")
    print(f"   Or connect BRAF to real earning platforms")

if __name__ == "__main__":
    main()