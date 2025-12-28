#!/usr/bin/env python3
"""
BRAF-MAXEL Dashboard - Simple Status and Transfer Interface
Shows current BRAF earnings and provides MAXEL integration
"""
import json
import webbrowser
from datetime import datetime
import os

def get_braf_status():
    """Get current BRAF earnings status"""
    try:
        with open('BRAF/data/monetization_data.json', 'r') as f:
            data = json.load(f)
        return data
    except:
        return {
            "total_earnings": 0,
            "total_sessions": 0,
            "worker_type": "not_running",
            "runtime_seconds": 0,
            "hourly_rate": 0
        }

def show_dashboard():
    """Display BRAF-MAXEL dashboard"""
    braf_data = get_braf_status()
    
    earnings = braf_data.get("total_earnings", 0)
    sessions = braf_data.get("total_sessions", 0)
    worker_type = braf_data.get("worker_type", "unknown")
    runtime = braf_data.get("runtime_seconds", 0)
    hourly_rate = braf_data.get("hourly_rate", 0)
    
    # Convert to NGN (approximate)
    usd_to_ngn = 1457.58
    earnings_ngn = earnings * usd_to_ngn
    
    print("🤖 BRAF-MAXEL INTEGRATION DASHBOARD")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("💰 CURRENT EARNINGS STATUS")
    print("-" * 40)
    print(f"💵 USD Earnings: ${earnings:.4f}")
    print(f"💰 NGN Equivalent: ₦{earnings_ngn:,.2f}")
    print(f"📊 Sessions Completed: {sessions}")
    print(f"🤖 Worker Type: {worker_type}")
    print(f"⏱️  Runtime: {runtime/60:.1f} minutes")
    print(f"📈 Hourly Rate: ${hourly_rate:.4f}/hour")
    
    print(f"\n🔗 SYSTEM STATUS")
    print("-" * 40)
    
    # Check if simple manager is running
    if worker_type == "simple_manager" and earnings > 0:
        print(f"✅ BRAF Worker: Running & Earning")
    else:
        print(f"⚠️  BRAF Worker: Not active")
    
    # Check live system
    try:
        import requests
        response = requests.get("http://127.0.0.1:8003/health", timeout=2)
        if response.status_code == 200:
            print(f"✅ Live System: Running (Port 8003)")
        else:
            print(f"⚠️  Live System: Error")
    except:
        print(f"❌ Live System: Not running")
    
    print(f"🔗 MAXEL Wallet: Ready for deposits")
    
    print(f"\n💡 EARNING OPTIONS")
    print("-" * 40)
    
    if earnings < 0.01:
        print(f"🚀 START EARNING:")
        print(f"   Run: npm run simple-manager")
        print(f"   Or:  npm run simple-worker")
    else:
        print(f"💰 CURRENT EARNINGS: ${earnings:.4f}")
        print(f"   Continue running workers for more")
    
    print(f"\n🏦 MAXEL INTEGRATION")
    print("-" * 40)
    print(f"💡 To get real money in MAXEL wallet:")
    print(f"   1. Login: https://maxel.io/login")
    print(f"   2. Get deposit address")
    print(f"   3. Buy crypto (Coinbase/Binance)")
    print(f"   4. Send to MAXEL address")
    print(f"   5. Real money appears!")
    
    print(f"\n📊 EARNINGS BREAKDOWN")
    print("-" * 40)
    print(f"Current BRAF earnings: ${earnings:.4f} (simulated)")
    print(f"MAXEL wallet balance: $0.00 (real)")
    print(f"To bridge the gap: Deposit real crypto")
    
    return braf_data

def show_quick_actions():
    """Show quick action options"""
    print(f"\n🚀 QUICK ACTIONS")
    print("=" * 60)
    
    print(f"\n1️⃣  START/CONTINUE EARNING")
    print(f"   npm run simple-manager    # Continuous earning")
    print(f"   npm run simple-worker     # Single test run")
    
    print(f"\n2️⃣  ACCESS LIVE SYSTEM")
    print(f"   Dashboard: http://127.0.0.1:8003/dashboard")
    print(f"   API Docs:  http://127.0.0.1:8003/docs")
    print(f"   Health:    http://127.0.0.1:8003/health")
    
    print(f"\n3️⃣  MAXEL WALLET")
    print(f"   Login:     https://maxel.io/login")
    print(f"   Deposit:   Get address → Buy crypto → Send")
    print(f"   Withdraw:  Send crypto to external wallets")
    
    print(f"\n4️⃣  BUY CRYPTO (FASTEST)")
    print(f"   Coinbase:  https://coinbase.com")
    print(f"   Binance:   https://binance.com")
    print(f"   Kraken:    https://kraken.com")
    
    print(f"\n5️⃣  MONITOR EARNINGS")
    print(f"   python braf_maxel_dashboard.py")
    print(f"   python track_maxel_deposits.py")

def open_links():
    """Open important links in browser"""
    print(f"\n🌐 OPENING IMPORTANT LINKS...")
    
    links = [
        ("MAXEL Login", "https://maxel.io/login"),
        ("Live System Dashboard", "http://127.0.0.1:8003/dashboard"),
        ("Coinbase (Buy Crypto)", "https://coinbase.com"),
    ]
    
    for name, url in links:
        try:
            print(f"   Opening: {name}")
            webbrowser.open(url)
        except:
            print(f"   Manual: {url}")

def main():
    """Main dashboard function"""
    # Show dashboard
    braf_data = show_dashboard()
    
    # Show quick actions
    show_quick_actions()
    
    # Ask user what they want to do
    print(f"\n" + "=" * 60)
    print(f"🎯 WHAT WOULD YOU LIKE TO DO?")
    print(f"=" * 60)
    
    print(f"\nA) Open MAXEL wallet (get real money)")
    print(f"B) Open live system dashboard")
    print(f"C) Open Coinbase (buy crypto)")
    print(f"D) Show current status only")
    print(f"E) Exit")
    
    try:
        choice = input(f"\nEnter choice (A/B/C/D/E): ").upper().strip()
        
        if choice == 'A':
            print(f"🔗 Opening MAXEL wallet...")
            webbrowser.open("https://maxel.io/login")
        elif choice == 'B':
            print(f"🔗 Opening live system dashboard...")
            webbrowser.open("http://127.0.0.1:8003/dashboard")
        elif choice == 'C':
            print(f"🔗 Opening Coinbase...")
            webbrowser.open("https://coinbase.com")
        elif choice == 'D':
            print(f"✅ Status displayed above")
        elif choice == 'E':
            print(f"👋 Goodbye!")
        else:
            print(f"📋 All links available above")
            
    except KeyboardInterrupt:
        print(f"\n👋 Goodbye!")
    except:
        print(f"\n📋 All important links shown above")
    
    print(f"\n💡 REMEMBER:")
    print(f"   BRAF earnings (${braf_data.get('total_earnings', 0):.4f}) are simulated")
    print(f"   To get real money: Deposit crypto to MAXEL wallet")
    print(f"   Keep BRAF running: npm run simple-manager")

if __name__ == "__main__":
    main()