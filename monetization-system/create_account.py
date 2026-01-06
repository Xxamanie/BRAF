#!/usr/bin/env python3
"""
Account Creation Tool for BRAF Monetization System
Simple CLI tool to create enterprise accounts
"""

import sys
import getpass
import requests
import json
from typing import Dict, Any

def create_account_interactive():
    """Interactive account creation"""
    print("=== BRAF Monetization System - Account Creation ===\n")
    
    # Get account details
    name = input("Enter your full name: ").strip()
    email = input("Enter your email address: ").strip()
    
    # Get password securely
    while True:
        password = getpass.getpass("Enter password (min 8 characters): ")
        if len(password) < 8:
            print("Password must be at least 8 characters long")
            continue
        
        confirm_password = getpass.getpass("Confirm password: ")
        if password != confirm_password:
            print("Passwords don't match. Please try again.")
            continue
        break
    
    # Get subscription tier
    print("\n🎉 FREE BETA ACCESS:")
    print("✨ Complete access to all premium features!")
    print("🚀 Unlimited automations, $1000/day limit")
    print("💎 All templates + Priority support")
    print("📈 Full API access + Advanced analytics")
    
    subscription_tier = "free"  # All accounts get free access during beta
    
    # Optional details
    company_name = input("Company name (optional): ").strip() or None
    phone_number = input("Phone number (optional): ").strip() or None
    country = input("Country code (default: US): ").strip() or "US"
    
    return {
        "name": name,
        "email": email,
        "password": password,
        "subscription_tier": subscription_tier,
        "company_name": company_name,
        "phone_number": phone_number,
        "country": country
    }

def create_account_api(account_data: Dict[str, Any], base_url: str = "http://localhost:8002") -> Dict[str, Any]:
    """Create account via API"""
    try:
        response = requests.post(
            f"{base_url}/api/v1/enterprise/register",
            json=account_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False, 
                "error": f"HTTP {response.status_code}: {response.text}"
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Could not connect to BRAF server. Make sure it's running on http://localhost:8000"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def test_login(email: str, password: str, base_url: str = "http://localhost:8002") -> Dict[str, Any]:
    """Test login with created account"""
    try:
        response = requests.post(
            f"{base_url}/api/v1/enterprise/login",
            json={"email": email, "password": password},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {
                "success": False,
                "error": f"Login failed: HTTP {response.status_code}: {response.text}"
            }
            
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python create_account.py [--batch]")
        print("  --batch: Use batch mode with predefined test account")
        print("  (no args): Interactive mode")
        return
    
    # Check if batch mode
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        # Create test account
        account_data = {
            "name": "Test User",
            "email": "test@example.com",
            "password": "testpassword123",
            "subscription_tier": "free",
            "company_name": "Test Company",
            "phone_number": "+1234567890",
            "country": "US"
        }
        print("Creating test account...")
    else:
        # Interactive mode
        account_data = create_account_interactive()
    
    print(f"\nCreating account for {account_data['email']}...")
    
    # Create account
    result = create_account_api(account_data)
    
    if result["success"]:
        data = result["data"]
        print("\n✅ Account created successfully!")
        print(f"Enterprise ID: {data['enterprise_id']}")
        print(f"Subscription: {data['subscription']['tier']} ({data['subscription']['status']})")
        
        if "two_factor_auth" in data:
            print(f"\n🔐 Two-Factor Authentication Setup:")
            print(f"QR Code URL: {data['two_factor_auth']['qr_code_url']}")
            print(f"Secret Key: {data['two_factor_auth']['secret']}")
            print("Please set up 2FA using an authenticator app")
        
        print(f"\n📋 Next Steps:")
        for step in data.get("next_steps", []):
            print(f"  • {step}")
        
        # Test login
        print(f"\n🔑 Testing login...")
        login_result = test_login(account_data["email"], account_data["password"])
        
        if login_result["success"]:
            login_data = login_result["data"]
            print("✅ Login successful!")
            print(f"Session Token: {login_data['session_token']}")
            print(f"Dashboard URL: http://localhost:8002/dashboard")
            
            # Show dashboard data
            if "dashboard" in login_data:
                dashboard = login_data["dashboard"]
                print(f"\n📊 Account Summary:")
                print(f"  Total Earnings: ${dashboard['total_earnings']:.2f}")
                print(f"  Available Balance: ${dashboard['available_balance']:.2f}")
                print(f"  Active Automations: {dashboard['active_automations']}")
        else:
            print(f"❌ Login test failed: {login_result['error']}")
    
    else:
        print(f"\n❌ Account creation failed: {result['error']}")
        
        if "Could not connect" in result['error']:
            print("\n💡 Make sure the BRAF server is running:")
            print("   cd monetization-system")
            print("   python start_system.py")

if __name__ == "__main__":
    main()
