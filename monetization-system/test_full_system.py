#!/usr/bin/env python3
"""
Test the complete BRAF Monetization System
Tests all major endpoints and functionality
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8003"

def test_endpoint(method, endpoint, data=None, expected_status=200):
    """Test an API endpoint"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url)
        elif method.upper() == "POST":
            response = requests.post(url, json=data)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
            
        print(f"{method.upper()} {endpoint} -> {response.status_code}")
        
        if response.status_code == expected_status:
            print(f"✅ Success: {endpoint}")
            if response.headers.get('content-type', '').startswith('application/json'):
                try:
                    result = response.json()
                    if isinstance(result, dict) and len(result) > 0:
                        print(f"   📊 Response keys: {list(result.keys())}")
                    elif isinstance(result, list) and len(result) > 0:
                        print(f"   📊 Response: {len(result)} items")
                except:
                    pass
            return True
        else:
            print(f"❌ Failed: {endpoint} (expected {expected_status}, got {response.status_code})")
            try:
                print(f"   Error: {response.text}")
            except:
                pass
            return False
            
    except Exception as e:
        print(f"❌ Error testing {endpoint}: {e}")
        return False

def main():
    """Test the complete system"""
    print("🧪 Testing BRAF Monetization System")
    print("=" * 50)
    
    # Test basic endpoints
    print("\n📡 Testing Basic Endpoints:")
    test_endpoint("GET", "/")
    test_endpoint("GET", "/health")
    test_endpoint("GET", "/docs", expected_status=200)
    
    # Test UI endpoints
    print("\n🌐 Testing UI Endpoints:")
    test_endpoint("GET", "/register")
    test_endpoint("GET", "/login")
    test_endpoint("GET", "/dashboard")
    
    # Test with existing test account
    print("\n🔐 Testing with Test Account:")
    enterprise_id = "e9e9d28b-62d1-4452-b0df-e1f1cf6e4721"  # From seeded data
    
    # Test API endpoints
    print("\n📊 Testing API Endpoints:")
    test_endpoint("GET", f"/api/v1/automation/list/{enterprise_id}")
    test_endpoint("GET", f"/api/v1/dashboard/earnings/{enterprise_id}")
    test_endpoint("GET", f"/api/v1/dashboard/withdrawals/{enterprise_id}")
    
    # Test automation creation
    print("\n🤖 Testing Automation Creation:")
    automation_data = {
        "template_type": "survey",
        "platform": "swagbucks",
        "config": {
            "platforms": ["swagbucks"],
            "max_surveys_per_session": 3,
            "daily_limit": 25.0
        }
    }
    test_endpoint("POST", f"/api/v1/automation/create/{enterprise_id}", automation_data)
    
    # Test withdrawal creation
    print("\n💸 Testing Withdrawal Creation:")
    withdrawal_data = {
        "amount": 100.0,
        "provider": "opay",
        "recipient": "+234XXXXXXXXXX"
    }
    test_endpoint("POST", f"/api/v1/withdrawal/create/{enterprise_id}", withdrawal_data)
    
    print("\n" + "=" * 50)
    print("🎉 System testing completed!")
    print("🌐 Visit the dashboard: http://127.0.0.1:8003/dashboard")
    print("📚 API Documentation: http://127.0.0.1:8003/docs")

if __name__ == "__main__":
    main()
