#!/usr/bin/env python3
"""
Test script for Maxel webhook server
"""

import requests
import json
import os
from datetime import datetime

# Configuration
WEBHOOK_URL = "http://localhost:8080/webhook"
HEALTH_URL = "http://localhost:8080/health"
MAXEL_SECRET = os.environ.get("MAXEL_SECRET", "default_secret")

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(HEALTH_URL)
        print(f"Health check status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_webhook_unauthorized():
    """Test webhook with wrong secret"""
    print("\nTesting unauthorized access...")
    
    payload = {
        "event_type": "payment_received",
        "payment_id": "test_123",
        "amount": 100.00,
        "currency": "USD"
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-Maxel-Secret": "wrong_secret"
    }
    
    try:
        response = requests.post(WEBHOOK_URL, json=payload, headers=headers)
        print(f"Unauthorized test status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 401
    except Exception as e:
        print(f"Unauthorized test failed: {e}")
        return False

def test_webhook_payment_received():
    """Test payment received webhook"""
    print("\nTesting payment received webhook...")
    
    payload = {
        "event_type": "payment_received",
        "payment_id": "pay_123456789",
        "amount": 250.75,
        "currency": "USD",
        "user_id": "user_987654321",
        "timestamp": datetime.now().isoformat()
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-Maxel-Secret": MAXEL_SECRET
    }
    
    try:
        response = requests.post(WEBHOOK_URL, json=payload, headers=headers)
        print(f"Payment received test status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Payment received test failed: {e}")
        return False

def test_webhook_payment_failed():
    """Test payment failed webhook"""
    print("\nTesting payment failed webhook...")
    
    payload = {
        "event_type": "payment_failed",
        "payment_id": "pay_failed_123",
        "amount": 100.00,
        "currency": "USD",
        "reason": "Insufficient funds",
        "user_id": "user_987654321",
        "timestamp": datetime.now().isoformat()
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-Maxel-Secret": MAXEL_SECRET
    }
    
    try:
        response = requests.post(WEBHOOK_URL, json=payload, headers=headers)
        print(f"Payment failed test status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Payment failed test failed: {e}")
        return False

def test_webhook_withdrawal_completed():
    """Test withdrawal completed webhook"""
    print("\nTesting withdrawal completed webhook...")
    
    payload = {
        "event_type": "withdrawal_completed",
        "withdrawal_id": "withdraw_789123",
        "amount": 500.00,
        "currency": "USD",
        "user_id": "user_987654321",
        "destination": "bank_account_123",
        "timestamp": datetime.now().isoformat()
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-Maxel-Secret": MAXEL_SECRET
    }
    
    try:
        response = requests.post(WEBHOOK_URL, json=payload, headers=headers)
        print(f"Withdrawal completed test status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Withdrawal completed test failed: {e}")
        return False

def test_webhook_unknown_event():
    """Test unknown event type"""
    print("\nTesting unknown event type...")
    
    payload = {
        "event_type": "unknown_event",
        "data": "some random data",
        "timestamp": datetime.now().isoformat()
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-Maxel-Secret": MAXEL_SECRET
    }
    
    try:
        response = requests.post(WEBHOOK_URL, json=payload, headers=headers)
        print(f"Unknown event test status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Unknown event test failed: {e}")
        return False

def run_all_tests():
    """Run all webhook tests"""
    print("=" * 50)
    print("MAXEL WEBHOOK SERVER TESTS")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Unauthorized Access", test_webhook_unauthorized),
        ("Payment Received", test_webhook_payment_received),
        ("Payment Failed", test_webhook_payment_failed),
        ("Withdrawal Completed", test_webhook_withdrawal_completed),
        ("Unknown Event", test_webhook_unknown_event)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed. Check the server logs.")

if __name__ == "__main__":
    print("Make sure the webhook server is running on localhost:8080")
    print("Start it with: python maxel_webhook_server.py")
    print("\nPress Enter to continue with tests...")
    input()
    
    run_all_tests()