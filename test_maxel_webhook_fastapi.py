#!/usr/bin/env python3
"""
Test script for FastAPI Maxel webhook server
"""

import asyncio
import aiohttp
import json
import os
from datetime import datetime
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8080"
MAXEL_SECRET = os.environ.get("MAXEL_SECRET", "default_secret")

class WebhookTester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health_check(self) -> bool:
        """Test the health check endpoint"""
        print("Testing health check...")
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                data = await response.json()
                print(f"Health check status: {response.status}")
                print(f"Response: {data}")
                return response.status == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    async def test_api_docs(self) -> bool:
        """Test API documentation endpoint"""
        print("\nTesting API docs...")
        try:
            async with self.session.get(f"{self.base_url}/docs") as response:
                print(f"API docs status: {response.status}")
                return response.status == 200
        except Exception as e:
            print(f"API docs test failed: {e}")
            return False
    
    async def test_webhook_unauthorized(self) -> bool:
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
            async with self.session.post(
                f"{self.base_url}/webhook",
                json=payload,
                headers=headers
            ) as response:
                data = await response.json()
                print(f"Unauthorized test status: {response.status}")
                print(f"Response: {data}")
                return response.status == 401
        except Exception as e:
            print(f"Unauthorized test failed: {e}")
            return False
    
    async def test_webhook_payment_received(self) -> bool:
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
            async with self.session.post(
                f"{self.base_url}/webhook",
                json=payload,
                headers=headers
            ) as response:
                data = await response.json()
                print(f"Payment received test status: {response.status}")
                print(f"Response: {data}")
                return response.status == 200
        except Exception as e:
            print(f"Payment received test failed: {e}")
            return False
    
    async def test_webhook_payment_failed(self) -> bool:
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
            async with self.session.post(
                f"{self.base_url}/webhook",
                json=payload,
                headers=headers
            ) as response:
                data = await response.json()
                print(f"Payment failed test status: {response.status}")
                print(f"Response: {data}")
                return response.status == 200
        except Exception as e:
            print(f"Payment failed test failed: {e}")
            return False
    
    async def test_webhook_withdrawal_completed(self) -> bool:
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
            async with self.session.post(
                f"{self.base_url}/webhook",
                json=payload,
                headers=headers
            ) as response:
                data = await response.json()
                print(f"Withdrawal completed test status: {response.status}")
                print(f"Response: {data}")
                return response.status == 200
        except Exception as e:
            print(f"Withdrawal completed test failed: {e}")
            return False
    
    async def test_webhook_invalid_payload(self) -> bool:
        """Test webhook with invalid payload"""
        print("\nTesting invalid payload...")
        
        payload = {
            "invalid_field": "invalid_value"
        }
        
        headers = {
            "Content-Type": "application/json",
            "X-Maxel-Secret": MAXEL_SECRET
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/webhook",
                json=payload,
                headers=headers
            ) as response:
                data = await response.json()
                print(f"Invalid payload test status: {response.status}")
                print(f"Response: {data}")
                return response.status == 422  # Validation error
        except Exception as e:
            print(f"Invalid payload test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all webhook tests"""
        print("=" * 60)
        print("MAXEL FASTAPI WEBHOOK SERVER TESTS")
        print("=" * 60)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("API Documentation", self.test_api_docs),
            ("Unauthorized Access", self.test_webhook_unauthorized),
            ("Payment Received", self.test_webhook_payment_received),
            ("Payment Failed", self.test_webhook_payment_failed),
            ("Withdrawal Completed", self.test_webhook_withdrawal_completed),
            ("Invalid Payload", self.test_webhook_invalid_payload)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = await test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"Test {test_name} crashed: {e}")
                results.append((test_name, False))
        
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        
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

async def main():
    """Main test function"""
    print("Make sure the FastAPI webhook server is running on localhost:8080")
    print("Start it with: python maxel_webhook_fastapi.py")
    print("\nPress Enter to continue with tests...")
    input()
    
    async with WebhookTester() as tester:
        await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())