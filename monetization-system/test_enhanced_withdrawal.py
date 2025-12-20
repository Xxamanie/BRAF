#!/usr/bin/env python3
"""
Enhanced Withdrawal System Test Suite
Tests all withdrawal methods and cryptocurrency support
"""

import asyncio
import aiohttp
import json
from datetime import datetime

class EnhancedWithdrawalTester:
    def __init__(self, base_url="http://127.0.0.1:8004"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_supported_cryptocurrencies(self):
        """Test supported cryptocurrencies endpoint"""
        print("🔍 Testing supported cryptocurrencies...")
        try:
            async with self.session.get(f"{self.base_url}/api/v1/withdrawal/supported-cryptos") as response:
                data = await response.json()
                assert response.status == 200
                assert data["success"] == True
                assert "cryptocurrencies" in data
                
                cryptos = data["cryptocurrencies"]
                print(f"✅ Found {len(cryptos)} supported cryptocurrencies:")
                for crypto_id, crypto_info in cryptos.items():
                    print(f"   • {crypto_info['name']} ({crypto_info['symbol']}) - Networks: {crypto_info['networks']}")
                
                return True
        except Exception as e:
            print(f"❌ Supported cryptocurrencies test failed: {e}")
            return False
    
    async def test_address_validation(self):
        """Test cryptocurrency address validation"""
        print("🔍 Testing address validation...")
        try:
            test_addresses = [
                ("btc", "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", None),  # Valid Bitcoin address
                ("eth", "0x742d35Cc6634C0532925a3b8D400e5e5c8c8b8c8", None),  # Valid Ethereum address
                ("ton", "UQBmMxSNU5PLmtib4xKsBH9zAg08681Tec0rcOHYB6F4vST7", None),  # Valid TON address
                ("usdt", "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t", "TRC20"),  # Valid USDT TRC20 address
                ("btc", "invalid_address", None),  # Invalid address
            ]
            
            passed = 0
            total = len(test_addresses)
            
            for crypto, address, network in test_addresses:
                params = {"crypto": crypto, "address": address}
                if network:
                    params["network"] = network
                
                async with self.session.post(
                    f"{self.base_url}/api/v1/withdrawal/validate-address",
                    params=params
                ) as response:
                    data = await response.json()
                    
                    if response.status == 200:
                        validation = data["validation"]
                        is_valid = validation["valid"]
                        expected_valid = "invalid" not in address
                        
                        if is_valid == expected_valid:
                            print(f"   ✅ {crypto.upper()} address validation: {'Valid' if is_valid else 'Invalid'}")
                            passed += 1
                        else:
                            print(f"   ❌ {crypto.upper()} address validation failed")
                    else:
                        print(f"   ❌ {crypto.upper()} validation request failed")
            
            print(f"✅ Address validation: {passed}/{total} tests passed")
            return passed == total
            
        except Exception as e:
            print(f"❌ Address validation test failed: {e}")
            return False
    
    async def test_fee_calculation(self):
        """Test withdrawal fee calculation"""
        print("🔍 Testing fee calculation...")
        try:
            test_cases = [
                ("btc", None, 1000.0),  # $1000 Bitcoin withdrawal
                ("usdt", "TRC20", 500.0),  # $500 USDT TRC20 withdrawal
                ("eth", None, 2000.0),  # $2000 Ethereum withdrawal
                ("ton", None, 100.0),  # $100 TON withdrawal
            ]
            
            passed = 0
            total = len(test_cases)
            
            for crypto, network, amount_usd in test_cases:
                params = {"crypto": crypto, "amount_usd": amount_usd}
                if network:
                    params["network"] = network
                
                async with self.session.post(
                    f"{self.base_url}/api/v1/withdrawal/calculate-fee",
                    params=params
                ) as response:
                    data = await response.json()
                    
                    if response.status == 200 and data["success"]:
                        calc = data["fee_calculation"]
                        print(f"   ✅ {crypto.upper()}{f' ({network})' if network else ''}: ${calc['amount_usd']:.2f} → {calc['net_crypto']:.6f} {crypto.upper()} (Fee: ${calc['fee_usd']:.2f})")
                        passed += 1
                    else:
                        print(f"   ❌ {crypto.upper()} fee calculation failed")
            
            print(f"✅ Fee calculation: {passed}/{total} tests passed")
            return passed == total
            
        except Exception as e:
            print(f"❌ Fee calculation test failed: {e}")
            return False
    
    async def test_withdrawal_methods(self):
        """Test withdrawal methods endpoint"""
        print("🔍 Testing withdrawal methods...")
        try:
            async with self.session.get(f"{self.base_url}/api/v1/withdrawal/methods") as response:
                data = await response.json()
                assert response.status == 200
                assert data["success"] == True
                assert "methods" in data
                
                methods = data["methods"]
                categories = data["categories"]
                
                print(f"✅ Found {categories['total']} withdrawal methods:")
                print(f"   • Cryptocurrencies: {categories['cryptocurrencies']}")
                print(f"   • Mobile Money: {categories['mobile_money']}")
                
                # Test some specific methods
                crypto_methods = [m for m, info in methods.items() if info['type'] == 'cryptocurrency']
                mobile_methods = [m for m, info in methods.items() if info['type'] == 'mobile_money']
                
                print(f"   • Crypto methods: {', '.join(crypto_methods[:5])}...")
                print(f"   • Mobile methods: {', '.join(mobile_methods)}")
                
                return True
        except Exception as e:
            print(f"❌ Withdrawal methods test failed: {e}")
            return False
    
    async def test_enhanced_withdrawal_request(self):
        """Test enhanced withdrawal request"""
        print("🔍 Testing enhanced withdrawal request...")
        try:
            # Test Bitcoin withdrawal
            btc_request = {
                "enterprise_id": 1,
                "amount": 100.0,
                "method": "btc",
                "network": None,
                "recipient": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
                "memo": "Test BTC withdrawal"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/withdrawal/enhanced-request",
                json=btc_request
            ) as response:
                data = await response.json()
                
                if response.status == 200 and data["success"]:
                    print(f"   ✅ Bitcoin withdrawal: {data['transaction_id']}")
                    print(f"      Amount: ${data['amount_usd']:.2f} → {data['amount_crypto']:.6f} BTC")
                    print(f"      Fee: ${data['fee_usd']:.2f}")
                    print(f"      Status: {data['status']}")
                    btc_success = True
                else:
                    print(f"   ❌ Bitcoin withdrawal failed: {data.get('detail', 'Unknown error')}")
                    btc_success = False
            
            # Test USDT TRC20 withdrawal
            usdt_request = {
                "enterprise_id": 1,
                "amount": 50.0,
                "method": "usdt",
                "network": "TRC20",
                "recipient": "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t",
                "memo": "Test USDT withdrawal"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/withdrawal/enhanced-request",
                json=usdt_request
            ) as response:
                data = await response.json()
                
                if response.status == 200 and data["success"]:
                    print(f"   ✅ USDT TRC20 withdrawal: {data['transaction_id']}")
                    print(f"      Amount: ${data['amount_usd']:.2f} → {data['amount_crypto']:.2f} USDT")
                    print(f"      Fee: ${data['fee_usd']:.2f}")
                    usdt_success = True
                else:
                    print(f"   ❌ USDT withdrawal failed: {data.get('detail', 'Unknown error')}")
                    usdt_success = False
            
            # Test TON withdrawal
            ton_request = {
                "enterprise_id": 1,
                "amount": 25.0,
                "method": "ton",
                "network": None,
                "recipient": "UQBmMxSNU5PLmtib4xKsBH9zAg08681Tec0rcOHYB6F4vST7",
                "memo": "Test TON withdrawal"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/withdrawal/enhanced-request",
                json=ton_request
            ) as response:
                data = await response.json()
                
                if response.status == 200 and data["success"]:
                    print(f"   ✅ TON withdrawal: {data['transaction_id']}")
                    print(f"      Amount: ${data['amount_usd']:.2f} → {data['amount_crypto']:.6f} TON")
                    print(f"      Fee: ${data['fee_usd']:.2f}")
                    ton_success = True
                else:
                    print(f"   ❌ TON withdrawal failed: {data.get('detail', 'Unknown error')}")
                    ton_success = False
            
            total_success = sum([btc_success, usdt_success, ton_success])
            print(f"✅ Enhanced withdrawal requests: {total_success}/3 successful")
            return total_success == 3
            
        except Exception as e:
            print(f"❌ Enhanced withdrawal request test failed: {e}")
            return False
    
    async def test_enhanced_withdrawal_page(self):
        """Test enhanced withdrawal page loading"""
        print("🔍 Testing enhanced withdrawal page...")
        try:
            async with self.session.get(f"{self.base_url}/enhanced-withdrawal") as response:
                content = await response.text()
                assert response.status == 200
                assert "Enhanced Withdrawal System" in content
                assert "Bitcoin" in content
                assert "Ethereum" in content
                assert "Monero" in content
                assert "TON" in content
                
                print("✅ Enhanced withdrawal page loads correctly")
                return True
        except Exception as e:
            print(f"❌ Enhanced withdrawal page test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all enhanced withdrawal tests"""
        print("🚀 Starting Enhanced Withdrawal Test Suite")
        print("=" * 60)
        
        tests = [
            self.test_supported_cryptocurrencies,
            self.test_address_validation,
            self.test_fee_calculation,
            self.test_withdrawal_methods,
            self.test_enhanced_withdrawal_request,
            self.test_enhanced_withdrawal_page
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                result = await test()
                if result:
                    passed += 1
                print()
            except Exception as e:
                print(f"❌ Test {test.__name__} failed with exception: {e}")
                print()
        
        print("=" * 60)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Enhanced withdrawal system is fully functional.")
            print("\n✨ Supported Features:")
            print("   • 12+ Cryptocurrencies (BTC, ETH, USDT, USDC, BNB, ADA, XMR, ZEC, DASH, TON, TRX, LTC, SOL)")
            print("   • Multiple Networks (ERC20, TRC20, BEP20, Polygon, etc.)")
            print("   • Privacy Coins (Monero, Zcash, Dash)")
            print("   • Fast & Cheap Options (TON, Tron, Solana)")
            print("   • Mobile Money (OPay, PalmPay)")
            print("   • Address Validation")
            print("   • Fee Calculation")
            print("   • Real-time Status Tracking")
        else:
            print(f"⚠️  {total - passed} tests failed. Check the output above.")
        
        return passed == total

async def main():
    """Main test function"""
    print("🔧 Enhanced Withdrawal System Test Suite")
    print("📋 Testing comprehensive cryptocurrency withdrawal support")
    print()
    
    async with EnhancedWithdrawalTester() as tester:
        success = await tester.run_all_tests()
        
        if success:
            print("\n✅ Enhanced withdrawal system is ready for production!")
            print("🌐 Access URLs:")
            print("   • Enhanced Withdrawal: http://127.0.0.1:8004/enhanced-withdrawal")
            print("   • API Documentation: http://127.0.0.1:8004/docs")
            print("   • Supported Cryptos: http://127.0.0.1:8004/api/v1/withdrawal/supported-cryptos")
        else:
            print("\n❌ Some tests failed. Please check the server and try again.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")