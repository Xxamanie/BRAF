#!/usr/bin/env python3
"""
Enhanced Dashboard Test Suite
Tests all dashboard functions and API endpoints
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

class DashboardTester:
    def __init__(self, base_url="http://127.0.0.1:8004"):
        self.base_url = base_url
        self.enterprise_id = 1
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health_check(self):
        """Test health check endpoint"""
        print("🔍 Testing health check...")
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                data = await response.json()
                assert response.status == 200
                assert data["status"] == "healthy"
                print("✅ Health check passed")
                return True
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    async def test_dashboard_page(self):
        """Test dashboard page loading"""
        print("🔍 Testing dashboard page...")
        try:
            async with self.session.get(f"{self.base_url}/") as response:
                content = await response.text()
                assert response.status == 200
                assert "BRAF Enhanced Dashboard" in content
                assert "Real-time" in content
                print("✅ Dashboard page loads correctly")
                return True
        except Exception as e:
            print(f"❌ Dashboard page test failed: {e}")
            return False
    
    async def test_realtime_data(self):
        """Test real-time dashboard data endpoint"""
        print("🔍 Testing real-time data endpoint...")
        try:
            async with self.session.get(f"{self.base_url}/api/v1/dashboard/realtime/{self.enterprise_id}") as response:
                data = await response.json()
                assert response.status == 200
                
                # Check required fields
                required_fields = [
                    "total_earnings", "available_balance", "active_operations",
                    "success_rate", "pending_payouts", "system_health"
                ]
                
                for field in required_fields:
                    assert field in data, f"Missing field: {field}"
                
                # Check system health structure
                health = data["system_health"]
                assert "cpu" in health
                assert "memory" in health
                assert "network" in health
                
                print(f"✅ Real-time data: ${data['total_earnings']:.2f} earnings, {data['active_operations']} operations")
                return True
        except Exception as e:
            print(f"❌ Real-time data test failed: {e}")
            return False
    
    async def test_create_automation(self):
        """Test automation creation"""
        print("🔍 Testing automation creation...")
        try:
            payload = {
                "enterprise_id": self.enterprise_id,
                "automation_type": "survey_automation",
                "platform": "swagbucks",
                "parameters": {
                    "max_surveys_per_day": 50,
                    "target_earnings": 100
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/dashboard/automation/create",
                json=payload
            ) as response:
                data = await response.json()
                assert response.status == 200
                assert "automation_id" in data
                assert "estimated_earnings" in data
                
                print(f"✅ Automation created: {data['automation_id']}, estimated earnings: ${data['estimated_earnings']}")
                return True
        except Exception as e:
            print(f"❌ Automation creation test failed: {e}")
            return False
    
    async def test_research_operation(self):
        """Test research operation start"""
        print("🔍 Testing research operation...")
        try:
            payload = {
                "enterprise_id": self.enterprise_id,
                "research_type": "platform_analysis",
                "parameters": {
                    "target_platforms": ["swagbucks", "youtube"],
                    "analysis_depth": "comprehensive"
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/dashboard/research/start",
                json=payload
            ) as response:
                data = await response.json()
                assert response.status == 200
                assert "research_id" in data
                assert "estimated_duration" in data
                
                print(f"✅ Research started: {data['research_id']}, duration: {data['estimated_duration']}s")
                return True
        except Exception as e:
            print(f"❌ Research operation test failed: {e}")
            return False
    
    async def test_system_optimization(self):
        """Test system optimization"""
        print("🔍 Testing system optimization...")
        try:
            payload = {
                "enterprise_id": self.enterprise_id,
                "optimization_type": "full"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/dashboard/intelligence/optimize",
                json=payload
            ) as response:
                data = await response.json()
                assert response.status == 200
                assert "optimization_id" in data
                assert "improvements" in data
                assert "estimated_impact" in data
                
                improvements = data["improvements"]
                impact = data["estimated_impact"]
                
                print(f"✅ Optimization completed: {len(improvements)} improvements")
                print(f"   💰 Earnings increase: ${impact['earnings_increase']}")
                print(f"   ⚡ Efficiency gain: {impact['efficiency_gain']}%")
                return True
        except Exception as e:
            print(f"❌ System optimization test failed: {e}")
            return False
    
    async def test_operation_management(self):
        """Test operation viewing and stopping"""
        print("🔍 Testing operation management...")
        try:
            operation_id = "op_001"
            
            # Test view operation
            async with self.session.get(
                f"{self.base_url}/api/v1/dashboard/operations/{operation_id}?enterprise_id={self.enterprise_id}"
            ) as response:
                data = await response.json()
                assert response.status == 200
                assert "operation_details" in data
                
                print(f"✅ Operation details retrieved: {operation_id}")
            
            # Test stop operation
            payload = {"enterprise_id": self.enterprise_id}
            async with self.session.post(
                f"{self.base_url}/api/v1/dashboard/operations/{operation_id}/stop",
                json=payload
            ) as response:
                data = await response.json()
                assert response.status == 200
                assert "status" in data
                assert data["status"] == "stopped"
                
                print(f"✅ Operation stopped: {operation_id}, earnings: ${data['final_earnings']}")
                return True
        except Exception as e:
            print(f"❌ Operation management test failed: {e}")
            return False
    
    async def test_earnings_data(self):
        """Test earnings data retrieval"""
        print("🔍 Testing earnings data...")
        try:
            async with self.session.get(f"{self.base_url}/api/v1/dashboard/earnings/{self.enterprise_id}") as response:
                data = await response.json()
                assert response.status == 200
                assert "recent_earnings" in data
                
                earnings = data["recent_earnings"]
                print(f"✅ Earnings data retrieved: {len(earnings)} recent earnings")
                return True
        except Exception as e:
            print(f"❌ Earnings data test failed: {e}")
            return False
    
    async def test_analytics_page(self):
        """Test analytics page"""
        print("🔍 Testing analytics page...")
        try:
            async with self.session.get(f"{self.base_url}/analytics") as response:
                content = await response.text()
                assert response.status == 200
                assert "BRAF Advanced Analytics" in content
                assert "Revenue Analytics" in content
                
                print("✅ Analytics page loads correctly")
                return True
        except Exception as e:
            print(f"❌ Analytics page test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all dashboard tests"""
        print("🚀 Starting Enhanced Dashboard Test Suite")
        print("=" * 50)
        
        tests = [
            self.test_health_check,
            self.test_dashboard_page,
            self.test_realtime_data,
            self.test_create_automation,
            self.test_research_operation,
            self.test_system_optimization,
            self.test_operation_management,
            self.test_earnings_data,
            self.test_analytics_page
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
        
        print("=" * 50)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Dashboard is fully functional.")
        else:
            print(f"⚠️  {total - passed} tests failed. Check the output above.")
        
        return passed == total

async def main():
    """Main test function"""
    print("🔧 Enhanced Dashboard Test Suite")
    print("📋 Testing all dashboard functions and API endpoints")
    print()
    
    async with DashboardTester() as tester:
        success = await tester.run_all_tests()
        
        if success:
            print("\n✅ All dashboard functions are working correctly!")
            print("🌐 Dashboard URL: http://127.0.0.1:8004")
            print("📊 Features tested:")
            print("   • Real-time data updates")
            print("   • Automation creation")
            print("   • Research operations")
            print("   • System optimization")
            print("   • Operation management")
            print("   • Analytics and reporting")
        else:
            print("\n❌ Some tests failed. Please check the server and try again.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
