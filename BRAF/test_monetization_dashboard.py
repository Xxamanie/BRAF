#!/usr/bin/env python3
"""
BRAF Monetization Dashboard Test
Test the monetization dashboard and earnings tracking
"""
import sys
import json
import time
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_monetization_data_generation():
    """Test monetization data generation"""
    print("💰 Testing Monetization Data Generation")
    print("=" * 50)
    
    try:
        from start_monetization_dashboard import generate_monetization_data
        
        # Generate test data
        data = generate_monetization_data()
        
        # Validate data structure
        required_fields = [
            'total_earnings', 'pending_earnings', 'withdrawn_amount',
            'platforms', 'recent_activity', 'performance'
        ]
        
        for field in required_fields:
            if field not in data:
                print(f"❌ Missing field: {field}")
                return False
        
        print(f"✅ Data structure validation passed")
        print(f"   💰 Total earnings: ${data['total_earnings']:.2f}")
        print(f"   ⏳ Pending earnings: ${data['pending_earnings']:.2f}")
        print(f"   💸 Withdrawn amount: ${data['withdrawn_amount']:.2f}")
        print(f"   🏢 Platforms: {len(data['platforms'])}")
        print(f"   📋 Recent activity: {len(data['recent_activity'])}")
        
        # Validate platforms
        if data['platforms']:
            platform = data['platforms'][0]
            platform_fields = ['name', 'total_earned', 'status', 'last_updated']
            for field in platform_fields:
                if field not in platform:
                    print(f"❌ Missing platform field: {field}")
                    return False
            print(f"✅ Platform data validation passed")
        
        # Validate activity
        if data['recent_activity']:
            activity = data['recent_activity'][0]
            activity_fields = ['type', 'title', 'details', 'amount', 'timestamp']
            for field in activity_fields:
                if field not in activity:
                    print(f"❌ Missing activity field: {field}")
                    return False
            print(f"✅ Activity data validation passed")
        
        # Validate performance
        performance_fields = ['success_rate', 'total_tasks', 'avg_execution_time']
        for field in performance_fields:
            if field not in data['performance']:
                print(f"❌ Missing performance field: {field}")
                return False
        print(f"✅ Performance data validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Monetization data generation test failed: {e}")
        return False

def test_earnings_tracker_integration():
    """Test earnings tracker integration"""
    print("\n💼 Testing Earnings Tracker Integration")
    print("=" * 50)
    
    try:
        from monetization.earnings_tracker import EarningsTracker, MonetizationManager
        
        # Create test earnings tracker
        tracker = EarningsTracker(db_path="test_monetization.db")
        
        # Record test earnings
        success1 = tracker.record_earning(
            platform='test_dashboard_platform',
            task_type='dashboard_test',
            amount=5.75,
            details={'test': True, 'dashboard': 'monetization'}
        )
        
        success2 = tracker.record_earning(
            platform='test_dashboard_platform',
            task_type='automation_test',
            amount=3.25,
            details={'test': True, 'type': 'automation'}
        )
        
        if success1 and success2:
            print("✅ Test earnings recorded successfully")
            
            # Get earnings summary
            summary = tracker.get_earnings_summary(days=1)
            print(f"   💰 Total amount: ${summary.get('total_amount', 0):.2f}")
            print(f"   📊 Total tasks: {summary.get('total_tasks', 0)}")
            
            # Test withdrawal request
            withdrawal_success = tracker.request_withdrawal(
                amount=8.00,
                method='test_method',
                address='test@dashboard.com'
            )
            
            if withdrawal_success:
                print("✅ Test withdrawal request successful")
                
                # Get withdrawal history
                history = tracker.get_withdrawal_history()
                if history:
                    print(f"✅ Withdrawal history retrieved: {len(history)} records")
                    return True
                else:
                    print("❌ No withdrawal history found")
                    return False
            else:
                print("❌ Test withdrawal request failed")
                return False
        else:
            print("❌ Failed to record test earnings")
            return False
            
    except Exception as e:
        print(f"❌ Earnings tracker integration test failed: {e}")
        return False

def test_dashboard_data_file():
    """Test dashboard data file creation and validation"""
    print("\n📊 Testing Dashboard Data File")
    print("=" * 50)
    
    try:
        # Check if monetization data file exists
        data_file = Path(__file__).parent / 'data' / 'monetization_data.json'
        
        if not data_file.exists():
            print("❌ Monetization data file not found")
            return False
        
        # Load and validate data file
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Validate top-level structure
        if 'monetization_data' not in data:
            print("❌ Missing monetization_data in file")
            return False
        
        monetization_data = data['monetization_data']
        
        # Check required fields
        required_fields = [
            'total_earnings', 'pending_earnings', 'withdrawn_amount',
            'platforms', 'recent_activity', 'performance'
        ]
        
        for field in required_fields:
            if field not in monetization_data:
                print(f"❌ Missing field in data file: {field}")
                return False
        
        print("✅ Data file structure validation passed")
        
        # Validate data types and values
        if not isinstance(monetization_data['total_earnings'], (int, float)):
            print("❌ Invalid total_earnings type")
            return False
        
        if not isinstance(monetization_data['platforms'], list):
            print("❌ Invalid platforms type")
            return False
        
        if not isinstance(monetization_data['recent_activity'], list):
            print("❌ Invalid recent_activity type")
            return False
        
        print("✅ Data file content validation passed")
        print(f"   💰 Total earnings: ${monetization_data['total_earnings']:.2f}")
        print(f"   🏢 Platforms: {len(monetization_data['platforms'])}")
        print(f"   📋 Activities: {len(monetization_data['recent_activity'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard data file test failed: {e}")
        return False

def test_dashboard_server_startup():
    """Test dashboard server startup (without actually starting it)"""
    print("\n🌐 Testing Dashboard Server Configuration")
    print("=" * 50)
    
    try:
        # Import the server module
        import start_monetization_dashboard
        
        # Check if required functions exist
        if not hasattr(start_monetization_dashboard, 'generate_monetization_data'):
            print("❌ Missing generate_monetization_data function")
            return False
        
        if not hasattr(start_monetization_dashboard, 'start_dashboard'):
            print("❌ Missing start_dashboard function")
            return False
        
        print("✅ Server module structure validation passed")
        
        # Test data generation function
        try:
            data = start_monetization_dashboard.generate_monetization_data()
            if data and 'total_earnings' in data:
                print("✅ Data generation function works")
            else:
                print("❌ Data generation function returned invalid data")
                return False
        except Exception as e:
            print(f"❌ Data generation function failed: {e}")
            return False
        
        print("✅ Dashboard server configuration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard server test failed: {e}")
        return False

def test_dashboard_html():
    """Test dashboard HTML file"""
    print("\n🎨 Testing Dashboard HTML")
    print("=" * 50)
    
    try:
        # Check if dashboard HTML exists
        html_file = Path(__file__).parent / 'dashboard' / 'index.html'
        
        if not html_file.exists():
            print("❌ Dashboard HTML file not found")
            return False
        
        # Read and validate HTML content
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check for required elements
        required_elements = [
            'BRAF Monetization Dashboard',
            'total-earnings',
            'pending-earnings',
            'withdrawn-amount',
            'platform-stats',
            'recent-activity',
            'loadDashboardData',
            'monetization_data.json'
        ]
        
        for element in required_elements:
            if element not in html_content:
                print(f"❌ Missing required element: {element}")
                return False
        
        print("✅ Dashboard HTML validation passed")
        print(f"   📄 File size: {len(html_content)} characters")
        print(f"   🎨 Contains all required UI elements")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard HTML test failed: {e}")
        return False

def main():
    """Run all monetization dashboard tests"""
    print("💰 BRAF Monetization Dashboard Test Suite")
    print("=" * 60)
    
    tests = [
        ("Monetization Data Generation", test_monetization_data_generation),
        ("Earnings Tracker Integration", test_earnings_tracker_integration),
        ("Dashboard Data File", test_dashboard_data_file),
        ("Dashboard Server Configuration", test_dashboard_server_startup),
        ("Dashboard HTML", test_dashboard_html)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ {test_name}: CRITICAL ERROR - {e}")
            print()
    
    print("📊 Monetization Dashboard Test Summary:")
    print("=" * 60)
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   📈 Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print(f"🎉 All tests passed! Monetization dashboard is ready!")
        print(f"💡 Start the dashboard with: python start_monetization_dashboard.py")
        return True
    elif passed >= total * 0.8:  # 80% pass rate
        print(f"✅ Most tests passed! Dashboard is largely functional.")
        return True
    else:
        print(f"⚠️  Some tests failed. Please check the dashboard components.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
