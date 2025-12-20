#!/usr/bin/env python3
"""
Simple test for account creation
"""

import requests
import json

def test_server():
    """Test if server is running"""
    try:
        response = requests.get("http://localhost:8003/api/status", timeout=5)
        print(f"✅ Server is running: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ Server not responding: {e}")
        return False

def test_registration():
    """Test account registration"""
    import time
    email = f"user{int(time.time())}@example.com"
    account_data = {
        "name": "Test User",
        "email": email,
        "password": "testpassword123",
        "subscription_tier": "free",
        "company_name": "Test Company",
        "phone_number": "+1234567890",
        "country": "US"
    }
    
    try:
        response = requests.post(
            "http://localhost:8003/api/v1/enterprise/register",
            json=account_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"Registration response: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Account created successfully!")
            print(f"Enterprise ID: {data.get('enterprise_id')}")
            data["test_email"] = email  # Store email for login test
            return data
        else:
            print(f"❌ Registration failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return None

def test_login(email, password):
    """Test login"""
    try:
        response = requests.post(
            "http://localhost:8003/api/v1/enterprise/login",
            json={"email": email, "password": password},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"Login response: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Login successful!")
            return data
        else:
            print(f"❌ Login failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Login error: {e}")
        return None

if __name__ == "__main__":
    print("🧪 Testing BRAF Account System")
    print("=" * 40)
    
    # Test server
    if not test_server():
        print("Server is not running. Start it with: python run_server.py")
        exit(1)
    
    # Test registration
    print("\n📝 Testing Registration...")
    reg_data = test_registration()
    
    if reg_data:
        # Test login
        print("\n🔐 Testing Login...")
        # Get the email from the registration data
        test_email = reg_data.get("test_email", "user@example.com")
        login_data = test_login(test_email, "testpassword123")
        
        if login_data:
            print(f"\n🎉 All tests passed!")
            print(f"🌐 Visit: http://localhost:8003/dashboard")
            print(f"🔑 Login with: test@example.com / testpassword123")
        else:
            print("\n❌ Login test failed")
    else:
        print("\n❌ Registration test failed")