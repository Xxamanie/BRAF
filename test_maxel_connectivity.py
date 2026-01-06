#!/usr/bin/env python3
"""
Test maxelpay API Connectivity
Check if maxelpay API is reachable and working with provided credentials
"""
import requests
import json

def test_MAXELPAY_connectivity():
    """Test maxelpay API connectivity"""
    print("🔍 Testing maxelpay API Connectivity")
    print("=" * 50)
    
    # maxelpay API credentials
    MAXELPAY_API_KEY = 'pk_Eq8N27HLVFDrPFd34j7a7cpIJd6PncsWMAXELPAY_SECRET_KEY'
    MAXELPAY_SECRET_KEY = 'sk_rI7pJyhIyaiU5js1BCpjYA53y5iS7Ny0'
    
    headers = {
        'Authorization': f'Bearer {MAXELPAY_API_KEY}',
        'X-Secret-Key': MAXELPAY_SECRET_KEY,
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    # Try different maxelpay endpoints
    endpoints = [
        'https://api.maxelpay.com/v1/status',
        'https://api.maxelpay.com/status', 
        'https://maxelpay.com/api/v1/status',
        'https://api-sandbox.maxelpay.com/v1/status',
        'https://api.maxelpay.com/v1/account/balance',
        'https://api.maxelpay.com/v1/currencies'
    ]
    
    working_endpoint = None
    
    for url in endpoints:
        try:
            print(f"\n🌐 Testing: {url}")
            response = requests.get(url, headers=headers, timeout=10)
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ SUCCESS!")
                print(f"   Response: {response.text[:200]}...")
                working_endpoint = url
                break
            elif response.status_code == 401:
                print(f"   🔑 Authentication issue - check API keys")
            elif response.status_code == 404:
                print(f"   ❌ Endpoint not found")
            else:
                print(f"   ⚠️  HTTP {response.status_code}: {response.text[:100]}")
                
        except requests.exceptions.SSLError as e:
            print(f"   🔒 SSL Error: {e}")
        except requests.exceptions.ConnectionError as e:
            print(f"   🌐 Connection Error: {e}")
        except requests.exceptions.Timeout as e:
            print(f"   ⏰ Timeout Error: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    if working_endpoint:
        print(f"\n✅ maxelpay API is accessible at: {working_endpoint}")
        return working_endpoint
    else:
        print(f"\n❌ maxelpay API is not accessible with current configuration")
        print(f"\n💡 Possible issues:")
        print(f"   1. API keys may be invalid or expired")
        print(f"   2. maxelpay API endpoints may have changed")
        print(f"   3. Network/firewall blocking requests")
        print(f"   4. maxelpay service may be down")
        return None

def test_real_MAXELPAY_deposit():
    """Test creating a real deposit to maxelpay wallet"""
    print(f"\n💰 Testing Real maxelpay Deposit")
    print("=" * 40)
    
    # This would be the actual API call to deposit money
    # For now, we'll simulate what should happen
    
    print("🚨 IMPORTANT: BRAF earnings are currently LOCAL TRACKING ONLY")
    print("   - Dashboard shows simulated earnings")
    print("   - No real cryptocurrency is generated")
    print("   - Transfer script only updates local JSON file")
    print("   - Real maxelpay wallet remains unchanged")
    
    print(f"\n💡 To get real money in maxelpay wallet:")
    print(f"   1. BRAF would need to connect to real earning platforms")
    print(f"   2. Complete actual paid tasks/surveys/work")
    print(f"   3. Receive real payments from those platforms")
    print(f"   4. Then transfer real money to maxelpay")
    
    return False

if __name__ == "__main__":
    working_endpoint = test_MAXELPAY_connectivity()
    test_real_MAXELPAY_deposit()
    
    if not working_endpoint:
        print(f"\n🔧 Next Steps:")
        print(f"   1. Verify maxelpay API keys are correct")
        print(f"   2. Check maxelpay documentation for correct endpoints")
        print(f"   3. Contact maxelpay support if needed")
        print(f"   4. Consider using maxelpay sandbox/test environment first")
