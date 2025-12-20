#!/usr/bin/env python3
"""
Test MAXEL API Connectivity
Check if MAXEL API is reachable and working with provided credentials
"""
import requests
import json

def test_maxel_connectivity():
    """Test MAXEL API connectivity"""
    print("🔍 Testing MAXEL API Connectivity")
    print("=" * 50)
    
    # MAXEL API credentials
    MAXEL_API_KEY = 'pk_Eq8N27HLVFDrPFd34j7a7cpIJd6PncsWMAXEL_SECRET_KEY'
    MAXEL_SECRET_KEY = 'sk_rI7pJyhIyaiU5js1BCpjYA53y5iS7Ny0'
    
    headers = {
        'Authorization': f'Bearer {MAXEL_API_KEY}',
        'X-Secret-Key': MAXEL_SECRET_KEY,
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    # Try different MAXEL endpoints
    endpoints = [
        'https://api.maxel.io/v1/status',
        'https://api.maxel.io/status', 
        'https://maxel.io/api/v1/status',
        'https://api-sandbox.maxel.io/v1/status',
        'https://api.maxel.io/v1/account/balance',
        'https://api.maxel.io/v1/currencies'
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
        print(f"\n✅ MAXEL API is accessible at: {working_endpoint}")
        return working_endpoint
    else:
        print(f"\n❌ MAXEL API is not accessible with current configuration")
        print(f"\n💡 Possible issues:")
        print(f"   1. API keys may be invalid or expired")
        print(f"   2. MAXEL API endpoints may have changed")
        print(f"   3. Network/firewall blocking requests")
        print(f"   4. MAXEL service may be down")
        return None

def test_real_maxel_deposit():
    """Test creating a real deposit to MAXEL wallet"""
    print(f"\n💰 Testing Real MAXEL Deposit")
    print("=" * 40)
    
    # This would be the actual API call to deposit money
    # For now, we'll simulate what should happen
    
    print("🚨 IMPORTANT: BRAF earnings are currently LOCAL TRACKING ONLY")
    print("   - Dashboard shows simulated earnings")
    print("   - No real cryptocurrency is generated")
    print("   - Transfer script only updates local JSON file")
    print("   - Real MAXEL wallet remains unchanged")
    
    print(f"\n💡 To get real money in MAXEL wallet:")
    print(f"   1. BRAF would need to connect to real earning platforms")
    print(f"   2. Complete actual paid tasks/surveys/work")
    print(f"   3. Receive real payments from those platforms")
    print(f"   4. Then transfer real money to MAXEL")
    
    return False

if __name__ == "__main__":
    working_endpoint = test_maxel_connectivity()
    test_real_maxel_deposit()
    
    if not working_endpoint:
        print(f"\n🔧 Next Steps:")
        print(f"   1. Verify MAXEL API keys are correct")
        print(f"   2. Check MAXEL documentation for correct endpoints")
        print(f"   3. Contact MAXEL support if needed")
        print(f"   4. Consider using MAXEL sandbox/test environment first")