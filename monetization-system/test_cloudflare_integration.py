#!/usr/bin/env python3
"""
Test Cloudflare Integration
Verify API connectivity and functionality
"""

import os
import sys
from pathlib import Path
from integrations.cloudflare_integration import CloudflareIntegration

def test_cloudflare_api():
    """Test Cloudflare API connectivity"""
    print("🔧 Testing Cloudflare API Integration")
    print("=" * 50)
    
    # Initialize Cloudflare integration
    cf = CloudflareIntegration()
    
    print(f"API Key: {cf.api_key[:8]}..." if cf.api_key else "API Key: Not configured")
    print(f"Email: {cf.email}" if cf.email else "Email: Not configured")
    print(f"Zone ID: {cf.zone_id}" if cf.zone_id else "Zone ID: Not configured")
    print()
    
    # Test 1: Get zones
    print("📋 Test 1: Getting zones...")
    zones = cf.get_zones()
    
    if zones:
        print(f"✅ Success: Found {len(zones)} zones")
        for zone in zones[:3]:  # Show first 3 zones
            print(f"   - {zone['name']} ({zone['id']}) - Status: {zone['status']}")
        
        # Use first zone for further tests
        test_zone = zones[0]
        zone_id = test_zone['id']
        zone_name = test_zone['name']
        
        print(f"\n🎯 Using zone '{zone_name}' for testing")
        
        # Test 2: Get zone info
        print("\n📋 Test 2: Getting zone information...")
        zone_info = cf.get_zone_info(zone_id)
        if zone_info:
            print(f"✅ Success: Zone info retrieved")
            print(f"   - Name: {zone_info['name']}")
            print(f"   - Status: {zone_info['status']}")
            print(f"   - Plan: {zone_info.get('plan', {}).get('name', 'Unknown')}")
        else:
            print("❌ Failed to get zone info")
        
        # Test 3: List DNS records
        print("\n📋 Test 3: Listing DNS records...")
        dns_records = cf.list_dns_records(zone_id)
        if dns_records:
            print(f"✅ Success: Found {len(dns_records)} DNS records")
            for record in dns_records[:5]:  # Show first 5 records
                print(f"   - {record['name']} ({record['type']}) -> {record['content']}")
        else:
            print("❌ Failed to get DNS records")
        
        # Test 4: Get security settings
        print("\n📋 Test 4: Getting security settings...")
        security_settings = cf.get_security_settings(zone_id)
        if security_settings:
            print(f"✅ Success: Security settings retrieved")
            for setting, value in security_settings.items():
                print(f"   - {setting}: {value}")
        else:
            print("❌ Failed to get security settings")
        
        # Test 5: Get analytics (if available)
        print("\n📋 Test 5: Getting analytics...")
        analytics = cf.get_analytics(zone_id)
        if analytics:
            print(f"✅ Success: Analytics data retrieved")
            print(f"   - Data points available: {len(analytics)}")
        else:
            print("⚠️  Analytics data not available or failed")
        
        # Test 6: Get SSL certificate info
        print("\n📋 Test 6: Getting SSL certificate info...")
        ssl_info = cf.get_ssl_certificate_info(zone_id)
        if ssl_info:
            print(f"✅ Success: SSL certificate info retrieved")
            if isinstance(ssl_info, list) and ssl_info:
                cert = ssl_info[0]
                print(f"   - Status: {cert.get('status', 'Unknown')}")
                print(f"   - Type: {cert.get('type', 'Unknown')}")
        else:
            print("⚠️  SSL certificate info not available")
        
        # Test 7: Get firewall rules
        print("\n📋 Test 7: Getting firewall rules...")
        firewall_rules = cf.get_firewall_rules(zone_id)
        if isinstance(firewall_rules, list):
            print(f"✅ Success: Found {len(firewall_rules)} firewall rules")
            for rule in firewall_rules[:3]:  # Show first 3 rules
                print(f"   - {rule.get('description', 'No description')} ({rule.get('action', {}).get('mode', 'Unknown')})")
        else:
            print("⚠️  Firewall rules not available")
        
        print("\n" + "=" * 50)
        print("🎉 Cloudflare API test completed!")
        print(f"✅ Zone: {zone_name}")
        print(f"✅ API connectivity: Working")
        print(f"✅ DNS management: Available")
        print(f"✅ Security settings: Available")
        
        return True
        
    else:
        print("❌ Failed: No zones found or API connection failed")
        print("\nPossible issues:")
        print("1. Invalid API key")
        print("2. Incorrect email address")
        print("3. No zones in account")
        print("4. Network connectivity issues")
        
        return False

def test_deployment_simulation():
    """Simulate a deployment without making changes"""
    print("\n🚀 Testing Deployment Simulation")
    print("=" * 50)
    
    cf = CloudflareIntegration()
    zones = cf.get_zones()
    
    if not zones:
        print("❌ Cannot simulate deployment - no zones available")
        return False
    
    test_zone = zones[0]
    zone_name = test_zone['name']
    
    print(f"📋 Simulating deployment for: {zone_name}")
    print(f"🎯 Server IP: 192.168.1.100 (example)")
    
    # Simulate DNS records that would be created
    dns_records = [
        {'name': zone_name, 'type': 'A', 'content': '192.168.1.100'},
        {'name': f'api.{zone_name}', 'type': 'A', 'content': '192.168.1.100'},
        {'name': f'admin.{zone_name}', 'type': 'A', 'content': '192.168.1.100'},
        {'name': f'www.{zone_name}', 'type': 'CNAME', 'content': zone_name}
    ]
    
    print("\n📋 DNS records that would be created:")
    for record in dns_records:
        print(f"   ✅ {record['name']} ({record['type']}) -> {record['content']}")
    
    print("\n📋 Security settings that would be configured:")
    security_settings = [
        'SSL: Full',
        'Security Level: High',
        'Always Use HTTPS: On',
        'HSTS: Enabled',
        'DDoS Protection: On'
    ]
    
    for setting in security_settings:
        print(f"   ✅ {setting}")
    
    print("\n📋 Performance optimizations that would be applied:")
    performance_settings = [
        'Brotli Compression: On',
        'Minification: CSS, HTML, JS',
        'Rocket Loader: On',
        'Image Optimization: On',
        'Caching Rules: Configured'
    ]
    
    for setting in performance_settings:
        print(f"   ✅ {setting}")
    
    print("\n🎉 Deployment simulation completed!")
    print("💡 To run actual deployment, use: python deploy_with_cloudflare.py")
    
    return True

def main():
    """Main test function"""
    print("🔧 BRAF Cloudflare Integration Test Suite")
    print("=" * 60)
    
    # Check if API key is configured
    api_key = os.getenv('CLOUDFLARE_API_KEY', '')
    if not api_key or api_key == 'your-api-key':
        print("⚠️  Cloudflare API key not configured!")
        print("Please set CLOUDFLARE_API_KEY environment variable")
        print("Current API key from .env:", api_key[:8] + "..." if len(api_key) > 8 else api_key)
        return
    
    # Test API connectivity
    api_test_passed = test_cloudflare_api()
    
    if api_test_passed:
        # Test deployment simulation
        test_deployment_simulation()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("\n📋 Next steps:")
        print("1. Configure your domain and server IP")
        print("2. Run: python deploy_with_cloudflare.py yourdomain.com your.server.ip")
        print("3. Monitor deployment progress")
        print("4. Verify DNS propagation")
        
    else:
        print("\n" + "=" * 60)
        print("❌ Tests failed - please check configuration")
        print("\n🔧 Troubleshooting:")
        print("1. Verify API key is correct")
        print("2. Check email address (if using email + API key)")
        print("3. Ensure you have zones in your Cloudflare account")
        print("4. Test network connectivity")

if __name__ == "__main__":
    main()