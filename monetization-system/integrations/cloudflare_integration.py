#!/usr/bin/env python3
"""
Cloudflare Integration Module
Provides DNS management, security, and CDN functionality
"""

import os
import requests
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class CloudflareIntegration:
    """Cloudflare API integration for DNS, security, and CDN management"""
    
    def __init__(self):
        self.api_key = os.getenv('CLOUDFLARE_API_KEY', 'c40ef9c9bf82658bb72b21fd80944dac')
        self.email = os.getenv('CLOUDFLARE_EMAIL', '')
        self.zone_id = os.getenv('CLOUDFLARE_ZONE_ID', '')
        self.base_url = "https://api.cloudflare.com/client/v4"
        
        self.headers = {
            'X-Auth-Email': self.email,
            'X-Auth-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        
        logger.info("Cloudflare integration initialized")
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make authenticated request to Cloudflare API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=self.headers, params=data)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=self.headers, json=data)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=self.headers, json=data)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=self.headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Cloudflare API request failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_zones(self) -> List[Dict]:
        """Get all zones associated with the account"""
        response = self._make_request('GET', '/zones')
        
        if response.get('success'):
            return response.get('result', [])
        else:
            logger.error(f"Failed to get zones: {response}")
            return []
    
    def get_zone_info(self, zone_id: Optional[str] = None) -> Optional[Dict]:
        """Get information about a specific zone"""
        zone_id = zone_id or self.zone_id
        if not zone_id:
            logger.error("No zone ID provided")
            return None
        
        response = self._make_request('GET', f'/zones/{zone_id}')
        
        if response.get('success'):
            return response.get('result')
        else:
            logger.error(f"Failed to get zone info: {response}")
            return None
    
    def list_dns_records(self, zone_id: Optional[str] = None, record_type: Optional[str] = None) -> List[Dict]:
        """List DNS records for a zone"""
        zone_id = zone_id or self.zone_id
        if not zone_id:
            logger.error("No zone ID provided")
            return []
        
        params = {}
        if record_type:
            params['type'] = record_type
        
        response = self._make_request('GET', f'/zones/{zone_id}/dns_records', params)
        
        if response.get('success'):
            return response.get('result', [])
        else:
            logger.error(f"Failed to list DNS records: {response}")
            return []
    
    def create_dns_record(self, name: str, record_type: str, content: str, 
                         ttl: int = 300, proxied: bool = False, 
                         zone_id: Optional[str] = None) -> Optional[Dict]:
        """Create a new DNS record"""
        zone_id = zone_id or self.zone_id
        if not zone_id:
            logger.error("No zone ID provided")
            return None
        
        data = {
            'type': record_type,
            'name': name,
            'content': content,
            'ttl': ttl,
            'proxied': proxied
        }
        
        response = self._make_request('POST', f'/zones/{zone_id}/dns_records', data)
        
        if response.get('success'):
            logger.info(f"Created DNS record: {name} -> {content}")
            return response.get('result')
        else:
            logger.error(f"Failed to create DNS record: {response}")
            return None
    
    def update_dns_record(self, record_id: str, name: str, record_type: str, 
                         content: str, ttl: int = 300, proxied: bool = False,
                         zone_id: Optional[str] = None) -> Optional[Dict]:
        """Update an existing DNS record"""
        zone_id = zone_id or self.zone_id
        if not zone_id:
            logger.error("No zone ID provided")
            return None
        
        data = {
            'type': record_type,
            'name': name,
            'content': content,
            'ttl': ttl,
            'proxied': proxied
        }
        
        response = self._make_request('PUT', f'/zones/{zone_id}/dns_records/{record_id}', data)
        
        if response.get('success'):
            logger.info(f"Updated DNS record: {name} -> {content}")
            return response.get('result')
        else:
            logger.error(f"Failed to update DNS record: {response}")
            return None
    
    def delete_dns_record(self, record_id: str, zone_id: Optional[str] = None) -> bool:
        """Delete a DNS record"""
        zone_id = zone_id or self.zone_id
        if not zone_id:
            logger.error("No zone ID provided")
            return False
        
        response = self._make_request('DELETE', f'/zones/{zone_id}/dns_records/{record_id}')
        
        if response.get('success'):
            logger.info(f"Deleted DNS record: {record_id}")
            return True
        else:
            logger.error(f"Failed to delete DNS record: {response}")
            return False
    
    def setup_braf_dns(self, domain: str, server_ip: str) -> Dict[str, Any]:
        """Setup DNS records for BRAF deployment"""
        results = {
            'main': None,
            'api': None,
            'admin': None,
            'www': None
        }
        
        # Main application record
        results['main'] = self.create_dns_record(
            name=domain,
            record_type='A',
            content=server_ip,
            ttl=300,
            proxied=True
        )
        
        # API subdomain
        results['api'] = self.create_dns_record(
            name=f'api.{domain}',
            record_type='A',
            content=server_ip,
            ttl=300,
            proxied=True
        )
        
        # Admin subdomain
        results['admin'] = self.create_dns_record(
            name=f'admin.{domain}',
            record_type='A',
            content=server_ip,
            ttl=300,
            proxied=True
        )
        
        # WWW redirect
        results['www'] = self.create_dns_record(
            name=f'www.{domain}',
            record_type='CNAME',
            content=domain,
            ttl=300,
            proxied=True
        )
        
        return results
    
    def get_security_settings(self, zone_id: Optional[str] = None) -> Dict:
        """Get security settings for a zone"""
        zone_id = zone_id or self.zone_id
        if not zone_id:
            logger.error("No zone ID provided")
            return {}
        
        settings = {}
        
        # Get SSL settings
        ssl_response = self._make_request('GET', f'/zones/{zone_id}/settings/ssl')
        if ssl_response.get('success'):
            settings['ssl'] = ssl_response.get('result', {}).get('value')
        
        # Get security level
        security_response = self._make_request('GET', f'/zones/{zone_id}/settings/security_level')
        if security_response.get('success'):
            settings['security_level'] = security_response.get('result', {}).get('value')
        
        # Get DDoS protection
        ddos_response = self._make_request('GET', f'/zones/{zone_id}/settings/ddos_protection')
        if ddos_response.get('success'):
            settings['ddos_protection'] = ddos_response.get('result', {}).get('value')
        
        return settings
    
    def configure_security(self, zone_id: Optional[str] = None) -> Dict[str, bool]:
        """Configure security settings for BRAF deployment"""
        zone_id = zone_id or self.zone_id
        if not zone_id:
            logger.error("No zone ID provided")
            return {}
        
        results = {}
        
        # Enable Full SSL
        ssl_data = {'value': 'full'}
        ssl_response = self._make_request('PATCH', f'/zones/{zone_id}/settings/ssl', ssl_data)
        results['ssl'] = ssl_response.get('success', False)
        
        # Set security level to High
        security_data = {'value': 'high'}
        security_response = self._make_request('PATCH', f'/zones/{zone_id}/settings/security_level', security_data)
        results['security_level'] = security_response.get('success', False)
        
        # Enable Always Use HTTPS
        https_data = {'value': 'on'}
        https_response = self._make_request('PATCH', f'/zones/{zone_id}/settings/always_use_https', https_data)
        results['always_https'] = https_response.get('success', False)
        
        # Enable HSTS
        hsts_data = {
            'value': {
                'enabled': True,
                'max_age': 31536000,
                'include_subdomains': True,
                'preload': True
            }
        }
        hsts_response = self._make_request('PATCH', f'/zones/{zone_id}/settings/security_header', hsts_data)
        results['hsts'] = hsts_response.get('success', False)
        
        return results
    
    def create_page_rule(self, url_pattern: str, actions: Dict, zone_id: Optional[str] = None) -> Optional[Dict]:
        """Create a page rule"""
        zone_id = zone_id or self.zone_id
        if not zone_id:
            logger.error("No zone ID provided")
            return None
        
        data = {
            'targets': [{'target': 'url', 'constraint': {'operator': 'matches', 'value': url_pattern}}],
            'actions': [{'id': key, 'value': value} for key, value in actions.items()],
            'status': 'active'
        }
        
        response = self._make_request('POST', f'/zones/{zone_id}/pagerules', data)
        
        if response.get('success'):
            logger.info(f"Created page rule for: {url_pattern}")
            return response.get('result')
        else:
            logger.error(f"Failed to create page rule: {response}")
            return None
    
    def setup_braf_page_rules(self, domain: str, zone_id: Optional[str] = None) -> Dict[str, Any]:
        """Setup page rules for BRAF deployment"""
        results = {}
        
        # API caching rule
        api_actions = {
            'cache_level': 'bypass',
            'browser_cache_ttl': 0
        }
        results['api_cache'] = self.create_page_rule(f'{domain}/api/*', api_actions, zone_id)
        
        # Static assets caching
        static_actions = {
            'cache_level': 'cache_everything',
            'browser_cache_ttl': 31536000,  # 1 year
            'edge_cache_ttl': 2592000       # 30 days
        }
        results['static_cache'] = self.create_page_rule(f'{domain}/static/*', static_actions, zone_id)
        
        # Admin security
        admin_actions = {
            'security_level': 'high',
            'cache_level': 'bypass'
        }
        results['admin_security'] = self.create_page_rule(f'admin.{domain}/*', admin_actions, zone_id)
        
        return results
    
    def get_analytics(self, zone_id: Optional[str] = None, since: Optional[datetime] = None) -> Dict:
        """Get analytics data for a zone"""
        zone_id = zone_id or self.zone_id
        if not zone_id:
            logger.error("No zone ID provided")
            return {}
        
        # Default to last 24 hours
        if not since:
            since = datetime.now() - timedelta(days=1)
        
        params = {
            'since': since.isoformat(),
            'until': datetime.now().isoformat()
        }
        
        response = self._make_request('GET', f'/zones/{zone_id}/analytics/dashboard', params)
        
        if response.get('success'):
            return response.get('result', {})
        else:
            logger.error(f"Failed to get analytics: {response}")
            return {}
    
    def purge_cache(self, files: Optional[List[str]] = None, zone_id: Optional[str] = None) -> bool:
        """Purge cache for specific files or entire zone"""
        zone_id = zone_id or self.zone_id
        if not zone_id:
            logger.error("No zone ID provided")
            return False
        
        if files:
            data = {'files': files}
        else:
            data = {'purge_everything': True}
        
        response = self._make_request('POST', f'/zones/{zone_id}/purge_cache', data)
        
        if response.get('success'):
            logger.info("Cache purged successfully")
            return True
        else:
            logger.error(f"Failed to purge cache: {response}")
            return False
    
    def get_firewall_rules(self, zone_id: Optional[str] = None) -> List[Dict]:
        """Get firewall rules for a zone"""
        zone_id = zone_id or self.zone_id
        if not zone_id:
            logger.error("No zone ID provided")
            return []
        
        response = self._make_request('GET', f'/zones/{zone_id}/firewall/rules')
        
        if response.get('success'):
            return response.get('result', [])
        else:
            logger.error(f"Failed to get firewall rules: {response}")
            return []
    
    def create_firewall_rule(self, expression: str, action: str, description: str = "", 
                           zone_id: Optional[str] = None) -> Optional[Dict]:
        """Create a firewall rule"""
        zone_id = zone_id or self.zone_id
        if not zone_id:
            logger.error("No zone ID provided")
            return None
        
        data = {
            'filter': {
                'expression': expression,
                'paused': False
            },
            'action': {
                'mode': action
            },
            'description': description
        }
        
        response = self._make_request('POST', f'/zones/{zone_id}/firewall/rules', data)
        
        if response.get('success'):
            logger.info(f"Created firewall rule: {description}")
            return response.get('result')
        else:
            logger.error(f"Failed to create firewall rule: {response}")
            return None
    
    def setup_braf_firewall(self, zone_id: Optional[str] = None) -> Dict[str, Any]:
        """Setup firewall rules for BRAF deployment"""
        results = {}
        
        # Block known bad IPs
        results['block_bad_ips'] = self.create_firewall_rule(
            expression='(cf.threat_score gt 14)',
            action='block',
            description='Block high threat score IPs',
            zone_id=zone_id
        )
        
        # Rate limit API endpoints
        results['rate_limit_api'] = self.create_firewall_rule(
            expression='(http.request.uri.path matches "^/api/.*")',
            action='challenge',
            description='Challenge API requests',
            zone_id=zone_id
        )
        
        # Protect admin area
        results['protect_admin'] = self.create_firewall_rule(
            expression='(http.host eq "admin.yourdomain.com")',
            action='challenge',
            description='Challenge admin access',
            zone_id=zone_id
        )
        
        return results
    
    def get_ssl_certificate_info(self, zone_id: Optional[str] = None) -> Dict:
        """Get SSL certificate information"""
        zone_id = zone_id or self.zone_id
        if not zone_id:
            logger.error("No zone ID provided")
            return {}
        
        response = self._make_request('GET', f'/zones/{zone_id}/ssl/certificate_packs')
        
        if response.get('success'):
            return response.get('result', [])
        else:
            logger.error(f"Failed to get SSL certificate info: {response}")
            return {}
    
    def deploy_braf_infrastructure(self, domain: str, server_ip: str, zone_id: Optional[str] = None) -> Dict:
        """Complete Cloudflare setup for BRAF deployment"""
        logger.info(f"Deploying BRAF infrastructure for {domain}")
        
        results = {
            'dns_records': {},
            'security_settings': {},
            'page_rules': {},
            'firewall_rules': {},
            'ssl_info': {}
        }
        
        # Setup DNS records
        results['dns_records'] = self.setup_braf_dns(domain, server_ip)
        
        # Configure security settings
        results['security_settings'] = self.configure_security(zone_id)
        
        # Setup page rules
        results['page_rules'] = self.setup_braf_page_rules(domain, zone_id)
        
        # Setup firewall rules
        results['firewall_rules'] = self.setup_braf_firewall(zone_id)
        
        # Get SSL certificate info
        results['ssl_info'] = self.get_ssl_certificate_info(zone_id)
        
        logger.info("BRAF infrastructure deployment completed")
        return results


def test_cloudflare_integration():
    """Test Cloudflare integration functionality"""
    cf = CloudflareIntegration()
    
    print("Testing Cloudflare Integration...")
    
    # Test getting zones
    zones = cf.get_zones()
    print(f"Found {len(zones)} zones")
    
    if zones:
        zone = zones[0]
        zone_id = zone['id']
        domain = zone['name']
        
        print(f"Testing with zone: {domain} ({zone_id})")
        
        # Test getting DNS records
        records = cf.list_dns_records(zone_id)
        print(f"Found {len(records)} DNS records")
        
        # Test getting security settings
        security = cf.get_security_settings(zone_id)
        print(f"Security settings: {security}")
        
        # Test getting analytics
        analytics = cf.get_analytics(zone_id)
        print(f"Analytics data available: {bool(analytics)}")
    
    print("Cloudflare integration test completed")


if __name__ == "__main__":
    test_cloudflare_integration()