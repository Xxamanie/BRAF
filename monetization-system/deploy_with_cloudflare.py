#!/usr/bin/env python3
"""
BRAF Deployment with Cloudflare Integration
Automated deployment with DNS, security, and CDN setup
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from integrations.cloudflare_integration import CloudflareIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BRAFCloudflareDeployment:
    """BRAF deployment with Cloudflare integration"""
    
    def __init__(self, domain: str, server_ip: str, zone_id: str = None):
        self.domain = domain
        self.server_ip = server_ip
        self.zone_id = zone_id
        self.cf = CloudflareIntegration()
        
        # Update environment with zone ID if provided
        if zone_id:
            os.environ['CLOUDFLARE_ZONE_ID'] = zone_id
            self.cf.zone_id = zone_id
        
        logger.info(f"Initializing BRAF Cloudflare deployment for {domain}")
    
    def validate_configuration(self) -> bool:
        """Validate Cloudflare configuration"""
        logger.info("Validating Cloudflare configuration...")
        
        if not self.cf.api_key or self.cf.api_key == 'your-api-key':
            logger.error("Cloudflare API key not configured")
            return False
        
        if not self.cf.email:
            logger.warning("Cloudflare email not configured - using API token mode")
        
        # Test API connectivity
        zones = self.cf.get_zones()
        if not zones:
            logger.error("Failed to connect to Cloudflare API or no zones found")
            return False
        
        logger.info(f"Found {len(zones)} zones in Cloudflare account")
        
        # Find zone for domain if not specified
        if not self.zone_id:
            for zone in zones:
                if zone['name'] == self.domain or self.domain.endswith(f".{zone['name']}"):
                    self.zone_id = zone['id']
                    self.cf.zone_id = zone['id']
                    logger.info(f"Found zone for {self.domain}: {self.zone_id}")
                    break
            
            if not self.zone_id:
                logger.error(f"No zone found for domain {self.domain}")
                return False
        
        logger.info("Cloudflare configuration validated successfully")
        return True
    
    def setup_dns_records(self) -> Dict[str, Any]:
        """Setup DNS records for BRAF deployment"""
        logger.info("Setting up DNS records...")
        
        results = {}
        
        # Check existing records first
        existing_records = self.cf.list_dns_records()
        existing_names = {record['name']: record for record in existing_records}
        
        # DNS records to create/update
        dns_config = [
            {'name': self.domain, 'type': 'A', 'content': self.server_ip, 'proxied': True},
            {'name': f'api.{self.domain}', 'type': 'A', 'content': self.server_ip, 'proxied': True},
            {'name': f'admin.{self.domain}', 'type': 'A', 'content': self.server_ip, 'proxied': True},
            {'name': f'www.{self.domain}', 'type': 'CNAME', 'content': self.domain, 'proxied': True},
            {'name': f'dashboard.{self.domain}', 'type': 'A', 'content': self.server_ip, 'proxied': True},
            {'name': f'monitoring.{self.domain}', 'type': 'A', 'content': self.server_ip, 'proxied': False}
        ]
        
        for record_config in dns_config:
            name = record_config['name']
            
            if name in existing_names:
                # Update existing record
                existing_record = existing_names[name]
                if (existing_record['content'] != record_config['content'] or 
                    existing_record['proxied'] != record_config['proxied']):
                    
                    result = self.cf.update_dns_record(
                        record_id=existing_record['id'],
                        name=name,
                        record_type=record_config['type'],
                        content=record_config['content'],
                        proxied=record_config['proxied']
                    )
                    results[name] = {'action': 'updated', 'success': bool(result)}
                else:
                    results[name] = {'action': 'unchanged', 'success': True}
            else:
                # Create new record
                result = self.cf.create_dns_record(
                    name=name,
                    record_type=record_config['type'],
                    content=record_config['content'],
                    proxied=record_config['proxied']
                )
                results[name] = {'action': 'created', 'success': bool(result)}
        
        logger.info(f"DNS setup completed: {len(results)} records processed")
        return results
    
    def configure_security_settings(self) -> Dict[str, bool]:
        """Configure Cloudflare security settings"""
        logger.info("Configuring security settings...")
        
        results = self.cf.configure_security()
        
        # Additional security configurations
        additional_settings = [
            ('browser_check', 'on'),
            ('challenge_ttl', 1800),
            ('development_mode', 'off'),
            ('email_obfuscation', 'on'),
            ('hotlink_protection', 'on'),
            ('ip_geolocation', 'on'),
            ('ipv6', 'on'),
            ('minify', {'css': 'on', 'html': 'on', 'js': 'on'}),
            ('mirage', 'on'),
            ('mobile_redirect', 'off'),
            ('opportunistic_encryption', 'on'),
            ('polish', 'lossless'),
            ('prefetch_preload', 'on'),
            ('response_buffering', 'on'),
            ('rocket_loader', 'on'),
            ('server_side_exclude', 'on'),
            ('sort_query_string_for_cache', 'on'),
            ('tls_1_3', 'on'),
            ('waf', 'on'),
            ('webp', 'on')
        ]
        
        for setting, value in additional_settings:
            try:
                data = {'value': value}
                response = self.cf._make_request('PATCH', f'/zones/{self.zone_id}/settings/{setting}', data)
                results[setting] = response.get('success', False)
            except Exception as e:
                logger.warning(f"Failed to configure {setting}: {e}")
                results[setting] = False
        
        logger.info(f"Security configuration completed: {sum(results.values())}/{len(results)} settings applied")
        return results
    
    def setup_page_rules(self) -> Dict[str, Any]:
        """Setup page rules for optimal performance"""
        logger.info("Setting up page rules...")
        
        page_rules = [
            {
                'url': f'{self.domain}/api/*',
                'actions': {
                    'cache_level': 'bypass',
                    'browser_cache_ttl': 0,
                    'security_level': 'high'
                },
                'description': 'API endpoints - no cache, high security'
            },
            {
                'url': f'{self.domain}/static/*',
                'actions': {
                    'cache_level': 'cache_everything',
                    'browser_cache_ttl': 31536000,  # 1 year
                    'edge_cache_ttl': 2592000       # 30 days
                },
                'description': 'Static assets - aggressive caching'
            },
            {
                'url': f'admin.{self.domain}/*',
                'actions': {
                    'security_level': 'high',
                    'cache_level': 'bypass',
                    'browser_check': 'on'
                },
                'description': 'Admin area - high security, no cache'
            },
            {
                'url': f'{self.domain}/docs*',
                'actions': {
                    'cache_level': 'cache_everything',
                    'browser_cache_ttl': 3600,      # 1 hour
                    'edge_cache_ttl': 86400         # 1 day
                },
                'description': 'Documentation - moderate caching'
            }
        ]
        
        results = {}
        for rule in page_rules:
            result = self.cf.create_page_rule(
                url_pattern=rule['url'],
                actions=rule['actions']
            )
            results[rule['description']] = {'success': bool(result), 'rule': result}
        
        logger.info(f"Page rules setup completed: {len(results)} rules created")
        return results
    
    def setup_firewall_rules(self) -> Dict[str, Any]:
        """Setup firewall rules for security"""
        logger.info("Setting up firewall rules...")
        
        firewall_rules = [
            {
                'expression': '(cf.threat_score gt 14)',
                'action': 'block',
                'description': 'Block high threat score IPs'
            },
            {
                'expression': '(http.request.uri.path matches "^/api/.*") and (rate(1m) > 100)',
                'action': 'challenge',
                'description': 'Rate limit API endpoints'
            },
            {
                'expression': '(http.host eq "admin.' + self.domain + '")',
                'action': 'challenge',
                'description': 'Challenge admin access'
            },
            {
                'expression': '(http.request.uri.path contains "/wp-admin") or (http.request.uri.path contains "/wp-login")',
                'action': 'block',
                'description': 'Block WordPress attack attempts'
            },
            {
                'expression': '(http.user_agent contains "bot") and not (cf.verified_bot_category in {"Search Engine" "Security"})',
                'action': 'challenge',
                'description': 'Challenge unverified bots'
            }
        ]
        
        results = {}
        for rule in firewall_rules:
            result = self.cf.create_firewall_rule(
                expression=rule['expression'],
                action=rule['action'],
                description=rule['description']
            )
            results[rule['description']] = {'success': bool(result), 'rule': result}
        
        logger.info(f"Firewall rules setup completed: {len(results)} rules created")
        return results
    
    def setup_ssl_configuration(self) -> Dict[str, Any]:
        """Setup SSL/TLS configuration"""
        logger.info("Configuring SSL/TLS settings...")
        
        ssl_settings = [
            ('ssl', 'full'),
            ('always_use_https', 'on'),
            ('min_tls_version', '1.2'),
            ('tls_1_3', 'on'),
            ('automatic_https_rewrites', 'on'),
            ('opportunistic_encryption', 'on')
        ]
        
        results = {}
        for setting, value in ssl_settings:
            try:
                data = {'value': value}
                response = self.cf._make_request('PATCH', f'/zones/{self.zone_id}/settings/{setting}', data)
                results[setting] = response.get('success', False)
            except Exception as e:
                logger.warning(f"Failed to configure SSL setting {setting}: {e}")
                results[setting] = False
        
        # Setup HSTS
        hsts_data = {
            'value': {
                'enabled': True,
                'max_age': 31536000,
                'include_subdomains': True,
                'preload': True
            }
        }
        hsts_response = self.cf._make_request('PATCH', f'/zones/{self.zone_id}/settings/security_header', hsts_data)
        results['hsts'] = hsts_response.get('success', False)
        
        logger.info(f"SSL configuration completed: {sum(results.values())}/{len(results)} settings applied")
        return results
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Optimize performance settings"""
        logger.info("Optimizing performance settings...")
        
        performance_settings = [
            ('brotli', 'on'),
            ('minify', {'css': 'on', 'html': 'on', 'js': 'on'}),
            ('rocket_loader', 'on'),
            ('mirage', 'on'),
            ('polish', 'lossless'),
            ('webp', 'on'),
            ('early_hints', 'on'),
            ('h2_prioritization', 'on')
        ]
        
        results = {}
        for setting, value in performance_settings:
            try:
                data = {'value': value}
                response = self.cf._make_request('PATCH', f'/zones/{self.zone_id}/settings/{setting}', data)
                results[setting] = response.get('success', False)
            except Exception as e:
                logger.warning(f"Failed to configure performance setting {setting}: {e}")
                results[setting] = False
        
        logger.info(f"Performance optimization completed: {sum(results.values())}/{len(results)} settings applied")
        return results
    
    def generate_deployment_report(self, deployment_results: Dict) -> str:
        """Generate deployment report"""
        report = f"""
# BRAF Cloudflare Deployment Report

**Domain:** {self.domain}
**Server IP:** {self.server_ip}
**Zone ID:** {self.zone_id}
**Deployment Time:** {datetime.now().isoformat()}

## DNS Records
"""
        
        for name, result in deployment_results.get('dns_records', {}).items():
            status = "✅" if result['success'] else "❌"
            report += f"- {status} {name} ({result['action']})\n"
        
        report += "\n## Security Settings\n"
        for setting, success in deployment_results.get('security_settings', {}).items():
            status = "✅" if success else "❌"
            report += f"- {status} {setting}\n"
        
        report += "\n## SSL/TLS Configuration\n"
        for setting, success in deployment_results.get('ssl_configuration', {}).items():
            status = "✅" if success else "❌"
            report += f"- {status} {setting}\n"
        
        report += "\n## Performance Optimization\n"
        for setting, success in deployment_results.get('performance_optimization', {}).items():
            status = "✅" if success else "❌"
            report += f"- {status} {setting}\n"
        
        report += f"""
## Access URLs
- **Main Application:** https://{self.domain}
- **API Documentation:** https://api.{self.domain}/docs
- **Admin Dashboard:** https://admin.{self.domain}
- **Monitoring:** https://monitoring.{self.domain}

## Next Steps
1. Verify DNS propagation (may take up to 24 hours)
2. Test SSL certificate installation
3. Configure application-specific settings
4. Setup monitoring and alerting
5. Perform security testing

## Support
- Cloudflare Dashboard: https://dash.cloudflare.com
- Zone ID: {self.zone_id}
- API Key: {self.cf.api_key[:8]}...
"""
        
        return report
    
    def deploy(self) -> Dict[str, Any]:
        """Execute complete Cloudflare deployment"""
        logger.info("Starting BRAF Cloudflare deployment...")
        
        # Validate configuration
        if not self.validate_configuration():
            raise Exception("Cloudflare configuration validation failed")
        
        deployment_results = {}
        
        try:
            # Setup DNS records
            deployment_results['dns_records'] = self.setup_dns_records()
            
            # Configure security settings
            deployment_results['security_settings'] = self.configure_security_settings()
            
            # Setup SSL configuration
            deployment_results['ssl_configuration'] = self.setup_ssl_configuration()
            
            # Setup page rules
            deployment_results['page_rules'] = self.setup_page_rules()
            
            # Setup firewall rules
            deployment_results['firewall_rules'] = self.setup_firewall_rules()
            
            # Optimize performance
            deployment_results['performance_optimization'] = self.optimize_performance()
            
            # Generate report
            report = self.generate_deployment_report(deployment_results)
            
            # Save report
            report_file = Path(f'cloudflare_deployment_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md')
            report_file.write_text(report)
            
            logger.info(f"Deployment completed successfully! Report saved to {report_file}")
            
            deployment_results['report_file'] = str(report_file)
            deployment_results['success'] = True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            deployment_results['success'] = False
            deployment_results['error'] = str(e)
        
        return deployment_results


def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy BRAF with Cloudflare integration')
    parser.add_argument('domain', help='Domain name for deployment')
    parser.add_argument('server_ip', help='Server IP address')
    parser.add_argument('--zone-id', help='Cloudflare Zone ID (optional)')
    parser.add_argument('--api-key', help='Cloudflare API Key (optional)')
    parser.add_argument('--email', help='Cloudflare Email (optional)')
    
    args = parser.parse_args()
    
    # Set environment variables if provided
    if args.api_key:
        os.environ['CLOUDFLARE_API_KEY'] = args.api_key
    if args.email:
        os.environ['CLOUDFLARE_EMAIL'] = args.email
    if args.zone_id:
        os.environ['CLOUDFLARE_ZONE_ID'] = args.zone_id
    
    # Create deployment instance
    deployment = BRAFCloudflareDeployment(
        domain=args.domain,
        server_ip=args.server_ip,
        zone_id=args.zone_id
    )
    
    # Execute deployment
    results = deployment.deploy()
    
    if results['success']:
        print(f"\n✅ Deployment completed successfully!")
        print(f"📄 Report saved to: {results['report_file']}")
        print(f"🌐 Access your application at: https://{args.domain}")
    else:
        print(f"\n❌ Deployment failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
