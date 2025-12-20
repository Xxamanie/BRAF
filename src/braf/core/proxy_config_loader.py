"""
Proxy configuration loader and validator for BRAF.

This module handles loading proxy configurations from various sources
and validates them for security and ethical compliance.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import yaml

from braf.core.models import ProxyConfig

logger = logging.getLogger(__name__)


class ProxyConfigValidator:
    """Validator for proxy configurations."""
    
    # Blocked proxy providers (known public/free proxies that shouldn't be used)
    BLOCKED_PROVIDERS = [
        "free-proxy-list.net",
        "proxylist.geonode.com", 
        "spys.one",
        "hidemy.name",
        "proxy-list.download"
    ]
    
    # Suspicious ports that might indicate compromised systems
    SUSPICIOUS_PORTS = [22, 23, 25, 53, 80, 110, 143, 443, 993, 995]
    
    def validate_proxy_config(self, config: Dict) -> Tuple[bool, List[str]]:
        """
        Validate proxy configuration for security and ethics.
        
        Args:
            config: Proxy configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Required fields
        required_fields = ["host", "port"]
        for field in required_fields:
            if field not in config:
                issues.append(f"Missing required field: {field}")
        
        if issues:  # Don't continue if basic fields are missing
            return False, issues
        
        # Validate host
        host_issues = self._validate_host(config["host"])
        issues.extend(host_issues)
        
        # Validate port
        port_issues = self._validate_port(config["port"])
        issues.extend(port_issues)
        
        # Validate proxy type
        if "type" in config:
            type_issues = self._validate_proxy_type(config["type"])
            issues.extend(type_issues)
        
        # Validate credentials if present
        if "username" in config or "password" in config:
            cred_issues = self._validate_credentials(config)
            issues.extend(cred_issues)
        
        # Check for blocked providers
        blocked_issues = self._check_blocked_providers(config["host"])
        issues.extend(blocked_issues)
        
        # Check for suspicious configurations
        suspicious_issues = self._check_suspicious_config(config)
        issues.extend(suspicious_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _validate_host(self, host: str) -> List[str]:
        """Validate proxy host."""
        issues = []
        
        if not host or not isinstance(host, str):
            issues.append("Host must be a non-empty string")
            return issues
        
        # Check for valid hostname/IP format
        if not re.match(r'^[a-zA-Z0-9.-]+$', host):
            issues.append(f"Invalid host format: {host}")
        
        # Check for localhost/private IPs (potential security risk)
        if host.lower() in ['localhost', '127.0.0.1', '0.0.0.0']:
            issues.append(f"Localhost proxy not allowed: {host}")
        
        # Check for private IP ranges
        if self._is_private_ip(host):
            issues.append(f"Private IP address not allowed: {host}")
        
        # Check length
        if len(host) > 253:  # DNS hostname limit
            issues.append(f"Host name too long: {len(host)} characters")
        
        return issues
    
    def _validate_port(self, port) -> List[str]:
        """Validate proxy port."""
        issues = []
        
        try:
            port_num = int(port)
        except (ValueError, TypeError):
            issues.append(f"Port must be a valid integer: {port}")
            return issues
        
        if port_num < 1 or port_num > 65535:
            issues.append(f"Port must be between 1 and 65535: {port_num}")
        
        # Warn about suspicious ports
        if port_num in self.SUSPICIOUS_PORTS:
            issues.append(f"Suspicious port detected: {port_num} (commonly used for other services)")
        
        return issues
    
    def _validate_proxy_type(self, proxy_type: str) -> List[str]:
        """Validate proxy type."""
        issues = []
        
        valid_types = ["http", "https", "socks4", "socks5"]
        if proxy_type.lower() not in valid_types:
            issues.append(f"Invalid proxy type: {proxy_type}. Must be one of: {valid_types}")
        
        return issues
    
    def _validate_credentials(self, config: Dict) -> List[str]:
        """Validate proxy credentials."""
        issues = []
        
        username = config.get("username")
        password = config.get("password")
        
        # Both or neither should be present
        if (username and not password) or (password and not username):
            issues.append("Both username and password must be provided together")
        
        if username:
            if not isinstance(username, str) or len(username) == 0:
                issues.append("Username must be a non-empty string")
            elif len(username) > 255:
                issues.append("Username too long (max 255 characters)")
        
        if password:
            if not isinstance(password, str) or len(password) == 0:
                issues.append("Password must be a non-empty string")
            elif len(password) > 255:
                issues.append("Password too long (max 255 characters)")
        
        return issues
    
    def _check_blocked_providers(self, host: str) -> List[str]:
        """Check if host is from a blocked provider."""
        issues = []
        
        for blocked_provider in self.BLOCKED_PROVIDERS:
            if blocked_provider.lower() in host.lower():
                issues.append(f"Blocked proxy provider detected: {blocked_provider}")
        
        return issues
    
    def _check_suspicious_config(self, config: Dict) -> List[str]:
        """Check for suspicious proxy configurations."""
        issues = []
        
        host = config["host"]
        port = config["port"]
        
        # Check for common free proxy patterns
        if any(keyword in host.lower() for keyword in ["free", "public", "open", "anonymous"]):
            issues.append("Suspicious host name suggests free/public proxy")
        
        # Check for residential proxy patterns (these should be from legitimate providers)
        if "residential" in host.lower() and not config.get("username"):
            issues.append("Residential proxy without authentication is suspicious")
        
        # Check for datacenter patterns
        if any(keyword in host.lower() for keyword in ["aws", "azure", "gcp", "digitalocean", "vultr"]):
            issues.append("Datacenter proxy detected - may have higher detection risk")
        
        return issues
    
    def _is_private_ip(self, host: str) -> bool:
        """Check if host is a private IP address."""
        try:
            # Simple regex check for private IP ranges
            private_patterns = [
                r'^10\.',                    # 10.0.0.0/8
                r'^172\.(1[6-9]|2[0-9]|3[01])\.',  # 172.16.0.0/12
                r'^192\.168\.',              # 192.168.0.0/16
                r'^169\.254\.',              # 169.254.0.0/16 (link-local)
            ]
            
            for pattern in private_patterns:
                if re.match(pattern, host):
                    return True
            
            return False
        except Exception:
            return False


class ProxyConfigLoader:
    """Loader for proxy configurations from various sources."""
    
    def __init__(self):
        """Initialize proxy config loader."""
        self.validator = ProxyConfigValidator()
    
    def load_from_file(self, file_path: str) -> Tuple[List[Dict], List[str]]:
        """
        Load proxy configurations from file.
        
        Args:
            file_path: Path to configuration file (JSON or YAML)
            
        Returns:
            Tuple of (valid_configs, error_messages)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return [], [f"Configuration file not found: {file_path}"]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    return [], [f"Unsupported file format: {file_path.suffix}"]
            
            return self._process_config_data(data)
            
        except Exception as e:
            return [], [f"Error loading configuration file: {e}"]
    
    def load_from_dict(self, config_data: Dict) -> Tuple[List[Dict], List[str]]:
        """
        Load proxy configurations from dictionary.
        
        Args:
            config_data: Configuration data dictionary
            
        Returns:
            Tuple of (valid_configs, error_messages)
        """
        return self._process_config_data(config_data)
    
    def load_from_url_list(self, proxy_urls: List[str]) -> Tuple[List[Dict], List[str]]:
        """
        Load proxy configurations from list of URLs.
        
        Args:
            proxy_urls: List of proxy URLs (e.g., "http://user:pass@host:port")
            
        Returns:
            Tuple of (valid_configs, error_messages)
        """
        configs = []
        errors = []
        
        for url in proxy_urls:
            try:
                config = self._parse_proxy_url(url)
                is_valid, issues = self.validator.validate_proxy_config(config)
                
                if is_valid:
                    configs.append(config)
                else:
                    errors.append(f"Invalid proxy URL {url}: {', '.join(issues)}")
                    
            except Exception as e:
                errors.append(f"Error parsing proxy URL {url}: {e}")
        
        return configs, errors
    
    def _process_config_data(self, data: Dict) -> Tuple[List[Dict], List[str]]:
        """Process configuration data and validate."""
        configs = []
        errors = []
        
        # Handle different configuration formats
        if "proxies" in data:
            proxy_list = data["proxies"]
        elif isinstance(data, list):
            proxy_list = data
        else:
            return [], ["Configuration must contain 'proxies' key or be a list"]
        
        if not isinstance(proxy_list, list):
            return [], ["Proxy configuration must be a list"]
        
        for i, proxy_config in enumerate(proxy_list):
            if not isinstance(proxy_config, dict):
                errors.append(f"Proxy config {i} must be a dictionary")
                continue
            
            is_valid, issues = self.validator.validate_proxy_config(proxy_config)
            
            if is_valid:
                configs.append(proxy_config)
            else:
                errors.append(f"Invalid proxy config {i}: {', '.join(issues)}")
        
        return configs, errors
    
    def _parse_proxy_url(self, url: str) -> Dict:
        """Parse proxy URL into configuration dictionary."""
        parsed = urlparse(url)
        
        if not parsed.hostname or not parsed.port:
            raise ValueError("URL must contain hostname and port")
        
        config = {
            "host": parsed.hostname,
            "port": parsed.port,
            "type": parsed.scheme or "http"
        }
        
        if parsed.username:
            config["username"] = parsed.username
        
        if parsed.password:
            config["password"] = parsed.password
        
        return config
    
    def create_proxy_objects(self, configs: List[Dict]) -> List[ProxyConfig]:
        """
        Create ProxyConfig objects from configuration dictionaries.
        
        Args:
            configs: List of validated configuration dictionaries
            
        Returns:
            List of ProxyConfig objects
        """
        proxy_objects = []
        
        for config in configs:
            proxy = ProxyConfig(
                host=config["host"],
                port=config["port"],
                username=config.get("username"),
                password=config.get("password"),
                proxy_type=config.get("type", "http")
            )
            proxy_objects.append(proxy)
        
        return proxy_objects


def load_proxy_configs_from_file(file_path: str) -> Tuple[List[ProxyConfig], List[str]]:
    """
    Convenience function to load and create proxy configs from file.
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        Tuple of (proxy_configs, error_messages)
    """
    loader = ProxyConfigLoader()
    configs, errors = loader.load_from_file(file_path)
    
    if errors:
        return [], errors
    
    proxy_objects = loader.create_proxy_objects(configs)
    return proxy_objects, []


def validate_proxy_config_dict(config: Dict) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate a single proxy configuration.
    
    Args:
        config: Proxy configuration dictionary
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    validator = ProxyConfigValidator()
    return validator.validate_proxy_config(config)