"""
Configuration management for BRAF.

This module handles loading and validation of configuration from YAML files
and environment variables using Pydantic settings.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseSettings, Field, validator


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(..., env="DATABASE_URL")
    pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    class Config:
        env_prefix = "DATABASE_"


class RedisConfig(BaseSettings):
    """Redis configuration settings."""
    
    url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    
    class Config:
        env_prefix = "REDIS_"


class C2Config(BaseSettings):
    """Command & Control server configuration."""
    
    host: str = Field(default="0.0.0.0", env="C2_HOST")
    port: int = Field(default=8000, env="C2_PORT")
    debug: bool = Field(default=False, env="C2_DEBUG")
    reload: bool = Field(default=False, env="C2_RELOAD")
    workers: int = Field(default=1, env="C2_WORKERS")
    
    class Config:
        env_prefix = "C2_"


class WorkerConfig(BaseSettings):
    """Worker node configuration."""
    
    id: str = Field(default="worker-001", env="WORKER_ID")
    max_concurrent_tasks: int = Field(default=5, env="WORKER_MAX_TASKS")
    fingerprint_pool_size: int = Field(default=5, env="WORKER_FINGERPRINT_POOL_SIZE")
    proxy_pool_size: int = Field(default=3, env="WORKER_PROXY_POOL_SIZE")
    heartbeat_interval: int = Field(default=30, env="WORKER_HEARTBEAT_INTERVAL")
    task_timeout: int = Field(default=300, env="WORKER_TASK_TIMEOUT")
    
    class Config:
        env_prefix = "WORKER_"


class BehavioralConfig(BaseSettings):
    """Behavioral emulation configuration."""
    
    # Mouse movement settings
    mouse_bezier_points: int = Field(default=4)
    mouse_noise_factor: float = Field(default=0.5)
    mouse_min_velocity: float = Field(default=0.5)
    mouse_max_velocity: float = Field(default=1.5)
    
    # Typing settings
    typing_wpm_min: int = Field(default=40)
    typing_wpm_max: int = Field(default=80)
    typing_error_rate: float = Field(default=0.02)
    typing_correction_probability: float = Field(default=0.8)
    
    # Delay settings
    action_delay_min: float = Field(default=0.1)
    action_delay_max: float = Field(default=0.3)
    page_load_timeout: int = Field(default=30)
    element_wait_timeout: int = Field(default=10)


class FingerprintConfig(BaseSettings):
    """Fingerprint management configuration."""
    
    pool_size: int = Field(default=5)
    rotation_interval: int = Field(default=3600)  # seconds
    consistency_mode: str = Field(default="session")  # session, profile, random
    
    @validator('consistency_mode')
    def validate_consistency_mode(cls, v):
        if v not in ['session', 'profile', 'random']:
            raise ValueError('consistency_mode must be one of: session, profile, random')
        return v


class ProxyConfig(BaseSettings):
    """Proxy configuration settings."""
    
    enabled: bool = Field(default=True)
    max_per_profile: int = Field(default=3)
    rotation_strategy: str = Field(default="round_robin")
    health_check_interval: int = Field(default=300)
    
    @validator('rotation_strategy')
    def validate_rotation_strategy(cls, v):
        if v not in ['round_robin', 'random', 'least_used']:
            raise ValueError('rotation_strategy must be one of: round_robin, random, least_used')
        return v


class CaptchaConfig(BaseSettings):
    """CAPTCHA solving configuration."""
    
    primary_service: str = Field(default="2captcha")
    api_key: Optional[str] = Field(default=None, env="CAPTCHA_API_KEY")
    fallback_ocr: bool = Field(default=True)
    test_mode: bool = Field(default=True)
    timeout: int = Field(default=120)
    
    @validator('primary_service')
    def validate_primary_service(cls, v):
        if v not in ['2captcha', 'anticaptcha', 'deathbycaptcha']:
            raise ValueError('primary_service must be one of: 2captcha, anticaptcha, deathbycaptcha')
        return v


class ComplianceConfig(BaseSettings):
    """Compliance and ethical constraints configuration."""
    
    max_requests_per_hour: int = Field(default=100, env="MAX_REQUESTS_PER_HOUR")
    max_requests_per_day: int = Field(default=1000, env="MAX_REQUESTS_PER_DAY")
    ethical_constraints_enabled: bool = Field(default=False, env="ETHICAL_CONSTRAINTS_ENABLED")
    mandatory_logging: bool = Field(default=True)
    auto_shutdown_on_violation: bool = Field(default=True)
    
    # Rate limits
    rate_limit_per_domain: int = Field(default=10)  # per minute
    rate_limit_per_ip: int = Field(default=50)      # per hour
    rate_limit_global: int = Field(default=500)     # per hour


class SecurityConfig(BaseSettings):
    """Security configuration settings."""
    
    # Encryption settings
    encryption_algorithm: str = Field(default="PBKDF2")
    encryption_iterations: int = Field(default=100000)
    encryption_key_length: int = Field(default=32)
    
    # Vault settings
    vault_enabled: bool = Field(default=False, env="VAULT_ENABLED")
    vault_url: str = Field(default="http://localhost:8200", env="VAULT_URL")
    vault_token: Optional[str] = Field(default=None, env="VAULT_TOKEN")
    vault_mount_path: str = Field(default="secret", env="VAULT_MOUNT_PATH")
    
    # TLS settings
    tls_enabled: bool = Field(default=False)
    tls_cert_file: Optional[str] = Field(default=None)
    tls_key_file: Optional[str] = Field(default=None)
    
    # Keys
    secret_key: str = Field(..., env="BRAF_SECRET_KEY")
    auth_token: str = Field(..., env="BRAF_AUTH_TOKEN")


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration."""
    
    # Prometheus settings
    prometheus_enabled: bool = Field(default=True)
    prometheus_port: int = Field(default=8001, env="PROMETHEUS_PORT")
    prometheus_metrics_path: str = Field(default="/metrics")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json")
    
    # ELK settings
    elk_enabled: bool = Field(default=True)
    elk_elasticsearch_url: str = Field(default="http://localhost:9200", env="ELK_ELASTICSEARCH_URL")
    elk_index_prefix: str = Field(default="braf")


class BrowserConfig(BaseSettings):
    """Browser automation configuration."""
    
    engine: str = Field(default="playwright")
    headless: bool = Field(default=True)
    stealth_mode: bool = Field(default=True)
    user_data_dir: str = Field(default="./data/browser-profiles")
    
    # Viewport settings
    viewport_width: int = Field(default=1920)
    viewport_height: int = Field(default=1080)
    
    # Launch options
    launch_args: List[str] = Field(default_factory=lambda: [
        "--no-sandbox",
        "--disable-blink-features=AutomationControlled",
        "--disable-features=VizDisplayCompositor"
    ])


class BRAFConfig(BaseSettings):
    """Main BRAF configuration combining all sub-configurations."""
    
    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    c2: C2Config = Field(default_factory=C2Config)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    behavioral: BehavioralConfig = Field(default_factory=BehavioralConfig)
    fingerprints: FingerprintConfig = Field(default_factory=FingerprintConfig)
    proxies: ProxyConfig = Field(default_factory=ProxyConfig)
    captcha: CaptchaConfig = Field(default_factory=CaptchaConfig)
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    return config_data or {}


def create_config(config_path: Optional[str] = None) -> BRAFConfig:
    """
    Create BRAF configuration from file and environment variables.
    
    Args:
        config_path: Optional path to YAML configuration file
        
    Returns:
        Validated BRAF configuration
    """
    config_data = {}
    
    # Load from YAML file if provided
    if config_path:
        config_data = load_config_from_yaml(config_path)
    
    # Create nested configuration objects
    config_dict = {}
    
    for section_name in ['database', 'redis', 'c2', 'worker', 'behavioral', 
                        'fingerprints', 'proxies', 'captcha', 'compliance', 
                        'security', 'monitoring', 'browser']:
        
        section_data = config_data.get(section_name, {})
        
        # Get the appropriate config class
        config_class = {
            'database': DatabaseConfig,
            'redis': RedisConfig,
            'c2': C2Config,
            'worker': WorkerConfig,
            'behavioral': BehavioralConfig,
            'fingerprints': FingerprintConfig,
            'proxies': ProxyConfig,
            'captcha': CaptchaConfig,
            'compliance': ComplianceConfig,
            'security': SecurityConfig,
            'monitoring': MonitoringConfig,
            'browser': BrowserConfig,
        }[section_name]
        
        # Create config instance with YAML data and env vars
        config_dict[section_name] = config_class(**section_data)
    
    return BRAFConfig(**config_dict)


# Global configuration instance
_config: Optional[BRAFConfig] = None


def init_config(config_path: Optional[str] = None) -> BRAFConfig:
    """Initialize global configuration."""
    global _config
    _config = create_config(config_path)
    return _config


def get_config() -> BRAFConfig:
    """Get global configuration instance."""
    if _config is None:
        raise RuntimeError("Configuration not initialized. Call init_config() first.")
    return _config
