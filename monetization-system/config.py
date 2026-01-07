import os
from dotenv import load_dotenv
from typing import Dict, List

# Load environment variables
load_dotenv()

class Config:
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/braf_monetization")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Application Configuration
    BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "jwt-secret-key")
    
    # Stripe Configuration
    STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
    STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY")
    STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
    
    # Mobile Money APIs
    OPAY_API_KEY = os.getenv("OPAY_API_KEY")
    OPAY_SECRET_KEY = os.getenv("OPAY_SECRET_KEY")
    PALMPAY_API_KEY = os.getenv("PALMPAY_API_KEY")
    PALMPAY_SECRET_KEY = os.getenv("PALMPAY_SECRET_KEY")
    
    # Mobile Money Secrets Dictionary
    MOBILE_MONEY_SECRETS = {
        "opay_ng": os.getenv("OPAY_SECRET_KEY"),
        "opay_ke": os.getenv("OPAY_SECRET_KEY"),
        "opay_gh": os.getenv("OPAY_SECRET_KEY"),
        "palmpay_ng": os.getenv("PALMPAY_SECRET_KEY")
    }
    
    # Cryptocurrency Configuration
    TRON_PRIVATE_KEY = os.getenv("TRON_PRIVATE_KEY")
    ETH_PRIVATE_KEY = os.getenv("ETH_PRIVATE_KEY")
    BTC_WALLET = os.getenv("BTC_WALLET")
    INFURA_PROJECT_ID = os.getenv("INFURA_PROJECT_ID")
    
    # Blockchain Networks
    TRON_NETWORK = os.getenv("TRON_NETWORK", "mainnet")
    ETH_NETWORK = os.getenv("ETH_NETWORK", "mainnet")
    BTC_NETWORK = os.getenv("BTC_NETWORK", "mainnet")
    GAS_PRICE_GWEI = int(os.getenv("GAS_PRICE_GWEI", "50"))
    CONFIRMATION_BLOCKS = int(os.getenv("CONFIRMATION_BLOCKS", "12"))
    
    # SSL Configuration
    SSL_CERT_PATH = os.getenv("SSL_CERT_PATH")
    SSL_KEY_PATH = os.getenv("SSL_KEY_PATH")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "1000"))
    MAX_WITHDRAWALS_PER_DAY = int(os.getenv("MAX_WITHDRAWALS_PER_DAY", "10"))
    
    # Monitoring
    PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
    GRAFANA_ENABLED = os.getenv("GRAFANA_ENABLED", "true").lower() == "true"
    SENTRY_DSN = os.getenv("SENTRY_DSN")
    
    # Email Configuration
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
    
    # Webhook URLs
    OPAY_WEBHOOK_URL = os.getenv("OPAY_WEBHOOK_URL", f"{BASE_URL}/api/v1/webhooks/opay")
    PALMPAY_WEBHOOK_URL = os.getenv("PALMPAY_WEBHOOK_URL", f"{BASE_URL}/api/v1/webhooks/palmpay")
    STRIPE_WEBHOOK_URL = os.getenv("STRIPE_WEBHOOK_URL", f"{BASE_URL}/api/v1/webhooks/stripe")
    
    # Platform Credentials (Base64 encoded for security)
    SWAGBUCKS_ACCOUNTS = os.getenv("SWAGBUCKS_ACCOUNTS")
    SURVEYJUNKIE_ACCOUNTS = os.getenv("SURVEYJUNKIE_ACCOUNTS")
    YOUTUBE_ACCOUNTS = os.getenv("YOUTUBE_ACCOUNTS")
    
    # Beta/Monetization Settings
    FREE_BETA_MODE = os.getenv("FREE_BETA_MODE", "false").lower() == "true"
    ENABLE_PAID_TIERS = os.getenv("ENABLE_PAID_TIERS", "false").lower() == "true"
    
    # Compliance Settings - REMOVED FOR UNLIMITED YIELD
    # MAX_DAILY_EARNINGS = int(os.getenv("MAX_DAILY_EARNINGS", "2000"))
    COMPLIANCE_CHECK_INTERVAL = 0  # Disabled compliance checks
    RISK_THRESHOLD_HIGH = 1.0  # Accept all risks
    RISK_THRESHOLD_MEDIUM = 1.0

    # Withdrawal Limits - REMOVED FOR UNLIMITED WITHDRAWALS
    MIN_WITHDRAWAL = {
        "opay": 0,  # No minimum
        "palmpay": 0,
        "crypto": 0
    }

    # API Rate Limits - REMOVED FOR UNLIMITED SCALING
    # RATE_LIMITS = {
    #     "withdrawal": f"{MAX_WITHDRAWALS_PER_DAY}/day",
    #     "api_calls": f"{MAX_REQUESTS_PER_MINUTE}/minute"
    # }

    # Unlimited scaling configuration
    UNLIMITED_MODE = True
    MAX_CONCURRENT_SESSIONS = 10000  # Support massive scaling
    
    # Subscription Tiers (FREE BETA - All tiers are free during beta period)
    SUBSCRIPTION_TIERS = {
        "free": {
            "price": 0,  # FREE during beta
            "templates": ["survey", "video", "content"],
            "max_automations": 100,  # Unlimited during beta
            "daily_earnings_limit": 1000,  # High limit during beta
            "features": ["full_access", "email_notifications", "api_access", "priority_support"]
        },
        # FUTURE PAID TIERS (Currently disabled - uncomment to re-enable)
        # "basic": {
        #     "price": 9900,  # $99 in cents
        #     "templates": ["survey"],
        #     "max_automations": 5,
        #     "daily_earnings_limit": 50,
        #     "features": ["basic_support", "email_notifications"]
        # },
        # "pro": {
        #     "price": 29900,  # $299 in cents
        #     "templates": ["survey", "video", "content"],
        #     "max_automations": 20,
        #     "daily_earnings_limit": 200,
        #     "features": ["priority_support", "advanced_analytics", "api_access"]
        # },
        # "enterprise": {
        #     "price": 99900,  # $999 in cents
        #     "templates": ["survey", "video", "content"],
        #     "max_automations": 100,
        #     "daily_earnings_limit": 1000,
        #     "features": ["dedicated_support", "custom_integrations", "white_label", "sla"]
        # }
    }
    
    # Browser Configuration
    CHROME_BIN = os.getenv("CHROME_BIN", "/usr/bin/chromium")
    CHROMEDRIVER_PATH = os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")
    
    # Celery Configuration
    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = REDIS_URL
    CELERY_TASK_SERIALIZER = "json"
    CELERY_ACCEPT_CONTENT = ["json"]
    CELERY_RESULT_SERIALIZER = "json"
    CELERY_TIMEZONE = "UTC"
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get database URL with proper formatting"""
        return cls.DATABASE_URL
    
    @classmethod
    def get_redis_url(cls) -> str:
        """Get Redis URL with proper formatting"""
        return cls.REDIS_URL
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment"""
        return cls.ENVIRONMENT.lower() == "production"
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development environment"""
        return cls.ENVIRONMENT.lower() == "development"
    
    @classmethod
    def get_subscription_tier(cls, tier_name: str) -> Dict:
        """Get subscription tier configuration"""
        return cls.SUBSCRIPTION_TIERS.get(tier_name, cls.SUBSCRIPTION_TIERS["basic"])
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate configuration and return list of missing required variables"""
        required_vars = []
        
        if not cls.SECRET_KEY or cls.SECRET_KEY == "your-secret-key-change-in-production":
            required_vars.append("SECRET_KEY")
        
        if cls.is_production():
            production_required = [
                "STRIPE_SECRET_KEY",
                "DATABASE_URL",
                "REDIS_URL",
                "ENCRYPTION_KEY"
            ]
            
            for var in production_required:
                if not getattr(cls, var, None):
                    required_vars.append(var)
        
        return required_vars
