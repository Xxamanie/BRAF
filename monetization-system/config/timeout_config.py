"""
BRAF Timeout Configuration Module
Centralized timeout settings for all system components
"""

import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class TimeoutConfig:
    """Centralized timeout configuration for BRAF system"""
    
    # Database Timeouts (seconds)
    DB_CONNECTION_TIMEOUT: int = int(os.getenv('DB_CONNECTION_TIMEOUT', '30'))
    DB_QUERY_TIMEOUT: int = int(os.getenv('DB_QUERY_TIMEOUT', '60'))
    DB_POOL_TIMEOUT: int = int(os.getenv('DB_POOL_TIMEOUT', '30'))
    DB_POOL_RECYCLE: int = int(os.getenv('DB_POOL_RECYCLE', '3600'))
    
    # Redis Timeouts (seconds)
    REDIS_SOCKET_TIMEOUT: int = int(os.getenv('REDIS_SOCKET_TIMEOUT', '5'))
    REDIS_SOCKET_CONNECT_TIMEOUT: int = int(os.getenv('REDIS_SOCKET_CONNECT_TIMEOUT', '5'))
    REDIS_HEALTH_CHECK_INTERVAL: int = int(os.getenv('REDIS_HEALTH_CHECK_INTERVAL', '30'))
    
    # RabbitMQ/Celery Timeouts (seconds)
    CELERY_BROKER_CONNECTION_TIMEOUT: int = int(os.getenv('CELERY_BROKER_CONNECTION_TIMEOUT', '30'))
    CELERY_TASK_TIME_LIMIT: int = int(os.getenv('CELERY_TASK_TIME_LIMIT', '3600'))
    CELERY_TASK_SOFT_TIME_LIMIT: int = int(os.getenv('CELERY_TASK_SOFT_TIME_LIMIT', '3300'))
    CELERY_RESULT_EXPIRES: int = int(os.getenv('CELERY_RESULT_EXPIRES', '3600'))
    
    # HTTP Client Timeouts (seconds)
    HTTP_CLIENT_TIMEOUT: int = int(os.getenv('HTTP_CLIENT_TIMEOUT', '60'))
    HTTP_CLIENT_CONNECT_TIMEOUT: int = int(os.getenv('HTTP_CLIENT_CONNECT_TIMEOUT', '10'))
    HTTP_CLIENT_READ_TIMEOUT: int = int(os.getenv('HTTP_CLIENT_READ_TIMEOUT', '60'))
    
    # Web Server Timeouts (seconds)
    UVICORN_TIMEOUT_KEEP_ALIVE: int = int(os.getenv('UVICORN_TIMEOUT_KEEP_ALIVE', '75'))
    UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN: int = int(os.getenv('UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN', '30'))
    
    # Browser Automation Timeouts (seconds)
    BROWSER_PAGE_LOAD_TIMEOUT: int = int(os.getenv('BROWSER_PAGE_LOAD_TIMEOUT', '60'))
    BROWSER_SCRIPT_TIMEOUT: int = int(os.getenv('BROWSER_SCRIPT_TIMEOUT', '30'))
    BROWSER_IMPLICIT_WAIT: int = int(os.getenv('BROWSER_IMPLICIT_WAIT', '10'))
    BROWSER_EXPLICIT_WAIT: int = int(os.getenv('BROWSER_EXPLICIT_WAIT', '30'))
    BROWSER_ELEMENT_TIMEOUT: int = int(os.getenv('BROWSER_ELEMENT_TIMEOUT', '15'))
    
    # Task Execution Timeouts (seconds)
    TASK_EXECUTION_TIMEOUT: int = int(os.getenv('TASK_EXECUTION_TIMEOUT', '3600'))
    TASK_RETRY_TIMEOUT: int = int(os.getenv('TASK_RETRY_TIMEOUT', '300'))
    TASK_MAX_RETRIES: int = int(os.getenv('TASK_MAX_RETRIES', '3'))
    
    # API Timeouts (seconds)
    API_REQUEST_TIMEOUT: int = int(os.getenv('API_REQUEST_TIMEOUT', '30'))
    API_LONG_REQUEST_TIMEOUT: int = int(os.getenv('API_LONG_REQUEST_TIMEOUT', '300'))
    API_UPLOAD_TIMEOUT: int = int(os.getenv('API_UPLOAD_TIMEOUT', '600'))
    
    # Cryptocurrency API Timeouts (seconds)
    CRYPTO_API_TIMEOUT: int = int(os.getenv('CRYPTO_API_TIMEOUT', '30'))
    CRYPTO_TRANSACTION_TIMEOUT: int = int(os.getenv('CRYPTO_TRANSACTION_TIMEOUT', '300'))
    CRYPTO_CONFIRMATION_TIMEOUT: int = int(os.getenv('CRYPTO_CONFIRMATION_TIMEOUT', '1800'))
    
    # Payment Provider Timeouts (seconds)
    PAYMENT_API_TIMEOUT: int = int(os.getenv('PAYMENT_API_TIMEOUT', '60'))
    PAYMENT_PROCESSING_TIMEOUT: int = int(os.getenv('PAYMENT_PROCESSING_TIMEOUT', '300'))
    
    # File Operation Timeouts (seconds)
    FILE_UPLOAD_TIMEOUT: int = int(os.getenv('FILE_UPLOAD_TIMEOUT', '300'))
    FILE_DOWNLOAD_TIMEOUT: int = int(os.getenv('FILE_DOWNLOAD_TIMEOUT', '600'))
    
    # Monitoring Timeouts (seconds)
    HEALTH_CHECK_TIMEOUT: int = int(os.getenv('HEALTH_CHECK_TIMEOUT', '10'))
    METRICS_COLLECTION_TIMEOUT: int = int(os.getenv('METRICS_COLLECTION_TIMEOUT', '30'))
    
    # Research Operation Timeouts (seconds)
    RESEARCH_TASK_TIMEOUT: int = int(os.getenv('RESEARCH_TASK_TIMEOUT', '7200'))  # 2 hours
    RESEARCH_DATA_COLLECTION_TIMEOUT: int = int(os.getenv('RESEARCH_DATA_COLLECTION_TIMEOUT', '1800'))  # 30 minutes
    RESEARCH_ANALYSIS_TIMEOUT: int = int(os.getenv('RESEARCH_ANALYSIS_TIMEOUT', '3600'))  # 1 hour
    
    # Intelligence System Timeouts (seconds)
    INTELLIGENCE_PROCESSING_TIMEOUT: int = int(os.getenv('INTELLIGENCE_PROCESSING_TIMEOUT', '600'))
    INTELLIGENCE_MODEL_TIMEOUT: int = int(os.getenv('INTELLIGENCE_MODEL_TIMEOUT', '300'))
    
    # Network Timeouts (seconds)
    NETWORK_CONNECT_TIMEOUT: int = int(os.getenv('NETWORK_CONNECT_TIMEOUT', '10'))
    NETWORK_READ_TIMEOUT: int = int(os.getenv('NETWORK_READ_TIMEOUT', '60'))
    NETWORK_WRITE_TIMEOUT: int = int(os.getenv('NETWORK_WRITE_TIMEOUT', '60'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert timeout configuration to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database-specific timeout configuration"""
        return {
            'pool_timeout': self.DB_POOL_TIMEOUT,
            'pool_recycle': self.DB_POOL_RECYCLE,
            'connect_args': {
                'connect_timeout': self.DB_CONNECTION_TIMEOUT,
                'command_timeout': self.DB_QUERY_TIMEOUT,
            }
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis-specific timeout configuration"""
        return {
            'socket_timeout': self.REDIS_SOCKET_TIMEOUT,
            'socket_connect_timeout': self.REDIS_SOCKET_CONNECT_TIMEOUT,
            'health_check_interval': self.REDIS_HEALTH_CHECK_INTERVAL,
            'socket_keepalive': True,
            'socket_keepalive_options': {
                'TCP_KEEPIDLE': 60,
                'TCP_KEEPINTVL': 10,
                'TCP_KEEPCNT': 6
            }
        }
    
    def get_celery_config(self) -> Dict[str, Any]:
        """Get Celery-specific timeout configuration"""
        return {
            'broker_connection_timeout': self.CELERY_BROKER_CONNECTION_TIMEOUT,
            'task_time_limit': self.CELERY_TASK_TIME_LIMIT,
            'task_soft_time_limit': self.CELERY_TASK_SOFT_TIME_LIMIT,
            'result_expires': self.CELERY_RESULT_EXPIRES,
            'broker_connection_retry': True,
            'broker_connection_max_retries': 10,
            'worker_prefetch_multiplier': 4,
            'task_acks_late': True,
            'task_reject_on_worker_lost': True,
        }
    
    def get_http_client_config(self) -> Dict[str, Any]:
        """Get HTTP client timeout configuration"""
        return {
            'timeout': self.HTTP_CLIENT_TIMEOUT,
            'connect_timeout': self.HTTP_CLIENT_CONNECT_TIMEOUT,
            'read_timeout': self.HTTP_CLIENT_READ_TIMEOUT,
        }
    
    def get_browser_config(self) -> Dict[str, Any]:
        """Get browser automation timeout configuration"""
        return {
            'page_load_timeout': self.BROWSER_PAGE_LOAD_TIMEOUT,
            'script_timeout': self.BROWSER_SCRIPT_TIMEOUT,
            'implicit_wait': self.BROWSER_IMPLICIT_WAIT,
            'explicit_wait': self.BROWSER_EXPLICIT_WAIT,
            'element_timeout': self.BROWSER_ELEMENT_TIMEOUT,
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API timeout configuration"""
        return {
            'request_timeout': self.API_REQUEST_TIMEOUT,
            'long_request_timeout': self.API_LONG_REQUEST_TIMEOUT,
            'upload_timeout': self.API_UPLOAD_TIMEOUT,
        }
    
    def validate(self) -> bool:
        """Validate timeout configuration values"""
        errors = []
        
        # Check for reasonable timeout values
        if self.DB_CONNECTION_TIMEOUT < 5 or self.DB_CONNECTION_TIMEOUT > 300:
            errors.append("DB_CONNECTION_TIMEOUT should be between 5 and 300 seconds")
        
        if self.CELERY_TASK_SOFT_TIME_LIMIT >= self.CELERY_TASK_TIME_LIMIT:
            errors.append("CELERY_TASK_SOFT_TIME_LIMIT should be less than CELERY_TASK_TIME_LIMIT")
        
        if self.HTTP_CLIENT_CONNECT_TIMEOUT > self.HTTP_CLIENT_TIMEOUT:
            errors.append("HTTP_CLIENT_CONNECT_TIMEOUT should be less than HTTP_CLIENT_TIMEOUT")
        
        if self.BROWSER_IMPLICIT_WAIT > self.BROWSER_EXPLICIT_WAIT:
            errors.append("BROWSER_IMPLICIT_WAIT should be less than BROWSER_EXPLICIT_WAIT")
        
        if errors:
            raise ValueError(f"Timeout configuration validation failed: {'; '.join(errors)}")
        
        return True


# Global timeout configuration instance
timeout_config = TimeoutConfig()

# Validate configuration on import
try:
    timeout_config.validate()
except ValueError as e:
    print(f"Warning: {e}")

# Export commonly used timeout groups
DATABASE_TIMEOUTS = timeout_config.get_database_config()
REDIS_TIMEOUTS = timeout_config.get_redis_config()
CELERY_TIMEOUTS = timeout_config.get_celery_config()
HTTP_TIMEOUTS = timeout_config.get_http_client_config()
BROWSER_TIMEOUTS = timeout_config.get_browser_config()
API_TIMEOUTS = timeout_config.get_api_config()


def get_timeout_for_operation(operation_type: str) -> int:
    """Get timeout value for specific operation type"""
    timeout_map = {
        'database_query': timeout_config.DB_QUERY_TIMEOUT,
        'redis_operation': timeout_config.REDIS_SOCKET_TIMEOUT,
        'celery_task': timeout_config.CELERY_TASK_TIME_LIMIT,
        'http_request': timeout_config.HTTP_CLIENT_TIMEOUT,
        'browser_action': timeout_config.BROWSER_EXPLICIT_WAIT,
        'api_call': timeout_config.API_REQUEST_TIMEOUT,
        'crypto_transaction': timeout_config.CRYPTO_TRANSACTION_TIMEOUT,
        'payment_processing': timeout_config.PAYMENT_PROCESSING_TIMEOUT,
        'file_upload': timeout_config.FILE_UPLOAD_TIMEOUT,
        'research_task': timeout_config.RESEARCH_TASK_TIMEOUT,
        'intelligence_processing': timeout_config.INTELLIGENCE_PROCESSING_TIMEOUT,
    }
    
    return timeout_map.get(operation_type, 30)  # Default 30 seconds


def create_timeout_context(operation_type: str, custom_timeout: int = None):
    """Create a timeout context manager for operations"""
    import signal
    from contextlib import contextmanager
    
    @contextmanager
    def timeout_context():
        timeout_value = custom_timeout or get_timeout_for_operation(operation_type)
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation '{operation_type}' timed out after {timeout_value} seconds")
        
        # Set the signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_value)
        
        try:
            yield
        finally:
            # Restore the old signal handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    return timeout_context()


# Example usage functions
def apply_database_timeouts(engine_config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply database timeout configuration to SQLAlchemy engine config"""
    engine_config.update(DATABASE_TIMEOUTS)
    return engine_config


def apply_redis_timeouts(redis_config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply Redis timeout configuration to Redis client config"""
    redis_config.update(REDIS_TIMEOUTS)
    return redis_config


def apply_celery_timeouts(celery_config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply Celery timeout configuration to Celery app config"""
    celery_config.update(CELERY_TIMEOUTS)
    return celery_config


if __name__ == "__main__":
    # Print current timeout configuration
    print("BRAF Timeout Configuration:")
    print("=" * 50)
    
    config_dict = timeout_config.to_dict()
    for key, value in sorted(config_dict.items()):
        print(f"{key}: {value} seconds")
    
    print("\nConfiguration validation:", "PASSED" if timeout_config.validate() else "FAILED")
