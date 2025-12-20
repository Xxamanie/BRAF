"""
Error Handling and Recovery Systems for BRAF.

This module provides comprehensive error handling, exponential backoff retry logic,
graceful degradation, and circuit breaker patterns for external service calls.
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(str, Enum):
    """Recovery strategies for different error types."""
    
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADE = "graceful_degrade"
    ESCALATE = "escalate"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    
    error: Exception
    operation: str
    component: str
    timestamp: datetime
    severity: ErrorSeverity
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_strategy: str = "exponential"  # exponential, linear, fixed


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3  # For half-open state
    timeout: float = 30.0


class RetryableError(Exception):
    """Base class for retryable errors."""
    pass


class NonRetryableError(Exception):
    """Base class for non-retryable errors."""
    pass


class ServiceUnavailableError(RetryableError):
    """Service temporarily unavailable."""
    pass


class RateLimitError(RetryableError):
    """Rate limit exceeded."""
    pass


class AuthenticationError(NonRetryableError):
    """Authentication failed."""
    pass


class ValidationError(NonRetryableError):
    """Input validation failed."""
    pass


class ExponentialBackoff:
    """Implements exponential backoff with jitter."""
    
    def __init__(self, config: RetryConfig):
        """Initialize exponential backoff."""
        self.config = config
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.config.backoff_strategy == "fixed":
            delay = self.config.base_delay
        elif self.config.backoff_strategy == "linear":
            delay = self.config.base_delay * attempt
        else:  # exponential
            delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        """Initialize circuit breaker."""
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.next_attempt_time: Optional[datetime] = None
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
            else:
                raise ServiceUnavailableError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            # Execute function with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Handle success
            self._on_success()
            return result
            
        except Exception as e:
            # Handle failure
            self._on_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now(timezone.utc) - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker {self.name} reset to CLOSED state")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self, error: Exception):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker {self.name} failed in HALF_OPEN, returning to OPEN")
        elif (self.state == CircuitBreakerState.CLOSED and 
              self.failure_count >= self.config.failure_threshold):
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker {self.name} opened due to {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state information."""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'success_threshold': self.config.success_threshold,
                'timeout': self.config.timeout
            }
        }


class ErrorClassifier:
    """Classifies errors and determines appropriate recovery strategies."""
    
    def __init__(self):
        """Initialize error classifier."""
        self.error_mappings = {
            # Network errors - retryable
            ConnectionError: (ErrorSeverity.MEDIUM, RecoveryStrategy.RETRY),
            TimeoutError: (ErrorSeverity.MEDIUM, RecoveryStrategy.RETRY),
            asyncio.TimeoutError: (ErrorSeverity.MEDIUM, RecoveryStrategy.RETRY),
            
            # HTTP errors
            # 5xx errors - retryable
            # 4xx errors - usually not retryable
            
            # Custom errors
            ServiceUnavailableError: (ErrorSeverity.HIGH, RecoveryStrategy.CIRCUIT_BREAK),
            RateLimitError: (ErrorSeverity.MEDIUM, RecoveryStrategy.RETRY),
            AuthenticationError: (ErrorSeverity.HIGH, RecoveryStrategy.ESCALATE),
            ValidationError: (ErrorSeverity.LOW, RecoveryStrategy.ESCALATE),
            
            # System errors
            MemoryError: (ErrorSeverity.CRITICAL, RecoveryStrategy.GRACEFUL_DEGRADE),
            OSError: (ErrorSeverity.HIGH, RecoveryStrategy.RETRY),
        }
    
    def classify_error(self, error: Exception) -> tuple[ErrorSeverity, RecoveryStrategy]:
        """Classify error and determine recovery strategy."""
        error_type = type(error)
        
        # Check direct mapping
        if error_type in self.error_mappings:
            return self.error_mappings[error_type]
        
        # Check inheritance hierarchy
        for mapped_type, (severity, strategy) in self.error_mappings.items():
            if isinstance(error, mapped_type):
                return severity, strategy
        
        # Check HTTP status codes if available
        if hasattr(error, 'status') or hasattr(error, 'status_code'):
            status = getattr(error, 'status', None) or getattr(error, 'status_code', None)
            if status:
                return self._classify_http_error(status)
        
        # Default classification
        return ErrorSeverity.MEDIUM, RecoveryStrategy.RETRY
    
    def _classify_http_error(self, status_code: int) -> tuple[ErrorSeverity, RecoveryStrategy]:
        """Classify HTTP errors by status code."""
        if 500 <= status_code < 600:
            # Server errors - retryable
            return ErrorSeverity.HIGH, RecoveryStrategy.RETRY
        elif status_code == 429:
            # Rate limit - retryable with backoff
            return ErrorSeverity.MEDIUM, RecoveryStrategy.RETRY
        elif status_code in [401, 403]:
            # Authentication/authorization - not retryable
            return ErrorSeverity.HIGH, RecoveryStrategy.ESCALATE
        elif 400 <= status_code < 500:
            # Client errors - usually not retryable
            return ErrorSeverity.LOW, RecoveryStrategy.ESCALATE
        else:
            # Other errors
            return ErrorSeverity.MEDIUM, RecoveryStrategy.RETRY


class GracefulDegradation:
    """Implements graceful degradation strategies."""
    
    def __init__(self):
        """Initialize graceful degradation."""
        self.fallback_handlers: Dict[str, Callable] = {}
        self.degraded_services: Set[str] = set()
        self.service_levels: Dict[str, str] = {}
    
    def register_fallback(self, service: str, fallback_handler: Callable):
        """Register fallback handler for a service."""
        self.fallback_handlers[service] = fallback_handler
        logger.info(f"Registered fallback handler for service: {service}")
    
    async def execute_with_fallback(
        self,
        service: str,
        primary_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with fallback on failure."""
        try:
            # Try primary function
            result = await primary_func(*args, **kwargs)
            
            # Mark service as healthy if it was degraded
            if service in self.degraded_services:
                self.degraded_services.remove(service)
                self.service_levels[service] = "normal"
                logger.info(f"Service {service} recovered to normal operation")
            
            return result
            
        except Exception as e:
            logger.warning(f"Primary function failed for service {service}: {e}")
            
            # Mark service as degraded
            self.degraded_services.add(service)
            self.service_levels[service] = "degraded"
            
            # Try fallback if available
            if service in self.fallback_handlers:
                try:
                    logger.info(f"Using fallback for service: {service}")
                    return await self.fallback_handlers[service](*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for service {service}: {fallback_error}")
                    raise
            else:
                # No fallback available
                raise ServiceUnavailableError(f"Service {service} unavailable and no fallback configured")
    
    def get_service_status(self) -> Dict[str, str]:
        """Get status of all services."""
        return self.service_levels.copy()
    
    def is_service_degraded(self, service: str) -> bool:
        """Check if service is currently degraded."""
        return service in self.degraded_services


class ErrorRecoveryManager:
    """Main error recovery manager coordinating all recovery strategies."""
    
    def __init__(self):
        """Initialize error recovery manager."""
        self.error_classifier = ErrorClassifier()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.graceful_degradation = GracefulDegradation()
        self.error_history: List[ErrorContext] = []
        self.recovery_handlers: Dict[RecoveryStrategy, Callable] = {}
        
        # Setup default recovery handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default recovery handlers."""
        self.recovery_handlers[RecoveryStrategy.RETRY] = self._handle_retry
        self.recovery_handlers[RecoveryStrategy.FALLBACK] = self._handle_fallback
        self.recovery_handlers[RecoveryStrategy.CIRCUIT_BREAK] = self._handle_circuit_break
        self.recovery_handlers[RecoveryStrategy.GRACEFUL_DEGRADE] = self._handle_graceful_degrade
        self.recovery_handlers[RecoveryStrategy.ESCALATE] = self._handle_escalate
    
    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig):
        """Register circuit breaker for a service."""
        self.circuit_breakers[name] = CircuitBreaker(name, config)
        logger.info(f"Registered circuit breaker: {name}")
    
    def register_fallback(self, service: str, fallback_handler: Callable):
        """Register fallback handler."""
        self.graceful_degradation.register_fallback(service, fallback_handler)
    
    async def execute_with_recovery(
        self,
        func: Callable,
        operation: str,
        component: str,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_name: Optional[str] = None,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with comprehensive error recovery."""
        retry_config = retry_config or RetryConfig()
        backoff = ExponentialBackoff(retry_config)
        
        for attempt in range(1, retry_config.max_attempts + 1):
            try:
                # Use circuit breaker if specified
                if circuit_breaker_name and circuit_breaker_name in self.circuit_breakers:
                    circuit_breaker = self.circuit_breakers[circuit_breaker_name]
                    return await circuit_breaker.call(func, *args, **kwargs)
                else:
                    return await func(*args, **kwargs)
                    
            except Exception as e:
                # Create error context
                error_context = ErrorContext(
                    error=e,
                    operation=operation,
                    component=component,
                    timestamp=datetime.now(timezone.utc),
                    severity=ErrorSeverity.MEDIUM,  # Will be updated by classifier
                    retry_count=attempt,
                    metadata={'args': str(args), 'kwargs': str(kwargs)}
                )
                
                # Classify error
                severity, strategy = self.error_classifier.classify_error(e)
                error_context.severity = severity
                
                # Log error
                self.error_history.append(error_context)
                logger.warning(
                    f"Error in {component}.{operation} (attempt {attempt}): {e} "
                    f"(severity: {severity.value}, strategy: {strategy.value})"
                )
                
                # Handle non-retryable errors immediately
                if isinstance(e, NonRetryableError) or attempt == retry_config.max_attempts:
                    await self._handle_error(error_context, strategy)
                    raise
                
                # Apply recovery strategy
                if strategy == RecoveryStrategy.RETRY:
                    delay = backoff.calculate_delay(attempt)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Handle other strategies
                    await self._handle_error(error_context, strategy)
                    raise
        
        # Should not reach here
        raise RuntimeError("Unexpected end of retry loop")
    
    async def _handle_error(self, error_context: ErrorContext, strategy: RecoveryStrategy):
        """Handle error using specified strategy."""
        if strategy in self.recovery_handlers:
            await self.recovery_handlers[strategy](error_context)
        else:
            logger.error(f"No handler for recovery strategy: {strategy.value}")
    
    async def _handle_retry(self, error_context: ErrorContext):
        """Handle retry strategy."""
        # Retry logic is handled in execute_with_recovery
        pass
    
    async def _handle_fallback(self, error_context: ErrorContext):
        """Handle fallback strategy."""
        # Fallback logic would be implemented here
        logger.info(f"Attempting fallback for {error_context.component}.{error_context.operation}")
    
    async def _handle_circuit_break(self, error_context: ErrorContext):
        """Handle circuit breaker strategy."""
        # Circuit breaker logic is handled in execute_with_recovery
        pass
    
    async def _handle_graceful_degrade(self, error_context: ErrorContext):
        """Handle graceful degradation strategy."""
        logger.warning(f"Gracefully degrading service due to: {error_context.error}")
        # Implementation would reduce service functionality
    
    async def _handle_escalate(self, error_context: ErrorContext):
        """Handle escalation strategy."""
        logger.error(f"Escalating error: {error_context.error}")
        # Implementation would notify administrators, trigger alerts, etc.
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        recent_errors = [
            error for error in self.error_history
            if datetime.now(timezone.utc) - error.timestamp <= timedelta(hours=24)
        ]
        
        error_counts = {}
        for error in recent_errors:
            error_type = type(error.error).__name__
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            'total_errors_24h': len(recent_errors),
            'error_types': error_counts,
            'circuit_breaker_states': {
                name: cb.get_state() for name, cb in self.circuit_breakers.items()
            },
            'degraded_services': list(self.graceful_degradation.degraded_services),
            'service_status': self.graceful_degradation.get_service_status()
        }


# Decorator for automatic error handling
def with_error_recovery(
    operation: str,
    component: str,
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker_name: Optional[str] = None
):
    """Decorator to add error recovery to functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            recovery_manager = get_error_recovery_manager()
            if recovery_manager:
                return await recovery_manager.execute_with_recovery(
                    func, operation, component, retry_config, circuit_breaker_name,
                    *args, **kwargs
                )
            else:
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# Global error recovery manager instance
_error_recovery_manager: Optional[ErrorRecoveryManager] = None


def get_error_recovery_manager() -> Optional[ErrorRecoveryManager]:
    """
    Get global error recovery manager instance.
    
    Returns:
        Error recovery manager instance or None if not initialized
    """
    return _error_recovery_manager


def init_error_recovery_manager() -> ErrorRecoveryManager:
    """
    Initialize global error recovery manager.
    
    Returns:
        Initialized error recovery manager
    """
    global _error_recovery_manager
    
    _error_recovery_manager = ErrorRecoveryManager()
    
    # Register default circuit breakers
    _error_recovery_manager.register_circuit_breaker(
        "captcha_service",
        CircuitBreakerConfig(failure_threshold=3, recovery_timeout=120.0)
    )
    
    _error_recovery_manager.register_circuit_breaker(
        "proxy_service",
        CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60.0)
    )
    
    _error_recovery_manager.register_circuit_breaker(
        "browser_service",
        CircuitBreakerConfig(failure_threshold=3, recovery_timeout=180.0)
    )
    
    return _error_recovery_manager