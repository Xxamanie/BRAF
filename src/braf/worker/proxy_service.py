"""
Proxy Service for BRAF worker nodes.

This service provides high-level proxy management integrated with the profile system,
ensuring ethical constraints and proper proxy assignment coordination.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from braf.core.models import ProxyConfig
from braf.core.proxy_rotator import ProxyPool, ProxyRotator, create_proxy_pool_from_config

logger = logging.getLogger(__name__)


class ProxyService:
    """High-level proxy management service for worker nodes."""
    
    def __init__(self, proxy_configs: List[Dict], max_ips_per_profile: int = 3):
        """
        Initialize proxy service.
        
        Args:
            proxy_configs: List of proxy configuration dictionaries
            max_ips_per_profile: Maximum IP addresses per profile (ethical constraint)
        """
        self.proxy_pool = create_proxy_pool_from_config(proxy_configs)
        self.proxy_rotator = ProxyRotator(self.proxy_pool, max_ips_per_profile)
        self.health_check_interval = 300  # 5 minutes
        self.health_check_task: Optional[asyncio.Task] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize proxy service and start health monitoring."""
        if self._initialized:
            return
        
        # Initial health check
        await self.proxy_pool.check_all_proxies_health()
        
        # Start background health monitoring
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        self._initialized = True
        logger.info(f"Proxy service initialized with {len(self.proxy_pool.proxies)} proxies")
    
    async def shutdown(self) -> None:
        """Shutdown proxy service and stop health monitoring."""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        self._initialized = False
        logger.info("Proxy service shutdown")
    
    async def get_proxy_for_profile(self, profile_id: str) -> Optional[ProxyConfig]:
        """
        Get proxy configuration for profile with ethical constraints.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Proxy configuration or None if constraints prevent assignment
        """
        if not self._initialized:
            await self.initialize()
        
        return await self.proxy_rotator.get_proxy_for_profile(profile_id)
    
    async def rotate_profile_proxy(self, profile_id: str) -> Optional[ProxyConfig]:
        """
        Force proxy rotation for profile (emergency use).
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            New proxy configuration or None if rotation not possible
        """
        if not self._initialized:
            await self.initialize()
        
        return await self.proxy_rotator.rotate_profile_proxy(profile_id)
    
    async def release_profile_proxies(self, profile_id: str) -> None:
        """
        Release all proxy assignments for profile.
        
        Args:
            profile_id: Profile identifier
        """
        await self.proxy_rotator.release_profile_proxies(profile_id)
    
    async def check_profile_proxy_limits(self, profile_id: str) -> Dict:
        """
        Check proxy usage limits for profile.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Limit status dictionary
        """
        stats = self.proxy_rotator.get_profile_proxy_stats(profile_id)
        
        return {
            "profile_id": profile_id,
            "current_ips": stats["unique_ip_addresses"],
            "max_ips": self.proxy_rotator.max_ips_per_profile,
            "at_limit": stats["unique_ip_addresses"] >= self.proxy_rotator.max_ips_per_profile,
            "can_rotate": stats["unique_ip_addresses"] < self.proxy_rotator.max_ips_per_profile,
            "assigned_proxies": stats["assigned_proxies"]
        }
    
    async def get_proxy_health_status(self) -> Dict:
        """
        Get health status of all proxies.
        
        Returns:
            Health status dictionary
        """
        pool_stats = self.proxy_pool.get_proxy_stats()
        
        proxy_details = []
        for proxy_id, proxy_config in self.proxy_pool.proxies.items():
            status = self.proxy_pool.health_status[proxy_id]
            
            proxy_details.append({
                "proxy_id": proxy_id,
                "host": proxy_config.host,
                "port": proxy_config.port,
                "is_healthy": status["is_healthy"],
                "ip_address": status.get("ip_address"),
                "response_time": status.get("response_time", 0.0),
                "failure_count": status.get("failure_count", 0),
                "last_check": status.get("last_check"),
                "usage_count": self.proxy_rotator.proxy_usage_count.get(proxy_id, 0)
            })
        
        return {
            "pool_stats": pool_stats,
            "proxy_details": proxy_details
        }
    
    async def get_service_stats(self) -> Dict:
        """
        Get comprehensive proxy service statistics.
        
        Returns:
            Service statistics dictionary
        """
        rotator_stats = self.proxy_rotator.get_rotator_stats()
        health_status = await self.get_proxy_health_status()
        
        return {
            "service_initialized": self._initialized,
            "health_check_interval": self.health_check_interval,
            "rotator_stats": rotator_stats,
            "health_status": health_status["pool_stats"],
            "ethical_compliance": {
                "max_ips_per_profile": self.proxy_rotator.max_ips_per_profile,
                "profiles_at_limit": rotator_stats["profiles_at_ip_limit"],
                "total_profiles": rotator_stats["total_profiles"],
                "compliance_rate": (
                    1.0 - (rotator_stats["profiles_at_ip_limit"] / max(1, rotator_stats["total_profiles"]))
                ) * 100
            }
        }
    
    async def validate_ethical_constraints(self) -> Dict:
        """
        Validate that ethical constraints are being enforced.
        
        Returns:
            Validation results dictionary
        """
        violations = []
        warnings = []
        
        # Check each profile's IP usage
        for profile_id in self.proxy_rotator.profile_ip_addresses:
            ip_count = len(self.proxy_rotator.profile_ip_addresses[profile_id])
            
            if ip_count > self.proxy_rotator.max_ips_per_profile:
                violations.append({
                    "profile_id": profile_id,
                    "violation": "IP_LIMIT_EXCEEDED",
                    "current_ips": ip_count,
                    "max_allowed": self.proxy_rotator.max_ips_per_profile
                })
            elif ip_count == self.proxy_rotator.max_ips_per_profile:
                warnings.append({
                    "profile_id": profile_id,
                    "warning": "AT_IP_LIMIT",
                    "current_ips": ip_count,
                    "max_allowed": self.proxy_rotator.max_ips_per_profile
                })
        
        return {
            "is_compliant": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "total_profiles_checked": len(self.proxy_rotator.profile_ip_addresses),
            "violation_count": len(violations),
            "warning_count": len(warnings)
        }
    
    async def emergency_cleanup(self) -> Dict:
        """
        Emergency cleanup of proxy assignments (compliance enforcement).
        
        Returns:
            Cleanup results dictionary
        """
        cleanup_results = {
            "profiles_cleaned": 0,
            "violations_fixed": 0,
            "inactive_profiles_cleaned": 0
        }
        
        # Fix IP limit violations
        violations_to_fix = []
        for profile_id in self.proxy_rotator.profile_ip_addresses:
            ip_count = len(self.proxy_rotator.profile_ip_addresses[profile_id])
            if ip_count > self.proxy_rotator.max_ips_per_profile:
                violations_to_fix.append(profile_id)
        
        for profile_id in violations_to_fix:
            await self.proxy_rotator.release_profile_proxies(profile_id)
            cleanup_results["violations_fixed"] += 1
            cleanup_results["profiles_cleaned"] += 1
        
        # Clean up inactive profiles
        inactive_cleaned = await self.proxy_rotator.cleanup_inactive_profiles(inactive_hours=1)
        cleanup_results["inactive_profiles_cleaned"] = inactive_cleaned
        cleanup_results["profiles_cleaned"] += inactive_cleaned
        
        logger.warning(f"Emergency cleanup completed: {cleanup_results}")
        return cleanup_results
    
    async def _health_check_loop(self) -> None:
        """Background task for periodic proxy health checks."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                logger.debug("Starting periodic proxy health check")
                await self.proxy_pool.check_all_proxies_health()
                
                # Log health summary
                stats = self.proxy_pool.get_proxy_stats()
                logger.info(f"Proxy health check completed: "
                           f"{stats['healthy_proxies']}/{stats['total_proxies']} healthy, "
                           f"avg response time: {stats['average_response_time']:.2f}s")
                
            except asyncio.CancelledError:
                logger.info("Proxy health check loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in proxy health check loop: {e}")
                # Continue the loop despite errors


class ProxyServiceManager:
    """Manager for proxy service lifecycle and configuration."""
    
    @staticmethod
    def create_from_config(config: Dict) -> ProxyService:
        """
        Create proxy service from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured proxy service
        """
        proxy_configs = config.get("proxies", [])
        max_ips_per_profile = config.get("max_ips_per_profile", 3)
        
        if not proxy_configs:
            logger.warning("No proxy configurations provided, creating empty service")
            proxy_configs = []
        
        service = ProxyService(proxy_configs, max_ips_per_profile)
        
        # Configure health check interval if specified
        if "health_check_interval" in config:
            service.health_check_interval = config["health_check_interval"]
        
        return service
    
    @staticmethod
    def create_test_service() -> ProxyService:
        """
        Create proxy service for testing with mock proxies.
        
        Returns:
            Test proxy service
        """
        test_proxies = [
            {
                "host": "proxy1.test.com",
                "port": 8080,
                "username": "test_user",
                "password": "test_pass",
                "type": "http"
            },
            {
                "host": "proxy2.test.com", 
                "port": 8080,
                "type": "http"
            },
            {
                "host": "proxy3.test.com",
                "port": 1080,
                "type": "socks5"
            }
        ]
        
        return ProxyService(test_proxies, max_ips_per_profile=3)


# Global proxy service instance
_proxy_service: Optional[ProxyService] = None


def get_proxy_service() -> Optional[ProxyService]:
    """
    Get global proxy service instance.
    
    Returns:
        Proxy service instance or None if not initialized
    """
    return _proxy_service


async def init_proxy_service(proxy_configs: List[Dict], max_ips_per_profile: int = 3) -> ProxyService:
    """
    Initialize global proxy service.
    
    Args:
        proxy_configs: List of proxy configuration dictionaries
        max_ips_per_profile: Maximum IP addresses per profile
        
    Returns:
        Initialized proxy service
    """
    global _proxy_service
    
    _proxy_service = ProxyService(proxy_configs, max_ips_per_profile)
    await _proxy_service.initialize()
    
    return _proxy_service


async def shutdown_proxy_service() -> None:
    """Shutdown global proxy service."""
    global _proxy_service
    
    if _proxy_service:
        await _proxy_service.shutdown()
        _proxy_service = None
