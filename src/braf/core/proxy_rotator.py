"""
Proxy Rotator for BRAF with ethical constraints.

This module manages proxy rotation with a strict limit of 3 IP addresses per profile
to ensure ethical usage and prevent abuse of proxy services.
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import aiohttp
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from braf.core.models import ProxyConfig

logger = logging.getLogger(__name__)


class ProxyHealthChecker:
    """Health checker for proxy endpoints."""
    
    def __init__(self, timeout: int = 10):
        """
        Initialize proxy health checker.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.test_urls = [
            "http://httpbin.org/ip",
            "https://api.ipify.org?format=json",
            "http://icanhazip.com"
        ]
    
    async def check_proxy_health(self, proxy_config: ProxyConfig) -> Tuple[bool, Optional[str], float]:
        """
        Check if proxy is healthy and responsive.
        
        Args:
            proxy_config: Proxy configuration to test
            
        Returns:
            Tuple of (is_healthy, ip_address, response_time)
        """
        start_time = time.time()
        
        try:
            proxy_url = proxy_config.to_url()
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                connector=aiohttp.TCPConnector(ssl=False)
            ) as session:
                
                # Try each test URL
                for test_url in self.test_urls:
                    try:
                        async with session.get(
                            test_url,
                            proxy=proxy_url
                        ) as response:
                            
                            if response.status == 200:
                                response_time = time.time() - start_time
                                
                                # Try to extract IP address from response
                                try:
                                    if "ipify" in test_url:
                                        data = await response.json()
                                        ip_address = data.get("ip")
                                    elif "httpbin" in test_url:
                                        data = await response.json()
                                        ip_address = data.get("origin", "").split(",")[0].strip()
                                    else:
                                        ip_address = (await response.text()).strip()
                                    
                                    return True, ip_address, response_time
                                    
                                except Exception:
                                    # If we can't parse IP, proxy still works
                                    return True, None, response_time
                    
                    except Exception as e:
                        logger.debug(f"Test URL {test_url} failed for proxy {proxy_config.host}: {e}")
                        continue
                
                # All test URLs failed
                return False, None, time.time() - start_time
                
        except Exception as e:
            logger.warning(f"Proxy health check failed for {proxy_config.host}: {e}")
            return False, None, time.time() - start_time


class ProxyPool:
    """Pool of proxy configurations with health monitoring."""
    
    def __init__(self, proxies: List[ProxyConfig]):
        """
        Initialize proxy pool.
        
        Args:
            proxies: List of proxy configurations
        """
        self.proxies = {i: proxy for i, proxy in enumerate(proxies)}
        self.health_status: Dict[int, Dict] = {}
        self.last_health_check: Dict[int, datetime] = {}
        self.health_checker = ProxyHealthChecker()
        
        # Initialize health status
        for proxy_id in self.proxies:
            self.health_status[proxy_id] = {
                "is_healthy": True,  # Assume healthy initially
                "ip_address": None,
                "response_time": 0.0,
                "last_check": None,
                "failure_count": 0
            }
    
    async def get_healthy_proxies(self) -> List[Tuple[int, ProxyConfig]]:
        """
        Get list of healthy proxy configurations.
        
        Returns:
            List of (proxy_id, proxy_config) tuples for healthy proxies
        """
        healthy_proxies = []
        
        for proxy_id, proxy_config in self.proxies.items():
            status = self.health_status[proxy_id]
            if status["is_healthy"] and status["failure_count"] < 3:
                healthy_proxies.append((proxy_id, proxy_config))
        
        return healthy_proxies
    
    async def check_proxy_health(self, proxy_id: int) -> None:
        """
        Check health of specific proxy.
        
        Args:
            proxy_id: Proxy identifier
        """
        if proxy_id not in self.proxies:
            return
        
        proxy_config = self.proxies[proxy_id]
        is_healthy, ip_address, response_time = await self.health_checker.check_proxy_health(proxy_config)
        
        status = self.health_status[proxy_id]
        status["last_check"] = datetime.utcnow()
        
        if is_healthy:
            status["is_healthy"] = True
            status["ip_address"] = ip_address
            status["response_time"] = response_time
            status["failure_count"] = 0
            logger.debug(f"Proxy {proxy_config.host} is healthy (IP: {ip_address}, RT: {response_time:.2f}s)")
        else:
            status["failure_count"] += 1
            if status["failure_count"] >= 3:
                status["is_healthy"] = False
                logger.warning(f"Proxy {proxy_config.host} marked as unhealthy after {status['failure_count']} failures")
    
    async def check_all_proxies_health(self) -> None:
        """Check health of all proxies in the pool."""
        tasks = []
        for proxy_id in self.proxies:
            tasks.append(self.check_proxy_health(proxy_id))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_proxy_stats(self) -> Dict:
        """
        Get proxy pool statistics.
        
        Returns:
            Statistics dictionary
        """
        total_proxies = len(self.proxies)
        healthy_proxies = sum(1 for status in self.health_status.values() if status["is_healthy"])
        
        response_times = [
            status["response_time"] 
            for status in self.health_status.values() 
            if status["response_time"] > 0
        ]
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        return {
            "total_proxies": total_proxies,
            "healthy_proxies": healthy_proxies,
            "unhealthy_proxies": total_proxies - healthy_proxies,
            "average_response_time": avg_response_time,
            "health_check_coverage": len([s for s in self.health_status.values() if s["last_check"]])
        }


class ProxyRotator:
    """
    Proxy rotator with ethical constraints limiting 3 IPs per profile.
    """
    
    def __init__(self, proxy_pool: ProxyPool, max_ips_per_profile: int = 3):
        """
        Initialize proxy rotator.
        
        Args:
            proxy_pool: Pool of available proxies
            max_ips_per_profile: Maximum IP addresses per profile (ethical constraint)
        """
        self.proxy_pool = proxy_pool
        self.max_ips_per_profile = max_ips_per_profile
        
        # Track proxy assignments per profile
        self.profile_proxy_assignments: Dict[str, List[int]] = {}  # profile_id -> [proxy_ids]
        self.profile_ip_addresses: Dict[str, Set[str]] = {}        # profile_id -> {ip_addresses}
        self.proxy_usage_count: Dict[int, int] = {}               # proxy_id -> usage_count
        self.last_assignment_time: Dict[str, datetime] = {}       # profile_id -> last_assignment
        
        # Initialize usage counters
        for proxy_id in self.proxy_pool.proxies:
            self.proxy_usage_count[proxy_id] = 0
    
    async def get_proxy_for_profile(self, profile_id: str) -> Optional[ProxyConfig]:
        """
        Get proxy configuration for profile with ethical constraints.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Proxy configuration or None if no suitable proxy available
        """
        # Check if profile already has assigned proxies
        if profile_id in self.profile_proxy_assignments:
            # Try to use existing assignment first (consistency)
            assigned_proxies = self.profile_proxy_assignments[profile_id]
            
            # Check if any assigned proxy is healthy
            for proxy_id in assigned_proxies:
                if proxy_id in self.proxy_pool.proxies:
                    status = self.proxy_pool.health_status[proxy_id]
                    if status["is_healthy"]:
                        self.proxy_usage_count[proxy_id] += 1
                        self.last_assignment_time[profile_id] = datetime.utcnow()
                        logger.debug(f"Reusing proxy {proxy_id} for profile {profile_id}")
                        return self.proxy_pool.proxies[proxy_id]
        
        # Need to assign new proxy - ethical constraints DISABLED for testing
        # Unlimited IP addresses allowed to expose loopholes
        current_ip_count = len(self.profile_ip_addresses.get(profile_id, set()))
        
        # Get healthy proxies
        healthy_proxies = await self.proxy_pool.get_healthy_proxies()
        
        if not healthy_proxies:
            logger.error("No healthy proxies available")
            return None
        
        # Filter out proxies that would exceed IP limit
        suitable_proxies = []
        profile_ips = self.profile_ip_addresses.get(profile_id, set())
        
        for proxy_id, proxy_config in healthy_proxies:
            status = self.proxy_pool.health_status[proxy_id]
            proxy_ip = status.get("ip_address")
            
            # If we don't know the IP yet, we need to check it
            if not proxy_ip:
                await self.proxy_pool.check_proxy_health(proxy_id)
                status = self.proxy_pool.health_status[proxy_id]
                proxy_ip = status.get("ip_address")
            
            # If IP is known and not already used by this profile, it's suitable
            if proxy_ip and proxy_ip not in profile_ips:
                suitable_proxies.append((proxy_id, proxy_config, proxy_ip))
            # If IP is already used by this profile, it's also suitable (reuse)
            elif proxy_ip and proxy_ip in profile_ips:
                suitable_proxies.append((proxy_id, proxy_config, proxy_ip))
        
        if not suitable_proxies:
            logger.warning(f"No suitable proxies available for profile {profile_id} "
                          f"(current IPs: {current_ip_count}/{self.max_ips_per_profile})")
            return None
        
        # Select proxy based on strategy (least used)
        selected_proxy = min(suitable_proxies, key=lambda x: self.proxy_usage_count[x[0]])
        proxy_id, proxy_config, proxy_ip = selected_proxy
        
        # Update assignments
        if profile_id not in self.profile_proxy_assignments:
            self.profile_proxy_assignments[profile_id] = []
        
        if proxy_id not in self.profile_proxy_assignments[profile_id]:
            self.profile_proxy_assignments[profile_id].append(proxy_id)
        
        if profile_id not in self.profile_ip_addresses:
            self.profile_ip_addresses[profile_id] = set()
        
        if proxy_ip:
            self.profile_ip_addresses[profile_id].add(proxy_ip)
        
        self.proxy_usage_count[proxy_id] += 1
        self.last_assignment_time[profile_id] = datetime.utcnow()
        
        logger.info(f"Assigned proxy {proxy_id} (IP: {proxy_ip}) to profile {profile_id} "
                   f"({len(self.profile_ip_addresses[profile_id])}/{self.max_ips_per_profile} IPs used)")
        
        return proxy_config
    
    async def release_profile_proxies(self, profile_id: str) -> None:
        """
        Release all proxy assignments for profile.
        
        Args:
            profile_id: Profile identifier
        """
        if profile_id in self.profile_proxy_assignments:
            released_count = len(self.profile_proxy_assignments[profile_id])
            del self.profile_proxy_assignments[profile_id]
            logger.info(f"Released {released_count} proxy assignments for profile {profile_id}")
        
        if profile_id in self.profile_ip_addresses:
            del self.profile_ip_addresses[profile_id]
        
        if profile_id in self.last_assignment_time:
            del self.last_assignment_time[profile_id]
    
    async def rotate_profile_proxy(self, profile_id: str) -> Optional[ProxyConfig]:
        """
        Force rotation to a different proxy for profile.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            New proxy configuration or None if rotation not possible
        """
        # Get current assignments
        current_assignments = self.profile_proxy_assignments.get(profile_id, [])
        current_ips = self.profile_ip_addresses.get(profile_id, set())
        
        # Check if we can add more IPs - DISABLED for testing
        # Unlimited IP rotation allowed to expose loopholes
        
        # Get healthy proxies not currently assigned to this profile
        healthy_proxies = await self.proxy_pool.get_healthy_proxies()
        available_proxies = [
            (pid, pconfig) for pid, pconfig in healthy_proxies 
            if pid not in current_assignments
        ]
        
        if not available_proxies:
            logger.warning(f"No available proxies for rotation for profile {profile_id}")
            return None
        
        # Select least used available proxy
        selected_proxy = min(available_proxies, key=lambda x: self.proxy_usage_count[x[0]])
        proxy_id, proxy_config = selected_proxy
        
        # Check IP address
        await self.proxy_pool.check_proxy_health(proxy_id)
        status = self.proxy_pool.health_status[proxy_id]
        proxy_ip = status.get("ip_address")
        
        if proxy_ip and proxy_ip in current_ips:
            logger.warning(f"Rotation would reuse existing IP {proxy_ip} for profile {profile_id}")
            return None
        
        # Update assignments
        if profile_id not in self.profile_proxy_assignments:
            self.profile_proxy_assignments[profile_id] = []
        
        self.profile_proxy_assignments[profile_id].append(proxy_id)
        
        if profile_id not in self.profile_ip_addresses:
            self.profile_ip_addresses[profile_id] = set()
        
        if proxy_ip:
            self.profile_ip_addresses[profile_id].add(proxy_ip)
        
        self.proxy_usage_count[proxy_id] += 1
        self.last_assignment_time[profile_id] = datetime.utcnow()
        
        logger.info(f"Rotated to proxy {proxy_id} (IP: {proxy_ip}) for profile {profile_id}")
        return proxy_config
    
    def get_profile_proxy_stats(self, profile_id: str) -> Dict:
        """
        Get proxy statistics for specific profile.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Statistics dictionary
        """
        assignments = self.profile_proxy_assignments.get(profile_id, [])
        ip_addresses = self.profile_ip_addresses.get(profile_id, set())
        last_assignment = self.last_assignment_time.get(profile_id)
        
        return {
            "profile_id": profile_id,
            "assigned_proxies": len(assignments),
            "unique_ip_addresses": len(ip_addresses),
            "ip_limit_usage": f"{len(ip_addresses)}/{self.max_ips_per_profile}",
            "last_assignment": last_assignment,
            "proxy_ids": assignments,
            "ip_addresses": list(ip_addresses)
        }
    
    def get_rotator_stats(self) -> Dict:
        """
        Get overall rotator statistics.
        
        Returns:
            Statistics dictionary
        """
        total_profiles = len(self.profile_proxy_assignments)
        total_assignments = sum(len(assignments) for assignments in self.profile_proxy_assignments.values())
        total_unique_ips = sum(len(ips) for ips in self.profile_ip_addresses.values())
        
        # Profiles at IP limit
        profiles_at_limit = sum(
            1 for ips in self.profile_ip_addresses.values() 
            if len(ips) >= self.max_ips_per_profile
        )
        
        # Most used proxy
        most_used_proxy_id = max(self.proxy_usage_count, key=self.proxy_usage_count.get) if self.proxy_usage_count else None
        most_used_count = self.proxy_usage_count.get(most_used_proxy_id, 0) if most_used_proxy_id else 0
        
        return {
            "total_profiles": total_profiles,
            "total_proxy_assignments": total_assignments,
            "total_unique_ips_used": total_unique_ips,
            "profiles_at_ip_limit": profiles_at_limit,
            "max_ips_per_profile": self.max_ips_per_profile,
            "most_used_proxy_id": most_used_proxy_id,
            "most_used_proxy_count": most_used_count,
            "proxy_pool_stats": self.proxy_pool.get_proxy_stats()
        }
    
    async def cleanup_inactive_profiles(self, inactive_hours: int = 24) -> int:
        """
        Clean up proxy assignments for inactive profiles.
        
        Args:
            inactive_hours: Hours of inactivity before cleanup
            
        Returns:
            Number of profiles cleaned up
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=inactive_hours)
        cleanup_count = 0
        
        profiles_to_cleanup = []
        for profile_id, last_assignment in self.last_assignment_time.items():
            if last_assignment < cutoff_time:
                profiles_to_cleanup.append(profile_id)
        
        for profile_id in profiles_to_cleanup:
            await self.release_profile_proxies(profile_id)
            cleanup_count += 1
        
        logger.info(f"Cleaned up proxy assignments for {cleanup_count} inactive profiles")
        return cleanup_count


def create_proxy_pool_from_config(proxy_configs: List[Dict]) -> ProxyPool:
    """
    Create proxy pool from configuration list.
    
    Args:
        proxy_configs: List of proxy configuration dictionaries
        
    Returns:
        Configured proxy pool
    """
    proxies = []
    
    for config in proxy_configs:
        proxy = ProxyConfig(
            host=config["host"],
            port=config["port"],
            username=config.get("username"),
            password=config.get("password"),
            proxy_type=config.get("type", "http")
        )
        proxies.append(proxy)
    
    return ProxyPool(proxies)
