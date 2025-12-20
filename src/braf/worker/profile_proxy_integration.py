"""
Integration module for Profile and Proxy services.

This module provides coordinated management of profiles and proxies,
ensuring consistent fingerprint-proxy combinations and ethical constraints.
"""

import logging
from typing import Dict, Optional, Tuple

from braf.core.models import BrowserFingerprint, ProxyConfig
from braf.worker.profile_service import get_profile_service
from braf.worker.proxy_service import get_proxy_service

logger = logging.getLogger(__name__)


class ProfileProxyCoordinator:
    """Coordinator for profile and proxy management."""
    
    def __init__(self):
        """Initialize profile-proxy coordinator."""
        self._active_sessions: Dict[str, Dict] = {}  # profile_id -> session_info
    
    async def start_coordinated_session(self, profile_id: str) -> Optional[Dict]:
        """
        Start a coordinated session with both profile and proxy assignment.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Session data with fingerprint and proxy, or None if failed
        """
        profile_service = get_profile_service()
        proxy_service = get_proxy_service()
        
        if not profile_service:
            logger.error("Profile service not initialized")
            return None
        
        try:
            # Start profile session (gets fingerprint)
            session_data = await profile_service.start_profile_session(profile_id)
            
            # Get proxy for profile (with ethical constraints)
            proxy_config = None
            if proxy_service:
                proxy_config = await proxy_service.get_proxy_for_profile(profile_id)
                
                if not proxy_config:
                    # Check if it's due to ethical constraints
                    limits = await proxy_service.check_profile_proxy_limits(profile_id)
                    if limits["at_limit"]:
                        logger.warning(f"Profile {profile_id} has reached IP limit, no proxy assigned")
                    else:
                        logger.warning(f"No healthy proxies available for profile {profile_id}")
            else:
                logger.info("Proxy service not available, session will run without proxy")
            
            # Update session data with proxy
            session_data["proxy_config"] = proxy_config
            
            # Store coordinated session
            self._active_sessions[profile_id] = {
                "session_id": session_data["session_id"],
                "started_at": session_data["started_at"],
                "fingerprint": session_data["fingerprint"],
                "proxy_config": proxy_config,
                "has_proxy": proxy_config is not None
            }
            
            logger.info(f"Started coordinated session for profile {profile_id} "
                       f"(proxy: {'assigned' if proxy_config else 'none'})")
            
            return session_data
            
        except Exception as e:
            logger.error(f"Failed to start coordinated session for profile {profile_id}: {e}")
            
            # Cleanup on failure
            if profile_service:
                await profile_service.end_profile_session(profile_id)
            
            return None
    
    async def end_coordinated_session(self, profile_id: str) -> None:
        """
        End coordinated session for profile.
        
        Args:
            profile_id: Profile identifier
        """
        profile_service = get_profile_service()
        proxy_service = get_proxy_service()
        
        # End profile session
        if profile_service:
            await profile_service.end_profile_session(profile_id)
        
        # Note: We don't release proxy assignments here as they should persist
        # across sessions for consistency. Proxies are released when profile
        # is deleted or during cleanup operations.
        
        # Remove from active sessions
        if profile_id in self._active_sessions:
            session_info = self._active_sessions[profile_id]
            logger.info(f"Ended coordinated session {session_info['session_id']} for profile {profile_id}")
            del self._active_sessions[profile_id]
    
    async def get_session_info(self, profile_id: str) -> Optional[Dict]:
        """
        Get active session information for profile.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Session information or None if no active session
        """
        return self._active_sessions.get(profile_id)
    
    async def rotate_session_proxy(self, profile_id: str) -> Optional[ProxyConfig]:
        """
        Rotate proxy for active session (emergency use).
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            New proxy configuration or None if rotation failed
        """
        proxy_service = get_proxy_service()
        
        if not proxy_service:
            logger.error("Proxy service not available for rotation")
            return None
        
        if profile_id not in self._active_sessions:
            logger.error(f"No active session for profile {profile_id}")
            return None
        
        try:
            new_proxy = await proxy_service.rotate_profile_proxy(profile_id)
            
            if new_proxy:
                # Update active session
                self._active_sessions[profile_id]["proxy_config"] = new_proxy
                self._active_sessions[profile_id]["has_proxy"] = True
                
                logger.info(f"Rotated proxy for profile {profile_id} session")
            
            return new_proxy
            
        except Exception as e:
            logger.error(f"Failed to rotate proxy for profile {profile_id}: {e}")
            return None
    
    async def validate_session_constraints(self, profile_id: str) -> Dict:
        """
        Validate that session adheres to ethical constraints.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Validation results dictionary
        """
        results = {
            "profile_id": profile_id,
            "has_active_session": profile_id in self._active_sessions,
            "fingerprint_consistent": False,
            "proxy_within_limits": False,
            "violations": [],
            "warnings": []
        }
        
        if not results["has_active_session"]:
            results["warnings"].append("No active session to validate")
            return results
        
        profile_service = get_profile_service()
        proxy_service = get_proxy_service()
        
        # Validate fingerprint consistency
        if profile_service:
            try:
                current_fingerprint = await profile_service.get_profile_fingerprint(profile_id)
                session_fingerprint = self._active_sessions[profile_id]["fingerprint"]
                
                if current_fingerprint and session_fingerprint:
                    # Compare key fingerprint attributes
                    if (current_fingerprint.user_agent == session_fingerprint.user_agent and
                        current_fingerprint.canvas_hash == session_fingerprint.canvas_hash):
                        results["fingerprint_consistent"] = True
                    else:
                        results["violations"].append("Fingerprint inconsistency detected in session")
                
            except Exception as e:
                results["violations"].append(f"Failed to validate fingerprint: {e}")
        
        # Validate proxy limits
        if proxy_service:
            try:
                limits = await proxy_service.check_profile_proxy_limits(profile_id)
                results["proxy_within_limits"] = not limits["at_limit"]
                
                if limits["at_limit"]:
                    results["warnings"].append(f"Profile at IP limit ({limits['current_ips']}/{limits['max_ips']})")
                
                if limits["current_ips"] > limits["max_ips"]:
                    results["violations"].append(f"Profile exceeds IP limit: {limits['current_ips']}/{limits['max_ips']}")
                
            except Exception as e:
                results["violations"].append(f"Failed to validate proxy limits: {e}")
        
        return results
    
    async def get_coordinator_stats(self) -> Dict:
        """
        Get coordinator statistics.
        
        Returns:
            Statistics dictionary
        """
        profile_service = get_profile_service()
        proxy_service = get_proxy_service()
        
        stats = {
            "active_sessions": len(self._active_sessions),
            "sessions_with_proxy": sum(1 for s in self._active_sessions.values() if s["has_proxy"]),
            "sessions_without_proxy": sum(1 for s in self._active_sessions.values() if not s["has_proxy"]),
            "profile_service_available": profile_service is not None,
            "proxy_service_available": proxy_service is not None
        }
        
        # Add service-specific stats if available
        if profile_service:
            try:
                profile_stats = await profile_service.get_profile_stats()
                stats["profile_stats"] = profile_stats
            except Exception as e:
                logger.warning(f"Failed to get profile stats: {e}")
        
        if proxy_service:
            try:
                proxy_stats = await proxy_service.get_service_stats()
                stats["proxy_stats"] = proxy_stats
            except Exception as e:
                logger.warning(f"Failed to get proxy stats: {e}")
        
        return stats
    
    async def emergency_compliance_check(self) -> Dict:
        """
        Emergency compliance check across all active sessions.
        
        Returns:
            Compliance check results
        """
        results = {
            "total_sessions_checked": len(self._active_sessions),
            "compliant_sessions": 0,
            "violation_sessions": 0,
            "violations": [],
            "actions_taken": []
        }
        
        for profile_id in list(self._active_sessions.keys()):
            try:
                validation = await self.validate_session_constraints(profile_id)
                
                if validation["violations"]:
                    results["violation_sessions"] += 1
                    results["violations"].extend([
                        {"profile_id": profile_id, "violation": v} 
                        for v in validation["violations"]
                    ])
                    
                    # Take corrective action for serious violations
                    if any("exceeds IP limit" in v for v in validation["violations"]):
                        await self.end_coordinated_session(profile_id)
                        results["actions_taken"].append(f"Terminated session for profile {profile_id} due to IP limit violation")
                
                else:
                    results["compliant_sessions"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to check compliance for profile {profile_id}: {e}")
                results["violations"].append({
                    "profile_id": profile_id,
                    "violation": f"Compliance check failed: {e}"
                })
        
        logger.info(f"Emergency compliance check completed: "
                   f"{results['compliant_sessions']}/{results['total_sessions_checked']} sessions compliant")
        
        return results


# Global coordinator instance
_coordinator: Optional[ProfileProxyCoordinator] = None


def get_profile_proxy_coordinator() -> ProfileProxyCoordinator:
    """
    Get global profile-proxy coordinator instance.
    
    Returns:
        Coordinator instance
    """
    global _coordinator
    
    if _coordinator is None:
        _coordinator = ProfileProxyCoordinator()
    
    return _coordinator


async def start_coordinated_session(profile_id: str) -> Optional[Dict]:
    """
    Convenience function to start coordinated session.
    
    Args:
        profile_id: Profile identifier
        
    Returns:
        Session data or None if failed
    """
    coordinator = get_profile_proxy_coordinator()
    return await coordinator.start_coordinated_session(profile_id)


async def end_coordinated_session(profile_id: str) -> None:
    """
    Convenience function to end coordinated session.
    
    Args:
        profile_id: Profile identifier
    """
    coordinator = get_profile_proxy_coordinator()
    await coordinator.end_coordinated_session(profile_id)