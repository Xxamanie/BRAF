"""
Profile Service for worker nodes.

This service provides a high-level interface for profile and fingerprint management
within worker nodes, ensuring consistency and proper lifecycle management.
"""

import logging
from typing import Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from braf.core.connection_pool import get_db_session
from braf.core.fingerprint_store import FingerprintStore
from braf.core.models import BrowserFingerprint, Profile, ProxyConfig
from braf.core.profile_manager import ProfileManager

logger = logging.getLogger(__name__)


class ProfileService:
    """High-level service for profile and fingerprint management."""
    
    def __init__(self, max_fingerprints: int = 5):
        """
        Initialize profile service.
        
        Args:
            max_fingerprints: Maximum fingerprints in pool (ethical constraint)
        """
        self.fingerprint_store = FingerprintStore(max_fingerprints)
        self.profile_manager = ProfileManager(self.fingerprint_store)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the profile service and fingerprint pool."""
        if self._initialized:
            return
        
        async with get_db_session() as session:
            await self.fingerprint_store.initialize_fingerprint_pool(session)
        
        self._initialized = True
        logger.info("Profile service initialized")
    
    async def create_profile(
        self, 
        profile_config: Optional[Dict] = None,
        proxy_config: Optional[ProxyConfig] = None
    ) -> Profile:
        """
        Create a new profile with assigned fingerprint.
        
        Args:
            profile_config: Optional profile configuration
            proxy_config: Optional proxy configuration
            
        Returns:
            Created profile
        """
        if not self._initialized:
            await self.initialize()
        
        async with get_db_session() as session:
            return await self.profile_manager.create_profile(
                session, profile_config, proxy_config
            )
    
    async def get_profile_fingerprint(self, profile_id: str) -> Optional[BrowserFingerprint]:
        """
        Get consistent fingerprint for profile.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Browser fingerprint or None if profile not found
        """
        async with get_db_session() as session:
            return await self.profile_manager.get_fingerprint(session, profile_id)
    
    async def start_profile_session(self, profile_id: str) -> Dict:
        """
        Start automation session for profile.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Session data with fingerprint and configuration
        """
        async with get_db_session() as session:
            return await self.profile_manager.start_session(session, profile_id)
    
    async def end_profile_session(self, profile_id: str) -> None:
        """
        End automation session for profile.
        
        Args:
            profile_id: Profile identifier
        """
        await self.profile_manager.end_session(profile_id)
    
    async def update_detection_score(self, profile_id: str, score: float) -> None:
        """
        Update detection risk score for profile.
        
        Args:
            profile_id: Profile identifier
            score: Detection score (0.0 to 1.0)
        """
        async with get_db_session() as session:
            await self.profile_manager.update_detection_score(session, profile_id, score)
    
    async def emergency_rotate_fingerprint(self, profile_id: str) -> BrowserFingerprint:
        """
        Emergency fingerprint rotation for high-risk profiles.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            New browser fingerprint
        """
        async with get_db_session() as session:
            return await self.profile_manager.rotate_profile_fingerprint(session, profile_id)
    
    async def store_profile_credentials(self, profile_id: str, credentials: Dict) -> None:
        """
        Store encrypted credentials for profile.
        
        Args:
            profile_id: Profile identifier
            credentials: Credentials to encrypt and store
        """
        async with get_db_session() as session:
            await self.profile_manager.store_credentials(session, profile_id, credentials)
    
    async def get_profile_credentials(self, profile_id: str) -> Optional[Dict]:
        """
        Retrieve decrypted credentials for profile.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Decrypted credentials or None if not found
        """
        async with get_db_session() as session:
            return await self.profile_manager.get_credentials(session, profile_id)
    
    async def get_fingerprint_stats(self) -> Dict:
        """
        Get fingerprint pool statistics.
        
        Returns:
            Statistics dictionary
        """
        async with get_db_session() as session:
            return await self.fingerprint_store.get_fingerprint_stats(session)
    
    async def get_profile_stats(self) -> Dict:
        """
        Get profile management statistics.
        
        Returns:
            Statistics dictionary
        """
        async with get_db_session() as session:
            return await self.profile_manager.get_profile_stats(session)
    
    async def rotate_fingerprint_pool(self) -> None:
        """
        Rotate all fingerprints in the pool (maintenance operation).
        """
        async with get_db_session() as session:
            fingerprint_list = await self.fingerprint_store.list_fingerprints(session)
            
            for fingerprint_id, usage_count, last_used in fingerprint_list:
                await self.fingerprint_store.rotate_fingerprint(session, fingerprint_id)
        
        logger.info("Rotated entire fingerprint pool")
    
    async def cleanup_old_profiles(self, days_inactive: int = 30) -> int:
        """
        Clean up profiles that haven't been used for specified days.
        
        Args:
            days_inactive: Number of days of inactivity before cleanup
            
        Returns:
            Number of profiles cleaned up
        """
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_inactive)
        cleanup_count = 0
        
        async with get_db_session() as session:
            profiles = await self.profile_manager.list_profiles(session, limit=1000)
            
            for profile in profiles:
                last_activity = profile.last_used or profile.created_at
                if last_activity < cutoff_date:
                    await self.profile_manager.delete_profile(session, profile.id)
                    cleanup_count += 1
        
        logger.info(f"Cleaned up {cleanup_count} inactive profiles")
        return cleanup_count


# Global profile service instance
_profile_service: Optional[ProfileService] = None


def get_profile_service() -> ProfileService:
    """
    Get global profile service instance.
    
    Returns:
        Profile service instance
    """
    global _profile_service
    
    if _profile_service is None:
        _profile_service = ProfileService()
    
    return _profile_service


async def init_profile_service(max_fingerprints: int = 5) -> ProfileService:
    """
    Initialize global profile service.
    
    Args:
        max_fingerprints: Maximum fingerprints in pool
        
    Returns:
        Initialized profile service
    """
    global _profile_service
    
    _profile_service = ProfileService(max_fingerprints)
    await _profile_service.initialize()
    
    return _profile_service
