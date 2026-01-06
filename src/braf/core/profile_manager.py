"""
Profile Manager for BRAF user profile lifecycle management.

This module handles profile creation, fingerprint assignment, session management,
and ensures consistency within sessions while providing variation across profiles.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from braf.core.database import ProfileModel, EncryptedCredentialModel
from braf.core.encryption import get_credential_vault
from braf.core.fingerprint_store import FingerprintStore
from braf.core.models import Profile, BrowserFingerprint, ProxyConfig

logger = logging.getLogger(__name__)


class ProfileManager:
    """Manager for user profiles with fingerprint and proxy assignment."""
    
    def __init__(self, fingerprint_store: FingerprintStore):
        """
        Initialize profile manager.
        
        Args:
            fingerprint_store: Fingerprint store instance
        """
        self.fingerprint_store = fingerprint_store
        self._active_sessions: Dict[str, Dict] = {}  # profile_id -> session_data
    
    async def create_profile(
        self, 
        session: AsyncSession,
        profile_config: Optional[Dict] = None,
        proxy_config: Optional[ProxyConfig] = None
    ) -> Profile:
        """
        Create a new user profile with assigned fingerprint.
        
        Args:
            session: Database session
            profile_config: Optional profile configuration
            proxy_config: Optional proxy configuration
            
        Returns:
            Created profile
        """
        profile_id = str(uuid.uuid4())
        
        # Get least used fingerprint for new profile
        fingerprint = await self.fingerprint_store.get_least_used_fingerprint(session)
        if not fingerprint:
            raise RuntimeError("No fingerprints available in pool")
        
        # Determine fingerprint ID from the fingerprint store
        fingerprint_result = await session.execute(
            select(ProfileModel.fingerprint_id)
            .where(ProfileModel.fingerprint_id.isnot(None))
            .limit(1)
        )
        
        # Get a fingerprint ID by querying the fingerprint store
        fingerprint_list = await self.fingerprint_store.list_fingerprints(session)
        if not fingerprint_list:
            raise RuntimeError("No fingerprints available")
        
        # Use least used fingerprint
        fingerprint_id = min(fingerprint_list, key=lambda x: x[1])[0]  # Sort by usage count
        
        # Create profile model
        profile_model = ProfileModel(
            id=uuid.UUID(profile_id),
            fingerprint_id=fingerprint_id,
            proxy_config=proxy_config.to_dict() if proxy_config else None,
            created_at=datetime.utcnow(),
            session_count=0,
            detection_score=0.0,
            metadata=profile_config or {}
        )
        
        session.add(profile_model)
        await session.commit()
        
        # Create profile object
        profile = Profile(
            id=profile_id,
            fingerprint_id=fingerprint_id,
            proxy_config=proxy_config,
            created_at=profile_model.created_at,
            session_count=0,
            detection_score=0.0,
            metadata=profile_config or {}
        )
        
        logger.info(f"Created profile {profile_id} with fingerprint {fingerprint_id}")
        return profile
    
    async def get_profile(self, session: AsyncSession, profile_id: str) -> Optional[Profile]:
        """
        Get profile by ID.
        
        Args:
            session: Database session
            profile_id: Profile identifier
            
        Returns:
            Profile or None if not found
        """
        result = await session.execute(
            select(ProfileModel).where(ProfileModel.id == uuid.UUID(profile_id))
        )
        profile_model = result.scalar_one_or_none()
        
        if not profile_model:
            return None
        
        # Convert proxy config
        proxy_config = None
        if profile_model.proxy_config:
            proxy_data = profile_model.proxy_config
            proxy_config = ProxyConfig(
                host=proxy_data["host"],
                port=proxy_data["port"],
                username=proxy_data.get("username"),
                password=proxy_data.get("password"),
                proxy_type=proxy_data.get("proxy_type", "http")
            )
        
        return Profile(
            id=str(profile_model.id),
            fingerprint_id=profile_model.fingerprint_id,
            proxy_config=proxy_config,
            created_at=profile_model.created_at,
            last_used=profile_model.last_used,
            session_count=profile_model.session_count,
            detection_score=profile_model.detection_score,
            metadata=profile_model.metadata or {}
        )
    
    async def get_fingerprint(self, session: AsyncSession, profile_id: str) -> Optional[BrowserFingerprint]:
        """
        Get consistent fingerprint for profile.
        
        Args:
            session: Database session
            profile_id: Profile identifier
            
        Returns:
            Browser fingerprint or None if profile not found
        """
        profile = await self.get_profile(session, profile_id)
        if not profile:
            return None
        
        return await self.fingerprint_store.get_fingerprint(session, profile.fingerprint_id)
    
    async def start_session(self, session: AsyncSession, profile_id: str) -> Dict:
        """
        Start a new session for profile.
        
        Args:
            session: Database session
            profile_id: Profile identifier
            
        Returns:
            Session data dictionary
            
        Raises:
            ValueError: If profile doesn't exist
        """
        profile = await self.get_profile(session, profile_id)
        if not profile:
            raise ValueError(f"Profile {profile_id} not found")
        
        # Update profile usage
        await session.execute(
            update(ProfileModel)
            .where(ProfileModel.id == uuid.UUID(profile_id))
            .values(
                last_used=datetime.utcnow(),
                session_count=ProfileModel.session_count + 1
            )
        )
        await session.commit()
        
        # Get fingerprint for session
        fingerprint = await self.get_fingerprint(session, profile_id)
        
        # Create session data
        session_data = {
            "profile_id": profile_id,
            "session_id": str(uuid.uuid4()),
            "started_at": datetime.utcnow(),
            "fingerprint": fingerprint,
            "proxy_config": profile.proxy_config,
            "behavioral_state": {
                "mouse_speed_factor": 1.0,
                "typing_speed_wpm": 60,
                "error_rate": 0.02,
                "last_action_time": None
            }
        }
        
        # Store active session
        self._active_sessions[profile_id] = session_data
        
        logger.info(f"Started session {session_data['session_id']} for profile {profile_id}")
        return session_data
    
    async def end_session(self, profile_id: str) -> None:
        """
        End active session for profile.
        
        Args:
            profile_id: Profile identifier
        """
        if profile_id in self._active_sessions:
            session_data = self._active_sessions[profile_id]
            session_duration = datetime.utcnow() - session_data["started_at"]
            
            logger.info(f"Ended session {session_data['session_id']} for profile {profile_id} "
                       f"(duration: {session_duration})")
            
            del self._active_sessions[profile_id]
    
    async def get_active_session(self, profile_id: str) -> Optional[Dict]:
        """
        Get active session data for profile.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Session data or None if no active session
        """
        return self._active_sessions.get(profile_id)
    
    async def update_detection_score(
        self, 
        session: AsyncSession, 
        profile_id: str, 
        detection_score: float
    ) -> None:
        """
        Update detection score for profile.
        
        Args:
            session: Database session
            profile_id: Profile identifier
            detection_score: New detection score (0.0 to 1.0)
        """
        await session.execute(
            update(ProfileModel)
            .where(ProfileModel.id == uuid.UUID(profile_id))
            .values(detection_score=detection_score)
        )
        await session.commit()
        
        # Update active session if exists
        if profile_id in self._active_sessions:
            self._active_sessions[profile_id]["detection_score"] = detection_score
        
        logger.info(f"Updated detection score for profile {profile_id}: {detection_score}")
    
    async def rotate_profile_fingerprint(
        self, 
        session: AsyncSession, 
        profile_id: str
    ) -> BrowserFingerprint:
        """
        Rotate fingerprint for profile (emergency use only).
        
        Args:
            session: Database session
            profile_id: Profile identifier
            
        Returns:
            New browser fingerprint
            
        Raises:
            ValueError: If profile doesn't exist
        """
        profile = await self.get_profile(session, profile_id)
        if not profile:
            raise ValueError(f"Profile {profile_id} not found")
        
        # Rotate the fingerprint
        new_fingerprint = await self.fingerprint_store.rotate_fingerprint(
            session, profile.fingerprint_id
        )
        
        # End any active session to force new fingerprint usage
        await self.end_session(profile_id)
        
        logger.warning(f"Rotated fingerprint for profile {profile_id} (emergency rotation)")
        return new_fingerprint
    
    async def store_credentials(
        self, 
        session: AsyncSession, 
        profile_id: str, 
        credentials: Dict
    ) -> None:
        """
        Store encrypted credentials for profile.
        
        Args:
            session: Database session
            profile_id: Profile identifier
            credentials: Credentials dictionary to encrypt and store
        """
        vault = get_credential_vault()
        encrypted_data, salt = vault.encrypt_credentials(credentials)
        
        # Check if credentials already exist
        result = await session.execute(
            select(EncryptedCredentialModel)
            .where(EncryptedCredentialModel.profile_id == uuid.UUID(profile_id))
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update existing credentials
            await session.execute(
                update(EncryptedCredentialModel)
                .where(EncryptedCredentialModel.profile_id == uuid.UUID(profile_id))
                .values(
                    encrypted_data=encrypted_data,
                    salt=salt,
                    updated_at=datetime.utcnow()
                )
            )
        else:
            # Create new credentials
            credential_model = EncryptedCredentialModel(
                profile_id=uuid.UUID(profile_id),
                encrypted_data=encrypted_data,
                salt=salt,
                created_at=datetime.utcnow()
            )
            session.add(credential_model)
        
        await session.commit()
        logger.info(f"Stored encrypted credentials for profile {profile_id}")
    
    async def get_credentials(
        self, 
        session: AsyncSession, 
        profile_id: str
    ) -> Optional[Dict]:
        """
        Retrieve and decrypt credentials for profile.
        
        Args:
            session: Database session
            profile_id: Profile identifier
            
        Returns:
            Decrypted credentials dictionary or None if not found
        """
        result = await session.execute(
            select(EncryptedCredentialModel)
            .where(EncryptedCredentialModel.profile_id == uuid.UUID(profile_id))
        )
        credential_model = result.scalar_one_or_none()
        
        if not credential_model:
            return None
        
        vault = get_credential_vault()
        try:
            credentials = vault.decrypt_credentials(
                credential_model.encrypted_data,
                credential_model.salt
            )
            logger.info(f"Retrieved credentials for profile {profile_id}")
            return credentials
        except ValueError as e:
            logger.error(f"Failed to decrypt credentials for profile {profile_id}: {e}")
            return None
    
    async def delete_profile(self, session: AsyncSession, profile_id: str) -> bool:
        """
        Delete profile and associated data.
        
        Args:
            session: Database session
            profile_id: Profile identifier
            
        Returns:
            True if profile was deleted, False if not found
        """
        # End any active session
        await self.end_session(profile_id)
        
        # Delete encrypted credentials
        await session.execute(
            delete(EncryptedCredentialModel)
            .where(EncryptedCredentialModel.profile_id == uuid.UUID(profile_id))
        )
        
        # Delete profile
        result = await session.execute(
            delete(ProfileModel)
            .where(ProfileModel.id == uuid.UUID(profile_id))
        )
        
        await session.commit()
        
        deleted = result.rowcount > 0
        if deleted:
            logger.info(f"Deleted profile {profile_id}")
        
        return deleted
    
    async def list_profiles(
        self, 
        session: AsyncSession, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[Profile]:
        """
        List profiles with pagination.
        
        Args:
            session: Database session
            limit: Maximum number of profiles to return
            offset: Number of profiles to skip
            
        Returns:
            List of profiles
        """
        result = await session.execute(
            select(ProfileModel)
            .order_by(ProfileModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        
        profiles = []
        for profile_model in result.scalars():
            # Convert proxy config
            proxy_config = None
            if profile_model.proxy_config:
                proxy_data = profile_model.proxy_config
                proxy_config = ProxyConfig(
                    host=proxy_data["host"],
                    port=proxy_data["port"],
                    username=proxy_data.get("username"),
                    password=proxy_data.get("password"),
                    proxy_type=proxy_data.get("proxy_type", "http")
                )
            
            profile = Profile(
                id=str(profile_model.id),
                fingerprint_id=profile_model.fingerprint_id,
                proxy_config=proxy_config,
                created_at=profile_model.created_at,
                last_used=profile_model.last_used,
                session_count=profile_model.session_count,
                detection_score=profile_model.detection_score,
                metadata=profile_model.metadata or {}
            )
            profiles.append(profile)
        
        return profiles
    
    async def get_profile_stats(self, session: AsyncSession) -> Dict:
        """
        Get profile statistics.
        
        Args:
            session: Database session
            
        Returns:
            Statistics dictionary
        """
        result = await session.execute(
            select(ProfileModel.session_count, ProfileModel.detection_score, ProfileModel.created_at)
        )
        profiles = result.all()
        
        if not profiles:
            return {
                "total_profiles": 0,
                "active_sessions": 0,
                "total_sessions": 0,
                "average_detection_score": 0.0,
                "high_risk_profiles": 0
            }
        
        session_counts = [p.session_count for p in profiles]
        detection_scores = [p.detection_score for p in profiles]
        
        return {
            "total_profiles": len(profiles),
            "active_sessions": len(self._active_sessions),
            "total_sessions": sum(session_counts),
            "average_detection_score": sum(detection_scores) / len(detection_scores),
            "high_risk_profiles": len([s for s in detection_scores if s > 0.7])
        }
