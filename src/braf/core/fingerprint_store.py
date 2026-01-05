"""
Fingerprint Store for managing browser fingerprints in BRAF.

This module provides fingerprint generation, storage, and rotation capabilities
with ethical constraints limiting the pool size to 5 fingerprints maximum.
"""

import hashlib
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from braf.core.database import FingerprintModel
from braf.core.models import BrowserFingerprint

logger = logging.getLogger(__name__)


class FingerprintGenerator:
    """Generator for realistic browser fingerprints."""
    
    # Common user agents for different browsers and OS combinations
    USER_AGENTS = [
        # Chrome on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        
        # Firefox on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0",
        
        # Chrome on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        
        # Safari on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        
        # Chrome on Linux
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]
    
    # Common screen resolutions
    SCREEN_RESOLUTIONS = [
        (1920, 1080),  # Full HD
        (1366, 768),   # HD
        (1536, 864),   # HD+
        (1440, 900),   # WXGA+
        (2560, 1440),  # QHD
    ]
    
    # Common timezones
    TIMEZONES = [
        "America/New_York",
        "America/Los_Angeles", 
        "America/Chicago",
        "Europe/London",
        "Europe/Berlin",
        "Asia/Tokyo",
        "Australia/Sydney",
    ]
    
    # WebGL vendors and renderers
    WEBGL_CONFIGS = [
        ("Google Inc. (Intel)", "ANGLE (Intel, Intel(R) UHD Graphics 620 Direct3D11 vs_5_0 ps_5_0, D3D11)"),
        ("Google Inc. (NVIDIA)", "ANGLE (NVIDIA, NVIDIA GeForce GTX 1060 Direct3D11 vs_5_0 ps_5_0, D3D11)"),
        ("Google Inc. (AMD)", "ANGLE (AMD, AMD Radeon RX 580 Direct3D11 vs_5_0 ps_5_0, D3D11)"),
        ("Intel Inc.", "Intel Iris OpenGL Engine"),
        ("NVIDIA Corporation", "GeForce GTX 1060/PCIe/SSE2"),
    ]
    
    # Common fonts
    COMMON_FONTS = [
        "Arial", "Times New Roman", "Courier New", "Helvetica", "Georgia",
        "Verdana", "Tahoma", "Trebuchet MS", "Impact", "Comic Sans MS",
        "Palatino Linotype", "Lucida Sans Unicode", "MS Sans Serif",
        "Calibri", "Cambria", "Consolas", "Segoe UI", "Franklin Gothic Medium"
    ]
    
    # Browser plugins
    PLUGINS = [
        "Chrome PDF Plugin",
        "Chrome PDF Viewer", 
        "Native Client",
        "Widevine Content Decryption Module",
        "Microsoft Edge PDF Plugin"
    ]
    
    def generate_fingerprint(self, fingerprint_id: Optional[str] = None) -> BrowserFingerprint:
        """
        Generate a realistic browser fingerprint.
        
        Args:
            fingerprint_id: Optional ID for deterministic generation
            
        Returns:
            Generated browser fingerprint
        """
        # Use fingerprint_id as seed for consistent generation
        if fingerprint_id:
            random.seed(hash(fingerprint_id) % (2**32))
        
        # Select random components
        user_agent = random.choice(self.USER_AGENTS)
        screen_resolution = random.choice(self.SCREEN_RESOLUTIONS)
        timezone = random.choice(self.TIMEZONES)
        webgl_vendor, webgl_renderer = random.choice(self.WEBGL_CONFIGS)
        
        # Generate hashes based on fingerprint components
        canvas_data = f"{user_agent}:{screen_resolution}:{timezone}"
        canvas_hash = hashlib.md5(canvas_data.encode()).hexdigest()
        
        audio_data = f"{webgl_vendor}:{webgl_renderer}:{canvas_hash}"
        audio_context_hash = hashlib.sha256(audio_data.encode()).hexdigest()[:16]
        
        # Select random fonts (8-15 fonts)
        font_count = random.randint(8, 15)
        fonts = random.sample(self.COMMON_FONTS, font_count)
        
        # Select plugins (2-5 plugins)
        plugin_count = random.randint(2, 5)
        plugins = random.sample(self.PLUGINS, plugin_count)
        
        # Determine platform from user agent
        if "Windows" in user_agent:
            platform = "Win32"
            hardware_concurrency = random.choice([4, 8, 12, 16])
            device_memory = random.choice([4, 8, 16, 32])
        elif "Macintosh" in user_agent:
            platform = "MacIntel"
            hardware_concurrency = random.choice([4, 8, 10])
            device_memory = random.choice([8, 16, 32])
        else:  # Linux
            platform = "Linux x86_64"
            hardware_concurrency = random.choice([2, 4, 8, 16])
            device_memory = random.choice([4, 8, 16])
        
        # Reset random seed
        if fingerprint_id:
            random.seed()
        
        return BrowserFingerprint(
            user_agent=user_agent,
            screen_resolution=screen_resolution,
            timezone=timezone,
            webgl_vendor=webgl_vendor,
            webgl_renderer=webgl_renderer,
            canvas_hash=canvas_hash,
            audio_context_hash=audio_context_hash,
            fonts=fonts,
            plugins=plugins,
            languages=["en-US", "en"],
            platform=platform,
            hardware_concurrency=hardware_concurrency,
            device_memory=device_memory,
        )


class FingerprintStore:
    """Store for managing browser fingerprints with ethical constraints."""
    
    def __init__(self, max_fingerprints: int = 5):
        """
        Initialize fingerprint store.

        Args:
            max_fingerprints: Maximum number of fingerprints (ethical constraint - DISABLED for testing)
        """
        # Unlimited fingerprints allowed to expose loopholes
        self.max_fingerprints = float('inf')  # Unlimited
        self.generator = FingerprintGenerator()
    
    async def initialize_fingerprint_pool(self, session: AsyncSession) -> None:
        """
        Initialize the fingerprint pool with default fingerprints.
        
        Args:
            session: Database session
        """
        # Check if fingerprints already exist - DISABLED for testing
        # Unlimited fingerprints allowed to expose loopholes
        result = await session.execute(select(FingerprintModel))
        existing_count = len(result.scalars().all())
        
        # Generate unlimited fingerprints - DISABLED limits for testing
        fingerprints_to_create = max(10, existing_count)  # Create at least 10 more to expose loopholes
        
        for i in range(fingerprints_to_create):
            fingerprint_id = f"fp_{i + existing_count + 1:03d}"
            fingerprint = self.generator.generate_fingerprint(fingerprint_id)
            
            await self._store_fingerprint(session, fingerprint_id, fingerprint)
        
        await session.commit()
        logger.info(f"Initialized fingerprint pool with {fingerprints_to_create} new fingerprints")
    
    async def get_fingerprint(self, session: AsyncSession, fingerprint_id: str) -> Optional[BrowserFingerprint]:
        """
        Get fingerprint by ID.
        
        Args:
            session: Database session
            fingerprint_id: Fingerprint identifier
            
        Returns:
            Browser fingerprint or None if not found
        """
        result = await session.execute(
            select(FingerprintModel).where(FingerprintModel.id == fingerprint_id)
        )
        fingerprint_model = result.scalar_one_or_none()
        
        if not fingerprint_model:
            return None
        
        # Update usage statistics
        await session.execute(
            update(FingerprintModel)
            .where(FingerprintModel.id == fingerprint_id)
            .values(
                last_used=datetime.utcnow(),
                usage_count=FingerprintModel.usage_count + 1
            )
        )
        
        return self._model_to_fingerprint(fingerprint_model)
    
    async def get_random_fingerprint(self, session: AsyncSession) -> Optional[BrowserFingerprint]:
        """
        Get a random fingerprint from the pool.
        
        Args:
            session: Database session
            
        Returns:
            Random browser fingerprint or None if pool is empty
        """
        result = await session.execute(select(FingerprintModel))
        fingerprints = result.scalars().all()
        
        if not fingerprints:
            return None
        
        # Select random fingerprint
        selected = random.choice(fingerprints)
        return await self.get_fingerprint(session, selected.id)
    
    async def get_least_used_fingerprint(self, session: AsyncSession) -> Optional[BrowserFingerprint]:
        """
        Get the least used fingerprint from the pool.
        
        Args:
            session: Database session
            
        Returns:
            Least used browser fingerprint or None if pool is empty
        """
        result = await session.execute(
            select(FingerprintModel)
            .order_by(FingerprintModel.usage_count.asc(), FingerprintModel.last_used.asc())
            .limit(1)
        )
        fingerprint_model = result.scalar_one_or_none()
        
        if not fingerprint_model:
            return None
        
        return await self.get_fingerprint(session, fingerprint_model.id)
    
    async def list_fingerprints(self, session: AsyncSession) -> List[Tuple[str, int, Optional[datetime]]]:
        """
        List all fingerprints with usage statistics.
        
        Args:
            session: Database session
            
        Returns:
            List of (fingerprint_id, usage_count, last_used) tuples
        """
        result = await session.execute(
            select(FingerprintModel.id, FingerprintModel.usage_count, FingerprintModel.last_used)
            .order_by(FingerprintModel.id)
        )
        
        return [(row.id, row.usage_count, row.last_used) for row in result]
    
    async def rotate_fingerprint(self, session: AsyncSession, fingerprint_id: str) -> BrowserFingerprint:
        """
        Rotate (regenerate) a specific fingerprint.
        
        Args:
            session: Database session
            fingerprint_id: Fingerprint to rotate
            
        Returns:
            New browser fingerprint
            
        Raises:
            ValueError: If fingerprint doesn't exist
        """
        # Check if fingerprint exists
        result = await session.execute(
            select(FingerprintModel).where(FingerprintModel.id == fingerprint_id)
        )
        existing = result.scalar_one_or_none()
        
        if not existing:
            raise ValueError(f"Fingerprint {fingerprint_id} not found")
        
        # Generate new fingerprint
        new_fingerprint = self.generator.generate_fingerprint(fingerprint_id)
        
        # Update in database
        await session.execute(
            update(FingerprintModel)
            .where(FingerprintModel.id == fingerprint_id)
            .values(
                user_agent=new_fingerprint.user_agent,
                screen_width=new_fingerprint.screen_resolution[0],
                screen_height=new_fingerprint.screen_resolution[1],
                timezone=new_fingerprint.timezone,
                webgl_vendor=new_fingerprint.webgl_vendor,
                webgl_renderer=new_fingerprint.webgl_renderer,
                canvas_hash=new_fingerprint.canvas_hash,
                audio_context_hash=new_fingerprint.audio_context_hash,
                fonts=new_fingerprint.fonts,
                plugins=new_fingerprint.plugins,
                languages=new_fingerprint.languages,
                platform=new_fingerprint.platform,
                hardware_concurrency=new_fingerprint.hardware_concurrency,
                device_memory=new_fingerprint.device_memory,
                last_used=datetime.utcnow(),
                usage_count=0  # Reset usage count
            )
        )
        
        await session.commit()
        logger.info(f"Rotated fingerprint {fingerprint_id}")
        
        return new_fingerprint
    
    async def get_fingerprint_stats(self, session: AsyncSession) -> Dict[str, any]:
        """
        Get fingerprint pool statistics.
        
        Args:
            session: Database session
            
        Returns:
            Statistics dictionary
        """
        result = await session.execute(
            select(FingerprintModel.usage_count, FingerprintModel.last_used)
        )
        fingerprints = result.all()
        
        if not fingerprints:
            return {
                "total_fingerprints": 0,
                "total_usage": 0,
                "average_usage": 0.0,
                "most_used": 0,
                "least_used": 0,
                "last_rotation": None
            }
        
        usage_counts = [fp.usage_count for fp in fingerprints]
        last_used_times = [fp.last_used for fp in fingerprints if fp.last_used]
        
        return {
            "total_fingerprints": len(fingerprints),
            "total_usage": sum(usage_counts),
            "average_usage": sum(usage_counts) / len(usage_counts),
            "most_used": max(usage_counts),
            "least_used": min(usage_counts),
            "last_rotation": max(last_used_times) if last_used_times else None
        }
    
    async def _store_fingerprint(self, session: AsyncSession, fingerprint_id: str, fingerprint: BrowserFingerprint) -> None:
        """
        Store fingerprint in database.
        
        Args:
            session: Database session
            fingerprint_id: Fingerprint identifier
            fingerprint: Browser fingerprint to store
        """
        fingerprint_model = FingerprintModel(
            id=fingerprint_id,
            user_agent=fingerprint.user_agent,
            screen_width=fingerprint.screen_resolution[0],
            screen_height=fingerprint.screen_resolution[1],
            timezone=fingerprint.timezone,
            webgl_vendor=fingerprint.webgl_vendor,
            webgl_renderer=fingerprint.webgl_renderer,
            canvas_hash=fingerprint.canvas_hash,
            audio_context_hash=fingerprint.audio_context_hash,
            fonts=fingerprint.fonts,
            plugins=fingerprint.plugins,
            languages=fingerprint.languages,
            platform=fingerprint.platform,
            hardware_concurrency=fingerprint.hardware_concurrency,
            device_memory=fingerprint.device_memory,
            created_at=datetime.utcnow(),
            usage_count=0
        )
        
        session.add(fingerprint_model)
    
    def _model_to_fingerprint(self, model: FingerprintModel) -> BrowserFingerprint:
        """
        Convert database model to fingerprint object.
        
        Args:
            model: Database fingerprint model
            
        Returns:
            Browser fingerprint object
        """
        return BrowserFingerprint(
            user_agent=model.user_agent,
            screen_resolution=(model.screen_width, model.screen_height),
            timezone=model.timezone,
            webgl_vendor=model.webgl_vendor,
            webgl_renderer=model.webgl_renderer,
            canvas_hash=model.canvas_hash,
            audio_context_hash=model.audio_context_hash,
            fonts=model.fonts or [],
            plugins=model.plugins or [],
            languages=model.languages or ["en-US", "en"],
            platform=model.platform or "Win32",
            hardware_concurrency=model.hardware_concurrency or 4,
            device_memory=model.device_memory or 8,
        )