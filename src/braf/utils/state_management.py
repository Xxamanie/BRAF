"""
State Management Module for BRAF.

This module provides browser session state management with database persistence,
including cookie management, session recovery, and emergency state saving.
"""

import asyncio
import json
import logging
import pickle
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from playwright.async_api import Page, BrowserContext
from sqlalchemy import Column, String, Text, DateTime, Integer, BLOB
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from braf.core.database import Base, get_database

logger = logging.getLogger(__name__)


class BrowserStateModel(Base):
    """Database model for browser session states."""
    
    __tablename__ = "browser_states"
    
    profile_id = Column(String(255), primary_key=True)
    cookies = Column(JSONB, nullable=False, default=list)
    local_storage = Column(JSONB, nullable=True)
    session_storage = Column(JSONB, nullable=True)
    page_url = Column(Text, nullable=True)
    viewport = Column(JSONB, nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    session_count = Column(Integer, default=0)
    metadata = Column(JSONB, default=dict)


async def load_cookies_from_db(profile_id: str, session: AsyncSession) -> List[Dict[str, Any]]:
    """
    Load cookies from database for a profile.
    
    Args:
        profile_id: Profile identifier
        session: Database session
        
    Returns:
        List of cookie dictionaries
    """
    try:
        from sqlalchemy import select
        
        stmt = select(BrowserStateModel).where(BrowserStateModel.profile_id == profile_id)
        result = await session.execute(stmt)
        state = result.scalar_one_or_none()
        
        if state and state.cookies:
            logger.info(f"Loaded {len(state.cookies)} cookies for profile {profile_id}")
            return state.cookies
        
        logger.info(f"No cookies found for profile {profile_id}")
        return []
        
    except Exception as e:
        logger.error(f"Failed to load cookies from database: {e}")
        return []


async def save_cookies_to_db(
    profile_id: str, 
    cookies: List[Dict[str, Any]], 
    session: AsyncSession,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Save cookies to database for a profile.
    
    Args:
        profile_id: Profile identifier
        cookies: List of cookie dictionaries
        session: Database session
        metadata: Optional metadata to store
        
    Returns:
        True if successful
    """
    try:
        from sqlalchemy import select
        
        # Check if state exists
        stmt = select(BrowserStateModel).where(BrowserStateModel.profile_id == profile_id)
        result = await session.execute(stmt)
        state = result.scalar_one_or_none()
        
        if state:
            # Update existing state
            state.cookies = cookies
            state.updated_at = datetime.now(timezone.utc)
            state.session_count += 1
            if metadata:
                state.metadata.update(metadata)
        else:
            # Create new state
            state = BrowserStateModel(
                profile_id=profile_id,
                cookies=cookies,
                session_count=1,
                metadata=metadata or {}
            )
            session.add(state)
        
        await session.commit()
        logger.info(f"Saved {len(cookies)} cookies for profile {profile_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save cookies to database: {e}")
        await session.rollback()
        return False


async def save_full_state_to_db(
    profile_id: str,
    page: Page,
    session: AsyncSession,
    include_storage: bool = True
) -> bool:
    """
    Save complete browser state including cookies, storage, and page info.
    
    Args:
        profile_id: Profile identifier
        page: Playwright page instance
        session: Database session
        include_storage: Whether to include local/session storage
        
    Returns:
        True if successful
    """
    try:
        from sqlalchemy import select
        
        # Get cookies
        context = page.context
        cookies = await context.cookies()
        
        # Get page URL and viewport
        page_url = page.url
        viewport = page.viewport_size
        
        # Get local and session storage if requested
        local_storage = None
        session_storage = None
        
        if include_storage:
            try:
                local_storage = await page.evaluate("""
                    () => {
                        const storage = {};
                        for (let i = 0; i < localStorage.length; i++) {
                            const key = localStorage.key(i);
                            storage[key] = localStorage.getItem(key);
                        }
                        return storage;
                    }
                """)
                
                session_storage = await page.evaluate("""
                    () => {
                        const storage = {};
                        for (let i = 0; i < sessionStorage.length; i++) {
                            const key = sessionStorage.key(i);
                            storage[key] = sessionStorage.getItem(key);
                        }
                        return storage;
                    }
                """)
            except Exception as e:
                logger.warning(f"Failed to capture storage: {e}")
        
        # Get user agent
        user_agent = await page.evaluate("() => navigator.userAgent")
        
        # Check if state exists
        stmt = select(BrowserStateModel).where(BrowserStateModel.profile_id == profile_id)
        result = await session.execute(stmt)
        state = result.scalar_one_or_none()
        
        if state:
            # Update existing state
            state.cookies = cookies
            state.local_storage = local_storage
            state.session_storage = session_storage
            state.page_url = page_url
            state.viewport = viewport
            state.user_agent = user_agent
            state.updated_at = datetime.now(timezone.utc)
            state.session_count += 1
        else:
            # Create new state
            state = BrowserStateModel(
                profile_id=profile_id,
                cookies=cookies,
                local_storage=local_storage,
                session_storage=session_storage,
                page_url=page_url,
                viewport=viewport,
                user_agent=user_agent,
                session_count=1
            )
            session.add(state)
        
        await session.commit()
        logger.info(f"Saved full browser state for profile {profile_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save full state: {e}")
        await session.rollback()
        return False


async def restore_state_from_db(
    profile_id: str,
    context: BrowserContext,
    page: Page,
    session: AsyncSession,
    restore_storage: bool = True
) -> bool:
    """
    Restore browser state from database.
    
    Args:
        profile_id: Profile identifier
        context: Browser context
        page: Playwright page instance
        session: Database session
        restore_storage: Whether to restore local/session storage
        
    Returns:
        True if successful
    """
    try:
        from sqlalchemy import select
        
        # Load state from database
        stmt = select(BrowserStateModel).where(BrowserStateModel.profile_id == profile_id)
        result = await session.execute(stmt)
        state = result.scalar_one_or_none()
        
        if not state:
            logger.warning(f"No saved state found for profile {profile_id}")
            return False
        
        # Restore cookies
        if state.cookies:
            await context.add_cookies(state.cookies)
            logger.info(f"Restored {len(state.cookies)} cookies")
        
        # Navigate to saved URL if available
        if state.page_url:
            await page.goto(state.page_url, wait_until="domcontentloaded")
            logger.info(f"Navigated to saved URL: {state.page_url}")
        
        # Restore storage if requested
        if restore_storage:
            if state.local_storage:
                await page.evaluate(f"""
                    (storage) => {{
                        for (const [key, value] of Object.entries(storage)) {{
                            localStorage.setItem(key, value);
                        }}
                    }}
                """, state.local_storage)
                logger.info("Restored local storage")
            
            if state.session_storage:
                await page.evaluate(f"""
                    (storage) => {{
                        for (const [key, value] of Object.entries(storage)) {{
                            sessionStorage.setItem(key, value);
                        }}
                    }}
                """, state.session_storage)
                logger.info("Restored session storage")
        
        logger.info(f"Successfully restored state for profile {profile_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to restore state: {e}")
        return False


async def emergency_save_state(
    profile_id: str,
    page: Page,
    session: AsyncSession,
    reason: str = "emergency"
) -> bool:
    """
    Emergency state save for crash recovery.
    
    Args:
        profile_id: Profile identifier
        page: Playwright page instance
        session: Database session
        reason: Reason for emergency save
        
    Returns:
        True if successful
    """
    try:
        logger.warning(f"Emergency state save triggered for profile {profile_id}: {reason}")
        
        # Save full state with emergency metadata
        metadata = {
            "emergency_save": True,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        success = await save_full_state_to_db(
            profile_id=profile_id,
            page=page,
            session=session,
            include_storage=True
        )
        
        if success:
            # Update metadata
            from sqlalchemy import select, update
            stmt = (
                update(BrowserStateModel)
                .where(BrowserStateModel.profile_id == profile_id)
                .values(metadata=BrowserStateModel.metadata.op('||')(metadata))
            )
            await session.execute(stmt)
            await session.commit()
            
            logger.info(f"Emergency state saved successfully for profile {profile_id}")
        
        return success
        
    except Exception as e:
        logger.error(f"Emergency state save failed: {e}")
        return False


async def scrape_with_state_management(
    profile_id: str,
    scrape_function,
    session: AsyncSession,
    auto_save_interval: int = 300,
    **scrape_kwargs
) -> Dict[str, Any]:
    """
    Execute scraping function with automatic state management.
    
    Args:
        profile_id: Profile identifier
        scrape_function: Async function to execute scraping
        session: Database session
        auto_save_interval: Auto-save interval in seconds
        **scrape_kwargs: Arguments to pass to scrape function
        
    Returns:
        Scraping result with state management info
    """
    from braf.core.browser import get_browser_instance_manager
    
    browser_manager = get_browser_instance_manager()
    browser_instance = None
    last_save_time = datetime.now(timezone.utc)
    
    try:
        # Create browser instance
        browser_instance = await browser_manager.get_instance(profile_id)
        page = browser_instance.page
        context = browser_instance.context
        
        # Restore previous state
        await restore_state_from_db(profile_id, context, page, session)
        
        # Setup auto-save task
        async def auto_save_task():
            nonlocal last_save_time
            while True:
                await asyncio.sleep(auto_save_interval)
                try:
                    await save_full_state_to_db(profile_id, page, session)
                    last_save_time = datetime.now(timezone.utc)
                    logger.info(f"Auto-saved state for profile {profile_id}")
                except Exception as e:
                    logger.error(f"Auto-save failed: {e}")
        
        # Start auto-save task
        save_task = asyncio.create_task(auto_save_task())
        
        try:
            # Execute scraping function
            result = await scrape_function(page=page, **scrape_kwargs)
            
            # Save final state
            await save_full_state_to_db(profile_id, page, session)
            
            return {
                "success": True,
                "result": result,
                "state_management": {
                    "profile_id": profile_id,
                    "state_saved": True,
                    "last_save": last_save_time.isoformat()
                }
            }
            
        finally:
            # Cancel auto-save task
            save_task.cancel()
            try:
                await save_task
            except asyncio.CancelledError:
                pass
        
    except Exception as e:
        logger.error(f"Scraping with state management failed: {e}")
        
        # Emergency save on error
        if browser_instance:
            try:
                await emergency_save_state(
                    profile_id, 
                    browser_instance.page, 
                    session, 
                    reason=f"error: {str(e)}"
                )
            except Exception as save_error:
                logger.error(f"Emergency save failed: {save_error}")
        
        return {
            "success": False,
            "error": str(e),
            "state_management": {
                "profile_id": profile_id,
                "emergency_save_attempted": True
            }
        }
        
    finally:
        # Release browser instance
        if browser_instance:
            await browser_manager.release_instance(browser_instance.id)


async def clear_state(profile_id: str, session: AsyncSession) -> bool:
    """
    Clear saved state for a profile.
    
    Args:
        profile_id: Profile identifier
        session: Database session
        
    Returns:
        True if successful
    """
    try:
        from sqlalchemy import delete
        
        stmt = delete(BrowserStateModel).where(BrowserStateModel.profile_id == profile_id)
        await session.execute(stmt)
        await session.commit()
        
        logger.info(f"Cleared state for profile {profile_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to clear state: {e}")
        await session.rollback()
        return False


async def get_state_info(profile_id: str, session: AsyncSession) -> Optional[Dict[str, Any]]:
    """
    Get information about saved state.
    
    Args:
        profile_id: Profile identifier
        session: Database session
        
    Returns:
        State information dictionary or None
    """
    try:
        from sqlalchemy import select
        
        stmt = select(BrowserStateModel).where(BrowserStateModel.profile_id == profile_id)
        result = await session.execute(stmt)
        state = result.scalar_one_or_none()
        
        if not state:
            return None
        
        return {
            "profile_id": state.profile_id,
            "cookie_count": len(state.cookies) if state.cookies else 0,
            "has_local_storage": bool(state.local_storage),
            "has_session_storage": bool(state.session_storage),
            "page_url": state.page_url,
            "created_at": state.created_at.isoformat() if state.created_at else None,
            "updated_at": state.updated_at.isoformat() if state.updated_at else None,
            "session_count": state.session_count,
            "metadata": state.metadata
        }
        
    except Exception as e:
        logger.error(f"Failed to get state info: {e}")
        return None
