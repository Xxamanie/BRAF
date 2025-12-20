"""
Unit tests for state management module.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from src.braf.utils.state_management import (
    load_cookies_from_db, save_cookies_to_db, save_full_state_to_db,
    restore_state_from_db, emergency_save_state, scrape_with_state_management,
    clear_state, get_state_info, BrowserStateModel
)


@pytest.fixture
def mock_session():
    """Mock database session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.add = Mock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def mock_page():
    """Mock Playwright page."""
    page = AsyncMock()
    page.url = "https://example.com"
    page.viewport_size = {"width": 1920, "height": 1080}
    page.evaluate = AsyncMock()
    page.goto = AsyncMock()
    
    # Mock context
    context = AsyncMock()
    context.cookies = AsyncMock(return_value=[
        {"name": "test_cookie", "value": "test_value", "domain": "example.com"}
    ])
    context.add_cookies = AsyncMock()
    page.context = context
    
    return page


@pytest.fixture
def sample_cookies():
    """Sample cookies data."""
    return [
        {"name": "session_id", "value": "abc123", "domain": "example.com"},
        {"name": "user_pref", "value": "dark_mode", "domain": "example.com"}
    ]


class TestStateManagement:
    """Test state management functions."""
    
    @pytest.mark.asyncio
    async def test_load_cookies_from_db_success(self, mock_session, sample_cookies):
        """Test successful cookie loading."""
        # Mock database result
        mock_state = Mock()
        mock_state.cookies = sample_cookies
        
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_state
        mock_session.execute.return_value = mock_result
        
        # Test
        cookies = await load_cookies_from_db("test_profile", mock_session)
        
        # Assertions
        assert cookies == sample_cookies
        mock_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_cookies_from_db_no_state(self, mock_session):
        """Test cookie loading when no state exists."""
        # Mock no result
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        # Test
        cookies = await load_cookies_from_db("test_profile", mock_session)
        
        # Assertions
        assert cookies == []
    
    @pytest.mark.asyncio
    async def test_save_cookies_to_db_new_state(self, mock_session, sample_cookies):
        """Test saving cookies for new profile."""
        # Mock no existing state
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        # Test
        success = await save_cookies_to_db("test_profile", sample_cookies, mock_session)
        
        # Assertions
        assert success is True
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_cookies_to_db_update_existing(self, mock_session, sample_cookies):
        """Test updating cookies for existing profile."""
        # Mock existing state
        mock_state = Mock()
        mock_state.cookies = []
        mock_state.session_count = 1
        mock_state.metadata = {}
        
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_state
        mock_session.execute.return_value = mock_result
        
        # Test
        success = await save_cookies_to_db("test_profile", sample_cookies, mock_session)
        
        # Assertions
        assert success is True
        assert mock_state.cookies == sample_cookies
        assert mock_state.session_count == 2
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_full_state_to_db(self, mock_session, mock_page):
        """Test saving full browser state."""
        # Mock no existing state
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        # Mock page evaluations
        mock_page.evaluate.side_effect = [
            {"key1": "value1"},  # localStorage
            {"key2": "value2"},  # sessionStorage
            "Mozilla/5.0 Test Agent"  # userAgent
        ]
        
        # Test
        success = await save_full_state_to_db("test_profile", mock_page, mock_session)
        
        # Assertions
        assert success is True
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_restore_state_from_db(self, mock_session, mock_page, sample_cookies):
        """Test restoring browser state."""
        # Mock existing state
        mock_state = Mock()
        mock_state.cookies = sample_cookies
        mock_state.page_url = "https://example.com/restored"
        mock_state.local_storage = {"key1": "value1"}
        mock_state.session_storage = {"key2": "value2"}
        
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_state
        mock_session.execute.return_value = mock_result
        
        # Test
        success = await restore_state_from_db(
            "test_profile", mock_page.context, mock_page, mock_session
        )
        
        # Assertions
        assert success is True
        mock_page.context.add_cookies.assert_called_once_with(sample_cookies)
        mock_page.goto.assert_called_once_with(
            "https://example.com/restored", wait_until="domcontentloaded"
        )
        assert mock_page.evaluate.call_count == 2  # localStorage and sessionStorage
    
    @pytest.mark.asyncio
    async def test_emergency_save_state(self, mock_session, mock_page):
        """Test emergency state saving."""
        # Mock no existing state for save_full_state_to_db
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        # Mock page evaluations
        mock_page.evaluate.side_effect = [
            {"key1": "value1"},  # localStorage
            {"key2": "value2"},  # sessionStorage
            "Mozilla/5.0 Test Agent"  # userAgent
        ]
        
        # Test
        success = await emergency_save_state(
            "test_profile", mock_page, mock_session, "test_emergency"
        )
        
        # Assertions
        assert success is True
        # Should call execute twice: once for save_full_state, once for metadata update
        assert mock_session.execute.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_clear_state(self, mock_session):
        """Test clearing state."""
        # Test
        success = await clear_state("test_profile", mock_session)
        
        # Assertions
        assert success is True
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_state_info(self, mock_session, sample_cookies):
        """Test getting state information."""
        # Mock existing state
        mock_state = Mock()
        mock_state.profile_id = "test_profile"
        mock_state.cookies = sample_cookies
        mock_state.local_storage = {"key": "value"}
        mock_state.session_storage = None
        mock_state.page_url = "https://example.com"
        mock_state.created_at = datetime.now(timezone.utc)
        mock_state.updated_at = datetime.now(timezone.utc)
        mock_state.session_count = 5
        mock_state.metadata = {"test": "data"}
        
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_state
        mock_session.execute.return_value = mock_result
        
        # Test
        info = await get_state_info("test_profile", mock_session)
        
        # Assertions
        assert info is not None
        assert info["profile_id"] == "test_profile"
        assert info["cookie_count"] == 2
        assert info["has_local_storage"] is True
        assert info["has_session_storage"] is False
        assert info["session_count"] == 5
    
    @pytest.mark.asyncio
    async def test_scrape_with_state_management(self, mock_session):
        """Test scraping with state management."""
        # Mock browser manager
        mock_browser_manager = AsyncMock()
        mock_browser_instance = Mock()
        mock_browser_instance.page = mock_page = AsyncMock()
        mock_browser_instance.context = AsyncMock()
        mock_browser_instance.id = "test_instance"
        
        mock_browser_manager.get_instance.return_value = mock_browser_instance
        mock_browser_manager.release_instance = AsyncMock()
        
        # Mock scrape function
        async def mock_scrape_function(page=None, **kwargs):
            return {"data": "scraped_data"}
        
        with patch('src.braf.utils.state_management.get_browser_instance_manager', 
                  return_value=mock_browser_manager):
            with patch('src.braf.utils.state_management.restore_state_from_db', 
                      return_value=True):
                with patch('src.braf.utils.state_management.save_full_state_to_db', 
                          return_value=True):
                    
                    # Test
                    result = await scrape_with_state_management(
                        "test_profile", mock_scrape_function, mock_session,
                        auto_save_interval=1, test_param="test_value"
                    )
        
        # Assertions
        assert result["success"] is True
        assert result["result"]["data"] == "scraped_data"
        assert "state_management" in result
        mock_browser_manager.get_instance.assert_called_once_with("test_profile")
        mock_browser_manager.release_instance.assert_called_once_with("test_instance")


if __name__ == "__main__":
    pytest.main([__file__])