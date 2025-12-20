"""
Unit tests for cost governance module.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone, timedelta

from src.braf.utils.cost_governance import (
    CostGovernor, CostTrackingModel, BudgetModel,
    get_cost_governor, init_cost_governor
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
def mock_redis():
    """Mock Redis client."""
    redis_client = AsyncMock()
    redis_client.setex = AsyncMock()
    redis_client.get = AsyncMock()
    redis_client.incrbyfloat = AsyncMock()
    redis_client.expire = AsyncMock()
    return redis_client


@pytest.fixture
def cost_governor(mock_redis):
    """Cost governor instance with mocked Redis."""
    return CostGovernor(mock_redis)


@pytest.fixture
def sample_budget():
    """Sample budget data."""
    return {
        "daily_limit": 10.0,
        "weekly_limit": 50.0,
        "monthly_limit": 200.0,
        "currency": "USD",
        "alert_threshold": 0.8,
        "enabled": True
    }


class TestCostGovernor:
    """Test cost governance functionality."""
    
    @pytest.mark.asyncio
    async def test_set_budget_new(self, cost_governor, mock_session):
        """Test setting budget for new profile."""
        # Mock no existing budget
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        # Test
        success = await cost_governor.set_budget(
            "test_profile", 10.0, 50.0, 200.0, mock_session
        )
        
        # Assertions
        assert success is True
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        cost_governor.redis_client.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_set_budget_update_existing(self, cost_governor, mock_session):
        """Test updating existing budget."""
        # Mock existing budget
        mock_budget = Mock()
        mock_budget.daily_limit = 5.0
        
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_budget
        mock_session.execute.return_value = mock_result
        
        # Test
        success = await cost_governor.set_budget(
            "test_profile", 10.0, 50.0, 200.0, mock_session
        )
        
        # Assertions
        assert success is True
        assert mock_budget.daily_limit == 10.0
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_budget_from_cache(self, cost_governor, mock_session, sample_budget):
        """Test getting budget from Redis cache."""
        # Mock Redis cache hit
        cost_governor.redis_client.get.return_value = json.dumps(sample_budget)
        
        # Test
        budget = await cost_governor.get_budget("test_profile", mock_session)
        
        # Assertions
        assert budget == sample_budget
        cost_governor.redis_client.get.assert_called_once_with("budget:test_profile")
        # Should not query database
        mock_session.execute.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_budget_from_database(self, cost_governor, mock_session, sample_budget):
        """Test getting budget from database when not cached."""
        # Mock Redis cache miss
        cost_governor.redis_client.get.return_value = None
        
        # Mock database result
        mock_budget = Mock()
        mock_budget.daily_limit = sample_budget["daily_limit"]
        mock_budget.weekly_limit = sample_budget["weekly_limit"]
        mock_budget.monthly_limit = sample_budget["monthly_limit"]
        mock_budget.currency = sample_budget["currency"]
        mock_budget.alert_threshold = sample_budget["alert_threshold"]
        mock_budget.enabled = sample_budget["enabled"]
        
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_budget
        mock_session.execute.return_value = mock_result
        
        # Test
        budget = await cost_governor.get_budget("test_profile", mock_session)
        
        # Assertions
        assert budget["daily_limit"] == sample_budget["daily_limit"]
        mock_session.execute.assert_called_once()
        # Should cache the result
        cost_governor.redis_client.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_track_cost(self, cost_governor, mock_session):
        """Test cost tracking."""
        # Test
        success = await cost_governor.track_cost(
            "test_profile", "page_load", 0.001, mock_session,
            metadata={"url": "https://example.com"}
        )
        
        # Assertions
        assert success is True
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        
        # Check Redis counter updates
        assert cost_governor.redis_client.incrbyfloat.call_count == 3  # daily, weekly, monthly
        assert cost_governor.redis_client.expire.call_count == 3
    
    @pytest.mark.asyncio
    async def test_get_spending(self, cost_governor):
        """Test getting spending amounts."""
        # Mock Redis responses
        cost_governor.redis_client.get.side_effect = ["5.50", "25.75", "150.25"]
        
        # Test
        daily = await cost_governor.get_spending("test_profile", "daily")
        weekly = await cost_governor.get_spending("test_profile", "weekly")
        monthly = await cost_governor.get_spending("test_profile", "monthly")
        
        # Assertions
        assert daily == 5.50
        assert weekly == 25.75
        assert monthly == 150.25
    
    @pytest.mark.asyncio
    async def test_check_before_scrape_approved(self, cost_governor, mock_session, sample_budget):
        """Test budget check that gets approved."""
        # Mock budget
        with patch.object(cost_governor, 'get_budget', return_value=sample_budget):
            with patch.object(cost_governor, 'estimate_operation_cost', return_value=1.0):
                with patch.object(cost_governor, 'get_spending') as mock_get_spending:
                    # Mock current spending (well under limits)
                    mock_get_spending.side_effect = [2.0, 10.0, 50.0]  # daily, weekly, monthly
                    
                    # Test
                    result = await cost_governor.check_before_scrape(
                        "test_profile", 100, mock_session
                    )
        
        # Assertions
        assert result["approved"] is True
        assert result["estimated_cost"] == 1.0
        assert "current_spending" in result
    
    @pytest.mark.asyncio
    async def test_check_before_scrape_daily_limit_exceeded(self, cost_governor, mock_session, sample_budget):
        """Test budget check that exceeds daily limit."""
        # Mock budget
        with patch.object(cost_governor, 'get_budget', return_value=sample_budget):
            with patch.object(cost_governor, 'estimate_operation_cost', return_value=2.0):
                with patch.object(cost_governor, 'get_spending') as mock_get_spending:
                    # Mock current spending (near daily limit)
                    mock_get_spending.side_effect = [9.5, 20.0, 100.0]  # daily, weekly, monthly
                    
                    # Test
                    result = await cost_governor.check_before_scrape(
                        "test_profile", 100, mock_session
                    )
        
        # Assertions
        assert result["approved"] is False
        assert result["reason"] == "daily_limit_exceeded"
        assert result["current_spending"] == 9.5
        assert result["limit"] == 10.0
    
    @pytest.mark.asyncio
    async def test_check_before_scrape_with_alerts(self, cost_governor, mock_session, sample_budget):
        """Test budget check that triggers alerts."""
        # Mock budget
        with patch.object(cost_governor, 'get_budget', return_value=sample_budget):
            with patch.object(cost_governor, 'estimate_operation_cost', return_value=1.0):
                with patch.object(cost_governor, 'get_spending') as mock_get_spending:
                    # Mock spending that will trigger alert threshold (80% of 10.0 = 8.0)
                    mock_get_spending.side_effect = [7.5, 15.0, 75.0]  # daily, weekly, monthly
                    
                    with patch.object(cost_governor, 'send_alert') as mock_send_alert:
                        # Test
                        result = await cost_governor.check_before_scrape(
                            "test_profile", 100, mock_session
                        )
        
        # Assertions
        assert result["approved"] is True
        assert "daily_threshold_reached" in result["alerts"]
        mock_send_alert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_estimate_operation_cost(self, cost_governor):
        """Test operation cost estimation."""
        # Test
        cost = await cost_governor.estimate_operation_cost(50)
        
        # Assertions
        assert cost > 0
        # Should include browser instance cost + action costs + page load costs
        expected_min = cost_governor.cost_estimates["browser_instance"]
        assert cost >= expected_min
    
    @pytest.mark.asyncio
    async def test_send_alert(self, cost_governor):
        """Test alert sending."""
        # Register mock callback
        mock_callback = AsyncMock()
        cost_governor.register_alert_callback(mock_callback)
        
        alert_types = ["daily_threshold_reached"]
        spending_data = {"daily": 8.0, "weekly": 20.0, "monthly": 100.0}
        
        # Test
        await cost_governor.send_alert("test_profile", alert_types, spending_data)
        
        # Assertions
        mock_callback.assert_called_once()
        call_args = mock_callback.call_args[0][0]
        assert call_args["profile_id"] == "test_profile"
        assert call_args["alert_types"] == alert_types
        
        # Should store alert in Redis
        cost_governor.redis_client.setex.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_cost_report(self, cost_governor, mock_session):
        """Test cost report generation."""
        # Mock database results
        mock_operation_result = Mock()
        mock_operation_result.operation_type = "page_load"
        mock_operation_result.total_cost = 5.0
        mock_operation_result.operation_count = 100
        
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([mock_operation_result]))
        
        mock_total_result = Mock()
        mock_total_result.scalar.return_value = 25.0
        
        mock_session.execute.side_effect = [mock_result, mock_total_result]
        
        # Mock other methods
        with patch.object(cost_governor, 'get_spending') as mock_get_spending:
            mock_get_spending.side_effect = [2.0, 10.0, 50.0]
            
            with patch.object(cost_governor, 'get_budget', return_value={"daily_limit": 10.0}):
                # Test
                report = await cost_governor.get_cost_report("test_profile", mock_session)
        
        # Assertions
        assert report["profile_id"] == "test_profile"
        assert report["total_cost"] == 25.0
        assert "page_load" in report["cost_by_operation"]
        assert report["cost_by_operation"]["page_load"]["total_cost"] == 5.0
        assert "current_spending" in report
        assert "budget" in report


class TestGlobalFunctions:
    """Test global cost governor functions."""
    
    def test_init_cost_governor(self, mock_redis):
        """Test cost governor initialization."""
        governor = init_cost_governor(mock_redis)
        
        assert governor is not None
        assert governor.redis_client == mock_redis
        assert get_cost_governor() == governor
    
    def test_get_cost_governor_none(self):
        """Test getting cost governor when not initialized."""
        # Reset global instance
        import src.braf.utils.cost_governance
        src.braf.utils.cost_governance._cost_governor = None
        
        governor = get_cost_governor()
        assert governor is None


if __name__ == "__main__":
    pytest.main([__file__])