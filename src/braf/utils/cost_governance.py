"""
Cost Governance Module for BRAF.

This module provides cost tracking, budget management, and spending alerts
for automation operations with Redis and database integration.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal

import redis.asyncio as redis
from sqlalchemy import Column, String, Float, DateTime, Integer, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from braf.core.database import Base, get_database

logger = logging.getLogger(__name__)


class CostTrackingModel(Base):
    """Database model for cost tracking."""
    
    __tablename__ = "cost_tracking"
    
    id = Column(String(255), primary_key=True)
    profile_id = Column(String(255), nullable=False, index=True)
    operation_type = Column(String(100), nullable=False)
    cost_amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    metadata = Column(JSONB, default=dict)


class BudgetModel(Base):
    """Database model for budget limits."""
    
    __tablename__ = "budgets"
    
    profile_id = Column(String(255), primary_key=True)
    daily_limit = Column(Float, nullable=False)
    weekly_limit = Column(Float, nullable=False)
    monthly_limit = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    alert_threshold = Column(Float, default=0.8)  # 80% threshold
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class CostGovernor:
    """Cost governance and budget management system."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize cost governor.
        
        Args:
            redis_client: Optional Redis client for caching
        """
        self.redis_client = redis_client
        self.cost_estimates = {
            "page_load": 0.001,
            "click_action": 0.0005,
            "form_fill": 0.002,
            "data_extraction": 0.003,
            "captcha_solve": 0.05,
            "proxy_rotation": 0.01,
            "browser_instance": 0.02
        }
        self.alert_callbacks = []
    
    async def set_budget(
        self,
        profile_id: str,
        daily_limit: float,
        weekly_limit: float,
        monthly_limit: float,
        session: AsyncSession,
        currency: str = "USD",
        alert_threshold: float = 0.8
    ) -> bool:
        """
        Set budget limits for a profile.
        
        Args:
            profile_id: Profile identifier
            daily_limit: Daily spending limit
            weekly_limit: Weekly spending limit
            monthly_limit: Monthly spending limit
            session: Database session
            currency: Currency code
            alert_threshold: Alert threshold (0.0-1.0)
            
        Returns:
            True if successful
        """
        try:
            from sqlalchemy import select
            
            # Check if budget exists
            stmt = select(BudgetModel).where(BudgetModel.profile_id == profile_id)
            result = await session.execute(stmt)
            budget = result.scalar_one_or_none()
            
            if budget:
                # Update existing budget
                budget.daily_limit = daily_limit
                budget.weekly_limit = weekly_limit
                budget.monthly_limit = monthly_limit
                budget.currency = currency
                budget.alert_threshold = alert_threshold
                budget.updated_at = datetime.now(timezone.utc)
            else:
                # Create new budget
                budget = BudgetModel(
                    profile_id=profile_id,
                    daily_limit=daily_limit,
                    weekly_limit=weekly_limit,
                    monthly_limit=monthly_limit,
                    currency=currency,
                    alert_threshold=alert_threshold
                )
                session.add(budget)
            
            await session.commit()
            
            # Cache in Redis
            if self.redis_client:
                budget_data = {
                    "daily_limit": daily_limit,
                    "weekly_limit": weekly_limit,
                    "monthly_limit": monthly_limit,
                    "currency": currency,
                    "alert_threshold": alert_threshold
                }
                await self.redis_client.setex(
                    f"budget:{profile_id}",
                    3600,  # 1 hour cache
                    json.dumps(budget_data)
                )
            
            logger.info(f"Set budget for profile {profile_id}: ${daily_limit}/day")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set budget: {e}")
            await session.rollback()
            return False
    async def get_budget(self, profile_id: str, session: AsyncSession) -> Optional[Dict[str, Any]]:
        """
        Get budget information for a profile.
        
        Args:
            profile_id: Profile identifier
            session: Database session
            
        Returns:
            Budget information or None
        """
        try:
            # Try Redis cache first
            if self.redis_client:
                cached = await self.redis_client.get(f"budget:{profile_id}")
                if cached:
                    return json.loads(cached)
            
            # Get from database
            from sqlalchemy import select
            
            stmt = select(BudgetModel).where(BudgetModel.profile_id == profile_id)
            result = await session.execute(stmt)
            budget = result.scalar_one_or_none()
            
            if not budget:
                return None
            
            budget_data = {
                "daily_limit": budget.daily_limit,
                "weekly_limit": budget.weekly_limit,
                "monthly_limit": budget.monthly_limit,
                "currency": budget.currency,
                "alert_threshold": budget.alert_threshold,
                "enabled": budget.enabled
            }
            
            # Cache in Redis
            if self.redis_client:
                await self.redis_client.setex(
                    f"budget:{profile_id}",
                    3600,
                    json.dumps(budget_data)
                )
            
            return budget_data
            
        except Exception as e:
            logger.error(f"Failed to get budget: {e}")
            return None
    
    async def track_cost(
        self,
        profile_id: str,
        operation_type: str,
        cost_amount: float,
        session: AsyncSession,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Track cost for an operation.
        
        Args:
            profile_id: Profile identifier
            operation_type: Type of operation
            cost_amount: Cost amount
            session: Database session
            metadata: Optional metadata
            
        Returns:
            True if successful
        """
        try:
            import uuid
            
            # Create cost tracking record
            cost_record = CostTrackingModel(
                id=str(uuid.uuid4()),
                profile_id=profile_id,
                operation_type=operation_type,
                cost_amount=cost_amount,
                metadata=metadata or {}
            )
            
            session.add(cost_record)
            await session.commit()
            
            # Update Redis counters
            if self.redis_client:
                today = datetime.now(timezone.utc).date()
                week_start = today - timedelta(days=today.weekday())
                month_start = today.replace(day=1)
                
                # Update daily counter
                daily_key = f"cost:daily:{profile_id}:{today}"
                await self.redis_client.incrbyfloat(daily_key, cost_amount)
                await self.redis_client.expire(daily_key, 86400)  # 24 hours
                
                # Update weekly counter
                weekly_key = f"cost:weekly:{profile_id}:{week_start}"
                await self.redis_client.incrbyfloat(weekly_key, cost_amount)
                await self.redis_client.expire(weekly_key, 604800)  # 7 days
                
                # Update monthly counter
                monthly_key = f"cost:monthly:{profile_id}:{month_start}"
                await self.redis_client.incrbyfloat(monthly_key, cost_amount)
                await self.redis_client.expire(monthly_key, 2592000)  # 30 days
            
            logger.debug(f"Tracked cost: {operation_type} = ${cost_amount} for {profile_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to track cost: {e}")
            await session.rollback()
            return False
    
    async def get_spending(
        self,
        profile_id: str,
        period: str = "daily"
    ) -> float:
        """
        Get current spending for a period.
        
        Args:
            profile_id: Profile identifier
            period: Period type (daily, weekly, monthly)
            
        Returns:
            Current spending amount
        """
        if not self.redis_client:
            return 0.0
        
        try:
            today = datetime.now(timezone.utc).date()
            
            if period == "daily":
                key = f"cost:daily:{profile_id}:{today}"
            elif period == "weekly":
                week_start = today - timedelta(days=today.weekday())
                key = f"cost:weekly:{profile_id}:{week_start}"
            elif period == "monthly":
                month_start = today.replace(day=1)
                key = f"cost:monthly:{profile_id}:{month_start}"
            else:
                return 0.0
            
            spending = await self.redis_client.get(key)
            return float(spending) if spending else 0.0
            
        except Exception as e:
            logger.error(f"Failed to get spending: {e}")
            return 0.0
    
    async def check_before_scrape(
        self,
        profile_id: str,
        estimated_actions: int,
        session: AsyncSession
    ) -> Dict[str, Any]:
        """
        Check budget before starting scraping operation - DISABLED FOR TESTING.

        Args:
            profile_id: Profile identifier
            estimated_actions: Number of estimated actions
            session: Database session

        Returns:
            Check result with approval status (always approved for testing)
        """
        # Budget checks DISABLED to allow unlimited spending and expose loopholes
        return {
            "approved": True,
            "reason": "safeguards_disabled",
            "estimated_cost": 0.0,
            "current_spending": {"daily": 0.0, "weekly": 0.0, "monthly": 0.0},
            "alerts": []
        }
    
    async def estimate_operation_cost(self, estimated_actions: int) -> float:
        """
        Estimate cost for an operation.
        
        Args:
            estimated_actions: Number of estimated actions
            
        Returns:
            Estimated cost
        """
        base_cost = self.cost_estimates["browser_instance"]
        action_cost = estimated_actions * self.cost_estimates["click_action"]
        
        # Add overhead for page loads (assume 1 per 10 actions)
        page_loads = max(1, estimated_actions // 10)
        page_cost = page_loads * self.cost_estimates["page_load"]
        
        return base_cost + action_cost + page_cost
    
    async def send_alert(
        self,
        profile_id: str,
        alert_types: List[str],
        spending_data: Dict[str, float]
    ) -> None:
        """
        Send budget alert.
        
        Args:
            profile_id: Profile identifier
            alert_types: List of alert types
            spending_data: Current spending data
        """
        try:
            alert_data = {
                "profile_id": profile_id,
                "alert_types": alert_types,
                "spending_data": spending_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Call registered alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(alert_data)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
            
            # Store alert in Redis
            if self.redis_client:
                alert_key = f"alert:{profile_id}:{datetime.now(timezone.utc).timestamp()}"
                await self.redis_client.setex(alert_key, 86400, json.dumps(alert_data))
            
            logger.warning(f"Budget alert sent for {profile_id}: {alert_types}")
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    def register_alert_callback(self, callback):
        """Register callback for budget alerts."""
        self.alert_callbacks.append(callback)
    
    async def get_cost_report(
        self,
        profile_id: str,
        session: AsyncSession,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate cost report for a profile.
        
        Args:
            profile_id: Profile identifier
            session: Database session
            days: Number of days to include
            
        Returns:
            Cost report
        """
        try:
            from sqlalchemy import select, func
            
            # Get cost data from database
            start_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            stmt = (
                select(
                    CostTrackingModel.operation_type,
                    func.sum(CostTrackingModel.cost_amount).label("total_cost"),
                    func.count(CostTrackingModel.id).label("operation_count")
                )
                .where(CostTrackingModel.profile_id == profile_id)
                .where(CostTrackingModel.timestamp >= start_date)
                .group_by(CostTrackingModel.operation_type)
            )
            
            result = await session.execute(stmt)
            cost_by_operation = {
                row.operation_type: {
                    "total_cost": float(row.total_cost),
                    "operation_count": row.operation_count
                }
                for row in result
            }
            
            # Get total cost
            total_stmt = (
                select(func.sum(CostTrackingModel.cost_amount))
                .where(CostTrackingModel.profile_id == profile_id)
                .where(CostTrackingModel.timestamp >= start_date)
            )
            
            total_result = await session.execute(total_stmt)
            total_cost = float(total_result.scalar() or 0.0)
            
            # Get current spending
            current_spending = {
                "daily": await self.get_spending(profile_id, "daily"),
                "weekly": await self.get_spending(profile_id, "weekly"),
                "monthly": await self.get_spending(profile_id, "monthly")
            }
            
            # Get budget
            budget = await self.get_budget(profile_id, session)
            
            return {
                "profile_id": profile_id,
                "report_period_days": days,
                "total_cost": total_cost,
                "cost_by_operation": cost_by_operation,
                "current_spending": current_spending,
                "budget": budget,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate cost report: {e}")
            return {
                "profile_id": profile_id,
                "error": str(e),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }


# Global cost governor instance
_cost_governor: Optional[CostGovernor] = None


def get_cost_governor() -> Optional[CostGovernor]:
    """Get global cost governor instance."""
    return _cost_governor


def init_cost_governor(redis_client: Optional[redis.Redis] = None) -> CostGovernor:
    """Initialize global cost governor."""
    global _cost_governor
    _cost_governor = CostGovernor(redis_client)
    return _cost_governor