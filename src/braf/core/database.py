"""
Database models and connection management for BRAF.

This module provides SQLAlchemy models that correspond to the Pydantic models
and handles database connections, sessions, and migrations.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func

Base = declarative_base()


class ProfileModel(Base):
    """Database model for user profiles."""
    
    __tablename__ = "profiles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    fingerprint_id = Column(String(255), nullable=False, index=True)
    proxy_config = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used = Column(DateTime(timezone=True), nullable=True)
    session_count = Column(Integer, default=0)
    detection_score = Column(Float, default=0.0)
    profile_metadata = Column(JSONB, default=dict)
    
    # Relationships
    tasks = relationship("AutomationTaskModel", back_populates="profile")
    compliance_logs = relationship("ComplianceLogModel", back_populates="profile")
    credentials = relationship("EncryptedCredentialModel", back_populates="profile")


class FingerprintModel(Base):
    """Database model for browser fingerprints."""
    
    __tablename__ = "fingerprints"
    
    id = Column(String(255), primary_key=True)
    user_agent = Column(Text, nullable=False)
    screen_width = Column(Integer, nullable=False)
    screen_height = Column(Integer, nullable=False)
    timezone = Column(String(100), nullable=False)
    webgl_vendor = Column(String(255), nullable=False)
    webgl_renderer = Column(String(255), nullable=False)
    canvas_hash = Column(String(255), nullable=False)
    audio_context_hash = Column(String(255), nullable=False)
    fonts = Column(JSONB, nullable=False)
    plugins = Column(JSONB, default=list)
    languages = Column(JSONB, default=list)
    platform = Column(String(50), default="Win32")
    hardware_concurrency = Column(Integer, default=4)
    device_memory = Column(Integer, default=8)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used = Column(DateTime(timezone=True), nullable=True)
    usage_count = Column(Integer, default=0)


class AutomationTaskModel(Base):
    """Database model for automation tasks."""
    
    __tablename__ = "automation_tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    profile_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id"), nullable=False)
    target_url = Column(Text, nullable=False)
    actions = Column(JSONB, nullable=False)
    constraints = Column(JSONB, nullable=False)
    priority = Column(Integer, default=0, index=True)
    status = Column(String(50), default="pending", index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    assigned_worker = Column(String(255), nullable=True, index=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    result = Column(JSONB, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    profile = relationship("ProfileModel", back_populates="tasks")


class ComplianceLogModel(Base):
    """Database model for compliance and audit logs."""
    
    __tablename__ = "compliance_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    action_type = Column(String(100), nullable=False, index=True)
    target_url = Column(Text, nullable=True)
    profile_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id"), nullable=False)
    worker_id = Column(String(255), nullable=False, index=True)
    detection_score = Column(Float, nullable=False)
    ethical_check_passed = Column(Boolean, nullable=False, index=True)
    authorization_token = Column(String(255), nullable=False)
    log_metadata = Column(JSONB, default=dict)
    
    # Relationships
    profile = relationship("ProfileModel", back_populates="compliance_logs")


class EncryptedCredentialModel(Base):
    """Database model for encrypted credential storage."""
    
    __tablename__ = "encrypted_credentials"
    
    profile_id = Column(UUID(as_uuid=True), ForeignKey("profiles.id"), primary_key=True)
    encrypted_data = Column(Text, nullable=False)  # Base64 encoded encrypted data
    salt = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    profile = relationship("ProfileModel", back_populates="credentials")


class WorkerNodeModel(Base):
    """Database model for worker node registration and status."""
    
    __tablename__ = "worker_nodes"
    
    id = Column(String(255), primary_key=True)
    status = Column(String(50), nullable=False, index=True)
    current_tasks = Column(Integer, default=0)
    max_tasks = Column(Integer, default=5)
    cpu_usage = Column(Float, default=0.0)
    memory_usage = Column(Float, default=0.0)
    last_heartbeat = Column(DateTime(timezone=True), server_default=func.now())
    version = Column(String(50), nullable=False)
    capabilities = Column(JSONB, default=list)
    configuration = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SystemMetricsModel(Base):
    """Database model for system performance metrics."""
    
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    metric_type = Column(String(100), nullable=False, index=True)
    metric_name = Column(String(255), nullable=False, index=True)
    value = Column(Float, nullable=False)
    labels = Column(JSONB, default=dict)
    worker_id = Column(String(255), nullable=True, index=True)


class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, database_url: str, echo: bool = False):
        """Initialize database manager with connection URL."""
        self.database_url = database_url
        self.echo = echo
        
        # Create async engine
        self.async_engine = create_async_engine(
            database_url,
            echo=echo,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        
        # Create async session factory
        self.async_session_factory = async_sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        # Create sync engine for migrations
        sync_url = database_url.replace("+asyncpg", "")
        self.sync_engine = create_engine(sync_url, echo=echo)
        self.sync_session_factory = sessionmaker(bind=self.sync_engine)
    
    async def get_async_session(self) -> AsyncSession:
        """Get async database session."""
        async with self.async_session_factory() as session:
            try:
                yield session
            finally:
                await session.close()
    
    def get_sync_session(self):
        """Get sync database session for migrations."""
        with self.sync_session_factory() as session:
            try:
                yield session
            finally:
                session.close()
    
    async def create_tables(self):
        """Create all database tables."""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self):
        """Drop all database tables."""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    async def close(self):
        """Close database connections."""
        await self.async_engine.dispose()
        self.sync_engine.dispose()


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


def init_database(database_url: str, echo: bool = False) -> DatabaseManager:
    """Initialize global database manager."""
    global db_manager
    db_manager = DatabaseManager(database_url, echo)
    return db_manager


def get_database() -> DatabaseManager:
    """Get global database manager instance."""
    if db_manager is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return db_manager


async def get_db_session() -> AsyncSession:
    """Dependency for getting database session in FastAPI."""
    db = get_database()
    async with db.async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()