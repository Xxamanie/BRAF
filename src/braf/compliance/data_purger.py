"""
Data Purging and Compliance Module for BRAF.

This module provides GDPR-compliant data purging, user data anonymization,
and compliance management with referential integrity maintenance.
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set

from sqlalchemy import Column, String, DateTime, Boolean, Text, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func

from braf.core.database import Base, get_database

logger = logging.getLogger(__name__)


class DataPurgeLogModel(Base):
    """Database model for data purge operations."""
    
    __tablename__ = "data_purge_logs"
    
    id = Column(String(255), primary_key=True)
    profile_id = Column(String(255), nullable=False, index=True)
    purge_type = Column(String(50), nullable=False)  # full, partial, anonymize
    requested_at = Column(DateTime(timezone=True), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    tables_affected = Column(JSONB, default=list)
    records_processed = Column(Integer, default=0)
    records_deleted = Column(Integer, default=0)
    records_anonymized = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    metadata = Column(JSONB, default=dict)


class ComplianceRequestModel(Base):
    """Database model for compliance requests."""
    
    __tablename__ = "compliance_requests"
    
    id = Column(String(255), primary_key=True)
    profile_id = Column(String(255), nullable=False, index=True)
    request_type = Column(String(50), nullable=False)  # deletion, export, anonymize
    status = Column(String(50), default="pending")
    requested_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    processed_at = Column(DateTime(timezone=True), nullable=True)
    requester_info = Column(JSONB, default=dict)
    verification_token = Column(String(255), nullable=True)
    verified = Column(Boolean, default=False)
    metadata = Column(JSONB, default=dict)


async def purge_user_data(
    profile_id: str,
    session: AsyncSession,
    purge_type: str = "full",
    verification_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Purge user data with GDPR compliance.
    
    Args:
        profile_id: Profile identifier to purge
        session: Database session
        purge_type: Type of purge (full, partial, anonymize)
        verification_token: Optional verification token
        
    Returns:
        Purge operation result
    """
    purge_id = str(uuid.uuid4())
    
    try:
        # Create purge log
        purge_log = DataPurgeLogModel(
            id=purge_id,
            profile_id=profile_id,
            purge_type=purge_type,
            requested_at=datetime.now(timezone.utc),
            status="running",
            metadata={"verification_token": verification_token}
        )
        session.add(purge_log)
        await session.commit()
        
        logger.info(f"Starting {purge_type} purge for profile {profile_id}")
        
        # Start purge operation
        purge_log.started_at = datetime.now(timezone.utc)
        
        if purge_type == "full":
            result = await _full_data_purge(profile_id, session, purge_log)
        elif purge_type == "partial":
            result = await _partial_data_purge(profile_id, session, purge_log)
        elif purge_type == "anonymize":
            result = await _anonymize_user_data(profile_id, session, purge_log)
        else:
            raise ValueError(f"Unknown purge type: {purge_type}")
        
        # Update purge log
        purge_log.completed_at = datetime.now(timezone.utc)
        purge_log.status = "completed" if result["success"] else "failed"
        purge_log.records_processed = result.get("records_processed", 0)
        purge_log.records_deleted = result.get("records_deleted", 0)
        purge_log.records_anonymized = result.get("records_anonymized", 0)
        purge_log.tables_affected = result.get("tables_affected", [])
        
        if not result["success"]:
            purge_log.error_message = result.get("error", "Unknown error")
        
        await session.commit()
        
        logger.info(f"Purge operation {purge_id} completed: {result}")
        return {
            "purge_id": purge_id,
            "success": result["success"],
            "records_processed": result.get("records_processed", 0),
            "records_deleted": result.get("records_deleted", 0),
            "records_anonymized": result.get("records_anonymized", 0),
            "tables_affected": result.get("tables_affected", [])
        }
        
    except Exception as e:
        logger.error(f"Purge operation failed: {e}")
        
        # Update purge log with error
        try:
            purge_log.status = "failed"
            purge_log.error_message = str(e)
            purge_log.completed_at = datetime.now(timezone.utc)
            await session.commit()
        except Exception as log_error:
            logger.error(f"Failed to update purge log: {log_error}")
        
        return {
            "purge_id": purge_id,
            "success": False,
            "error": str(e)
        }


async def _full_data_purge(
    profile_id: str,
    session: AsyncSession,
    purge_log: DataPurgeLogModel
) -> Dict[str, Any]:
    """Perform full data purge for a profile."""
    from braf.core.database import (
        ProfileModel, AutomationTaskModel, ComplianceLogModel,
        EncryptedCredentialModel, SystemMetricsModel
    )
    from src.braf.utils.state_management import BrowserStateModel
    from src.braf.utils.cost_governance import CostTrackingModel
    
    tables_affected = []
    records_deleted = 0
    
    try:
        # Delete in order to maintain referential integrity
        
        # 1. Delete browser states
        stmt = delete(BrowserStateModel).where(BrowserStateModel.profile_id == profile_id)
        result = await session.execute(stmt)
        if result.rowcount > 0:
            tables_affected.append("browser_states")
            records_deleted += result.rowcount
        
        # 2. Delete cost tracking
        stmt = delete(CostTrackingModel).where(CostTrackingModel.profile_id == profile_id)
        result = await session.execute(stmt)
        if result.rowcount > 0:
            tables_affected.append("cost_tracking")
            records_deleted += result.rowcount
        
        # 3. Delete system metrics
        stmt = delete(SystemMetricsModel).where(SystemMetricsModel.worker_id == profile_id)
        result = await session.execute(stmt)
        if result.rowcount > 0:
            tables_affected.append("system_metrics")
            records_deleted += result.rowcount
        
        # 4. Delete encrypted credentials
        stmt = delete(EncryptedCredentialModel).where(EncryptedCredentialModel.profile_id == profile_id)
        result = await session.execute(stmt)
        if result.rowcount > 0:
            tables_affected.append("encrypted_credentials")
            records_deleted += result.rowcount
        
        # 5. Delete compliance logs
        stmt = delete(ComplianceLogModel).where(ComplianceLogModel.profile_id == profile_id)
        result = await session.execute(stmt)
        if result.rowcount > 0:
            tables_affected.append("compliance_logs")
            records_deleted += result.rowcount
        
        # 6. Delete automation tasks
        stmt = delete(AutomationTaskModel).where(AutomationTaskModel.profile_id == profile_id)
        result = await session.execute(stmt)
        if result.rowcount > 0:
            tables_affected.append("automation_tasks")
            records_deleted += result.rowcount
        
        # 7. Delete profile (main record)
        stmt = delete(ProfileModel).where(ProfileModel.id == profile_id)
        result = await session.execute(stmt)
        if result.rowcount > 0:
            tables_affected.append("profiles")
            records_deleted += result.rowcount
        
        await session.commit()
        
        return {
            "success": True,
            "records_processed": records_deleted,
            "records_deleted": records_deleted,
            "records_anonymized": 0,
            "tables_affected": tables_affected
        }
        
    except Exception as e:
        await session.rollback()
        logger.error(f"Full purge failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "records_processed": records_deleted,
            "tables_affected": tables_affected
        }


async def _partial_data_purge(
    profile_id: str,
    session: AsyncSession,
    purge_log: DataPurgeLogModel
) -> Dict[str, Any]:
    """Perform partial data purge (keep profile, remove sensitive data)."""
    from braf.core.database import ComplianceLogModel, EncryptedCredentialModel
    from src.braf.utils.state_management import BrowserStateModel
    from src.braf.utils.cost_governance import CostTrackingModel
    
    tables_affected = []
    records_deleted = 0
    
    try:
        # Delete sensitive data but keep profile and task history
        
        # 1. Delete browser states (contains cookies and storage)
        stmt = delete(BrowserStateModel).where(BrowserStateModel.profile_id == profile_id)
        result = await session.execute(stmt)
        if result.rowcount > 0:
            tables_affected.append("browser_states")
            records_deleted += result.rowcount
        
        # 2. Delete encrypted credentials
        stmt = delete(EncryptedCredentialModel).where(EncryptedCredentialModel.profile_id == profile_id)
        result = await session.execute(stmt)
        if result.rowcount > 0:
            tables_affected.append("encrypted_credentials")
            records_deleted += result.rowcount
        
        # 3. Delete old compliance logs (keep recent ones)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
        stmt = (
            delete(ComplianceLogModel)
            .where(ComplianceLogModel.profile_id == profile_id)
            .where(ComplianceLogModel.timestamp < cutoff_date)
        )
        result = await session.execute(stmt)
        if result.rowcount > 0:
            tables_affected.append("compliance_logs")
            records_deleted += result.rowcount
        
        # 4. Delete old cost tracking (keep recent ones)
        stmt = (
            delete(CostTrackingModel)
            .where(CostTrackingModel.profile_id == profile_id)
            .where(CostTrackingModel.timestamp < cutoff_date)
        )
        result = await session.execute(stmt)
        if result.rowcount > 0:
            tables_affected.append("cost_tracking")
            records_deleted += result.rowcount
        
        await session.commit()
        
        return {
            "success": True,
            "records_processed": records_deleted,
            "records_deleted": records_deleted,
            "records_anonymized": 0,
            "tables_affected": tables_affected
        }
        
    except Exception as e:
        await session.rollback()
        logger.error(f"Partial purge failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "records_processed": records_deleted,
            "tables_affected": tables_affected
        }


async def _anonymize_user_data(
    profile_id: str,
    session: AsyncSession,
    purge_log: DataPurgeLogModel
) -> Dict[str, Any]:
    """Anonymize user data while preserving analytics."""
    from braf.core.database import ProfileModel, AutomationTaskModel, ComplianceLogModel
    from src.braf.utils.state_management import BrowserStateModel
    
    tables_affected = []
    records_anonymized = 0
    
    try:
        # Generate anonymous ID
        anonymous_id = f"anon_{hashlib.sha256(profile_id.encode()).hexdigest()[:16]}"
        
        # 1. Anonymize profile
        stmt = (
            update(ProfileModel)
            .where(ProfileModel.id == profile_id)
            .values(
                fingerprint_id=f"anon_fingerprint_{anonymous_id}",
                proxy_config=None,
                profile_metadata={"anonymized": True, "anonymized_at": datetime.now(timezone.utc).isoformat()}
            )
        )
        result = await session.execute(stmt)
        if result.rowcount > 0:
            tables_affected.append("profiles")
            records_anonymized += result.rowcount
        
        # 2. Anonymize automation tasks (remove sensitive URLs and data)
        stmt = (
            update(AutomationTaskModel)
            .where(AutomationTaskModel.profile_id == profile_id)
            .values(
                target_url="https://anonymized.example.com",
                result={"anonymized": True},
                error_message=None
            )
        )
        result = await session.execute(stmt)
        if result.rowcount > 0:
            tables_affected.append("automation_tasks")
            records_anonymized += result.rowcount
        
        # 3. Anonymize compliance logs
        stmt = (
            update(ComplianceLogModel)
            .where(ComplianceLogModel.profile_id == profile_id)
            .values(
                target_url="https://anonymized.example.com",
                authorization_token="anonymized_token",
                log_metadata={"anonymized": True}
            )
        )
        result = await session.execute(stmt)
        if result.rowcount > 0:
            tables_affected.append("compliance_logs")
            records_anonymized += result.rowcount
        
        # 4. Delete browser states (too sensitive to anonymize)
        stmt = delete(BrowserStateModel).where(BrowserStateModel.profile_id == profile_id)
        result = await session.execute(stmt)
        if result.rowcount > 0:
            tables_affected.append("browser_states")
        
        await session.commit()
        
        return {
            "success": True,
            "records_processed": records_anonymized,
            "records_deleted": 0,
            "records_anonymized": records_anonymized,
            "tables_affected": tables_affected,
            "anonymous_id": anonymous_id
        }
        
    except Exception as e:
        await session.rollback()
        logger.error(f"Anonymization failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "records_processed": records_anonymized,
            "tables_affected": tables_affected
        }


async def create_compliance_request(
    profile_id: str,
    request_type: str,
    session: AsyncSession,
    requester_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a compliance request (GDPR deletion, export, etc.).
    
    Args:
        profile_id: Profile identifier
        request_type: Type of request (deletion, export, anonymize)
        session: Database session
        requester_info: Information about requester
        
    Returns:
        Request creation result
    """
    try:
        request_id = str(uuid.uuid4())
        verification_token = str(uuid.uuid4())
        
        compliance_request = ComplianceRequestModel(
            id=request_id,
            profile_id=profile_id,
            request_type=request_type,
            requester_info=requester_info or {},
            verification_token=verification_token
        )
        
        session.add(compliance_request)
        await session.commit()
        
        logger.info(f"Created compliance request {request_id} for profile {profile_id}")
        
        return {
            "success": True,
            "request_id": request_id,
            "verification_token": verification_token,
            "status": "pending"
        }
        
    except Exception as e:
        logger.error(f"Failed to create compliance request: {e}")
        await session.rollback()
        return {
            "success": False,
            "error": str(e)
        }


async def verify_compliance_request(
    request_id: str,
    verification_token: str,
    session: AsyncSession
) -> bool:
    """
    Verify a compliance request.
    
    Args:
        request_id: Request identifier
        verification_token: Verification token
        session: Database session
        
    Returns:
        True if verified successfully
    """
    try:
        stmt = (
            select(ComplianceRequestModel)
            .where(ComplianceRequestModel.id == request_id)
            .where(ComplianceRequestModel.verification_token == verification_token)
        )
        result = await session.execute(stmt)
        request = result.scalar_one_or_none()
        
        if not request:
            return False
        
        request.verified = True
        request.processed_at = datetime.now(timezone.utc)
        await session.commit()
        
        logger.info(f"Verified compliance request {request_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to verify compliance request: {e}")
        return False


async def process_compliance_requests(session: AsyncSession, max_requests: int = 10) -> Dict[str, Any]:
    """
    Process pending compliance requests.
    
    Args:
        session: Database session
        max_requests: Maximum requests to process
        
    Returns:
        Processing result
    """
    try:
        # Get pending verified requests
        stmt = (
            select(ComplianceRequestModel)
            .where(ComplianceRequestModel.status == "pending")
            .where(ComplianceRequestModel.verified == True)
            .limit(max_requests)
        )
        result = await session.execute(stmt)
        requests = result.scalars().all()
        
        processed = 0
        failed = 0
        
        for request in requests:
            try:
                # Update status to processing
                request.status = "processing"
                await session.commit()
                
                # Process based on request type
                if request.request_type == "deletion":
                    result = await purge_user_data(
                        request.profile_id,
                        session,
                        purge_type="full",
                        verification_token=request.verification_token
                    )
                elif request.request_type == "anonymize":
                    result = await purge_user_data(
                        request.profile_id,
                        session,
                        purge_type="anonymize",
                        verification_token=request.verification_token
                    )
                elif request.request_type == "export":
                    # TODO: Implement data export
                    result = {"success": False, "error": "Export not implemented"}
                else:
                    result = {"success": False, "error": f"Unknown request type: {request.request_type}"}
                
                # Update request status
                if result["success"]:
                    request.status = "completed"
                    processed += 1
                else:
                    request.status = "failed"
                    request.metadata["error"] = result.get("error", "Unknown error")
                    failed += 1
                
                await session.commit()
                
            except Exception as e:
                logger.error(f"Failed to process compliance request {request.id}: {e}")
                request.status = "failed"
                request.metadata["error"] = str(e)
                failed += 1
                await session.commit()
        
        return {
            "success": True,
            "total_requests": len(requests),
            "processed": processed,
            "failed": failed
        }
        
    except Exception as e:
        logger.error(f"Failed to process compliance requests: {e}")
        return {
            "success": False,
            "error": str(e)
        }


async def get_data_retention_report(session: AsyncSession) -> Dict[str, Any]:
    """
    Generate data retention report.
    
    Args:
        session: Database session
        
    Returns:
        Data retention report
    """
    try:
        from braf.core.database import ProfileModel, AutomationTaskModel, ComplianceLogModel
        from src.braf.utils.state_management import BrowserStateModel
        from src.braf.utils.cost_governance import CostTrackingModel
        
        # Count records by table
        tables_info = {}
        
        # Profiles
        stmt = select(func.count(ProfileModel.id))
        result = await session.execute(stmt)
        tables_info["profiles"] = {"count": result.scalar()}
        
        # Automation tasks
        stmt = select(func.count(AutomationTaskModel.id))
        result = await session.execute(stmt)
        tables_info["automation_tasks"] = {"count": result.scalar()}
        
        # Compliance logs
        stmt = select(func.count(ComplianceLogModel.id))
        result = await session.execute(stmt)
        tables_info["compliance_logs"] = {"count": result.scalar()}
        
        # Browser states
        stmt = select(func.count(BrowserStateModel.profile_id))
        result = await session.execute(stmt)
        tables_info["browser_states"] = {"count": result.scalar()}
        
        # Cost tracking
        stmt = select(func.count(CostTrackingModel.id))
        result = await session.execute(stmt)
        tables_info["cost_tracking"] = {"count": result.scalar()}
        
        # Purge logs
        stmt = select(func.count(DataPurgeLogModel.id))
        result = await session.execute(stmt)
        tables_info["data_purge_logs"] = {"count": result.scalar()}
        
        # Compliance requests
        stmt = select(func.count(ComplianceRequestModel.id))
        result = await session.execute(stmt)
        tables_info["compliance_requests"] = {"count": result.scalar()}
        
        # Get old data counts (older than 90 days)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)
        
        old_data = {}
        
        # Old compliance logs
        stmt = (
            select(func.count(ComplianceLogModel.id))
            .where(ComplianceLogModel.timestamp < cutoff_date)
        )
        result = await session.execute(stmt)
        old_data["compliance_logs"] = result.scalar()
        
        # Old cost tracking
        stmt = (
            select(func.count(CostTrackingModel.id))
            .where(CostTrackingModel.timestamp < cutoff_date)
        )
        result = await session.execute(stmt)
        old_data["cost_tracking"] = result.scalar()
        
        return {
            "success": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tables_info": tables_info,
            "old_data_counts": old_data,
            "retention_cutoff_date": cutoff_date.isoformat(),
            "total_records": sum(info["count"] for info in tables_info.values())
        }
        
    except Exception as e:
        logger.error(f"Failed to generate retention report: {e}")
        return {
            "success": False,
            "error": str(e),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
