from celery import Celery
from datetime import datetime, timedelta
import asyncio
import logging
from typing import Dict, List

from config import Config
from templates.survey_automation import SurveyAutomation
from templates.video_monetization import VideoPlatformAutomation
from payments.mobile_money import MobileMoneyWithdrawal
from payments.crypto_withdrawal import CryptoWithdrawal
from compliance.automation_checker import ComplianceChecker
from security.authentication import SecurityManager

# Initialize Celery
celery = Celery(
    'braf_monetization',
    broker=Config.REDIS_URL,
    backend=Config.REDIS_URL,
    include=['worker']
)

# Celery configuration
celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@celery.task(bind=True, name='process_survey_automation')
def process_survey_automation(self, enterprise_id: str, config: Dict):
    """Process survey automation task"""
    try:
        logger.info(f"Starting survey automation for enterprise {enterprise_id}")
        
        # Initialize survey automation
        survey_automation = SurveyAutomation()
        
        # Run automation in async context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            survey_automation.run_automation(config.get('accounts', []))
        )
        
        loop.close()
        
        logger.info(f"Survey automation completed for enterprise {enterprise_id}")
        return {
            "success": True,
            "enterprise_id": enterprise_id,
            "earnings": result,
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as exc:
        logger.error(f"Survey automation failed for enterprise {enterprise_id}: {str(exc)}")
        self.retry(countdown=60, max_retries=3)

@celery.task(bind=True, name='process_video_automation')
def process_video_automation(self, enterprise_id: str, config: Dict):
    """Process video automation task"""
    try:
        logger.info(f"Starting video automation for enterprise {enterprise_id}")
        
        # Initialize video automation
        video_automation = VideoPlatformAutomation()
        
        # Run automation in async context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            video_automation.watch_videos(
                platform=config.get('platform', 'youtube'),
                device_type=config.get('device_type', 'desktop'),
                count=config.get('video_count', 50)
            )
        )
        
        loop.close()
        
        logger.info(f"Video automation completed for enterprise {enterprise_id}")
        return {
            "success": True,
            "enterprise_id": enterprise_id,
            "earnings": result,
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as exc:
        logger.error(f"Video automation failed for enterprise {enterprise_id}: {str(exc)}")
        self.retry(countdown=60, max_retries=3)

@celery.task(bind=True, name='process_withdrawal')
def process_withdrawal(self, withdrawal_data: Dict):
    """Process withdrawal request"""
    try:
        logger.info(f"Processing withdrawal: {withdrawal_data['transaction_id']}")
        
        withdrawal_type = withdrawal_data.get('type')
        
        if withdrawal_type == 'opay':
            mobile_money = MobileMoneyWithdrawal()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                mobile_money.withdraw_opay(
                    amount=withdrawal_data['amount'],
                    phone_number=withdrawal_data['phone_number'],
                    country=withdrawal_data.get('country', 'NG')
                )
            )
            loop.close()
            
        elif withdrawal_type == 'crypto':
            crypto_withdrawal = CryptoWithdrawal()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                crypto_withdrawal.process_withdrawal(
                    enterprise_id=withdrawal_data['enterprise_id'],
                    amount=withdrawal_data['amount'],
                    cryptocurrency=withdrawal_data['cryptocurrency'],
                    network=withdrawal_data['network'],
                    wallet_address=withdrawal_data['wallet_address']
                )
            )
            loop.close()
            
        else:
            raise ValueError(f"Unsupported withdrawal type: {withdrawal_type}")
        
        logger.info(f"Withdrawal processed successfully: {withdrawal_data['transaction_id']}")
        return result
        
    except Exception as exc:
        logger.error(f"Withdrawal processing failed: {str(exc)}")
        self.retry(countdown=300, max_retries=3)  # Retry after 5 minutes

@celery.task(name='compliance_check')
def compliance_check(enterprise_id: str, activity_data: Dict):
    """Perform compliance check on automation activity"""
    try:
        logger.info(f"Running compliance check for enterprise {enterprise_id}")
        
        compliance_checker = ComplianceChecker()
        
        # Check automation compliance
        compliance_result = compliance_checker.check_automation_compliance(
            template_type=activity_data.get('template_type'),
            automation_config=activity_data.get('config', {})
        )
        
        # Monitor activity patterns
        activity_monitoring = compliance_checker.monitor_activity(
            enterprise_id=enterprise_id,
            activities=activity_data.get('activities', [])
        )
        
        # Log compliance results
        logger.info(f"Compliance check completed for enterprise {enterprise_id}")
        
        return {
            "enterprise_id": enterprise_id,
            "compliance_score": compliance_result.get('score', 0),
            "violations": compliance_result.get('violations', []),
            "risk_level": activity_monitoring.get('risk_level', 'low'),
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as exc:
        logger.error(f"Compliance check failed for enterprise {enterprise_id}: {str(exc)}")
        return {
            "enterprise_id": enterprise_id,
            "error": str(exc),
            "checked_at": datetime.utcnow().isoformat()
        }

@celery.task(name='security_monitoring')
def security_monitoring(enterprise_id: str, activity: Dict):
    """Monitor for suspicious security activities"""
    try:
        logger.info(f"Running security monitoring for enterprise {enterprise_id}")
        
        security_manager = SecurityManager()
        
        # Detect suspicious activity
        suspicious_activity = security_manager.detect_suspicious_activity(
            enterprise_id=enterprise_id,
            activity=activity
        )
        
        if suspicious_activity:
            logger.warning(f"Suspicious activity detected for enterprise {enterprise_id}")
            # Send alert notification
            send_security_alert.delay(enterprise_id, suspicious_activity)
        
        return {
            "enterprise_id": enterprise_id,
            "suspicious_activity": suspicious_activity is not None,
            "monitored_at": datetime.utcnow().isoformat()
        }
        
    except Exception as exc:
        logger.error(f"Security monitoring failed for enterprise {enterprise_id}: {str(exc)}")
        return {"error": str(exc)}

@celery.task(name='send_security_alert')
def send_security_alert(enterprise_id: str, alert_data: Dict):
    """Send security alert notification"""
    try:
        logger.warning(f"Sending security alert for enterprise {enterprise_id}")
        
        # Implementation for sending alerts (email, SMS, webhook, etc.)
        # This would integrate with your notification system
        
        return {
            "enterprise_id": enterprise_id,
            "alert_sent": True,
            "sent_at": datetime.utcnow().isoformat()
        }
        
    except Exception as exc:
        logger.error(f"Failed to send security alert: {str(exc)}")
        return {"error": str(exc)}

@celery.task(name='cleanup_old_data')
def cleanup_old_data():
    """Clean up old logs and temporary data"""
    try:
        logger.info("Starting data cleanup task")
        
        # Clean up old log files
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        # Implementation for cleaning up old data
        # This would clean logs, temporary files, expired sessions, etc.
        
        logger.info("Data cleanup completed")
        return {
            "success": True,
            "cleaned_at": datetime.utcnow().isoformat()
        }
        
    except Exception as exc:
        logger.error(f"Data cleanup failed: {str(exc)}")
        return {"error": str(exc)}

# Periodic tasks
celery.conf.beat_schedule = {
    'cleanup-old-data': {
        'task': 'cleanup_old_data',
        'schedule': 86400.0,  # Run daily
    },
}

if __name__ == '__main__':
    celery.start()
