#!/usr/bin/env python3
"""
Import Automation Targets Module
Imports predefined automation targets and templates into the system
"""

import sys
import logging
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.models import Automation, Enterprise
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Predefined automation targets
AUTOMATION_TARGETS = [
    {
        "template_type": "survey",
        "platform": "swagbucks",
        "config": {
            "target_surveys_per_day": 10,
            "min_reward": 50,
            "max_duration_minutes": 15,
            "categories": ["general", "shopping", "technology"]
        },
        "description": "Swagbucks survey automation"
    },
    {
        "template_type": "survey",
        "platform": "surveyjunkie",
        "config": {
            "target_surveys_per_day": 8,
            "min_reward": 100,
            "max_duration_minutes": 20,
            "categories": ["lifestyle", "health", "finance"]
        },
        "description": "Survey Junkie automation"
    },
    {
        "template_type": "video",
        "platform": "youtube",
        "config": {
            "target_videos_per_day": 20,
            "min_watch_duration_seconds": 30,
            "max_watch_duration_seconds": 300,
            "categories": ["entertainment", "education", "technology"]
        },
        "description": "YouTube video viewing automation"
    },
    {
        "template_type": "content",
        "platform": "medium",
        "config": {
            "target_articles_per_day": 5,
            "min_read_duration_seconds": 60,
            "engagement_actions": ["clap", "comment", "follow"],
            "categories": ["technology", "business", "self-improvement"]
        },
        "description": "Medium content engagement automation"
    },
    {
        "template_type": "survey",
        "platform": "prolific",
        "config": {
            "target_surveys_per_day": 5,
            "min_reward": 200,
            "max_duration_minutes": 30,
            "categories": ["academic", "research", "psychology"]
        },
        "description": "Prolific academic survey automation"
    }
]

def import_targets():
    """Import automation targets into the database"""
    try:
        logger.info("📥 Importing automation targets...")
        
        # Get database URL from config
        database_url = Config.get_database_url()
        engine = create_engine(database_url)
        SessionLocal = sessionmaker(bind=engine)
        
        session = SessionLocal()
        
        try:
            # Get demo enterprise or create one
            demo_enterprise = session.query(Enterprise).filter_by(email="demo@braf.local").first()
            
            if not demo_enterprise:
                logger.warning("⚠️ Demo enterprise not found, creating one...")
                from database.models import Enterprise
                demo_enterprise = Enterprise(
                    name="Demo Enterprise",
                    email="demo@braf.local",
                    password_hash="demo_hash",
                    salt="demo_salt",
                    subscription_tier="free",
                    subscription_status="active",
                    kyc_level=1,
                    company_name="BRAF Demo Company",
                    phone_number="+1234567890",
                    country="US"
                )
                session.add(demo_enterprise)
                session.commit()
                logger.info("✅ Demo enterprise created")
            
            # Import automation targets
            imported_count = 0
            skipped_count = 0
            
            for target_data in AUTOMATION_TARGETS:
                # Check if automation already exists
                existing = session.query(Automation).filter_by(
                    enterprise_id=demo_enterprise.id,
                    template_type=target_data["template_type"],
                    platform=target_data["platform"]
                ).first()
                
                if existing:
                    logger.info(f"⏭️ Skipping existing automation: {target_data['platform']} - {target_data['template_type']}")
                    skipped_count += 1
                    continue
                
                # Create new automation
                automation = Automation(
                    enterprise_id=demo_enterprise.id,
                    template_type=target_data["template_type"],
                    platform=target_data["platform"],
                    status="inactive",  # Start as inactive
                    config=target_data["config"],
                    earnings_today=0.0,
                    earnings_total=0.0,
                    success_rate=0.0
                )
                
                session.add(automation)
                imported_count += 1
                logger.info(f"✅ Imported: {target_data['platform']} - {target_data['template_type']}")
            
            session.commit()
            
            logger.info(f"📊 Import Summary:")
            logger.info(f"   ✅ Imported: {imported_count} automations")
            logger.info(f"   ⏭️ Skipped: {skipped_count} existing automations")
            logger.info(f"   📈 Total: {imported_count + skipped_count} automation targets")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Target import failed: {e}")
            session.rollback()
            return False
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def list_targets():
    """List all automation targets in the system"""
    try:
        logger.info("📋 Listing automation targets...")
        
        database_url = Config.get_database_url()
        engine = create_engine(database_url)
        SessionLocal = sessionmaker(bind=engine)
        
        session = SessionLocal()
        
        try:
            automations = session.query(Automation).all()
            
            if not automations:
                logger.info("📭 No automation targets found in the system")
                return
            
            logger.info(f"🎯 Found {len(automations)} automation targets:")
            for automation in automations:
                logger.info(f"   🆔 {automation.id}")
                logger.info(f"   🏢 Enterprise: {automation.enterprise_id}")
                logger.info(f"   📝 Type: {automation.template_type}")
                logger.info(f"   🌐 Platform: {automation.platform}")
                logger.info(f"   📊 Status: {automation.status}")
                logger.info(f"   💰 Total Earnings: ${automation.earnings_total:.2f}")
                logger.info(f"   📈 Success Rate: {automation.success_rate:.1f}%")
                logger.info("   " + "-" * 40)
                
        except Exception as e:
            logger.error(f"❌ Failed to list targets: {e}")
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")

def clear_targets():
    """Clear all automation targets (use with caution)"""
    try:
        logger.warning("⚠️ Clearing all automation targets...")
        
        database_url = Config.get_database_url()
        engine = create_engine(database_url)
        SessionLocal = sessionmaker(bind=engine)
        
        session = SessionLocal()
        
        try:
            count = session.query(Automation).delete()
            session.commit()
            logger.info(f"✅ Cleared {count} automation targets")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to clear targets: {e}")
            session.rollback()
            return False
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def main():
    """Main entry point for target import"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Import automation targets for BRAF system")
    parser.add_argument("--list", action="store_true", help="List all automation targets")
    parser.add_argument("--clear", action="store_true", help="Clear all automation targets")
    parser.add_argument("--import", dest="do_import", action="store_true", help="Import automation targets")
    
    args = parser.parse_args()
    
    if args.list:
        list_targets()
        return
    
    if args.clear:
        if clear_targets():
            logger.info("✅ Targets cleared successfully")
        else:
            logger.error("❌ Failed to clear targets")
            sys.exit(1)
        return
    
    # Default action: import targets
    if import_targets():
        logger.info("✅ Automation targets imported successfully")
    else:
        logger.error("❌ Failed to import automation targets")
        sys.exit(1)

if __name__ == "__main__":
    main()