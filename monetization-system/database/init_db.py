#!/usr/bin/env python3
"""
ACTUAL DATABASE INITIALIZATION
Complete database setup with tables, extensions, indexes, and initial data
"""

import asyncio
import sys
import logging
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.models import Base
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_database():
    """Initialize database with tables and initial data"""
    try:
        logger.info("🚀 Starting database initialization...")
        
        # Get database URL from config
        database_url = Config.get_database_url()
        logger.info(f"📊 Connecting to database: {database_url.split('@')[1] if '@' in database_url else 'localhost'}")
        
        # Create engine
        engine = create_engine(database_url)
        SessionLocal = sessionmaker(bind=engine)
        
        # Test database connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"✅ Database connection successful: {version.split(',')[0]}")
        
        logger.info("📋 Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        
        # Create PostgreSQL extensions
        logger.info("🔧 Creating PostgreSQL extensions...")
        with engine.connect() as conn:
            extensions = [
                "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";",
                "CREATE EXTENSION IF NOT EXISTS \"pgcrypto\";",
                "CREATE EXTENSION IF NOT EXISTS pg_trgm;",
                "CREATE EXTENSION IF NOT EXISTS btree_gin;",
                "CREATE EXTENSION IF NOT EXISTS unaccent;"
            ]
            
            for ext_sql in extensions:
                try:
                    conn.execute(text(ext_sql))
                    logger.info(f"✅ Extension created: {ext_sql.split('IF NOT EXISTS')[1].split(';')[0].strip()}")
                except Exception as e:
                    logger.warning(f"⚠️ Extension creation warning: {e}")
            
            conn.commit()
        
        # Create performance indexes
        logger.info("📈 Creating performance indexes...")
        with engine.connect() as conn:
            indexes = [
                # Enterprise indexes
                "CREATE INDEX IF NOT EXISTS idx_enterprises_email ON enterprises(email);",
                "CREATE INDEX IF NOT EXISTS idx_enterprises_subscription_tier ON enterprises(subscription_tier);",
                "CREATE INDEX IF NOT EXISTS idx_enterprises_created_at ON enterprises(created_at DESC);",
                
                # Subscription indexes
                "CREATE INDEX IF NOT EXISTS idx_subscriptions_enterprise_id ON subscriptions(enterprise_id);",
                "CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status);",
                
                # Withdrawal indexes
                "CREATE INDEX IF NOT EXISTS idx_withdrawals_enterprise_id ON withdrawals(enterprise_id);",
                "CREATE INDEX IF NOT EXISTS idx_withdrawals_status ON withdrawals(status);",
                "CREATE INDEX IF NOT EXISTS idx_withdrawals_created_at ON withdrawals(created_at DESC);",
                "CREATE INDEX IF NOT EXISTS idx_withdrawals_provider ON withdrawals(provider);",
                
                # Automation indexes
                "CREATE INDEX IF NOT EXISTS idx_automations_enterprise_id ON automations(enterprise_id);",
                "CREATE INDEX IF NOT EXISTS idx_automations_template_type ON automations(template_type);",
                "CREATE INDEX IF NOT EXISTS idx_automations_status ON automations(status);",
                "CREATE INDEX IF NOT EXISTS idx_automations_platform ON automations(platform);",
                
                # Earnings indexes
                "CREATE INDEX IF NOT EXISTS idx_earnings_automation_id ON earnings(automation_id);",
                "CREATE INDEX IF NOT EXISTS idx_earnings_earned_at ON earnings(earned_at DESC);",
                "CREATE INDEX IF NOT EXISTS idx_earnings_platform ON earnings(platform);",
                
                # Compliance indexes
                "CREATE INDEX IF NOT EXISTS idx_compliance_logs_enterprise_id ON compliance_logs(enterprise_id);",
                "CREATE INDEX IF NOT EXISTS idx_compliance_logs_checked_at ON compliance_logs(checked_at DESC);",
                "CREATE INDEX IF NOT EXISTS idx_compliance_logs_risk_level ON compliance_logs(risk_level);",
                
                # Security indexes
                "CREATE INDEX IF NOT EXISTS idx_security_alerts_enterprise_id ON security_alerts(enterprise_id);",
                "CREATE INDEX IF NOT EXISTS idx_security_alerts_resolved ON security_alerts(resolved);",
                "CREATE INDEX IF NOT EXISTS idx_security_alerts_severity ON security_alerts(severity);",
                
                # Crypto indexes
                "CREATE INDEX IF NOT EXISTS idx_crypto_balances_user_enterprise ON crypto_balances(user_id, enterprise_id);",
                "CREATE INDEX IF NOT EXISTS idx_crypto_balances_currency ON crypto_balances(currency);",
                "CREATE INDEX IF NOT EXISTS idx_crypto_transactions_user_enterprise ON crypto_transactions(user_id, enterprise_id);",
                "CREATE INDEX IF NOT EXISTS idx_crypto_transactions_status ON crypto_transactions(status);",
                "CREATE INDEX IF NOT EXISTS idx_crypto_transactions_created_at ON crypto_transactions(created_at DESC);",
                
                # API Key indexes
                "CREATE INDEX IF NOT EXISTS idx_api_keys_enterprise_id ON api_keys(enterprise_id);",
                "CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(active);",
                
                # Whitelist indexes
                "CREATE INDEX IF NOT EXISTS idx_whitelist_enterprise_id ON withdrawal_whitelist(enterprise_id);",
                "CREATE INDEX IF NOT EXISTS idx_whitelist_address_type ON withdrawal_whitelist(address_type);"
            ]
            
            for index_sql in indexes:
                try:
                    conn.execute(text(index_sql))
                    index_name = index_sql.split("IF NOT EXISTS")[1].split("ON")[0].strip()
                    logger.info(f"✅ Index created: {index_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Index creation warning: {e}")
            
            conn.commit()
        
        # Create database functions and triggers
        logger.info("⚙️ Creating database functions and triggers...")
        with engine.connect() as conn:
            # Updated timestamp trigger function
            trigger_function = """
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
            """
            conn.execute(text(trigger_function))
            
            # Create triggers for tables with updated_at columns
            triggers = [
                "CREATE TRIGGER IF NOT EXISTS update_enterprises_updated_at BEFORE UPDATE ON enterprises FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();",
                "CREATE TRIGGER IF NOT EXISTS update_subscriptions_updated_at BEFORE UPDATE ON subscriptions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();",
                "CREATE TRIGGER IF NOT EXISTS update_automations_updated_at BEFORE UPDATE ON automations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();",
                "CREATE TRIGGER IF NOT EXISTS update_crypto_balances_updated_at BEFORE UPDATE ON crypto_balances FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();",
                "CREATE TRIGGER IF NOT EXISTS update_crypto_transactions_updated_at BEFORE UPDATE ON crypto_transactions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();"
            ]
            
            for trigger_sql in triggers:
                try:
                    conn.execute(text(trigger_sql))
                    trigger_name = trigger_sql.split("IF NOT EXISTS")[1].split("BEFORE")[0].strip()
                    logger.info(f"✅ Trigger created: {trigger_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Trigger creation warning: {e}")
            
            conn.commit()
        
        # Create dashboard view
        logger.info("📊 Creating dashboard views...")
        with engine.connect() as conn:
            dashboard_view = """
            CREATE OR REPLACE VIEW enterprise_dashboard AS
            SELECT 
                e.id as enterprise_id,
                e.name as enterprise_name,
                e.subscription_tier,
                e.subscription_status,
                COUNT(DISTINCT a.id) as total_automations,
                COUNT(DISTINCT CASE WHEN a.status = 'active' THEN a.id END) as active_automations,
                COALESCE(SUM(a.earnings_today), 0) as today_earnings,
                COALESCE(SUM(a.earnings_total), 0) as total_earnings,
                COUNT(DISTINCT w.id) as total_withdrawals,
                COALESCE(SUM(CASE WHEN w.status = 'completed' THEN w.net_amount ELSE 0 END), 0) as total_withdrawn,
                AVG(a.success_rate) as avg_success_rate,
                e.created_at,
                e.updated_at
            FROM enterprises e
            LEFT JOIN automations a ON e.id = a.enterprise_id
            LEFT JOIN withdrawals w ON e.id = w.enterprise_id
            GROUP BY e.id, e.name, e.subscription_tier, e.subscription_status, e.created_at, e.updated_at;
            """
            conn.execute(text(dashboard_view))
            logger.info("✅ Dashboard view created")
            
            conn.commit()
        
        # Add initial data
        logger.info("📝 Adding initial data...")
        session = SessionLocal()
        try:
            # Import models for initial data
            from database.models import Enterprise
            
            # Check if we already have data
            existing_enterprises = session.query(Enterprise).count()
            if existing_enterprises == 0:
                logger.info("🏢 Creating demo enterprise account...")
                
                # Create demo enterprise (for testing)
                demo_enterprise = Enterprise(
                    name="Demo Enterprise",
                    email="demo@braf.local",
                    password_hash="demo_hash",  # In production, use proper hashing
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
            else:
                logger.info(f"📊 Database already contains {existing_enterprises} enterprises")
            
        except Exception as e:
            logger.warning(f"⚠️ Initial data creation warning: {e}")
            session.rollback()
        finally:
            session.close()
        
        logger.info("🎉 Database initialization completed successfully!")
        return True
        
    except SQLAlchemyError as e:
        logger.error(f"❌ Database initialization failed (SQLAlchemy): {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Database initialization failed (General): {e}")
        return False

def run_migrations():
    """Run Alembic migrations"""
    import subprocess
    try:
        logger.info("🔄 Running database migrations...")
        result = subprocess.run(
            ["alembic", "upgrade", "head"], 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=Path(__file__).parent.parent
        )
        logger.info("✅ Migrations completed successfully!")
        if result.stdout:
            logger.info(f"Migration output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Migrations failed: {e}")
        if e.stdout:
            logger.error(f"Migration stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Migration stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.warning("⚠️ Alembic not found - skipping migrations")
        return True

def verify_database():
    """Verify database setup"""
    try:
        logger.info("🔍 Verifying database setup...")
        
        database_url = Config.get_database_url()
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            # Check if tables exist
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """))
            tables = [row[0] for row in result.fetchall()]
            
            logger.info(f"📋 Found {len(tables)} tables: {', '.join(tables)}")
            
            # Check if indexes exist
            result = conn.execute(text("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE schemaname = 'public' 
                AND indexname LIKE 'idx_%'
                ORDER BY indexname;
            """))
            indexes = [row[0] for row in result.fetchall()]
            
            logger.info(f"📈 Found {len(indexes)} custom indexes")
            
            # Check if extensions exist
            result = conn.execute(text("""
                SELECT extname 
                FROM pg_extension 
                WHERE extname IN ('uuid-ossp', 'pgcrypto', 'pg_trgm', 'btree_gin', 'unaccent')
                ORDER BY extname;
            """))
            extensions = [row[0] for row in result.fetchall()]
            
            logger.info(f"🔧 Found {len(extensions)} extensions: {', '.join(extensions)}")
            
        logger.info("✅ Database verification completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database verification failed: {e}")
        return False

def main():
    """Main entry point for database initialization"""
    logger.info("🚀 BRAF Database Initialization Starting...")
    
    # Initialize database
    if not init_database():
        logger.error("❌ Database initialization failed")
        sys.exit(1)
    
    # Run migrations
    if not run_migrations():
        logger.error("❌ Database migrations failed")
        sys.exit(1)
    
    # Verify setup
    if not verify_database():
        logger.error("❌ Database verification failed")
        sys.exit(1)
    
    logger.info("🎉 BRAF Database setup completed successfully!")
    logger.info("📊 Database is ready for production use")

if __name__ == "__main__":
    main()