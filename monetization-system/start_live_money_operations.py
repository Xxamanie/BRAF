#!/usr/bin/env python3
"""
Start Live Money Operations
Launches the complete BRAF monetization system with real money integrations
"""

import os
import sys
import time
import logging
import asyncio
from datetime import datetime
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import our live integration system
from live_integration_orchestrator import live_orchestrator
from main import app
import uvicorn
from threading import Thread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_operations.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class LiveMoneyOperationsManager:
    """Manages the complete live money operations system"""
    
    def __init__(self):
        self.web_server_thread = None
        self.is_running = False
        
        # Load production environment
        self._load_production_env()
        
        # Validate configuration
        self._validate_configuration()
    
    def _load_production_env(self):
        """Load production environment variables"""
        env_file = Path(__file__).parent / '.env.production'
        
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
            
            logger.info("Production environment loaded")
        else:
            logger.warning("Production environment file not found - using defaults")
    
    def _validate_configuration(self):
        """Validate that required configuration is present"""
        required_configs = {
            'Payment Providers': [
                'OPAY_MERCHANT_ID', 'OPAY_API_KEY', 'OPAY_SECRET_KEY',
                'PALMPAY_MERCHANT_ID', 'PALMPAY_API_KEY', 'PALMPAY_SECRET_KEY'
            ],
            'Earning Platforms': [
                'SWAGBUCKS_API_KEY', 'YOUTUBE_API_KEY'
            ],
            'Browser Automation': [
                'PROXY_USERNAME', 'CAPTCHA_API_KEY'
            ],
            'Currency Exchange': [
                'FIXER_API_KEY', 'CURRENCY_API_KEY'
            ]
        }
        
        missing_configs = {}
        demo_mode_reasons = []
        
        for category, configs in required_configs.items():
            missing = [config for config in configs if not os.getenv(config)]
            if missing:
                missing_configs[category] = missing
                demo_mode_reasons.append(f"{category}: {', '.join(missing)}")
        
        if missing_configs:
            logger.warning("⚠️  RUNNING IN DEMO MODE - Missing live credentials:")
            for category, missing in missing_configs.items():
                logger.warning(f"  {category}: {', '.join(missing)}")
            logger.warning("  Real money operations will be simulated")
            logger.warning("  See LIVE_INTEGRATION_GUIDE.md for setup instructions")
        else:
            logger.info("✅ All live credentials configured - REAL MONEY MODE ACTIVE")
    
    def start_web_server(self):
        """Start the web server in a separate thread"""
        def run_server():
            try:
                host = os.getenv('HOST', '0.0.0.0')
                port = int(os.getenv('PORT', 8003))
                
                logger.info(f"Starting web server on {host}:{port}")
                
                uvicorn.run(
                    app,
                    host=host,
                    port=port,
                    log_level="info",
                    access_log=True
                )
            except Exception as e:
                logger.error(f"Web server error: {e}")
        
        self.web_server_thread = Thread(target=run_server, daemon=True)
        self.web_server_thread.start()
        
        # Wait for server to start
        time.sleep(3)
        logger.info("Web server started successfully")
    
    def start_live_operations(self):
        """Start live money-making operations"""
        try:
            logger.info("🚀 Starting Live Money Operations...")
            
            # Start the orchestrator
            result = live_orchestrator.start_live_operations()
            
            if result['status'] == 'started':
                logger.info("✅ Live operations started successfully")
                logger.info(f"   Configuration: {result['configuration']}")
                return True
            else:
                logger.error(f"❌ Failed to start live operations: {result['message']}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting live operations: {e}")
            return False
    
    def display_startup_info(self):
        """Display startup information and instructions"""
        print("\n" + "="*80)
        print("🚀 BRAF LIVE MONEY OPERATIONS SYSTEM")
        print("="*80)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 Web Interface: http://localhost:{os.getenv('PORT', 8003)}")
        print(f"📊 Admin Dashboard: http://localhost:{os.getenv('PORT', 8003)}/dashboard")
        print(f"💰 Live Operations: {'ACTIVE' if not self._is_demo_mode() else 'DEMO MODE'}")
        print("="*80)
        
        if self._is_demo_mode():
            print("⚠️  DEMO MODE ACTIVE")
            print("   - No real money will be processed")
            print("   - All transactions are simulated")
            print("   - See LIVE_INTEGRATION_GUIDE.md for live setup")
        else:
            print("💰 REAL MONEY MODE ACTIVE")
            print("   - Real money transactions enabled")
            print("   - Live payment processing active")
            print("   - Automated earning operations running")
        
        print("\n📋 AVAILABLE OPERATIONS:")
        print("   • Survey completion automation")
        print("   • Video monetization")
        print("   • Real-time currency conversion")
        print("   • OPay/PalmPay withdrawals")
        print("   • Automated earnings optimization")
        
        print("\n🔧 MANAGEMENT COMMANDS:")
        print("   • View stats: GET /api/v1/live/stats")
        print("   • Process withdrawal: POST /api/v1/withdrawal/request")
        print("   • Check balance: GET /api/v1/dashboard/earnings/{enterprise_id}")
        
        print("\n📱 MOBILE MONEY SUPPORT:")
        print("   • OPay: Nigerian Naira (NGN)")
        print("   • PalmPay: Nigerian Naira (NGN)")
        print("   • Real-time USD to NGN conversion")
        
        print("="*80)
        print("💡 TIP: Monitor operations at the web dashboard")
        print("🛑 Press Ctrl+C to stop all operations")
        print("="*80 + "\n")
    
    def _is_demo_mode(self) -> bool:
        """Check if running in demo mode"""
        required_live_configs = [
            'OPAY_API_KEY', 'PALMPAY_API_KEY', 
            'SWAGBUCKS_API_KEY', 'PROXY_USERNAME'
        ]
        return not all(os.getenv(config) for config in required_live_configs)
    
    def run(self):
        """Run the complete live money operations system"""
        try:
            self.is_running = True
            
            # Display startup information
            self.display_startup_info()
            
            # Start web server
            self.start_web_server()
            
            # Start live operations
            if not self.start_live_operations():
                logger.error("Failed to start live operations")
                return False
            
            # Keep the main thread alive
            logger.info("🎯 System is now running - monitoring operations...")
            
            while self.is_running:
                try:
                    # Display periodic stats
                    stats = live_orchestrator.get_live_stats()
                    
                    if stats['is_running']:
                        logger.info(
                            f"💰 Stats: ${stats['total_earned_usd']:.2f} earned, "
                            f"${stats['available_balance_usd']:.2f} available, "
                            f"{stats['success_rate']}% success rate, "
                            f"${stats['hourly_rate_usd']:.2f}/hour"
                        )
                    
                    # Sleep for 5 minutes between status updates
                    time.sleep(300)
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Shutdown requested by user")
                    break
                except Exception as e:
                    logger.error(f"❌ Monitoring error: {e}")
                    time.sleep(60)  # Wait 1 minute on error
            
            return True
            
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested")
            return True
        except Exception as e:
            logger.error(f"❌ System error: {e}")
            return False
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Gracefully shutdown all operations"""
        logger.info("🛑 Shutting down live operations...")
        
        try:
            # Stop live operations
            if live_orchestrator.is_running:
                result = live_orchestrator.stop_live_operations()
                logger.info(f"Live operations stopped: {result['message']}")
            
            # Mark as not running
            self.is_running = False
            
            logger.info("✅ Shutdown completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

def main():
    """Main entry point"""
    try:
        # Create and run the live operations manager
        manager = LiveMoneyOperationsManager()
        success = manager.run()
        
        if success:
            print("\n✅ Live money operations completed successfully")
        else:
            print("\n❌ Live money operations failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
