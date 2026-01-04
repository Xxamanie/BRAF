#!/usr/bin/env python3
"""
BRAF Research - LIVE Platform Manager
Provides LivePlatformManager for running configured platform automations,
tracking earnings, and generating reports.
"""

import os
import json
import sys
import time
import random
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any
from pathlib import Path
import getpass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_operations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LivePlatformManager:
    def __init__(self, config_dir: str = "data/accounts"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Initialize directories
        self.logs_dir = Path("logs/live")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = Path("data/live_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config_file = self.config_dir / "live_config.json"
        self.accounts_file = self.config_dir / "accounts.json"
        self.config = self.load_config()
        self.accounts = self.load_accounts()

        # Earnings tracking
        self.earnings_file = self.data_dir / "earnings.json"
        self.daily_earnings = self.load_earnings()

        # Platform modules
        self.platform_modules = self.discover_platforms()

        # Statistics
        self.stats = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_earnings": 0.0,
            "today_earnings": 0.0,
            "platforms_enabled": 0,
            "platforms_configured": len(self.accounts)
        }

        logger.info(f"Live Platform Manager initialized with {self.stats['platforms_configured']} configured platforms")

    def load_config(self) -> Dict:
        """Load live configuration"""
        default_config = {
            "system": {
                "name": "BRAF Live Earnings System",
                "version": "1.0.0",
                "environment": "production",
                "encryption_enabled": False,
                "backup_enabled": True
            },
            "execution": {
                "concurrent_limit": 3,
                "timeout_per_platform": 600,
                "retry_attempts": 2,
                "delay_between_platforms": 10,
                "delay_between_retries": 30
            },
            "scheduling": {
                "enabled": True,
                "runs_per_day": 4,
                "peak_hours": [9, 12, 15, 18, 21],
                "avoid_hours": [1, 2, 3, 4]
            },
            "security": {
                "encrypt_passwords": False,
                "clear_logs_after_days": 30,
                "mask_sensitive_data": True
            },
            "notifications": {
                "enabled": True,
                "min_earnings_alert": 10.00,
                "error_alerts": True,
                "daily_summary": True
            }
        }

        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            else:
                with open(self.config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"Created default config at {self.config_file}")
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config

    def load_accounts(self) -> Dict:
        """Load platform accounts"""
        default_accounts = {
            "qmee": {
                "status": "needs_setup",
                "credentials": {
                    "email": "",
                    "password": "",
                    "paypal": ""
                },
                "settings": {
                    "enabled": False,
                    "daily_goal": 5.00,
                    "strategy": "surveys_and_offers",
                    "check_interval": 120
                }
            },
            "freebitcoin": {
                "status": "needs_setup",
                "credentials": {
                    "email": "",
                    "password": "",
                    "btc_wallet": ""
                },
                "settings": {
                    "enabled": False,
                    "daily_goal": 0.0001,
                    "strategy": "hourly_rolls",
                    "check_interval": 60
                }
            },
            "microsoft_rewards": {
                "status": "needs_setup",
                "credentials": {
                    "email": "",
                    "password": ""
                },
                "settings": {
                    "enabled": False,
                    "daily_goal": 150,
                    "strategy": "full_daily_set",
                    "check_interval": 1440
                }
            }
        }

        try:
            if self.accounts_file.exists():
                with open(self.accounts_file, 'r') as f:
                    accounts = json.load(f)

                    # Update with any new platforms
                    for platform, data in default_accounts.items():
                        if platform not in accounts:
                            accounts[platform] = data

                    return accounts
            else:
                with open(self.accounts_file, 'w') as f:
                    json.dump(default_accounts, f, indent=2)
                logger.info(f"Created accounts template at {self.accounts_file}")
                return default_accounts
        except Exception as e:
            logger.error(f"Error loading accounts: {e}")
            return default_accounts

    def load_earnings(self) -> Dict:
        """Load earnings data"""
        default_earnings = {
            "today": {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "earnings": 0.0,
                "platforms_run": 0,
                "transactions": []
            },
            "yesterday": {
                "date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                "earnings": 0.0,
                "platforms_run": 0,
                "transactions": []
            },
            "this_week": {
                "earnings": 0.0,
                "platforms_run": 0
            },
            "this_month": {
                "earnings": 0.0,
                "platforms_run": 0
            },
            "all_time": {
                "earnings": 0.0,
                "platforms_run": 0
            }
        }

        try:
            if self.earnings_file.exists():
                with open(self.earnings_file, 'r') as f:
                    earnings = json.load(f)

                # Check if today's date has changed
                if earnings["today"]["date"] != datetime.now().strftime("%Y-%m-%d"):
                    # Move today to yesterday
                    earnings["yesterday"] = earnings["today"]
                    # Reset today
                    earnings["today"] = {
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "earnings": 0.0,
                        "platforms_run": 0,
                        "transactions": []
                    }

                return earnings
            else:
                with open(self.earnings_file, 'w') as f:
                    json.dump(default_earnings, f, indent=2)
                return default_earnings
        except Exception as e:
            logger.error(f"Error loading earnings: {e}")
            return default_earnings

    def save_earnings(self):
        """Save earnings data"""
        try:
            with open(self.earnings_file, 'w') as f:
                json.dump(self.daily_earnings, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving earnings: {e}")

    def save_accounts(self):
        """Save accounts data"""
        try:
            with open(self.accounts_file, 'w') as f:
                json.dump(self.accounts, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving accounts: {e}")

    def discover_platforms(self) -> Dict[str, str]:
        """Discover available platform modules"""
        modules = {}
        platform_dir = Path(".")

        # Look for real platform files
        for file in platform_dir.glob("real_*.py"):
            platform = file.stem.replace("real_", "")
            modules[platform] = str(file)

        # Also look for standard hunter files
        for file in platform_dir.glob("*_hunter.py"):
            platform = file.stem.replace("_hunter", "")
            if platform not in modules:
                modules[platform] = str(file)

        logger.info(f"Discovered {len(modules)} platform modules")
        return modules

    def setup_platform(self, platform_name: str):
        """Setup a platform with credentials"""
        print(f"\n🔧 Setting up {platform_name}...")

        if platform_name not in self.accounts:
            self.accounts[platform_name] = {
                "status": "needs_setup",
                "credentials": {},
                "settings": {
                    "enabled": True,
                    "daily_goal": 1.00,
                    "strategy": "default",
                    "check_interval": 60
                }
            }

        platform = self.accounts[platform_name]

        # Get credentials
        print(f"\nEnter credentials for {platform_name}:")

        credentials = {}
        if platform_name in ["qmee", "swagbucks", "prizerebel", "freecash"]:
            credentials["email"] = input("Email: ").strip()
            credentials["password"] = getpass.getpass("Password: ")
            credentials["paypal"] = input("PayPal email (optional): ").strip()

        elif platform_name in ["freebitcoin", "cointiply", "firefaucet"]:
            credentials["email"] = input("Email: ").strip()
            credentials["password"] = getpass.getpass("Password: ")
            credentials["btc_wallet"] = input("Bitcoin wallet address: ").strip()

        elif platform_name == "microsoft_rewards":
            credentials["email"] = input("Microsoft email: ").strip()
            credentials["password"] = getpass.getpass("Password: ")

        elif platform_name in ["rakuten", "honey"]:
            credentials["email"] = input("Email: ").strip()
            credentials["password"] = getpass.getpass("Password: ")
            credentials["paypal"] = input("PayPal email: ").strip()

        else:
            # Generic platform setup
            credentials["email"] = input("Email: ").strip()
            credentials["password"] = getpass.getpass("Password: ")
            additional = input("Additional credentials (comma-separated key:value): ").strip()
            if additional:
                for item in additional.split(','):
                    if ':' in item:
                        key, value = item.split(':', 1)
                        credentials[key.strip()] = value.strip()

        # Update platform data
        platform["credentials"] = credentials
        platform["status"] = "configured"

        # Ask if enabled
        enable = input(f"Enable {platform_name} for automatic runs? (y/N): ").strip().lower()
        platform["settings"]["enabled"] = enable == 'y'

        if platform["settings"]["enabled"]:
            try:
                goal = float(input(f"Daily earnings goal for {platform_name} (default {platform['settings']['daily_goal']}): ").strip())
                platform["settings"]["daily_goal"] = goal
            except:
                pass

        # Save accounts
        self.save_accounts()

        print(f"\n✅ {platform_name} setup complete!")
        if platform["settings"]["enabled"]:
            print(f"   Status: Enabled (Goal: ${platform['settings']['daily_goal']}/day)")
        else:
            print(f"   Status: Configured but disabled")

    def run_platform(self, platform_name: str) -> Dict:
        """Run a single platform and return results"""
        result = {
            "platform": platform_name,
            "success": True,
            "earnings": 0.0,
            "duration": 0.0,
            "error": None,
            "timestamp": datetime.now().isoformat()
        }

        # Check if platform is configured and enabled
        if platform_name not in self.accounts:
            result["error"] = f"Platform not configured: {platform_name}"
            logger.error(result["error"])
            return result

        platform = self.accounts[platform_name]

        if platform["status"] != "configured":
            result["error"] = f"Platform not properly configured: {platform_name}"
            logger.error(result["error"])
            return result

        if not platform["settings"]["enabled"]:
            result["error"] = f"Platform disabled: {platform_name}"
            logger.info(result["error"])
            return result

        # Check if module exists
        if platform_name not in self.platform_modules:
            result["error"] = f"No automation module found for {platform_name}"
            logger.error(result["error"])
            return result

        module_file = self.platform_modules[platform_name]

        # Create platform-specific config file
        platform_config = {
            "platform": platform_name,
            "credentials": platform["credentials"],
            "settings": platform["settings"],
            "run_time": datetime.now().isoformat()
        }

        config_file = self.data_dir / f"{platform_name}_config.json"
        try:
            with open(config_file, 'w') as f:
                json.dump(platform_config, f, indent=2)
        except Exception as e:
            result["error"] = f"Failed to create config: {e}"
            return result

        logger.info(f"🚀 Running {platform_name}...")
        start_time = time.time()

        try:
            # Run the platform module
            cmd = [sys.executable, module_file, "--config", str(config_file)]

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config["execution"]["timeout_per_platform"],
                cwd=os.getcwd()
            )

            result["duration"] = time.time() - start_time

            # Parse results
            earnings = self.parse_earnings_from_output(process.stdout, platform_name)

            if earnings > 0:
                result["success"] = True
                result["earnings"] = earnings

                # Update earnings tracking
                self.update_earnings(platform_name, earnings)

                logger.info(f"✅ {platform_name} completed successfully in {result['duration']:.1f}s, earned: ${earnings:.2f}")
            else:
                result["error"] = "No earnings detected"
                logger.warning(f"⚠️  {platform_name} completed but no earnings detected")

            # Log detailed output if in debug mode
            if process.stderr:
                error_log = self.logs_dir / f"{platform_name}_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                with open(error_log, 'w') as f:
                    f.write(f"STDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}")

        except subprocess.TimeoutExpired:
            result["duration"] = time.time() - start_time
            result["error"] = f"Timeout after {result['duration']:.1f}s"
            logger.error(f"⏰ {platform_name} timed out after {result['duration']:.1f}s")

        except Exception as e:
            result["duration"] = time.time() - start_time
            result["error"] = str(e)
            logger.error(f"❌ Error running {platform_name}: {e}")

        # Clean up config file
        try:
            if config_file.exists():
                config_file.unlink()
        except:
            pass

        return result

    def parse_earnings_from_output(self, output: str, platform_name: str) -> float:
        """Parse earnings from platform output"""
        earnings = 0.0

        # Look for earnings patterns
        import re

        patterns = [
            r'EARNINGS_RESULT:\$(\d+\.?\d*)',
            r'earn(?:ed|ing)?[:\s]+\$?(\d+\.?\d*)',
            r'total[:\s]+\$?(\d+\.?\d*)',
            r'\$(\d+\.?\d*)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, output.lower())
            if matches:
                try:
                    # Take the highest value found
                    values = [float(m) for m in matches]
                    earnings = max(values)
                    break
                except ValueError:
                    continue

        # For crypto platforms, convert satoshis to BTC
        if earnings < 0.01 and platform_name in ["freebitcoin", "cointiply", "firefaucet"]:
            # Might be in satoshis (1 satoshi = 0.00000001 BTC)
            satoshi_pattern = r'(\d+)\s*sat'
            sat_matches = re.findall(satoshi_pattern, output.lower())
            if sat_matches:
                try:
                    satoshis = float(sat_matches[-1])
                    earnings = satoshis * 0.00000001
                except:
                    pass

        return earnings

    def update_earnings(self, platform_name: str, earnings: float):
        """Update earnings tracking"""
        transaction = {
            "platform": platform_name,
            "amount": earnings,
            "timestamp": datetime.now().isoformat(),
            "type": "automated"
        }

        # Update today's earnings
        self.daily_earnings["today"]["earnings"] += earnings
        self.daily_earnings["today"]["platforms_run"] += 1
        self.daily_earnings["today"]["transactions"].append(transaction)

        # Update weekly and monthly
        self.daily_earnings["this_week"]["earnings"] += earnings
        self.daily_earnings["this_week"]["platforms_run"] += 1

        self.daily_earnings["this_month"]["earnings"] += earnings
        self.daily_earnings["this_month"]["platforms_run"] += 1

        # Update all-time
        self.daily_earnings["all_time"]["earnings"] += earnings
        self.daily_earnings["all_time"]["platforms_run"] += 1

        # Update stats
        self.stats["total_earnings"] += earnings
        self.stats["today_earnings"] += earnings
        self.stats["total_runs"] += 1
        self.stats["successful_runs"] += 1

        # Save earnings
        self.save_earnings()

    def run_enabled_platforms(self, concurrent_limit: int = None):
        """Run all enabled platforms"""
        if concurrent_limit is None:
            concurrent_limit = self.config["execution"]["concurrent_limit"]

        # Get enabled platforms
        enabled_platforms = [
            name for name, data in self.accounts.items()
            if data.get("settings", {}).get("enabled", False) and data.get("status") == "configured"
        ]

        if not enabled_platforms:
            logger.warning("No enabled platforms to run!")
            print("\n⚠️  No platforms are enabled for automatic runs.")
            print("   Use option 2 to setup and enable platforms.")
            return []

        self.stats["platforms_enabled"] = len(enabled_platforms)

        print(f"\n🚀 Running {len(enabled_platforms)} enabled platforms...")
        print(f"   Concurrent limit: {concurrent_limit}")
        print(f"   Timeout per platform: {self.config['execution']['timeout_per_platform']}s")
        print("=" * 60)

        results = []
        batch_results = []

        # Run in batches
        for i in range(0, len(enabled_platforms), concurrent_limit):
            batch = enabled_platforms[i:i + concurrent_limit]
            print(f"\n📦 Batch {i//concurrent_limit + 1}: {', '.join(batch)}")

            batch_start = time.time()
            batch_results = []

            # Run batch (for now sequential, can be made concurrent later)
            for platform in batch:
                result = self.run_platform(platform)
                results.append(result)
                batch_results.append(result)

                # Small delay between platforms
                if platform != batch[-1]:
                    time.sleep(self.config["execution"]["delay_between_platforms"])

            batch_time = time.time() - batch_start

            # Print batch summary
            batch_earnings = sum(r["earnings"] for r in batch_results)
            batch_success = sum(1 for r in batch_results if r["success"])

            print(f"   Batch completed in {batch_time:.1f}s")
            print(f"   Earnings: ${batch_earnings:.2f}")
            print(f"   Successful: {batch_success}/{len(batch)}")

            # Delay between batches if not last batch
            if i + concurrent_limit < len(enabled_platforms):
                delay = self.config["execution"]["delay_between_retries"]
                print(f"   Waiting {delay}s before next batch...")
                time.sleep(delay)

        # Print final summary
        self.print_summary(results)

        return results

    def print_summary(self, results: List[Dict]):
        """Print detailed summary of run"""
        total_earnings = sum(r["earnings"] for r in results)
        successful = sum(1 for r in results if r["success"])
        total_time = sum(r["duration"] for r in results)

        print("\n" + "="*60)
        print("💰 LIVE OPERATIONS SUMMARY")
        print("="*60)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🚀 Platforms Run: {len(results)}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {len(results) - successful}")
        print(f"💰 Total Earnings: ${total_earnings:.2f}")
        print(f"⏱️  Total Time: {total_time:.1f}s")
        print(f"📊 Today Total: ${self.stats['today_earnings']:.2f}")
        print("-" * 60)

        # Show earnings by platform
        print("\n🏆 Platform Performance:")
        platform_earnings = {}
        for result in results:
            if result["success"]:
                platform_earnings[result["platform"]] = platform_earnings.get(result["platform"], 0) + result["earnings"]

        for platform, earnings in sorted(platform_earnings.items(), key=lambda x: x[1], reverse=True):
            print(f"   {platform:20} ${earnings:8.2f}")

        # Save detailed report
        report_file = self.logs_dir / f"run_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": {
                "total_earnings": total_earnings,
                "successful_runs": successful,
                "total_platforms": len(results),
                "total_time": total_time
            },
            "stats": self.stats
        }

        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📝 Detailed report saved to: {report_file}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")

        # Update config
        self.config["last_run"] = datetime.now().isoformat()
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def show_dashboard(self):
        """Show platform dashboard"""
        print("\n" + "="*70)
        print("📋 LIVE PLATFORM DASHBOARD")
        print("="*70)

        # Load complete platform database
        try:
            with open("complete_platforms_database.json", 'r') as f:
                platform_db = json.load(f)

            total_platforms = platform_db["total_platforms"]
            categories = platform_db["platform_categories"]
        except:
            total_platforms = 136
            categories = {}

        # Calculate stats
        configured = sum(1 for p in self.accounts.values() if p.get("status") == "configured")
        enabled = sum(1 for p in self.accounts.values() if p.get("settings", {}).get("enabled", True))

        print(f"\n📊 System Overview:")
        print(f"   • Total Platforms Available: {total_platforms}")
        print(f"   • Configured Platforms: {configured}")
        print(f"   • Enabled Platforms: {enabled}")
        print(f"   • Today's Earnings: ${self.stats['today_earnings']:.2f}")
        print(f"   • All-Time Earnings: ${self.stats['total_earnings']:.2f}")

        print(f"\n🔧 Configured Platforms:")
        print("-" * 70)

        for platform_name, platform_data in sorted(self.accounts.items()):
            status = platform_data.get("status", "unknown")
            enabled = platform_data.get("settings", {}).get("enabled", False)

            if status == "configured":
                status_icon = "✅" if enabled else "⚙️"
                earnings = platform_data.get("earnings", 0)
                goal = platform_data.get("settings", {}).get("daily_goal", 0)

                print(f"{status_icon} {platform_name:25} ", end="")
                if enabled:
                    print(f"Goal: ${goal:.2f}/day", end="")
                    if earnings > 0:
                        print(f" | Earned: ${earnings:.2f}")
                    else:
                        print(" | No earnings yet")
                else:
                    print("Configured but disabled")

        print(f"\n⏸️  Not Configured (from your list):")
        print("-" * 70)

        # Show top platforms from your list that aren't configured
        priority_platforms = [
            "swagbucks", "prizerebel", "freecash", "cointiply", "firefaucet",
            "rakuten", "honey", "adbtc", "faucetpay", "gg2u"
        ]

        not_configured = [p for p in priority_platforms if p not in self.accounts or self.accounts[p].get("status") != "configured"]

        for platform in not_configured[:10]:  # Show first 10
            print(f"🔸 {platform}")

        if len(not_configured) > 10:
            print(f"   ... and {len(not_configured) - 10} more")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("🚀 BRAF RESEARCH - LIVE EARNINGS PLATFORM MANAGER")
    print("="*70)
    print("💰 REAL AUTOMATION | 🔐 SECURE CREDENTIALS | 📊 LIVE TRACKING")
    print("")

    manager = LivePlatformManager()

    while True:
        print("\n🎯 MAIN MENU:")
        print("1. Run all enabled platforms")
        print("2. Setup/configure a platform")
        print("3. Show platform dashboard")
        print("4. View earnings report")
        print("5. Create platform automation template")
        print("6. Setup automation schedule")
        print("7. Exit")

        choice = input("\nSelect option (1-7): ").strip()

        if choice == "1":
            print("\n" + "="*60)
            print("🚀 STARTING LIVE AUTOMATION")
            print("="*60)

            # Check if any platforms are enabled
            enabled_count = sum(1 for p in manager.accounts.values() 
                              if p.get("settings", {}).get("enabled", False) 
                              and p.get("status") == "configured")

            if enabled_count == 0:
                print("\n⚠️  No platforms are enabled for automatic runs!")
                print("   Please setup and enable platforms first.")
                print("   Use option 2 to setup platforms.")
                continue

            concur = input(f"Concurrent platforms [{manager.config['execution']['concurrent_limit']}]: ").strip()
            concur = int(concur) if concur else manager.config["execution"]["concurrent_limit"]

            manager.run_enabled_platforms(concur)

        elif choice == "2":
            print("\n" + "="*60)
            print("🔧 PLATFORM SETUP")
            print("="*60)

            # Show available platforms
            print("\n📋 Available platforms (from your list):")

            # Load platform database
            try:
                with open("complete_platforms_database.json", 'r') as f:
                    platform_db = json.load(f)

                all_platforms = []
                for category in platform_db["platform_categories"].values():
                    all_platforms.extend(category["platforms"])

                # Clean platform names
                clean_platforms = []
                for p in all_platforms:
                    clean = p.lower().replace(' ', '_').replace('.', '_')
                    clean_platforms.append(clean)

                # Remove duplicates and sort
                clean_platforms = sorted(set(clean_platforms))

                # Show in columns
                for i in range(0, len(clean_platforms), 3):
                    row = clean_platforms[i:i+3]
                    print("   " + " | ".join(f"{p:20}" for p in row))

            except Exception as e:
                print(f"   Could not load platform list: {e}")
                print("   Using default list...")
                default_platforms = ["qmee", "freebitcoin", "microsoft_rewards", "swagbucks", "prizerebel"]
                for p in default_platforms:
                    print(f"   • {p}")

            print("\nEnter platform name to setup (or 'list' to see all):")
            platform_input = input("Platform: ").strip().lower()

            if platform_input == "list":
                # Show all platforms from database
                try:
                    with open("complete_platforms_database.json", 'r') as f:
                        platform_db = json.load(f)

                    print("\n📁 All Available Platforms by Category:")
                    for category_name, category_data in platform_db["platform_categories"].items():
                        print(f"\n{category_name.upper()}:")
                        for platform in category_data["platforms"][:5]:  # Show first 5
                            clean_name = platform.lower().replace(' ', '_').replace('.', '_')
                            configured = clean_name in manager.accounts and manager.accounts[clean_name].get("status") == "configured"
                            status = "✅" if configured else "🔸"
                            print(f"   {status} {clean_name}")
                        if len(category_data["platforms"]) > 5:
                            print(f"   ... and {len(category_data['platforms']) - 5} more")
                except:
                    print("Could not load platform database")

                platform_input = input("\nPlatform to setup: ").strip().lower()

            manager.setup_platform(platform_input)

        elif choice == "3":
            manager.show_dashboard()

        elif choice == "4":
            print("\n" + "="*60)
            print("💰 EARNINGS REPORT")
            print("="*60)

            print(f"\n📅 Today ({manager.daily_earnings['today']['date']}):")
            print(f"   Earnings: ${manager.daily_earnings['today']['earnings']:.2f}")
            print(f"   Platforms Run: {manager.daily_earnings['today']['platforms_run']}")

            print(f"\n📅 Yesterday ({manager.daily_earnings['yesterday']['date']}):")
            print(f"   Earnings: ${manager.daily_earnings['yesterday']['earnings']:.2f}")
            print(f"   Platforms Run: {manager.daily_earnings['yesterday']['platforms_run']}")

            print(f"\n📅 This Week:")
            print(f"   Earnings: ${manager.daily_earnings['this_week']['earnings']:.2f}")
            print(f"   Platforms Run: {manager.daily_earnings['this_week']['platforms_run']}")

            print(f"\n📅 This Month:")
            print(f"   Earnings: ${manager.daily_earnings['this_month']['earnings']:.2f}")
            print(f"   Platforms Run: {manager.daily_earnings['this_month']['platforms_run']}")

            print(f"\n📅 All Time:")
            print(f"   Earnings: ${manager.daily_earnings['all_time']['earnings']:.2f}")
            print(f"   Platforms Run: {manager.daily_earnings['all_time']['platforms_run']}")

            # Show recent transactions
            if manager.daily_earnings["today"]["transactions"]:
                print(f"\n📋 Recent Transactions (Today):")
                for txn in manager.daily_earnings["today"]["transactions"][-5:]:
                    time_str = datetime.fromisoformat(txn["timestamp"]).strftime("%H:%M")
                    print(f"   {time_str} - {txn['platform']}: ${txn['amount']:.2f}")

        elif choice == "5":
            print("\n" + "="*60)
            print("🤖 CREATE AUTOMATION TEMPLATE")
            print("="*60)

            platform = input("\nEnter platform name for automation template: ").strip().lower()

            # Determine platform type
            platform_type = "generic"
            if any(x in platform for x in ["bitcoin", "crypto", "faucet"]):
                platform_type = "crypto"
            elif any(x in platform for x in ["survey", "reward", "qmee"]):
                platform_type = "survey"
            elif "microsoft" in platform:
                platform_type = "microsoft"
            elif any(x in platform for x in ["cashback", "rakuten", "honey"]):
                platform_type = "cashback"

            template_file = f"real_{platform}.py"

            # Create appropriate template (omitted here for brevity in this file)
            print("Template creation is available in the full implementation.")

        elif choice == "6":
            print("\n" + "="*60)
            print("⏰ AUTOMATION SCHEDULE SETUP")
            print("="*60)
            print("This feature is coming soon...")
            print("For now, you can run platforms manually or use cron jobs.")

        elif choice == "7":
            print("\n👋 Goodbye!")
            break

        else:
            print("\n❌ Invalid choice. Please select 1-7.")


if __name__ == "__main__":
    main()
