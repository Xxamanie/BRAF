#!/usr/bin/env python3
"""
BRAF Earnings Tracker
Track and manage earnings from various automation tasks
"""
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class EarningsTracker:
    """Track earnings from automation tasks"""
    
    def __init__(self, db_path: str = "earnings.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize earnings database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create earnings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS earnings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    currency TEXT DEFAULT 'USD',
                    status TEXT DEFAULT 'pending',
                    details TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create platforms table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS platforms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    api_key TEXT,
                    status TEXT DEFAULT 'active',
                    total_earned REAL DEFAULT 0.0,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create withdrawal requests table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS withdrawals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    amount REAL NOT NULL,
                    currency TEXT DEFAULT 'USD',
                    method TEXT NOT NULL,
                    address TEXT,
                    status TEXT DEFAULT 'pending',
                    requested_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    processed_at TEXT,
                    transaction_id TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Earnings database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def record_earning(self, platform: str, task_type: str, amount: float, 
                      currency: str = 'USD', details: Dict = None) -> bool:
        """Record a new earning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO earnings (timestamp, platform, task_type, amount, currency, details)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                platform,
                task_type,
                amount,
                currency,
                json.dumps(details) if details else None
            ))
            
            # Update platform total
            cursor.execute('''
                INSERT OR REPLACE INTO platforms (name, total_earned, last_updated)
                VALUES (?, 
                    COALESCE((SELECT total_earned FROM platforms WHERE name = ?), 0) + ?,
                    ?)
            ''', (platform, platform, amount, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Recorded earning: {amount} {currency} from {platform}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record earning: {e}")
            return False
    
    def get_earnings_summary(self, days: int = 30) -> Dict:
        """Get earnings summary for specified period"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Total earnings
            cursor.execute('''
                SELECT SUM(amount), COUNT(*) FROM earnings 
                WHERE timestamp >= ? AND timestamp <= ?
            ''', (start_date.isoformat(), end_date.isoformat()))
            
            total_amount, total_tasks = cursor.fetchone()
            total_amount = total_amount or 0.0
            total_tasks = total_tasks or 0
            
            # Earnings by platform
            cursor.execute('''
                SELECT platform, SUM(amount), COUNT(*) FROM earnings 
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY platform
                ORDER BY SUM(amount) DESC
            ''', (start_date.isoformat(), end_date.isoformat()))
            
            platform_earnings = []
            for row in cursor.fetchall():
                platform_earnings.append({
                    'platform': row[0],
                    'amount': row[1],
                    'tasks': row[2]
                })
            
            # Earnings by task type
            cursor.execute('''
                SELECT task_type, SUM(amount), COUNT(*) FROM earnings 
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY task_type
                ORDER BY SUM(amount) DESC
            ''', (start_date.isoformat(), end_date.isoformat()))
            
            task_earnings = []
            for row in cursor.fetchall():
                task_earnings.append({
                    'task_type': row[0],
                    'amount': row[1],
                    'tasks': row[2]
                })
            
            # Daily earnings
            cursor.execute('''
                SELECT DATE(timestamp) as date, SUM(amount) FROM earnings 
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                LIMIT 7
            ''', (start_date.isoformat(), end_date.isoformat()))
            
            daily_earnings = []
            for row in cursor.fetchall():
                daily_earnings.append({
                    'date': row[0],
                    'amount': row[1]
                })
            
            conn.close()
            
            return {
                'period_days': days,
                'total_amount': total_amount,
                'total_tasks': total_tasks,
                'average_per_task': total_amount / total_tasks if total_tasks > 0 else 0,
                'platform_breakdown': platform_earnings,
                'task_breakdown': task_earnings,
                'daily_earnings': daily_earnings
            }
            
        except Exception as e:
            logger.error(f"Failed to get earnings summary: {e}")
            return {}
    
    def request_withdrawal(self, amount: float, method: str, address: str = None, 
                          currency: str = 'USD') -> bool:
        """Request a withdrawal"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO withdrawals (amount, currency, method, address)
                VALUES (?, ?, ?, ?)
            ''', (amount, currency, method, address))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Withdrawal requested: {amount} {currency} via {method}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to request withdrawal: {e}")
            return False
    
    def get_withdrawal_history(self) -> List[Dict]:
        """Get withdrawal history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM withdrawals 
                ORDER BY requested_at DESC
            ''')
            
            withdrawals = []
            for row in cursor.fetchall():
                withdrawals.append({
                    'id': row[0],
                    'amount': row[1],
                    'currency': row[2],
                    'method': row[3],
                    'address': row[4],
                    'status': row[5],
                    'requested_at': row[6],
                    'processed_at': row[7],
                    'transaction_id': row[8]
                })
            
            conn.close()
            return withdrawals
            
        except Exception as e:
            logger.error(f"Failed to get withdrawal history: {e}")
            return []
    
    def get_platform_stats(self) -> List[Dict]:
        """Get platform statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM platforms 
                ORDER BY total_earned DESC
            ''')
            
            platforms = []
            for row in cursor.fetchall():
                platforms.append({
                    'id': row[0],
                    'name': row[1],
                    'status': row[3],
                    'total_earned': row[4],
                    'last_updated': row[5]
                })
            
            conn.close()
            return platforms
            
        except Exception as e:
            logger.error(f"Failed to get platform stats: {e}")
            return []

class MonetizationManager:
    """Manage monetization workflows"""
    
    def __init__(self):
        self.earnings_tracker = EarningsTracker()
        self.active_tasks = {}
    
    def register_platform(self, platform_name: str, config: Dict) -> bool:
        """Register a new earning platform"""
        try:
            # Store platform configuration
            self.active_tasks[platform_name] = {
                'config': config,
                'status': 'active',
                'last_run': None,
                'total_earned': 0.0
            }
            
            logger.info(f"Platform registered: {platform_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register platform {platform_name}: {e}")
            return False
    
    def run_earning_task(self, platform: str, task_type: str, task_config: Dict) -> Dict:
        """Run an earning task"""
        try:
            start_time = datetime.now()
            
            # Simulate task execution (replace with actual implementation)
            success = True
            amount_earned = task_config.get('expected_earning', 0.0)
            
            if success and amount_earned > 0:
                # Record the earning
                self.earnings_tracker.record_earning(
                    platform=platform,
                    task_type=task_type,
                    amount=amount_earned,
                    details={
                        'task_config': task_config,
                        'execution_time': (datetime.now() - start_time).total_seconds()
                    }
                )
            
            return {
                'success': success,
                'platform': platform,
                'task_type': task_type,
                'amount_earned': amount_earned,
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Failed to run earning task: {e}")
            return {
                'success': False,
                'error': str(e),
                'platform': platform,
                'task_type': task_type
            }
    
    def get_dashboard_data(self) -> Dict:
        """Get data for monetization dashboard"""
        try:
            earnings_summary = self.earnings_tracker.get_earnings_summary()
            platform_stats = self.earnings_tracker.get_platform_stats()
            withdrawal_history = self.earnings_tracker.get_withdrawal_history()
            
            return {
                'earnings_summary': earnings_summary,
                'platform_stats': platform_stats,
                'withdrawal_history': withdrawal_history[:10],  # Last 10 withdrawals
                'active_platforms': len(self.active_tasks),
                'total_platforms': len(platform_stats)
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {}

def main():
    """Test monetization system"""
    print("💰 Testing BRAF Monetization System")
    
    # Initialize manager
    manager = MonetizationManager()
    
    # Register test platform
    manager.register_platform('test_platform', {
        'api_key': 'test_key',
        'base_url': 'https://api.example.com'
    })
    
    # Run test earning task
    result = manager.run_earning_task(
        platform='test_platform',
        task_type='survey_completion',
        task_config={
            'expected_earning': 2.50,
            'task_id': 'survey_123'
        }
    )
    
    print(f"Task result: {result}")
    
    # Get dashboard data
    dashboard = manager.get_dashboard_data()
    print(f"Dashboard data: {dashboard}")
    
    # Request test withdrawal
    manager.earnings_tracker.request_withdrawal(
        amount=10.0,
        method='paypal',
        address='test@example.com'
    )
    
    print("✅ Monetization system test completed")

if __name__ == "__main__":
    main()