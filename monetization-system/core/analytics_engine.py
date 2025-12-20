#!/usr/bin/env python3
"""
BRAF Analytics Engine
Advanced analytics and performance optimization for the Browser Automation Framework
"""
import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict, Counter
import statistics

logger = logging.getLogger(__name__)

class BRAFAnalytics:
    """Advanced analytics engine for BRAF performance optimization"""
    
    def __init__(self, db_path: str = None):
        """Initialize analytics engine with SQLite database"""
        self.db_path = db_path or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'braf_analytics.db'
        )
        self._init_database()
    
    def _init_database(self):
        """Initialize analytics database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    scraper_used TEXT NOT NULL,
                    scraper_recommended TEXT,
                    success BOOLEAN NOT NULL,
                    execution_time REAL,
                    error_message TEXT,
                    decision_score REAL,
                    complexity_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS domain_performance (
                    domain TEXT PRIMARY KEY,
                    total_executions INTEGER DEFAULT 0,
                    http_successes INTEGER DEFAULT 0,
                    http_failures INTEGER DEFAULT 0,
                    browser_successes INTEGER DEFAULT 0,
                    browser_failures INTEGER DEFAULT 0,
                    avg_http_time REAL DEFAULT 0,
                    avg_browser_time REAL DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS decision_accuracy (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    recommended_scraper TEXT NOT NULL,
                    actual_scraper TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    accuracy_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_executions_domain ON executions(domain)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_executions_timestamp ON executions(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_executions_scraper ON executions(scraper_used)')
    
    def record_execution(self, result: Dict, recommended_scraper: str = None, 
                        decision_score: float = None, complexity_score: float = None):
        """Record execution result for analytics"""
        url = result.get('url', '')
        domain = self._extract_domain(url)
        scraper_used = result.get('scraper_used', 'unknown')
        success = result.get('success', False)
        execution_time = result.get('execution_time')
        error_message = result.get('error') if not success else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO executions 
                (url, domain, scraper_used, scraper_recommended, success, execution_time, 
                 error_message, decision_score, complexity_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (url, domain, scraper_used, recommended_scraper, success, 
                  execution_time, error_message, decision_score, complexity_score,
                  json.dumps(result.get('metadata', {}))))
            
            # Update domain performance
            self._update_domain_performance(conn, domain, scraper_used, success, execution_time)
            
            # Record decision accuracy if we have recommendation
            if recommended_scraper:
                accuracy_score = 1.0 if recommended_scraper == scraper_used and success else 0.0
                conn.execute('''
                    INSERT INTO decision_accuracy 
                    (url, recommended_scraper, actual_scraper, success, accuracy_score)
                    VALUES (?, ?, ?, ?, ?)
                ''', (url, recommended_scraper, scraper_used, success, accuracy_score))
    
    def _update_domain_performance(self, conn, domain: str, scraper_used: str, 
                                  success: bool, execution_time: float):
        """Update domain performance statistics"""
        # Get current stats
        cursor = conn.execute(
            'SELECT * FROM domain_performance WHERE domain = ?', (domain,)
        )
        row = cursor.fetchone()
        
        if row:
            # Update existing record
            (_, total_exec, http_succ, http_fail, browser_succ, browser_fail, 
             avg_http, avg_browser, _) = row
            
            total_exec += 1
            
            if scraper_used == 'http':
                if success:
                    http_succ += 1
                else:
                    http_fail += 1
                
                # Update average time
                if execution_time and http_succ + http_fail > 0:
                    total_http = http_succ + http_fail
                    avg_http = ((avg_http * (total_http - 1)) + execution_time) / total_http
            
            elif scraper_used == 'browser':
                if success:
                    browser_succ += 1
                else:
                    browser_fail += 1
                
                # Update average time
                if execution_time and browser_succ + browser_fail > 0:
                    total_browser = browser_succ + browser_fail
                    avg_browser = ((avg_browser * (total_browser - 1)) + execution_time) / total_browser
            
            conn.execute('''
                UPDATE domain_performance 
                SET total_executions=?, http_successes=?, http_failures=?, 
                    browser_successes=?, browser_failures=?, avg_http_time=?, 
                    avg_browser_time=?, last_updated=?
                WHERE domain=?
            ''', (total_exec, http_succ, http_fail, browser_succ, browser_fail,
                  avg_http, avg_browser, datetime.now(), domain))
        
        else:
            # Create new record
            http_succ = 1 if scraper_used == 'http' and success else 0
            http_fail = 1 if scraper_used == 'http' and not success else 0
            browser_succ = 1 if scraper_used == 'browser' and success else 0
            browser_fail = 1 if scraper_used == 'browser' and not success else 0
            avg_http = execution_time if scraper_used == 'http' and execution_time else 0
            avg_browser = execution_time if scraper_used == 'browser' and execution_time else 0
            
            conn.execute('''
                INSERT INTO domain_performance 
                (domain, total_executions, http_successes, http_failures, 
                 browser_successes, browser_failures, avg_http_time, avg_browser_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (domain, 1, http_succ, http_fail, browser_succ, browser_fail,
                  avg_http, avg_browser))
    
    def get_performance_report(self, days: int = 30) -> Dict:
        """Generate comprehensive performance report"""
        since_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Overall statistics
            cursor = conn.execute('''
                SELECT 
                    COUNT(*) as total_executions,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN scraper_used = 'http' THEN 1 ELSE 0 END) as http_used,
                    SUM(CASE WHEN scraper_used = 'browser' THEN 1 ELSE 0 END) as browser_used,
                    AVG(CASE WHEN scraper_used = 'http' AND execution_time IS NOT NULL 
                        THEN execution_time END) as avg_http_time,
                    AVG(CASE WHEN scraper_used = 'browser' AND execution_time IS NOT NULL 
                        THEN execution_time END) as avg_browser_time
                FROM executions 
                WHERE timestamp >= ?
            ''', (since_date,))
            
            overall_stats = dict(zip([col[0] for col in cursor.description], cursor.fetchone()))
            
            # Success rates by scraper
            cursor = conn.execute('''
                SELECT 
                    scraper_used,
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    AVG(execution_time) as avg_time
                FROM executions 
                WHERE timestamp >= ?
                GROUP BY scraper_used
            ''', (since_date,))
            
            scraper_performance = {}
            for row in cursor.fetchall():
                scraper, total, successful, avg_time = row
                scraper_performance[scraper] = {
                    'total_executions': total,
                    'successful': successful,
                    'success_rate': (successful / total * 100) if total > 0 else 0,
                    'avg_execution_time': avg_time or 0
                }
            
            # Top performing domains
            cursor = conn.execute('''
                SELECT 
                    domain,
                    COUNT(*) as total_executions,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    AVG(execution_time) as avg_time
                FROM executions 
                WHERE timestamp >= ?
                GROUP BY domain
                ORDER BY total_executions DESC
                LIMIT 10
            ''', (since_date,))
            
            top_domains = []
            for row in cursor.fetchall():
                domain, total, successful, avg_time = row
                top_domains.append({
                    'domain': domain,
                    'total_executions': total,
                    'successful': successful,
                    'success_rate': (successful / total * 100) if total > 0 else 0,
                    'avg_execution_time': avg_time or 0
                })
            
            # Decision accuracy
            cursor = conn.execute('''
                SELECT 
                    AVG(accuracy_score) as avg_accuracy,
                    COUNT(*) as total_decisions
                FROM decision_accuracy 
                WHERE timestamp >= ?
            ''', (since_date,))
            
            accuracy_row = cursor.fetchone()
            decision_accuracy = {
                'average_accuracy': (accuracy_row[0] * 100) if accuracy_row[0] else 0,
                'total_decisions': accuracy_row[1] or 0
            }
            
            # Error analysis
            cursor = conn.execute('''
                SELECT 
                    error_message,
                    COUNT(*) as count
                FROM executions 
                WHERE timestamp >= ? AND success = 0 AND error_message IS NOT NULL
                GROUP BY error_message
                ORDER BY count DESC
                LIMIT 5
            ''', (since_date,))
            
            common_errors = [
                {'error': row[0], 'count': row[1]} 
                for row in cursor.fetchall()
            ]
        
        return {
            'report_period_days': days,
            'generated_at': datetime.now().isoformat(),
            'overall_statistics': overall_stats,
            'scraper_performance': scraper_performance,
            'top_domains': top_domains,
            'decision_accuracy': decision_accuracy,
            'common_errors': common_errors
        }
    
    def get_domain_insights(self, domain: str) -> Dict:
        """Get detailed insights for a specific domain"""
        with sqlite3.connect(self.db_path) as conn:
            # Domain performance
            cursor = conn.execute('''
                SELECT * FROM domain_performance WHERE domain = ?
            ''', (domain,))
            
            perf_row = cursor.fetchone()
            if not perf_row:
                return {'error': f'No data found for domain: {domain}'}
            
            (_, total_exec, http_succ, http_fail, browser_succ, browser_fail,
             avg_http, avg_browser, last_updated) = perf_row
            
            # Recent executions
            cursor = conn.execute('''
                SELECT url, scraper_used, success, execution_time, timestamp
                FROM executions 
                WHERE domain = ?
                ORDER BY timestamp DESC
                LIMIT 10
            ''', (domain,))
            
            recent_executions = [
                {
                    'url': row[0],
                    'scraper_used': row[1],
                    'success': bool(row[2]),
                    'execution_time': row[3],
                    'timestamp': row[4]
                }
                for row in cursor.fetchall()
            ]
            
            # Calculate success rates
            http_total = http_succ + http_fail
            browser_total = browser_succ + browser_fail
            
            return {
                'domain': domain,
                'total_executions': total_exec,
                'http_performance': {
                    'total': http_total,
                    'successes': http_succ,
                    'failures': http_fail,
                    'success_rate': (http_succ / http_total * 100) if http_total > 0 else 0,
                    'avg_execution_time': avg_http
                },
                'browser_performance': {
                    'total': browser_total,
                    'successes': browser_succ,
                    'failures': browser_fail,
                    'success_rate': (browser_succ / browser_total * 100) if browser_total > 0 else 0,
                    'avg_execution_time': avg_browser
                },
                'recommendation': self._get_domain_recommendation(http_total, http_succ, browser_total, browser_succ, avg_http, avg_browser),
                'recent_executions': recent_executions,
                'last_updated': last_updated
            }
    
    def _get_domain_recommendation(self, http_total: int, http_succ: int, 
                                  browser_total: int, browser_succ: int,
                                  avg_http: float, avg_browser: float) -> Dict:
        """Generate recommendation for domain based on performance data"""
        if http_total == 0 and browser_total == 0:
            return {'scraper': 'http', 'reason': 'No historical data, defaulting to HTTP', 'confidence': 0.5}
        
        http_success_rate = (http_succ / http_total) if http_total > 0 else 0
        browser_success_rate = (browser_succ / browser_total) if browser_total > 0 else 0
        
        # If one scraper has significantly better success rate
        if abs(http_success_rate - browser_success_rate) > 0.2:
            if http_success_rate > browser_success_rate:
                return {
                    'scraper': 'http',
                    'reason': f'HTTP has better success rate ({http_success_rate:.1%} vs {browser_success_rate:.1%})',
                    'confidence': 0.8
                }
            else:
                return {
                    'scraper': 'browser',
                    'reason': f'Browser has better success rate ({browser_success_rate:.1%} vs {http_success_rate:.1%})',
                    'confidence': 0.8
                }
        
        # If success rates are similar, consider speed
        if http_success_rate > 0.8 and browser_success_rate > 0.8:
            if avg_http > 0 and avg_browser > 0:
                if avg_http < avg_browser * 0.7:  # HTTP is significantly faster
                    return {
                        'scraper': 'http',
                        'reason': f'Both work well, but HTTP is faster ({avg_http:.1f}s vs {avg_browser:.1f}s)',
                        'confidence': 0.7
                    }
        
        # Default recommendation
        return {
            'scraper': 'http',
            'reason': 'Insufficient data for strong recommendation, defaulting to HTTP',
            'confidence': 0.5
        }
    
    def get_optimization_suggestions(self) -> List[Dict]:
        """Generate optimization suggestions based on analytics"""
        suggestions = []
        
        with sqlite3.connect(self.db_path) as conn:
            # Find domains with poor HTTP performance but no browser attempts
            cursor = conn.execute('''
                SELECT domain, http_successes, http_failures, browser_successes, browser_failures
                FROM domain_performance
                WHERE http_failures > http_successes AND browser_successes + browser_failures = 0
                ORDER BY http_failures DESC
                LIMIT 5
            ''')
            
            for row in cursor.fetchall():
                domain, http_succ, http_fail, browser_succ, browser_fail = row
                suggestions.append({
                    'type': 'scraper_switch',
                    'domain': domain,
                    'suggestion': f'Try browser scraper for {domain}',
                    'reason': f'HTTP has poor success rate ({http_succ}/{http_succ + http_fail})',
                    'priority': 'high'
                })
            
            # Find domains with slow browser performance but good HTTP potential
            cursor = conn.execute('''
                SELECT domain, avg_browser_time, browser_successes, browser_failures, 
                       http_successes, http_failures
                FROM domain_performance
                WHERE avg_browser_time > 10 AND browser_successes > 0 
                      AND (http_successes + http_failures = 0 OR http_successes > 0)
                ORDER BY avg_browser_time DESC
                LIMIT 5
            ''')
            
            for row in cursor.fetchall():
                domain, avg_browser, browser_succ, browser_fail, http_succ, http_fail = row
                suggestions.append({
                    'type': 'performance_optimization',
                    'domain': domain,
                    'suggestion': f'Consider HTTP scraper for {domain}',
                    'reason': f'Browser is slow ({avg_browser:.1f}s average)',
                    'priority': 'medium'
                })
            
            # Find frequently failing URLs
            cursor = conn.execute('''
                SELECT url, COUNT(*) as failure_count
                FROM executions
                WHERE success = 0 AND timestamp >= datetime('now', '-7 days')
                GROUP BY url
                HAVING failure_count >= 3
                ORDER BY failure_count DESC
                LIMIT 5
            ''')
            
            for row in cursor.fetchall():
                url, failure_count = row
                suggestions.append({
                    'type': 'url_investigation',
                    'url': url,
                    'suggestion': f'Investigate recurring failures for {url}',
                    'reason': f'{failure_count} failures in the last 7 days',
                    'priority': 'high'
                })
        
        return suggestions
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse
        try:
            return urlparse(url).netloc.lower()
        except:
            return 'unknown'
    
    def export_data(self, output_file: str, format: str = 'json'):
        """Export analytics data"""
        report = self.get_performance_report(days=90)  # 3 months of data
        
        if format.lower() == 'json':
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Analytics data exported to {output_file}")
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old analytics data"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'DELETE FROM executions WHERE timestamp < ?', (cutoff_date,)
            )
            deleted_executions = cursor.rowcount
            
            cursor = conn.execute(
                'DELETE FROM decision_accuracy WHERE timestamp < ?', (cutoff_date,)
            )
            deleted_decisions = cursor.rowcount
        
        logger.info(f"Cleaned up {deleted_executions} execution records and "
                   f"{deleted_decisions} decision records older than {days_to_keep} days")

def demo_analytics():
    """Demonstrate analytics capabilities"""
    print("📊 BRAF Analytics Engine Demo")
    print("=" * 40)
    
    analytics = BRAFAnalytics()
    
    # Simulate some execution data
    sample_results = [
        {
            'url': 'https://httpbin.org/html',
            'scraper_used': 'http',
            'success': True,
            'execution_time': 1.2,
            'metadata': {'test': True}
        },
        {
            'url': 'https://example.com',
            'scraper_used': 'http',
            'success': True,
            'execution_time': 0.8
        },
        {
            'url': 'https://app.example.com/dashboard',
            'scraper_used': 'browser',
            'success': True,
            'execution_time': 4.5
        },
        {
            'url': 'https://broken-site.com',
            'scraper_used': 'http',
            'success': False,
            'error': 'Connection timeout'
        }
    ]
    
    print("📝 Recording sample execution data...")
    for result in sample_results:
        analytics.record_execution(result, 'http', 0.7, 0.3)
    
    print("\n📈 Generating performance report...")
    report = analytics.get_performance_report(days=1)
    
    print(f"📊 Performance Report:")
    print(f"   Total executions: {report['overall_statistics']['total_executions']}")
    print(f"   Successful: {report['overall_statistics']['successful']}")
    print(f"   HTTP used: {report['overall_statistics']['http_used']}")
    print(f"   Browser used: {report['overall_statistics']['browser_used']}")
    
    if report['scraper_performance']:
        print(f"\n🔧 Scraper Performance:")
        for scraper, perf in report['scraper_performance'].items():
            print(f"   {scraper.upper()}: {perf['success_rate']:.1f}% success rate, "
                  f"{perf['avg_execution_time']:.2f}s average")
    
    print(f"\n🎯 Decision Accuracy: {report['decision_accuracy']['average_accuracy']:.1f}%")
    
    # Domain insights
    print(f"\n🌐 Domain Insights for example.com:")
    insights = analytics.get_domain_insights('example.com')
    if 'error' not in insights:
        print(f"   Total executions: {insights['total_executions']}")
        print(f"   HTTP success rate: {insights['http_performance']['success_rate']:.1f}%")
        print(f"   Recommendation: {insights['recommendation']['scraper']} "
              f"({insights['recommendation']['reason']})")
    
    # Optimization suggestions
    print(f"\n💡 Optimization Suggestions:")
    suggestions = analytics.get_optimization_suggestions()
    if suggestions:
        for suggestion in suggestions[:3]:  # Show top 3
            print(f"   • {suggestion['suggestion']} - {suggestion['reason']}")
    else:
        print("   No specific suggestions at this time")

if __name__ == "__main__":
    demo_analytics()