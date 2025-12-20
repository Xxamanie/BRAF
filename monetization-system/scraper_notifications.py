#!/usr/bin/env python3
"""
Scraper Notification System
Send notifications when scraping jobs complete or fail.
"""
import json
import os
import smtplib
import requests
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ScraperNotifications:
    """Handle notifications for scraper events"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or os.path.join(os.path.dirname(__file__), 'scraper_urls.json')
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load notification configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    return data.get('notifications', {})
        except Exception as e:
            logger.error(f"Failed to load notification config: {e}")
        
        return {}
    
    def send_email_notification(self, subject: str, body: str, is_html: bool = False) -> bool:
        """Send email notification"""
        email_config = self.config.get('email', {})
        
        if not email_config.get('enabled', False):
            logger.info("Email notifications disabled")
            return True
        
        try:
            # Email configuration
            smtp_server = email_config.get('smtp_server', 'smtp.gmail.com')
            smtp_port = email_config.get('smtp_port', 587)
            sender_email = email_config.get('sender_email')
            sender_password = email_config.get('sender_password')
            recipient_emails = email_config.get('recipients', [])
            
            if not all([sender_email, sender_password, recipient_emails]):
                logger.warning("Email configuration incomplete")
                return False
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = ', '.join(recipient_emails)
            
            # Add body
            if is_html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            logger.info(f"Email notification sent to {len(recipient_emails)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def send_webhook_notification(self, data: Dict) -> bool:
        """Send webhook notification"""
        webhook_url = self.config.get('webhook_url')
        
        if not webhook_url:
            logger.info("Webhook notifications not configured")
            return True
        
        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(webhook_url, json=data, headers=headers, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Webhook notification sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False
    
    def send_slack_notification(self, message: str, color: str = "good") -> bool:
        """Send Slack notification"""
        slack_webhook = self.config.get('slack_webhook')
        
        if not slack_webhook:
            logger.info("Slack notifications not configured")
            return True
        
        try:
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "text": message,
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
            
            response = requests.post(slack_webhook, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("Slack notification sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def notify_success(self, stats: Dict):
        """Send success notification"""
        # Email notification
        subject = "✅ Web Scraper - Job Completed Successfully"
        body = self.format_success_message(stats)
        self.send_email_notification(subject, body)
        
        # Slack notification
        slack_message = f"✅ Scraper completed: {stats.get('pages_scraped', 0)} pages scraped in {stats.get('duration_seconds', 0):.1f}s"
        self.send_slack_notification(slack_message, "good")
        
        # Webhook notification
        webhook_data = {
            "event": "scraper_success",
            "timestamp": datetime.now().isoformat(),
            "stats": stats
        }
        self.send_webhook_notification(webhook_data)
    
    def notify_failure(self, error: str, stats: Dict = None):
        """Send failure notification"""
        # Email notification
        subject = "❌ Web Scraper - Job Failed"
        body = self.format_failure_message(error, stats)
        self.send_email_notification(subject, body)
        
        # Slack notification
        slack_message = f"❌ Scraper failed: {error}"
        self.send_slack_notification(slack_message, "danger")
        
        # Webhook notification
        webhook_data = {
            "event": "scraper_failure",
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "stats": stats or {}
        }
        self.send_webhook_notification(webhook_data)
    
    def format_success_message(self, stats: Dict) -> str:
        """Format success notification message"""
        return f"""
Web Scraper Job Completed Successfully

📊 Statistics:
• Pages Scraped: {stats.get('pages_scraped', 0)}
• Pages Failed: {stats.get('pages_failed', 0)}
• Total Content: {stats.get('total_content_length', 0):,} characters
• Duration: {stats.get('duration_seconds', 0):.1f} seconds
• Success Rate: {stats.get('success_rate', 0):.1f}%

⏰ Timing:
• Start Time: {stats.get('start_time', 'Unknown')}
• End Time: {stats.get('end_time', 'Unknown')}

🔗 URLs Processed: {stats.get('urls_processed', 0)}

This is an automated notification from the web scraper system.
        """.strip()
    
    def format_failure_message(self, error: str, stats: Dict = None) -> str:
        """Format failure notification message"""
        stats = stats or {}
        
        return f"""
Web Scraper Job Failed

❌ Error: {error}

📊 Partial Statistics:
• Pages Scraped: {stats.get('pages_scraped', 0)}
• Pages Failed: {stats.get('pages_failed', 0)}
• Errors: {len(stats.get('errors', []))}

⏰ Timing:
• Start Time: {stats.get('start_time', 'Unknown')}
• Duration: {stats.get('duration_seconds', 0):.1f} seconds

Please check the scraper logs for more details.

This is an automated notification from the web scraper system.
        """.strip()

def test_notifications():
    """Test notification system"""
    print("🧪 Testing Notification System")
    print("=" * 30)
    
    notifier = ScraperNotifications()
    
    # Test success notification
    test_stats = {
        'pages_scraped': 5,
        'pages_failed': 1,
        'total_content_length': 15000,
        'duration_seconds': 45.2,
        'success_rate': 83.3,
        'start_time': datetime.now().isoformat(),
        'end_time': datetime.now().isoformat(),
        'urls_processed': 6
    }
    
    print("📧 Testing success notification...")
    notifier.notify_success(test_stats)
    
    print("📧 Testing failure notification...")
    notifier.notify_failure("Test error message", test_stats)
    
    print("✅ Notification tests completed")

if __name__ == "__main__":
    test_notifications()