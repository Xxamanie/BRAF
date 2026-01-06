#!/usr/bin/env python3
"""
Seed sample data for BRAF Monetization System
Creates sample automations and earnings for testing
"""

from database.service import DatabaseService
from database.models import Earning
from datetime import datetime, timedelta
import random

def seed_sample_data(enterprise_id: str):
    """Seed sample data for an enterprise"""
    
    with DatabaseService() as db:
        # Create sample automations
        automations_data = [
            {
                "enterprise_id": enterprise_id,
                "template_type": "survey",
                "platform": "swagbucks",
                "config": {
                    "platforms": ["swagbucks", "surveyjunkie"],
                    "max_surveys_per_session": 5,
                    "daily_limit": 50.0
                }
            },
            {
                "enterprise_id": enterprise_id,
                "template_type": "video",
                "platform": "youtube",
                "config": {
                    "platform": "youtube",
                    "video_count": 50,
                    "device_type": "desktop"
                }
            },
            {
                "enterprise_id": enterprise_id,
                "template_type": "content",
                "platform": "general",
                "config": {
                    "content_type": "blog",
                    "posts_per_day": 3,
                    "topics": ["technology", "reviews", "lifestyle"]
                }
            }
        ]
        
        created_automations = []
        for automation_data in automations_data:
            automation = db.create_automation(automation_data)
            created_automations.append(automation)
            print(f"✅ Created automation: {automation.template_type} - {automation.platform}")
        
        # Create sample earnings for the past 30 days
        earnings_data = []
        for i in range(30):  # 30 days of data
            date = datetime.utcnow() - timedelta(days=i)
            
            # Random earnings for each automation
            for automation in created_automations:
                if automation.template_type == "survey":
                    # Survey earnings: $2-15 per survey, 3-8 surveys per day
                    num_surveys = random.randint(3, 8)
                    for _ in range(num_surveys):
                        earning_data = {
                            "automation_id": automation.id,
                            "amount": round(random.uniform(2.0, 15.0), 2),
                            "currency": "USD",
                            "platform": "swagbucks",
                            "task_type": "survey_completion",
                            "task_details": {"survey_id": f"SV_{random.randint(1000, 9999)}"}
                        }
                        earning = db.record_earning(earning_data)
                        # Update the earned_at timestamp
                        earning.earned_at = date - timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
                        
                elif automation.template_type == "video":
                    # Video earnings: $0.50-5 per video, 20-50 videos per day
                    num_videos = random.randint(20, 50)
                    for _ in range(num_videos):
                        earning_data = {
                            "automation_id": automation.id,
                            "amount": round(random.uniform(0.5, 5.0), 2),
                            "currency": "USD",
                            "platform": "youtube",
                            "task_type": "video_viewing",
                            "task_details": {"video_id": f"VID_{random.randint(1000, 9999)}"}
                        }
                        earning = db.record_earning(earning_data)
                        earning.earned_at = date - timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
                        
                elif automation.template_type == "content":
                    # Content earnings: $5-50 per post, 1-3 posts per day
                    num_posts = random.randint(1, 3)
                    for _ in range(num_posts):
                        earning_data = {
                            "automation_id": automation.id,
                            "amount": round(random.uniform(5.0, 50.0), 2),
                            "currency": "USD",
                            "platform": "general",
                            "task_type": "content_creation",
                            "task_details": {"post_id": f"POST_{random.randint(1000, 9999)}"}
                        }
                        earning = db.record_earning(earning_data)
                        earning.earned_at = date - timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
        
        # Update automation statistics
        for automation in created_automations:
            # Calculate total earnings for this automation
            earnings = db.db.query(Earning).filter(Earning.automation_id == automation.id).all()
            total_earnings = sum(e.amount for e in earnings)
            today_earnings = sum(e.amount for e in earnings if e.earned_at.date() == datetime.utcnow().date())
            
            # Update automation
            automation.earnings_total = total_earnings
            automation.earnings_today = today_earnings
            automation.success_rate = random.uniform(0.85, 0.98)  # 85-98% success rate
            automation.last_run = datetime.utcnow() - timedelta(minutes=random.randint(5, 120))
            
        db.db.commit()
        
        # Create some sample withdrawals with proper currency conversion
        total_earnings = sum(a.earnings_total for a in created_automations)
        withdrawn_amount_usd = total_earnings * 0.3  # Withdraw 30% of earnings
        
        # Convert USD to NGN for OPay (approximate rate)
        exchange_rate = 750  # 1 USD = 750 NGN
        withdrawn_amount_ngn = withdrawn_amount_usd * exchange_rate
        fee_ngn = withdrawn_amount_ngn * 0.015  # 1.5% fee
        net_amount_ngn = withdrawn_amount_ngn - fee_ngn
        
        withdrawal_data = {
            "enterprise_id": enterprise_id,
            "transaction_id": f"WD_{random.randint(10000, 99999)}",
            "amount": round(withdrawn_amount_usd, 2),  # Store original USD amount
            "fee": round(fee_ngn, 2),
            "net_amount": round(net_amount_ngn, 2),
            "currency": "NGN",  # OPay uses NGN
            "provider": "opay",
            "recipient": "+234XXXXXXXXXX",
            "status": "completed"
        }
        
        withdrawal = db.create_withdrawal(withdrawal_data)
        withdrawal.completed_at = datetime.utcnow() - timedelta(days=random.randint(1, 7))
        db.db.commit()
        
        print(f"✅ Created sample data:")
        print(f"   📊 {len(created_automations)} automations")
        print(f"   💰 ${total_earnings:.2f} total earnings")
        print(f"   💸 ${withdrawn_amount_usd:.2f} USD withdrawn (₦{withdrawn_amount_ngn:.0f} NGN)")
        print(f"   💵 ${total_earnings - withdrawn_amount_usd:.2f} available balance")

def main():
    """Main function"""
    print("🌱 Seeding sample data for BRAF Monetization System")
    
    # Get enterprise ID from user input or use test account
    enterprise_id = input("Enter enterprise ID (or press Enter for test account): ").strip()
    
    if not enterprise_id:
        # Use test account
        with DatabaseService() as db:
            enterprise = db.get_enterprise_by_email("test@example.com")
            if enterprise:
                enterprise_id = enterprise.id
                print(f"Using test account: {enterprise_id}")
            else:
                print("❌ Test account not found. Please create an account first.")
                return
    
    try:
        seed_sample_data(enterprise_id)
        print("\n🎉 Sample data seeded successfully!")
        print("🌐 Visit the dashboard to see your data: http://127.0.0.1:8003/dashboard")
        
    except Exception as e:
        print(f"❌ Error seeding data: {e}")

if __name__ == "__main__":
    main()
