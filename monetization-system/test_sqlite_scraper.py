#!/usr/bin/env python3
"""
Test SQLite integration with the web scraper
"""
import asyncio
import sys
import os

# Add the automation directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'automation'))

from ethical_web_scraper import (
    init_database, 
    save_scraped_data, 
    get_scraped_data, 
    view_scraped_data,
    ScrapedPage
)
from datetime import datetime
import json

def test_database_operations():
    """Test basic database operations"""
    print("🧪 Testing SQLite Database Operations")
    print("=" * 40)
    
    # Test 1: Initialize database
    print("1. Initializing database...")
    init_database()
    
    # Test 2: Create sample data
    print("2. Creating sample scraped data...")
    sample_data = ScrapedPage(
        url="https://example.com/test",
        domain="example.com",
        title="Test Page Title",
        content="This is sample content for testing the SQLite database integration. " * 10,
        html="<html><body>Test HTML content</body></html>",
        screenshots={"full_page": b"fake_screenshot_data"},
        links=["https://example.com/link1", "https://example.com/link2"],
        metadata={"description": "Test page", "keywords": "test,sqlite,scraper"},
        collected_at=datetime.now(),
        data_hash="test_hash_123"
    )
    
    # Test 3: Save data
    print("3. Saving data to database...")
    success = save_scraped_data(sample_data)
    if success:
        print("   ✅ Data saved successfully")
    else:
        print("   ❌ Failed to save data")
    
    # Test 4: Retrieve data
    print("4. Retrieving data from database...")
    retrieved_data = get_scraped_data(limit=5)
    print(f"   📊 Retrieved {len(retrieved_data)} records")
    
    for item in retrieved_data:
        print(f"   - {item['domain']}: {item['title'][:30]}...")
    
    # Test 5: View formatted data
    print("\n5. Viewing formatted data:")
    view_scraped_data()
    
    print("\n✅ All database tests completed!")

async def test_real_scraping():
    """Test actual scraping with SQLite storage"""
    print("\n🌐 Testing Real Web Scraping with SQLite")
    print("=" * 40)
    
    try:
        from ethical_web_scraper import RealWebScraper
        
        # Test with a simple, fast-loading page
        test_url = "https://httpbin.org/html"
        
        async with RealWebScraper(headless=True) as scraper:
            print(f"Scraping test URL: {test_url}")
            page_data = await scraper.scrape_page(test_url)
            
            print(f"✓ Title: {page_data.title}")
            print(f"✓ Content length: {len(page_data.content)} chars")
            print(f"✓ Domain: {page_data.domain}")
            
            # Save to database
            if save_scraped_data(page_data):
                print("✅ Successfully saved to SQLite database")
            else:
                print("❌ Failed to save to database")
        
        # Show updated database contents
        print("\nUpdated database contents:")
        view_scraped_data()
        
    except ImportError:
        print("⚠️  Playwright not available, skipping real scraping test")
    except Exception as e:
        print(f"❌ Scraping test failed: {e}")

if __name__ == "__main__":
    # Run database tests
    test_database_operations()
    
    # Run scraping test if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--with-scraping":
        asyncio.run(test_real_scraping())
    else:
        print("\n💡 Run with --with-scraping to test actual web scraping")