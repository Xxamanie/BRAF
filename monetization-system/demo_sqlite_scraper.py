#!/usr/bin/env python3
"""
Demo: Web Scraper with SQLite Database Integration
"""
import asyncio
import sys
import os
from datetime import datetime

# Add automation directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'automation'))

try:
    from ethical_web_scraper import (
        RealWebScraper, 
        init_database, 
        save_scraped_data, 
        get_scraped_data,
        view_scraped_data
    )
    PLAYWRIGHT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Playwright not available: {e}")
    PLAYWRIGHT_AVAILABLE = False

def demo_database_only():
    """Demo database functionality without scraping"""
    print("🗄️  SQLite Database Demo")
    print("=" * 30)
    
    # Initialize database
    print("1. Initializing SQLite database...")
    init_database()
    
    # Show current data
    print("\n2. Current database contents:")
    view_scraped_data()
    
    # Search functionality
    print("\n3. Testing search functionality...")
    from database_manager import ScraperDatabaseManager
    manager = ScraperDatabaseManager()
    
    # Search for content
    results = manager.search_content("test")
    print(f"   Found {len(results)} results for 'test'")
    
    # Get stats
    stats = manager.get_stats()
    print(f"\n4. Database statistics:")
    print(f"   📊 Total records: {stats['total_records']}")
    print(f"   🌐 Unique domains: {stats['unique_domains']}")
    print(f"   💾 Database size: {stats['database_size_mb']} MB")

async def demo_with_scraping():
    """Demo with actual web scraping"""
    if not PLAYWRIGHT_AVAILABLE:
        print("❌ Playwright not available for scraping demo")
        return
    
    print("\n🌐 Web Scraping + SQLite Demo")
    print("=" * 35)
    
    # Simple test URLs that should work reliably
    test_urls = [
        "https://httpbin.org/html",  # Simple test page
        "https://example.com",       # Basic example page
    ]
    
    async with RealWebScraper(headless=True) as scraper:
        for url in test_urls:
            print(f"\n📥 Scraping: {url}")
            try:
                # Scrape the page
                page_data = await scraper.scrape_page(url)
                
                print(f"   ✓ Title: {page_data.title}")
                print(f"   ✓ Content: {len(page_data.content)} characters")
                print(f"   ✓ Links: {len(page_data.links)} found")
                print(f"   ✓ Domain: {page_data.domain}")
                
                # Save to database
                if save_scraped_data(page_data):
                    print(f"   ✅ Saved to SQLite database")
                else:
                    print(f"   ❌ Failed to save to database")
                
            except Exception as e:
                print(f"   ❌ Failed to scrape {url}: {e}")
    
    # Show updated database
    print("\n📊 Updated database contents:")
    view_scraped_data()

def interactive_demo():
    """Interactive demo menu"""
    print("\n🎯 Interactive SQLite Scraper Demo")
    print("=" * 35)
    print("1. Database operations only")
    print("2. Web scraping + database (requires Playwright)")
    print("3. View database contents")
    print("4. Search database")
    print("5. Database statistics")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nSelect option (0-5): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                demo_database_only()
            elif choice == "2":
                if PLAYWRIGHT_AVAILABLE:
                    asyncio.run(demo_with_scraping())
                else:
                    print("❌ Playwright not available")
            elif choice == "3":
                print("\n📋 Database Contents:")
                view_scraped_data()
            elif choice == "4":
                query = input("Enter search term: ").strip()
                if query:
                    from database_manager import ScraperDatabaseManager
                    manager = ScraperDatabaseManager()
                    results = manager.search_content(query)
                    print(f"\n🔍 Found {len(results)} results for '{query}':")
                    for result in results:
                        print(f"   {result['domain']}: {result['title']}")
            elif choice == "5":
                from database_manager import ScraperDatabaseManager
                manager = ScraperDatabaseManager()
                stats = manager.get_stats()
                print(f"\n📈 Database Statistics:")
                print(f"   Total records: {stats['total_records']}")
                print(f"   Unique domains: {stats['unique_domains']}")
                print(f"   Database size: {stats['database_size_mb']} MB")
                print(f"   Recent records: {stats['recent_records']}")
            else:
                print("❌ Invalid option")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 SQLite Web Scraper Demo")
    print("=" * 30)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--database-only":
            demo_database_only()
        elif sys.argv[1] == "--with-scraping":
            asyncio.run(demo_with_scraping())
        elif sys.argv[1] == "--interactive":
            interactive_demo()
        else:
            print("Usage:")
            print("  python demo_sqlite_scraper.py --database-only")
            print("  python demo_sqlite_scraper.py --with-scraping")
            print("  python demo_sqlite_scraper.py --interactive")
    else:
        # Default: run database demo
        demo_database_only()
        print("\n💡 Run with --interactive for full demo menu")
        print("💡 Run with --with-scraping to test actual web scraping")
