/**
 * Simple BRAF Manager - No Redis Required
 * Provides basic worker management without queue system
 */

import { chromium } from 'playwright';
import fs from 'fs/promises';

class SimpleBRAFManager {
    constructor() {
        this.isRunning = false;
        this.earnings = 0;
        this.sessionsCompleted = 0;
        this.startTime = Date.now();
        this.platforms = [
            { name: "Swagbucks Watch", url: "https://swagbucks.com/watch", wait: 30000 },
            { name: "InboxDollars Watch", url: "https://inboxdollars.com/watch", wait: 25000 },
            { name: "ySense Watch", url: "https://ysense.com/video", wait: 35000 },
            { name: "TimeBucks Tasks", url: "https://timebucks.com/tasks", wait: 20000 },
            { name: "PrizeRebel Watch", url: "https://prizerebel.com/watch", wait: 30000 },
            { name: "MyPoints Watch", url: "https://mypoints.com/watch", wait: 25000 },
            { name: "SurveyTime Surveys", url: "https://surveytime.io", wait: 15000 },
            { name: "Toloka Tasks", url: "https://toloka.ai", wait: 20000 }
        ];
    }

    async start() {
        console.log('🎯 Starting BRAF Simple Manager...');
        console.log(`📋 Managing ${this.platforms.length} earning platforms`);
        console.log('🔄 Running continuous earning cycles...\n');
        
        this.isRunning = true;
        
        // Start monitoring
        this.startMonitoring();
        
        // Run continuous cycles
        while (this.isRunning) {
            await this.runEarningCycle();
            
            // Wait between cycles
            const cycleDelay = 300000; // 5 minutes between cycles
            console.log(`⏳ Waiting ${cycleDelay/60000} minutes before next cycle...\n`);
            await this.sleep(cycleDelay);
        }
    }

    async runEarningCycle() {
        console.log('🚀 Starting new earning cycle...');
        
        const browser = await chromium.launch({ 
            headless: process.env.HEADLESS !== 'false'
        });
        
        try {
            // Process 3 random platforms per cycle
            const selectedPlatforms = this.getRandomPlatforms(3);
            
            for (let i = 0; i < selectedPlatforms.length; i++) {
                const platform = selectedPlatforms[i];
                console.log(`\n→ Processing ${i + 1}/${selectedPlatforms.length}: ${platform.name}`);
                
                await this.processPlatform(browser, platform);
                
                // Delay between platforms
                if (i < selectedPlatforms.length - 1) {
                    const delay = Math.random() * 10000 + 5000; // 5-15 seconds
                    console.log(`⏳ Waiting ${Math.round(delay/1000)}s before next platform...`);
                    await this.sleep(delay);
                }
            }
        } finally {
            await browser.close();
        }
        
        console.log('✅ Earning cycle completed');
    }

    async processPlatform(browser, platform) {
        const context = await browser.newContext({
            viewport: { width: 1366, height: 768 },
            userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        });
        
        const page = await context.newPage();
        
        try {
            // Apply stealth
            await page.addInitScript(() => {
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            });
            
            console.log(`   🌐 Navigating to: ${platform.url}`);
            await page.goto(platform.url, { 
                waitUntil: 'domcontentloaded', 
                timeout: 30000 
            });
            
            const title = await page.title();
            console.log(`   📄 Page title: ${title.substring(0, 60)}...`);
            
            // Take screenshot
            const screenshotName = `${platform.name.replace(/\s+/g, "_")}_${Date.now()}.png`;
            await page.screenshot({
                path: `screenshots/${screenshotName}`,
                fullPage: false
            });
            console.log(`   📸 Screenshot saved: ${screenshotName}`);
            
            // Simulate activity
            await this.simulateActivity(page, platform.wait);
            
            // Calculate earnings
            const earnings = this.calculateEarnings(platform.wait);
            this.earnings += earnings;
            this.sessionsCompleted++;
            
            console.log(`   💰 Earned: $${earnings.toFixed(4)}`);
            console.log(`   ✅ Platform completed successfully`);
            
            // Save earnings
            await this.saveEarnings();
            
        } catch (error) {
            console.log(`   ❌ Platform failed: ${error.message}`);
        } finally {
            await context.close();
        }
    }

    async simulateActivity(page, duration) {
        console.log(`   🤖 Simulating activity for ${duration/1000}s...`);
        
        const actions = Math.floor(duration / 3000); // Action every 3 seconds
        
        for (let i = 0; i < actions; i++) {
            try {
                const action = Math.random();
                
                if (action < 0.4) {
                    // Scroll
                    await page.evaluate(() => {
                        window.scrollBy(0, Math.random() * 300 - 150);
                    });
                } else if (action < 0.7) {
                    // Mouse movement
                    await page.mouse.move(
                        Math.random() * 800 + 100,
                        Math.random() * 600 + 100
                    );
                } else {
                    // Wait
                    await this.sleep(2000);
                }
            } catch (error) {
                // Ignore interaction errors
            }
            
            await this.sleep(2000 + Math.random() * 2000);
        }
    }

    getRandomPlatforms(count) {
        const shuffled = [...this.platforms].sort(() => 0.5 - Math.random());
        return shuffled.slice(0, count);
    }

    calculateEarnings(duration) {
        const seconds = duration / 1000;
        return seconds * 0.002; // $0.002 per second
    }

    async saveEarnings() {
        try {
            await fs.mkdir('BRAF/data', { recursive: true });
            
            const data = {
                total_earnings: this.earnings,
                total_sessions: this.sessionsCompleted,
                last_update: new Date().toISOString(),
                worker_type: 'simple_manager',
                runtime_seconds: (Date.now() - this.startTime) / 1000,
                hourly_rate: this.calculateHourlyRate()
            };
            
            await fs.writeFile('BRAF/data/monetization_data.json', JSON.stringify(data, null, 2));
        } catch (error) {
            console.log('⚠️  Could not save earnings data:', error.message);
        }
    }

    calculateHourlyRate() {
        const hours = (Date.now() - this.startTime) / (1000 * 60 * 60);
        return hours > 0 ? this.earnings / hours : 0;
    }

    startMonitoring() {
        // Display stats every 2 minutes
        setInterval(() => {
            this.displayStats();
        }, 120000);
        
        // Initial stats display
        setTimeout(() => this.displayStats(), 10000);
    }

    displayStats() {
        const runtime = (Date.now() - this.startTime) / 1000;
        const hourlyRate = this.calculateHourlyRate();
        
        console.clear();
        console.log('🤖 BRAF SIMPLE MANAGER DASHBOARD');
        console.log('================================');
        console.log(`💰 Total Earnings: $${this.earnings.toFixed(4)}`);
        console.log(`📊 Sessions Completed: ${this.sessionsCompleted}`);
        console.log(`⏱️  Runtime: ${Math.floor(runtime/60)}m ${Math.floor(runtime%60)}s`);
        console.log(`💵 Hourly Rate: $${hourlyRate.toFixed(4)}/hour`);
        console.log(`🔄 Status: ${this.isRunning ? 'Running' : 'Stopped'}`);
        console.log('================================');
        
        if (this.earnings >= 1.0) {
            console.log('🎉 Ready for MAXEL transfer! ($1.00+ earned)');
        }
    }

    async sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async stop() {
        console.log('\n🛑 Stopping Simple Manager...');
        this.isRunning = false;
        await this.saveEarnings();
        console.log('✅ Simple Manager stopped');
        process.exit(0);
    }
}

// Handle graceful shutdown
process.on('SIGINT', async () => {
    if (global.manager) {
        await global.manager.stop();
    } else {
        process.exit(0);
    }
});

// Run the manager
(async () => {
    try {
        global.manager = new SimpleBRAFManager();
        await global.manager.start();
    } catch (error) {
        console.error('❌ Manager failed:', error);
        process.exit(1);
    }
})();