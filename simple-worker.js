/**
 * Simple BRAF Worker - No Redis Required
 * Basic browser automation for immediate testing
 */

import { chromium } from 'playwright';
import fs from 'fs/promises';
import path from 'path';

const jobs = [
    {
        name: "Swagbucks Watch",
        url: "https://swagbucks.com/watch",
        wait: 5000
    },
    {
        name: "InboxDollars Watch", 
        url: "https://inboxdollars.com/watch",
        wait: 4000
    },
    {
        name: "ySense Watch",
        url: "https://ysense.com/video", 
        wait: 6000
    },
    {
        name: "TimeBucks Tasks",
        url: "https://timebucks.com/tasks",
        wait: 3000
    }
];

class SimpleBRAFWorker {
    constructor() {
        this.earnings = 0;
        this.sessionsCompleted = 0;
        this.startTime = Date.now();
    }

    async start() {
        console.log('🚀 BRAF Simple Worker starting...');
        console.log('📋 Jobs to process:', jobs.length);
        
        const browser = await chromium.launch({ 
            headless: process.env.HEADLESS !== 'false'
        });
        
        try {
            for (let i = 0; i < jobs.length; i++) {
                const job = jobs[i];
                console.log(`\n→ Processing ${i + 1}/${jobs.length}: ${job.name}`);
                
                await this.processJob(browser, job);
                
                // Random delay between jobs
                const delay = Math.random() * 3000 + 2000; // 2-5 seconds
                console.log(`⏳ Waiting ${Math.round(delay/1000)}s before next job...`);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        } finally {
            await browser.close();
        }
        
        await this.showFinalStats();
        console.log('\n✅ BRAF Simple Worker completed successfully!');
    }

    async processJob(browser, job) {
        const context = await browser.newContext({
            viewport: { width: 1366, height: 768 },
            userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        });
        
        const page = await context.newPage();
        
        try {
            // Apply basic stealth
            await page.addInitScript(() => {
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            });
            
            console.log(`   🌐 Navigating to: ${job.url}`);
            await page.goto(job.url, { 
                waitUntil: 'domcontentloaded', 
                timeout: 30000 
            });
            
            const title = await page.title();
            console.log(`   📄 Page title: ${title}`);
            
            // Take screenshot
            const screenshotName = `${job.name.replace(/\s+/g, "_")}.png`;
            await page.screenshot({
                path: `screenshots/${screenshotName}`,
                fullPage: false
            });
            console.log(`   📸 Screenshot saved: ${screenshotName}`);
            
            // Simulate human behavior
            await this.simulateHumanActivity(page, job.wait);
            
            // Calculate earnings (simple simulation)
            const earnings = this.calculateEarnings(job.wait);
            this.earnings += earnings;
            this.sessionsCompleted++;
            
            console.log(`   💰 Earned: $${earnings.toFixed(4)}`);
            console.log(`   ✅ Job completed successfully`);
            
        } catch (error) {
            console.log(`   ❌ Job failed: ${error.message}`);
        } finally {
            await context.close();
        }
    }

    async simulateHumanActivity(page, duration) {
        console.log(`   🤖 Simulating human activity for ${duration/1000}s...`);
        
        const actions = Math.floor(duration / 2000); // Action every 2 seconds
        
        for (let i = 0; i < actions; i++) {
            const action = Math.random();
            
            try {
                if (action < 0.4) {
                    // Random scroll
                    const scrollAmount = Math.random() * 300 - 150;
                    await page.evaluate((amount) => {
                        window.scrollBy(0, amount);
                    }, scrollAmount);
                } else if (action < 0.7) {
                    // Random mouse movement
                    await page.mouse.move(
                        Math.random() * 800 + 100,
                        Math.random() * 600 + 100
                    );
                } else {
                    // Random pause (do nothing)
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }
            } catch (error) {
                // Ignore interaction errors
            }
            
            // Wait between actions
            await new Promise(resolve => setTimeout(resolve, 1500 + Math.random() * 1000));
        }
    }

    calculateEarnings(duration) {
        // Simple earnings calculation: $0.001 per second
        const seconds = duration / 1000;
        return seconds * 0.001;
    }

    async showFinalStats() {
        const runtime = (Date.now() - this.startTime) / 1000;
        const hourlyRate = (this.earnings / runtime) * 3600;
        
        console.log('\n' + '='.repeat(50));
        console.log('📊 FINAL STATISTICS');
        console.log('='.repeat(50));
        console.log(`💰 Total Earnings: $${this.earnings.toFixed(4)}`);
        console.log(`📈 Sessions Completed: ${this.sessionsCompleted}`);
        console.log(`⏱️  Runtime: ${Math.round(runtime)}s`);
        console.log(`💵 Hourly Rate: $${hourlyRate.toFixed(4)}/hour`);
        console.log('='.repeat(50));
        
        // Save earnings data
        await this.saveEarningsData();
    }

    async saveEarningsData() {
        try {
            // Ensure BRAF directory exists
            await fs.mkdir('BRAF/data', { recursive: true });
            
            const data = {
                total_earnings: this.earnings,
                total_sessions: this.sessionsCompleted,
                last_update: new Date().toISOString(),
                worker_type: 'simple_worker',
                runtime_seconds: (Date.now() - this.startTime) / 1000
            };
            
            await fs.writeFile('BRAF/data/monetization_data.json', JSON.stringify(data, null, 2));
            console.log('💾 Earnings data saved to BRAF/data/monetization_data.json');
        } catch (error) {
            console.log('⚠️  Could not save earnings data:', error.message);
        }
    }
}

// Run the worker
(async () => {
    try {
        const worker = new SimpleBRAFWorker();
        await worker.start();
    } catch (error) {
        console.error('❌ Worker failed:', error);
        process.exit(1);
    }
})();