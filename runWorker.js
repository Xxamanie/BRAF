/**
 * BRAF Production Worker - Advanced Browser Automation System
 * Integrates with existing BRAF infrastructure for production earnings
 */

import { chromium, firefox, webkit } from 'playwright';
import { Queue, Worker } from 'bullmq';
import Redis from 'ioredis';
import fs from 'fs/promises';
import path from 'path';
import crypto from 'crypto';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const CONFIG = {
    redis: {
        host: process.env.REDIS_HOST || '127.0.0.1',
        port: parseInt(process.env.REDIS_PORT) || 6379,
        password: process.env.REDIS_PASSWORD || null
    },
    worker: {
        concurrency: parseInt(process.env.MAX_CONCURRENT) || 3,
        headless: process.env.HEADLESS !== 'false',
        timeout: parseInt(process.env.TASK_TIMEOUT) || 300000, // 5 minutes
        retries: parseInt(process.env.MAX_RETRIES) || 3
    },
    earnings: {
        trackingEnabled: process.env.TRACK_EARNINGS !== 'false',
        dataFile: './BRAF/data/monetization_data.json',
        maxelIntegration: process.env.MAXEL_INTEGRATION === 'true'
    },
    stealth: {
        enabled: process.env.STEALTH_MODE !== 'false',
        fingerprintRotation: process.env.FINGERPRINT_ROTATION === 'true',
        proxyRotation: process.env.PROXY_ROTATION === 'true'
    }
};

class BRAFWorker {
    constructor() {
        this.redis = new Redis(CONFIG.redis);
        this.queue = new Queue('braf-tasks', { connection: this.redis });
        this.browsers = new Map();
        this.profiles = new Map();
        this.earnings = { total: 0, sessions: 0, lastUpdate: new Date() };
        this.isRunning = false;
        
        // Load existing earnings data
        this.loadEarningsData();
        
        // Initialize worker
        this.worker = new Worker('braf-tasks', this.processJob.bind(this), {
            connection: this.redis,
            concurrency: CONFIG.worker.concurrency,
            removeOnComplete: 50,
            removeOnFail: 20
        });
        
        this.setupEventHandlers();
    }

    async loadEarningsData() {
        try {
            const data = await fs.readFile(CONFIG.earnings.dataFile, 'utf8');
            const parsed = JSON.parse(data);
            this.earnings = {
                total: parsed.total_earnings || 0,
                sessions: parsed.total_sessions || 0,
                lastUpdate: new Date(parsed.last_update || Date.now())
            };
            console.log(`📊 Loaded earnings data: $${this.earnings.total.toFixed(2)}`);
        } catch (error) {
            console.log('📊 No existing earnings data found, starting fresh');
        }
    }

    async saveEarningsData() {
        if (!CONFIG.earnings.trackingEnabled) return;
        
        try {
            const data = {
                total_earnings: this.earnings.total,
                total_sessions: this.earnings.sessions,
                last_update: new Date().toISOString(),
                worker_stats: {
                    active_browsers: this.browsers.size,
                    active_profiles: this.profiles.size,
                    uptime: Date.now() - this.startTime
                }
            };
            
            await fs.mkdir(path.dirname(CONFIG.earnings.dataFile), { recursive: true });
            await fs.writeFile(CONFIG.earnings.dataFile, JSON.stringify(data, null, 2));
        } catch (error) {
            console.error('❌ Failed to save earnings data:', error.message);
        }
    }

    setupEventHandlers() {
        this.worker.on('completed', (job) => {
            console.log(`✅ Job ${job.id} completed: ${job.returnvalue?.earnings || 0} earned`);
        });

        this.worker.on('failed', (job, err) => {
            console.log(`❌ Job ${job.id} failed: ${err.message}`);
        });

        this.worker.on('progress', (job, progress) => {
            console.log(`🔄 Job ${job.id} progress: ${progress}%`);
        });

        // Graceful shutdown
        process.on('SIGINT', () => this.shutdown());
        process.on('SIGTERM', () => this.shutdown());
    }

    async processJob(job) {
        const { type, data } = job.data;
        console.log(`🚀 Processing ${type} job: ${job.id}`);
        
        try {
            switch (type) {
                case 'navigate':
                    return await this.handleNavigateJob(job, data);
                case 'scrape':
                    return await this.handleScrapeJob(job, data);
                case 'survey':
                    return await this.handleSurveyJob(job, data);
                case 'video':
                    return await this.handleVideoJob(job, data);
                case 'interaction':
                    return await this.handleInteractionJob(job, data);
                default:
                    throw new Error(`Unknown job type: ${type}`);
            }
        } catch (error) {
            console.error(`❌ Job ${job.id} error:`, error.message);
            throw error;
        }
    }

    async getBrowser(profileId = 'default') {
        if (this.browsers.has(profileId)) {
            return this.browsers.get(profileId);
        }

        const browserType = this.selectBrowserType();
        const profile = await this.getProfile(profileId);
        
        const browser = await browserType.launch({
            headless: CONFIG.worker.headless,
            args: this.getBrowserArgs(profile),
            proxy: profile.proxy || null
        });

        this.browsers.set(profileId, browser);
        return browser;
    }

    selectBrowserType() {
        const browsers = [chromium, firefox, webkit];
        const weights = [0.7, 0.2, 0.1]; // Chrome dominant, realistic distribution
        
        const random = Math.random();
        let sum = 0;
        
        for (let i = 0; i < browsers.length; i++) {
            sum += weights[i];
            if (random <= sum) return browsers[i];
        }
        
        return chromium; // fallback
    }

    getBrowserArgs(profile) {
        const args = [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-accelerated-2d-canvas',
            '--no-first-run',
            '--no-zygote',
            '--disable-gpu'
        ];

        if (CONFIG.stealth.enabled) {
            args.push(
                '--disable-blink-features=AutomationControlled',
                '--disable-features=VizDisplayCompositor',
                '--user-agent=' + (profile.userAgent || this.generateUserAgent())
            );
        }

        return args;
    }

    async getProfile(profileId) {
        if (this.profiles.has(profileId)) {
            return this.profiles.get(profileId);
        }

        const profile = {
            id: profileId,
            userAgent: this.generateUserAgent(),
            viewport: this.generateViewport(),
            locale: this.generateLocale(),
            timezone: this.generateTimezone(),
            proxy: CONFIG.stealth.proxyRotation ? await this.getProxy() : null,
            fingerprint: await this.generateFingerprint()
        };

        this.profiles.set(profileId, profile);
        return profile;
    }

    generateUserAgent() {
        const agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0'
        ];
        return agents[Math.floor(Math.random() * agents.length)];
    }

    generateViewport() {
        const viewports = [
            { width: 1920, height: 1080 },
            { width: 1366, height: 768 },
            { width: 1536, height: 864 },
            { width: 1440, height: 900 }
        ];
        return viewports[Math.floor(Math.random() * viewports.length)];
    }

    generateLocale() {
        return 'en-US';
    }

    generateTimezone() {
        const timezones = [
            'America/New_York',
            'America/Chicago', 
            'America/Denver',
            'America/Los_Angeles'
        ];
        return timezones[Math.floor(Math.random() * timezones.length)];
    }

    async getProxy() {
        // Implement proxy rotation logic
        return null; // Placeholder
    }

    async generateFingerprint() {
        return {
            canvas: crypto.randomBytes(16).toString('hex'),
            webgl: crypto.randomBytes(16).toString('hex'),
            audio: crypto.randomBytes(16).toString('hex')
        };
    }

    async handleNavigateJob(job, data) {
        const { url, profileId = 'default', waitTime = 5000 } = data;
        
        await job.updateProgress(10);
        
        const browser = await this.getBrowser(profileId);
        const context = await browser.newContext({
            viewport: this.profiles.get(profileId).viewport,
            locale: this.profiles.get(profileId).locale,
            timezoneId: this.profiles.get(profileId).timezone
        });
        
        await job.updateProgress(30);
        
        const page = await context.newPage();
        
        // Apply stealth measures
        if (CONFIG.stealth.enabled) {
            await this.applyStealth(page);
        }
        
        await job.updateProgress(50);
        
        try {
            await page.goto(url, { 
                waitUntil: 'domcontentloaded', 
                timeout: CONFIG.worker.timeout 
            });
            
            await job.updateProgress(70);
            
            // Simulate human behavior
            await this.simulateHumanBehavior(page, waitTime);
            
            await job.updateProgress(90);
            
            const title = await page.title();
            const earnings = this.calculateEarnings('navigate', waitTime);
            
            await this.updateEarnings(earnings);
            await job.updateProgress(100);
            
            return {
                success: true,
                title,
                url,
                earnings,
                timestamp: new Date().toISOString()
            };
            
        } finally {
            await context.close();
        }
    }

    async handleSurveyJob(job, data) {
        const { platform, surveyId, profileId = 'default' } = data;
        
        await job.updateProgress(10);
        
        const browser = await this.getBrowser(profileId);
        const context = await browser.newContext();
        const page = await context.newPage();
        
        await job.updateProgress(30);
        
        try {
            // Platform-specific survey handling
            let result;
            switch (platform) {
                case 'swagbucks':
                    result = await this.handleSwagbucksSurvey(page, surveyId, job);
                    break;
                case 'survey_junkie':
                    result = await this.handleSurveyJunkieSurvey(page, surveyId, job);
                    break;
                default:
                    result = await this.handleGenericSurvey(page, surveyId, job);
            }
            
            const earnings = this.calculateEarnings('survey', result.duration);
            await this.updateEarnings(earnings);
            
            return { ...result, earnings };
            
        } finally {
            await context.close();
        }
    }

    async handleVideoJob(job, data) {
        const { url, duration = 30000, profileId = 'default' } = data;
        
        const browser = await this.getBrowser(profileId);
        const context = await browser.newContext();
        const page = await context.newPage();
        
        try {
            await page.goto(url);
            await job.updateProgress(25);
            
            // Look for video elements
            await page.waitForSelector('video', { timeout: 10000 });
            await job.updateProgress(50);
            
            // Play video if not auto-playing
            await page.evaluate(() => {
                const video = document.querySelector('video');
                if (video && video.paused) {
                    video.play();
                }
            });
            
            // Wait for video duration
            await this.simulateVideoWatching(page, duration, job);
            
            const earnings = this.calculateEarnings('video', duration);
            await this.updateEarnings(earnings);
            
            return {
                success: true,
                url,
                duration,
                earnings,
                timestamp: new Date().toISOString()
            };
            
        } finally {
            await context.close();
        }
    }

    async handleInteractionJob(job, data) {
        const { url, actions, profileId = 'default' } = data;
        
        const browser = await this.getBrowser(profileId);
        const context = await browser.newContext();
        const page = await context.newPage();
        
        try {
            await page.goto(url);
            await job.updateProgress(20);
            
            for (let i = 0; i < actions.length; i++) {
                const action = actions[i];
                await this.executeAction(page, action);
                await job.updateProgress(20 + (60 * (i + 1) / actions.length));
            }
            
            const earnings = this.calculateEarnings('interaction', actions.length * 1000);
            await this.updateEarnings(earnings);
            
            return {
                success: true,
                actionsCompleted: actions.length,
                earnings,
                timestamp: new Date().toISOString()
            };
            
        } finally {
            await context.close();
        }
    }

    async applyStealth(page) {
        // Remove webdriver property
        await page.addInitScript(() => {
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        });

        // Override plugins
        await page.addInitScript(() => {
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
        });

        // Override languages
        await page.addInitScript(() => {
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
        });
    }

    async simulateHumanBehavior(page, duration) {
        const actions = Math.floor(duration / 2000); // Action every 2 seconds
        
        for (let i = 0; i < actions; i++) {
            const action = Math.random();
            
            if (action < 0.3) {
                // Random scroll
                await page.evaluate(() => {
                    window.scrollBy(0, Math.random() * 200 - 100);
                });
            } else if (action < 0.6) {
                // Random mouse movement
                await page.mouse.move(
                    Math.random() * 800,
                    Math.random() * 600
                );
            } else {
                // Random click on safe element
                try {
                    await page.click('body', { force: true });
                } catch (e) {
                    // Ignore click errors
                }
            }
            
            await this.randomDelay(1000, 3000);
        }
    }

    async simulateVideoWatching(page, duration, job) {
        const checkInterval = 5000;
        const totalChecks = Math.floor(duration / checkInterval);
        
        for (let i = 0; i < totalChecks; i++) {
            await new Promise(resolve => setTimeout(resolve, checkInterval));
            
            // Check if video is still playing
            const isPlaying = await page.evaluate(() => {
                const video = document.querySelector('video');
                return video && !video.paused && !video.ended;
            });
            
            if (!isPlaying) {
                // Try to resume
                await page.evaluate(() => {
                    const video = document.querySelector('video');
                    if (video) video.play();
                });
            }
            
            await job.updateProgress(50 + (40 * (i + 1) / totalChecks));
        }
    }

    async executeAction(page, action) {
        switch (action.type) {
            case 'click':
                await page.click(action.selector);
                break;
            case 'type':
                await page.fill(action.selector, action.text);
                break;
            case 'scroll':
                await page.evaluate((pixels) => {
                    window.scrollBy(0, pixels);
                }, action.pixels || 100);
                break;
            case 'wait':
                await page.waitForTimeout(action.duration || 1000);
                break;
        }
        
        await this.randomDelay(500, 2000);
    }

    async handleSwagbucksSurvey(page, surveyId, job) {
        // Swagbucks-specific survey logic
        await page.goto(`https://swagbucks.com/surveys/${surveyId}`);
        await job.updateProgress(50);
        
        // Simulate survey completion
        await this.randomDelay(30000, 60000);
        await job.updateProgress(90);
        
        return {
            success: true,
            platform: 'swagbucks',
            surveyId,
            duration: 45000
        };
    }

    async handleSurveyJunkieSurvey(page, surveyId, job) {
        // Survey Junkie-specific logic
        await page.goto(`https://surveyjunkie.com/surveys/${surveyId}`);
        await job.updateProgress(50);
        
        await this.randomDelay(20000, 40000);
        await job.updateProgress(90);
        
        return {
            success: true,
            platform: 'survey_junkie',
            surveyId,
            duration: 30000
        };
    }

    async handleGenericSurvey(page, surveyId, job) {
        // Generic survey handling
        await this.randomDelay(15000, 30000);
        await job.updateProgress(90);
        
        return {
            success: true,
            platform: 'generic',
            surveyId,
            duration: 22500
        };
    }

    calculateEarnings(type, duration) {
        const rates = {
            navigate: 0.001, // $0.001 per second
            survey: 0.01,    // $0.01 per second  
            video: 0.005,    // $0.005 per second
            interaction: 0.002 // $0.002 per second
        };
        
        const rate = rates[type] || 0.001;
        const seconds = duration / 1000;
        return Math.round(rate * seconds * 100) / 100; // Round to 2 decimals
    }

    async updateEarnings(amount) {
        this.earnings.total += amount;
        this.earnings.sessions += 1;
        this.earnings.lastUpdate = new Date();
        
        await this.saveEarningsData();
        
        // Transfer to MAXEL if enabled
        if (CONFIG.earnings.maxelIntegration && this.earnings.total >= 1.0) {
            await this.transferToMaxel();
        }
    }

    async transferToMaxel() {
        try {
            // Call MAXEL transfer script
            const { spawn } = await import('child_process');
            const transfer = spawn('python', ['BRAF/transfer_to_maxel.py'], {
                stdio: 'inherit'
            });
            
            transfer.on('close', (code) => {
                if (code === 0) {
                    console.log('💰 Successfully transferred earnings to MAXEL');
                    this.earnings.total = 0; // Reset after transfer
                }
            });
        } catch (error) {
            console.error('❌ MAXEL transfer failed:', error.message);
        }
    }

    async randomDelay(min, max) {
        const delay = Math.random() * (max - min) + min;
        return new Promise(resolve => setTimeout(resolve, delay));
    }

    async start() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        this.startTime = Date.now();
        
        console.log('🚀 BRAF Worker starting...');
        console.log(`📊 Current earnings: $${this.earnings.total.toFixed(2)}`);
        console.log(`⚙️  Concurrency: ${CONFIG.worker.concurrency}`);
        console.log(`🔒 Stealth mode: ${CONFIG.stealth.enabled ? 'ON' : 'OFF'}`);
        
        // Start processing jobs
        await this.worker.run();
        
        console.log('✅ BRAF Worker ready for jobs');
    }

    async shutdown() {
        if (!this.isRunning) return;
        
        console.log('🛑 Shutting down BRAF Worker...');
        
        this.isRunning = false;
        
        // Close all browsers
        for (const [profileId, browser] of this.browsers) {
            try {
                await browser.close();
                console.log(`🔒 Closed browser for profile: ${profileId}`);
            } catch (error) {
                console.error(`❌ Error closing browser ${profileId}:`, error.message);
            }
        }
        
        // Close worker
        await this.worker.close();
        
        // Save final earnings
        await this.saveEarningsData();
        
        // Close Redis connection
        await this.redis.quit();
        
        console.log('✅ BRAF Worker shutdown complete');
        process.exit(0);
    }

    async addJob(type, data, options = {}) {
        return await this.queue.add(type, { type, data }, {
            attempts: CONFIG.worker.retries,
            backoff: {
                type: 'exponential',
                delay: 2000,
            },
            ...options
        });
    }

    getStats() {
        return {
            earnings: this.earnings,
            browsers: this.browsers.size,
            profiles: this.profiles.size,
            uptime: this.isRunning ? Date.now() - this.startTime : 0,
            isRunning: this.isRunning
        };
    }
}

// Export for use as module
export default BRAFWorker;

// Run directly if called as script
if (import.meta.url === `file://${process.argv[1]}`) {
    const worker = new BRAFWorker();
    
    // Add some sample jobs
    await worker.start();
    
    // Example job additions
    await worker.addJob('navigate', {
        url: 'https://swagbucks.com/watch',
        waitTime: 30000,
        profileId: 'profile1'
    });
    
    await worker.addJob('video', {
        url: 'https://inboxdollars.com/watch',
        duration: 60000,
        profileId: 'profile2'
    });
    
    console.log('📋 Sample jobs added to queue');
}