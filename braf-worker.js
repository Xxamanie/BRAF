#!/usr/bin/env node
/**
 * BRAF Worker - Production Browser Automation with BullMQ
 * Enhanced queue-based browser automation system
 */

import { Queue, Worker, QueueScheduler } from 'bullmq';
import { chromium } from 'playwright';
import dotenv from 'dotenv';

// Load environment configuration
dotenv.config();

// Redis connection configuration
const redisConnection = {
    connection: {
        host: process.env.REDIS_HOST || '127.0.0.1',
        port: parseInt(process.env.REDIS_PORT) || 6379,
        maxRetriesPerRequest: 3,
        retryDelayOnFailover: 100,
        lazyConnect: true
    }
};

// Initialize queue and scheduler
const queue = new Queue('braf-tasks', redisConnection);
const scheduler = new QueueScheduler('braf-tasks', redisConnection);

console.log('🚀 BRAF Worker Starting...');
console.log(`📊 Max Concurrent Jobs: ${process.env.MAX_CONCURRENT || 3}`);
console.log(`🎭 Headless Mode: ${process.env.HEADLESS === 'true'}`);
console.log(`🔗 Redis: ${process.env.REDIS_HOST || '127.0.0.1'}:${process.env.REDIS_PORT || 6379}`);

// Enhanced browser automation worker
const worker = new Worker('braf-tasks', async (job) => {
    const startTime = Date.now();
    console.log(`🔄 Processing job ${job.id}: ${job.name}`);
    
    let browser = null;
    let page = null;
    
    try {
        // Launch browser with enhanced configuration
        browser = await chromium.launch({
            headless: process.env.HEADLESS === 'true',
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--no-first-run',
                '--no-zygote',
                '--disable-gpu',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding'
            ]
        });
        
        // Create new page with stealth configuration
        page = await browser.newPage();
        
        // Enhanced stealth measures
        await page.evaluateOnNewDocument(() => {
            // Hide webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false,
            });
            
            // Mock plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            
            // Mock languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
            
            // Override permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        });
        
        // Set realistic viewport and user agent
        await page.setViewportSize({ width: 1920, height: 1080 });
        await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');
        
        // Process different job types
        let result = {};
        
        switch (job.name) {
            case 'navigate':
                result = await handleNavigateJob(page, job.data);
                break;
            case 'scrape':
                result = await handleScrapeJob(page, job.data);
                break;
            case 'interact':
                result = await handleInteractJob(page, job.data);
                break;
            case 'monitor':
                result = await handleMonitorJob(page, job.data);
                break;
            default:
                result = await handleGenericJob(page, job.data);
        }
        
        const duration = Date.now() - startTime;
        console.log(`✅ Job ${job.id} completed in ${duration}ms`);
        
        return {
            success: true,
            duration,
            result,
            timestamp: new Date().toISOString()
        };
        
    } catch (error) {
        const duration = Date.now() - startTime;
        console.error(`❌ Job ${job.id} failed after ${duration}ms:`, error.message);
        
        return {
            success: false,
            error: error.message,
            duration,
            timestamp: new Date().toISOString()
        };
        
    } finally {
        // Cleanup resources
        if (page) await page.close().catch(() => {});
        if (browser) await browser.close().catch(() => {});
    }
}, {
    concurrency: parseInt(process.env.MAX_CONCURRENT) || 3,
    removeOnComplete: 10,
    removeOnFail: 5
});

// Job handlers for different automation tasks

async function handleNavigateJob(page, data) {
    const { url, waitUntil = 'domcontentloaded', timeout = 30000 } = data;
    
    await page.goto(url, { waitUntil, timeout });
    
    return {
        url,
        title: await page.title(),
        status: 'navigated'
    };
}

async function handleScrapeJob(page, data) {
    const { url, selectors = {}, waitFor } = data;
    
    if (url) {
        await page.goto(url, { waitUntil: 'domcontentloaded' });
    }
    
    if (waitFor) {
        await page.waitForSelector(waitFor, { timeout: 10000 });
    }
    
    const results = {};
    
    for (const [key, selector] of Object.entries(selectors)) {
        try {
            const element = await page.$(selector);
            if (element) {
                results[key] = await element.textContent();
            }
        } catch (error) {
            results[key] = null;
        }
    }
    
    return {
        url: page.url(),
        data: results,
        status: 'scraped'
    };
}

async function handleInteractJob(page, data) {
    const { url, actions = [] } = data;
    
    if (url) {
        await page.goto(url, { waitUntil: 'domcontentloaded' });
    }
    
    const results = [];
    
    for (const action of actions) {
        try {
            switch (action.type) {
                case 'click':
                    await page.click(action.selector);
                    results.push({ action: 'click', selector: action.selector, success: true });
                    break;
                case 'type':
                    await page.fill(action.selector, action.text);
                    results.push({ action: 'type', selector: action.selector, success: true });
                    break;
                case 'wait':
                    await page.waitForTimeout(action.duration || 1000);
                    results.push({ action: 'wait', duration: action.duration, success: true });
                    break;
                default:
                    results.push({ action: action.type, success: false, error: 'Unknown action' });
            }
        } catch (error) {
            results.push({ action: action.type, success: false, error: error.message });
        }
    }
    
    return {
        url: page.url(),
        actions: results,
        status: 'interacted'
    };
}

async function handleMonitorJob(page, data) {
    const { url, checks = [] } = data;
    
    if (url) {
        await page.goto(url, { waitUntil: 'domcontentloaded' });
    }
    
    const results = {
        url: page.url(),
        title: await page.title(),
        timestamp: new Date().toISOString(),
        checks: []
    };
    
    for (const check of checks) {
        try {
            const element = await page.$(check.selector);
            results.checks.push({
                name: check.name,
                selector: check.selector,
                found: !!element,
                text: element ? await element.textContent() : null
            });
        } catch (error) {
            results.checks.push({
                name: check.name,
                selector: check.selector,
                found: false,
                error: error.message
            });
        }
    }
    
    return results;
}

async function handleGenericJob(page, data) {
    const { url } = data;
    
    if (url) {
        await page.goto(url, { waitUntil: 'domcontentloaded' });
        console.log(`📄 Visited: ${url}`);
        
        return {
            url,
            title: await page.title(),
            status: 'visited'
        };
    }
    
    return { status: 'no_action' };
}

// Event handlers
worker.on('completed', (job, result) => {
    console.log(`✅ Job ${job.id} completed successfully`);
});

worker.on('failed', (job, err) => {
    console.log(`❌ Job ${job.id} failed: ${err.message}`);
});

worker.on('progress', (job, progress) => {
    console.log(`📊 Job ${job.id} progress: ${progress}%`);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
    console.log('🛑 Received SIGTERM, shutting down gracefully...');
    await worker.close();
    await scheduler.close();
    process.exit(0);
});

process.on('SIGINT', async () => {
    console.log('🛑 Received SIGINT, shutting down gracefully...');
    await worker.close();
    await scheduler.close();
    process.exit(0);
});

// Export queue for external job submission
export { queue };

console.log('🎯 BRAF Worker ready and waiting for jobs...');
console.log('📋 Supported job types: navigate, scrape, interact, monitor');
console.log('🔄 Use Ctrl+C to stop the worker');