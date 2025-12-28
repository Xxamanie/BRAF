#!/usr/bin/env node
/**
 * BRAF Job Manager - Submit and manage automation jobs
 */

import { Queue } from 'bullmq';
import dotenv from 'dotenv';

dotenv.config();

const redisConnection = {
    connection: {
        host: process.env.REDIS_HOST || '127.0.0.1',
        port: parseInt(process.env.REDIS_PORT) || 6379
    }
};

const queue = new Queue('braf-tasks', redisConnection);

// Job submission functions
export async function addNavigateJob(url, options = {}) {
    const job = await queue.add('navigate', {
        url,
        waitUntil: options.waitUntil || 'domcontentloaded',
        timeout: options.timeout || 30000
    });
    
    console.log(`📝 Navigate job added: ${job.id} -> ${url}`);
    return job;
}

export async function addScrapeJob(url, selectors, options = {}) {
    const job = await queue.add('scrape', {
        url,
        selectors,
        waitFor: options.waitFor
    });
    
    console.log(`📝 Scrape job added: ${job.id} -> ${url}`);
    return job;
}

export async function addInteractJob(url, actions, options = {}) {
    const job = await queue.add('interact', {
        url,
        actions
    });
    
    console.log(`📝 Interact job added: ${job.id} -> ${url}`);
    return job;
}

export async function addMonitorJob(url, checks, options = {}) {
    const job = await queue.add('monitor', {
        url,
        checks
    });
    
    console.log(`📝 Monitor job added: ${job.id} -> ${url}`);
    return job;
}

// Demo jobs for testing
export async function runDemoJobs() {
    console.log('🎯 Running BRAF Demo Jobs...');
    
    // 1. Simple navigation
    await addNavigateJob('https://httpbin.org/user-agent');
    
    // 2. Data scraping
    await addScrapeJob('https://httpbin.org/json', {
        slideshow_title: 'pre',
        content: 'body'
    });
    
    // 3. Form interaction
    await addInteractJob('https://httpbin.org/forms/post', [
        { type: 'type', selector: 'input[name="custname"]', text: 'BRAF Test User' },
        { type: 'type', selector: 'input[name="custtel"]', text: '123-456-7890' },
        { type: 'wait', duration: 1000 },
        { type: 'click', selector: 'input[type="submit"]' }
    ]);
    
    // 4. Monitoring check
    await addMonitorJob('https://httpbin.org/status/200', [
        { name: 'status_check', selector: 'body' },
        { name: 'title_check', selector: 'title' }
    ]);
    
    console.log('✅ Demo jobs submitted to queue');
}

// Queue management functions
export async function getQueueStats() {
    const waiting = await queue.getWaiting();
    const active = await queue.getActive();
    const completed = await queue.getCompleted();
    const failed = await queue.getFailed();
    
    return {
        waiting: waiting.length,
        active: active.length,
        completed: completed.length,
        failed: failed.length
    };
}

export async function clearQueue() {
    await queue.clean(0, 1000, 'completed');
    await queue.clean(0, 1000, 'failed');
    console.log('🧹 Queue cleaned');
}

// CLI interface
if (import.meta.url === `file://${process.argv[1]}`) {
    const command = process.argv[2];
    
    switch (command) {
        case 'demo':
            await runDemoJobs();
            break;
        case 'stats':
            const stats = await getQueueStats();
            console.log('📊 Queue Statistics:');
            console.log(`   Waiting: ${stats.waiting}`);
            console.log(`   Active: ${stats.active}`);
            console.log(`   Completed: ${stats.completed}`);
            console.log(`   Failed: ${stats.failed}`);
            break;
        case 'clear':
            await clearQueue();
            break;
        case 'navigate':
            const url = process.argv[3];
            if (url) {
                await addNavigateJob(url);
            } else {
                console.log('Usage: node braf-jobs.js navigate <url>');
            }
            break;
        default:
            console.log('BRAF Job Manager');
            console.log('Commands:');
            console.log('  demo     - Run demo jobs');
            console.log('  stats    - Show queue statistics');
            console.log('  clear    - Clear completed/failed jobs');
            console.log('  navigate <url> - Add navigation job');
    }
    
    process.exit(0);
}