#!/usr/bin/env node
/**
 * BRAF Earning Tasks - Queue legitimate earning platform jobs
 * Automated task submission for real earning opportunities
 */

import { Queue } from 'bullmq';
import dotenv from 'dotenv';

dotenv.config();

const queue = new Queue('braf-tasks', {
    connection: { 
        host: process.env.REDIS_HOST || '127.0.0.1', 
        port: parseInt(process.env.REDIS_PORT) || 6379 
    }
});

// Legitimate earning platforms with real earning potential
const earningPlatforms = [
    // Video/Watch Platforms
    { name: "Swagbucks Watch", url: "https://swagbucks.com/watch", category: "video", earning: "$0.01-0.05" },
    { name: "InboxDollars Watch", url: "https://inboxdollars.com/watch", category: "video", earning: "$0.01-0.03" },
    { name: "ySense Watch", url: "https://ysense.com/video", category: "video", earning: "$0.005-0.02" },
    { name: "MyPoints Watch", url: "https://mypoints.com/watch", category: "video", earning: "$0.01-0.04" },
    { name: "FusionCash Videos", url: "https://fusioncash.net/videos", category: "video", earning: "$0.01-0.03" },
    { name: "PrizeRebel Watch", url: "https://prizerebel.com/watch", category: "video", earning: "$0.005-0.02" },
    { name: "iRazoo Videos", url: "https://irazoo.com/videos", category: "video", earning: "$0.01-0.02" },
    
    // Survey Platforms
    { name: "SurveyTime Surveys", url: "https://surveytime.io", category: "survey", earning: "$1.00" },
    { name: "AttaPoll Surveys", url: "https://attapoll.com", category: "survey", earning: "$0.10-2.50" },
    { name: "Surveyeah", url: "https://surveyeah.com", category: "survey", earning: "$0.50-3.00" },
    { name: "TGM Research", url: "https://tgmresearch.com", category: "survey", earning: "$1.00-5.00" },
    { name: "Quest Mindshare", url: "https://questmindshare.com", category: "survey", earning: "$0.50-4.00" },
    { name: "OpinionWorld Surveys", url: "https://opinionworld.com", category: "survey", earning: "$0.25-2.00" },
    
    // Task/Micro-work Platforms
    { name: "TimeBucks Tasks", url: "https://timebucks.com/tasks", category: "tasks", earning: "$0.01-0.50" },
    { name: "Toloka Tasks", url: "https://toloka.ai", category: "tasks", earning: "$0.02-1.00" },
    { name: "Qmee Tasks", url: "https://qmee.com", category: "tasks", earning: "$0.05-0.25" },
    { name: "Freecash Tasks", url: "https://freecash.com", category: "tasks", earning: "$0.01-2.00" },
    { name: "CashCrate Videos", url: "https://cashcrate.com", category: "tasks", earning: "$0.01-0.10" },
    { name: "Paid2YouTube Tasks", url: "https://paid2youtube.com", category: "tasks", earning: "$0.005-0.05" },
    { name: "CashInStyle Tasks", url: "https://cashinstyle.com", category: "tasks", earning: "$0.01-0.20" },
    
    // Ad/Reward Platforms
    { name: "AdWallet Ads", url: "https://adwallet.com", category: "ads", earning: "$0.01-0.10" }
];

// Enhanced job submission with earning tracking
async function submitEarningJobs() {
    console.log('💰 BRAF Earning Platform Job Scheduler');
    console.log('=' .repeat(60));
    console.log(`📅 ${new Date().toISOString()}`);
    console.log(`🎯 Submitting ${earningPlatforms.length} legitimate earning tasks`);
    
    let totalJobs = 0;
    let estimatedEarnings = 0;
    
    for (const platform of earningPlatforms) {
        try {
            // Calculate estimated earnings (conservative estimate)
            const minEarning = parseFloat(platform.earning.split('-')[0].replace('$', ''));
            estimatedEarnings += minEarning;
            
            // Submit job with enhanced metadata
            const job = await queue.add(platform.name, {
                platform: platform.name,
                url: platform.url,
                category: platform.category,
                earning_potential: platform.earning,
                staySeconds: getOptimalStayTime(platform.category),
                actions: getOptimalActions(platform.category),
                priority: getPlatformPriority(platform.category),
                retry_attempts: 3,
                timeout: 180000, // 3 minutes max per task
                metadata: {
                    submitted_at: new Date().toISOString(),
                    estimated_earning: minEarning,
                    platform_type: platform.category,
                    automation_level: 'ethical'
                }
            }, {
                // Job options
                removeOnComplete: 50,
                removeOnFail: 10,
                attempts: 3,
                backoff: {
                    type: 'exponential',
                    delay: 5000
                }
            });
            
            console.log(`✅ ${platform.name} (${platform.category}) - Job ID: ${job.id}`);
            totalJobs++;
            
        } catch (error) {
            console.error(`❌ Failed to submit ${platform.name}: ${error.message}`);
        }
    }
    
    console.log('\n📊 Job Submission Summary:');
    console.log(`   Total Jobs Queued: ${totalJobs}`);
    console.log(`   Estimated Min Earnings: $${estimatedEarnings.toFixed(2)}`);
    console.log(`   Estimated Max Earnings: $${(estimatedEarnings * 3).toFixed(2)}`);
    console.log(`   Average per Task: $${(estimatedEarnings / totalJobs).toFixed(3)}`);
    
    return {
        jobs_submitted: totalJobs,
        estimated_min_earnings: estimatedEarnings,
        estimated_max_earnings: estimatedEarnings * 3,
        platforms_by_category: groupPlatformsByCategory()
    };
}

// Get optimal stay time based on platform category
function getOptimalStayTime(category) {
    const stayTimes = {
        'video': 120,    // 2 minutes for video watching
        'survey': 300,   // 5 minutes for surveys
        'tasks': 180,    // 3 minutes for micro-tasks
        'ads': 60        // 1 minute for ad viewing
    };
    
    return stayTimes[category] || 120;
}

// Get optimal actions for each platform type
function getOptimalActions(category) {
    const actions = {
        'video': [
            { type: 'wait', duration: 5000 },
            { type: 'scroll', direction: 'down', amount: 200 },
            { type: 'wait', duration: 30000 },
            { type: 'click', selector: '.video-item, .watch-button', optional: true }
        ],
        'survey': [
            { type: 'wait', duration: 3000 },
            { type: 'click', selector: '.survey-start, .begin-survey', optional: true },
            { type: 'wait', duration: 10000 },
            { type: 'scroll', direction: 'down', amount: 300 }
        ],
        'tasks': [
            { type: 'wait', duration: 2000 },
            { type: 'scroll', direction: 'down', amount: 150 },
            { type: 'wait', duration: 5000 },
            { type: 'click', selector: '.task-item, .available-task', optional: true }
        ],
        'ads': [
            { type: 'wait', duration: 10000 },
            { type: 'scroll', direction: 'down', amount: 100 },
            { type: 'wait', duration: 20000 }
        ]
    };
    
    return actions[category] || actions['tasks'];
}

// Get platform priority (higher number = higher priority)
function getPlatformPriority(category) {
    const priorities = {
        'survey': 10,    // Highest earning potential
        'tasks': 8,      // Good earning potential
        'video': 6,      // Medium earning potential
        'ads': 4         // Lower earning potential
    };
    
    return priorities[category] || 5;
}

// Group platforms by category for reporting
function groupPlatformsByCategory() {
    const grouped = {};
    
    earningPlatforms.forEach(platform => {
        if (!grouped[platform.category]) {
            grouped[platform.category] = [];
        }
        grouped[platform.category].push(platform.name);
    });
    
    return grouped;
}

// Schedule recurring job submission
async function scheduleRecurringJobs(intervalMinutes = 60) {
    console.log(`🔄 Scheduling recurring job submission every ${intervalMinutes} minutes`);
    
    // Submit initial batch
    await submitEarningJobs();
    
    // Schedule recurring submissions
    setInterval(async () => {
        console.log('\n🔄 Submitting recurring earning jobs...');
        await submitEarningJobs();
    }, intervalMinutes * 60 * 1000);
}

// Get queue statistics
async function getEarningStats() {
    const waiting = await queue.getWaiting();
    const active = await queue.getActive();
    const completed = await queue.getCompleted();
    const failed = await queue.getFailed();
    
    // Calculate earnings from completed jobs
    let totalEarnings = 0;
    for (const job of completed) {
        if (job.data.metadata && job.data.metadata.estimated_earning) {
            totalEarnings += job.data.metadata.estimated_earning;
        }
    }
    
    return {
        queue_stats: {
            waiting: waiting.length,
            active: active.length,
            completed: completed.length,
            failed: failed.length
        },
        earnings: {
            estimated_total: totalEarnings,
            completed_tasks: completed.length,
            average_per_task: completed.length > 0 ? totalEarnings / completed.length : 0
        }
    };
}

// CLI interface
if (import.meta.url === `file://${process.argv[1]}`) {
    const command = process.argv[2];
    
    switch (command) {
        case 'submit':
            await submitEarningJobs();
            break;
        case 'schedule':
            const interval = parseInt(process.argv[3]) || 60;
            await scheduleRecurringJobs(interval);
            break;
        case 'stats':
            const stats = await getEarningStats();
            console.log('📊 BRAF Earning Statistics:');
            console.log('Queue Status:');
            console.log(`   Waiting: ${stats.queue_stats.waiting}`);
            console.log(`   Active: ${stats.queue_stats.active}`);
            console.log(`   Completed: ${stats.queue_stats.completed}`);
            console.log(`   Failed: ${stats.queue_stats.failed}`);
            console.log('Earnings:');
            console.log(`   Estimated Total: $${stats.earnings.estimated_total.toFixed(2)}`);
            console.log(`   Completed Tasks: ${stats.earnings.completed_tasks}`);
            console.log(`   Average per Task: $${stats.earnings.average_per_task.toFixed(3)}`);
            break;
        default:
            console.log('BRAF Earning Task Scheduler');
            console.log('Commands:');
            console.log('  submit           - Submit all earning platform jobs');
            console.log('  schedule [mins]  - Schedule recurring submissions (default: 60 min)');
            console.log('  stats           - Show earning statistics');
            console.log('\nExample:');
            console.log('  node earning-tasks.js submit');
            console.log('  node earning-tasks.js schedule 30');
    }
    
    process.exit(0);
}

export { submitEarningJobs, scheduleRecurringJobs, getEarningStats };