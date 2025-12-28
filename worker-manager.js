/**
 * BRAF Worker Manager - Control and monitor the worker system
 */

import BRAFWorker from './runWorker.js';
import { Queue } from 'bullmq';
import Redis from 'ioredis';

class WorkerManager {
    constructor() {
        this.worker = new BRAFWorker();
        this.redis = new Redis({
            host: process.env.REDIS_HOST || '127.0.0.1',
            port: parseInt(process.env.REDIS_PORT) || 6379
        });
        this.queue = new Queue('braf-tasks', { connection: this.redis });
    }

    async start() {
        console.log('🎯 Starting BRAF Worker Manager...');
        await this.worker.start();
        
        // Add initial earning tasks
        await this.loadEarningTasks();
        
        // Start monitoring
        this.startMonitoring();
    }

    async loadEarningTasks() {
        const tasks = [
            // Video watching tasks
            {
                type: 'video',
                data: {
                    url: 'https://swagbucks.com/watch',
                    duration: 30000,
                    profileId: 'video_profile_1'
                },
                options: { repeat: { every: 300000 } } // Every 5 minutes
            },
            {
                type: 'video', 
                data: {
                    url: 'https://inboxdollars.com/watch',
                    duration: 45000,
                    profileId: 'video_profile_2'
                },
                options: { repeat: { every: 420000 } } // Every 7 minutes
            },
            
            // Survey tasks
            {
                type: 'survey',
                data: {
                    platform: 'swagbucks',
                    surveyId: 'auto_detect',
                    profileId: 'survey_profile_1'
                },
                options: { repeat: { every: 900000 } } // Every 15 minutes
            },
            {
                type: 'survey',
                data: {
                    platform: 'survey_junkie', 
                    surveyId: 'auto_detect',
                    profileId: 'survey_profile_2'
                },
                options: { repeat: { every: 1200000 } } // Every 20 minutes
            },
            
            // Navigation tasks
            {
                type: 'navigate',
                data: {
                    url: 'https://ysense.com/video',
                    waitTime: 60000,
                    profileId: 'nav_profile_1'
                },
                options: { repeat: { every: 600000 } } // Every 10 minutes
            },
            {
                type: 'navigate',
                data: {
                    url: 'https://timebucks.com/tasks',
                    waitTime: 45000,
                    profileId: 'nav_profile_2'
                },
                options: { repeat: { every: 480000 } } // Every 8 minutes
            },
            
            // Interaction tasks
            {
                type: 'interaction',
                data: {
                    url: 'https://prizerebel.com/watch',
                    actions: [
                        { type: 'scroll', pixels: 200 },
                        { type: 'wait', duration: 5000 },
                        { type: 'scroll', pixels: 300 },
                        { type: 'wait', duration: 3000 }
                    ],
                    profileId: 'interaction_profile_1'
                },
                options: { repeat: { every: 720000 } } // Every 12 minutes
            }
        ];

        console.log('📋 Loading earning tasks...');
        
        for (const task of tasks) {
            try {
                await this.worker.addJob(task.type, task.data, task.options);
                console.log(`✅ Added ${task.type} task for ${task.data.url || task.data.platform}`);
            } catch (error) {
                console.error(`❌ Failed to add ${task.type} task:`, error.message);
            }
        }
        
        console.log(`🎯 ${tasks.length} earning tasks loaded successfully`);
    }

    startMonitoring() {
        // Monitor every 30 seconds
        setInterval(async () => {
            await this.displayStats();
        }, 30000);
        
        // Initial stats display
        setTimeout(() => this.displayStats(), 5000);
    }

    async displayStats() {
        const stats = this.worker.getStats();
        const queueStats = await this.getQueueStats();
        
        console.clear();
        console.log('🤖 BRAF WORKER DASHBOARD');
        console.log('========================');
        console.log(`💰 Total Earnings: $${stats.earnings.total.toFixed(4)}`);
        console.log(`📊 Sessions Completed: ${stats.earnings.sessions}`);
        console.log(`🕒 Last Update: ${stats.earnings.lastUpdate.toLocaleTimeString()}`);
        console.log(`⏱️  Uptime: ${this.formatUptime(stats.uptime)}`);
        console.log(`🌐 Active Browsers: ${stats.browsers}`);
        console.log(`👤 Active Profiles: ${stats.profiles}`);
        console.log(`📋 Queue Status:`);
        console.log(`   - Waiting: ${queueStats.waiting}`);
        console.log(`   - Active: ${queueStats.active}`);
        console.log(`   - Completed: ${queueStats.completed}`);
        console.log(`   - Failed: ${queueStats.failed}`);
        console.log('========================');
        
        // Calculate hourly rate
        if (stats.uptime > 0) {
            const hours = stats.uptime / (1000 * 60 * 60);
            const hourlyRate = stats.earnings.total / hours;
            console.log(`📈 Current Rate: $${hourlyRate.toFixed(4)}/hour`);
        }
    }

    async getQueueStats() {
        try {
            const waiting = await this.queue.getWaiting();
            const active = await this.queue.getActive();
            const completed = await this.queue.getCompleted();
            const failed = await this.queue.getFailed();
            
            return {
                waiting: waiting.length,
                active: active.length,
                completed: completed.length,
                failed: failed.length
            };
        } catch (error) {
            return { waiting: 0, active: 0, completed: 0, failed: 0 };
        }
    }

    formatUptime(ms) {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes % 60}m`;
        } else if (minutes > 0) {
            return `${minutes}m ${seconds % 60}s`;
        } else {
            return `${seconds}s`;
        }
    }

    async addCustomJob(type, data) {
        try {
            const job = await this.worker.addJob(type, data);
            console.log(`✅ Added custom ${type} job: ${job.id}`);
            return job;
        } catch (error) {
            console.error(`❌ Failed to add custom job:`, error.message);
            throw error;
        }
    }

    async pauseWorker() {
        await this.queue.pause();
        console.log('⏸️  Worker paused');
    }

    async resumeWorker() {
        await this.queue.resume();
        console.log('▶️  Worker resumed');
    }

    async clearQueue() {
        await this.queue.obliterate();
        console.log('🗑️  Queue cleared');
    }

    async shutdown() {
        console.log('🛑 Shutting down Worker Manager...');
        await this.worker.shutdown();
        await this.redis.quit();
        process.exit(0);
    }
}

// CLI Interface
if (import.meta.url === `file://${process.argv[1]}`) {
    const manager = new WorkerManager();
    
    // Handle command line arguments
    const command = process.argv[2];
    
    switch (command) {
        case 'start':
            await manager.start();
            break;
            
        case 'pause':
            await manager.pauseWorker();
            break;
            
        case 'resume':
            await manager.resumeWorker();
            break;
            
        case 'clear':
            await manager.clearQueue();
            break;
            
        case 'stats':
            await manager.displayStats();
            break;
            
        case 'add-video':
            const url = process.argv[3] || 'https://swagbucks.com/watch';
            await manager.addCustomJob('video', { url, duration: 30000 });
            break;
            
        case 'add-survey':
            const platform = process.argv[3] || 'swagbucks';
            await manager.addCustomJob('survey', { platform, surveyId: 'auto' });
            break;
            
        default:
            console.log('🤖 BRAF Worker Manager');
            console.log('Usage: node worker-manager.js [command]');
            console.log('');
            console.log('Commands:');
            console.log('  start          Start the worker with earning tasks');
            console.log('  pause          Pause job processing');
            console.log('  resume         Resume job processing');
            console.log('  clear          Clear all jobs from queue');
            console.log('  stats          Show current statistics');
            console.log('  add-video [url]    Add a video watching job');
            console.log('  add-survey [platform]  Add a survey job');
            console.log('');
            console.log('Examples:');
            console.log('  node worker-manager.js start');
            console.log('  node worker-manager.js add-video https://inboxdollars.com/watch');
            console.log('  node worker-manager.js add-survey survey_junkie');
            break;
    }
    
    // Handle graceful shutdown
    process.on('SIGINT', () => manager.shutdown());
    process.on('SIGTERM', () => manager.shutdown());
}

export default WorkerManager;