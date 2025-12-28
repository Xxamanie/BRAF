#!/usr/bin/env node
/**
 * BRAF System Startup - Complete automation system launcher
 */

import { spawn } from 'child_process';
import dotenv from 'dotenv';
import { Queue } from 'bullmq';

dotenv.config();

console.log('🚀 BRAF System Startup');
console.log('=' .repeat(60));
console.log(`📅 ${new Date().toISOString()}`);

// Configuration
const REDIS_HOST = process.env.REDIS_HOST || '127.0.0.1';
const REDIS_PORT = process.env.REDIS_PORT || 6379;
const MAX_CONCURRENT = process.env.MAX_CONCURRENT || 3;

// Check Redis connection
async function checkRedis() {
    console.log('🔗 Checking Redis connection...');
    
    try {
        const queue = new Queue('test', {
            connection: { host: REDIS_HOST, port: REDIS_PORT }
        });
        
        await queue.add('test', { test: true });
        await queue.close();
        
        console.log('✅ Redis connection successful');
        return true;
    } catch (error) {
        console.log('❌ Redis connection failed:', error.message);
        console.log('💡 Make sure Redis is running:');
        console.log('   Windows: Download from https://redis.io/download');
        console.log('   Docker: docker run -d -p 6379:6379 redis:alpine');
        return false;
    }
}

// Start BRAF worker
function startWorker() {
    console.log('\n🤖 Starting BRAF Worker...');
    
    const worker = spawn('node', ['braf-worker.js'], {
        stdio: 'inherit',
        env: { ...process.env }
    });
    
    worker.on('error', (error) => {
        console.error('❌ Worker error:', error);
    });
    
    worker.on('exit', (code) => {
        console.log(`🛑 Worker exited with code ${code}`);
    });
    
    return worker;
}

// Submit earning tasks
async function submitEarningTasks() {
    console.log('\n💰 Submitting earning tasks...');
    
    try {
        const { submitEarningJobs } = await import('./earning-tasks.js');
        const result = await submitEarningJobs();
        
        console.log('✅ Earning tasks submitted successfully');
        return result;
    } catch (error) {
        console.error('❌ Failed to submit earning tasks:', error.message);
        return null;
    }
}

// Monitor system status
async function monitorSystem() {
    const queue = new Queue('braf-tasks', {
        connection: { host: REDIS_HOST, port: REDIS_PORT }
    });
    
    setInterval(async () => {
        try {
            const waiting = await queue.getWaiting();
            const active = await queue.getActive();
            const completed = await queue.getCompleted();
            const failed = await queue.getFailed();
            
            console.log(`\n📊 System Status [${new Date().toLocaleTimeString()}]:`);
            console.log(`   Queue: ${waiting.length} waiting, ${active.length} active`);
            console.log(`   Completed: ${completed.length}, Failed: ${failed.length}`);
            
            // Calculate estimated earnings
            let totalEarnings = 0;
            for (const job of completed) {
                if (job.data.metadata?.estimated_earning) {
                    totalEarnings += job.data.metadata.estimated_earning;
                }
            }
            
            if (totalEarnings > 0) {
                console.log(`   💰 Estimated Earnings: $${totalEarnings.toFixed(2)}`);
            }
            
        } catch (error) {
            console.error('❌ Monitor error:', error.message);
        }
    }, 30000); // Update every 30 seconds
}

// Main startup sequence
async function main() {
    // 1. Check Redis
    const redisOk = await checkRedis();
    if (!redisOk) {
        console.log('\n🛑 Cannot start without Redis. Please install and start Redis first.');
        process.exit(1);
    }
    
    // 2. Start worker
    const worker = startWorker();
    
    // 3. Wait a moment for worker to initialize
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // 4. Submit initial earning tasks
    await submitEarningTasks();
    
    // 5. Start monitoring
    await monitorSystem();
    
    // 6. Setup graceful shutdown
    process.on('SIGINT', () => {
        console.log('\n🛑 Shutting down BRAF system...');
        worker.kill('SIGTERM');
        process.exit(0);
    });
    
    console.log('\n🎯 BRAF System Running');
    console.log('📋 Commands:');
    console.log('   Ctrl+C - Stop system');
    console.log('   Check logs above for real-time status');
    console.log('\n💡 Tips:');
    console.log('   - Monitor earnings in real-time above');
    console.log('   - System auto-submits new tasks periodically');
    console.log('   - Check individual platform dashboards for payouts');
}

// Handle command line arguments
const command = process.argv[2];

switch (command) {
    case 'worker-only':
        console.log('🤖 Starting worker only...');
        if (await checkRedis()) {
            startWorker();
        }
        break;
    case 'tasks-only':
        console.log('💰 Submitting tasks only...');
        if (await checkRedis()) {
            await submitEarningTasks();
            process.exit(0);
        }
        break;
    case 'status':
        console.log('📊 Checking system status...');
        if (await checkRedis()) {
            const { getEarningStats } = await import('./earning-tasks.js');
            const stats = await getEarningStats();
            console.log('System Statistics:', stats);
            process.exit(0);
        }
        break;
    default:
        // Full system startup
        main().catch(console.error);
}