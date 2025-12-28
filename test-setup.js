/**
 * Test setup to verify all dependencies are ready
 */

import { chromium } from 'playwright';
import Redis from 'ioredis';

async function testSetup() {
    console.log('🔍 Testing BRAF Worker Setup...\n');
    
    let allGood = true;
    
    // Test 1: Playwright
    try {
        console.log('1. Testing Playwright...');
        const browser = await chromium.launch({ headless: true });
        await browser.close();
        console.log('   ✅ Playwright working');
    } catch (error) {
        console.log('   ❌ Playwright failed:', error.message);
        allGood = false;
    }
    
    // Test 2: Redis (optional for basic worker)
    try {
        console.log('2. Testing Redis connection...');
        const redis = new Redis({
            host: '127.0.0.1',
            port: 6379,
            lazyConnect: true,
            maxRetriesPerRequest: 1
        });
        
        await redis.connect();
        await redis.ping();
        await redis.quit();
        console.log('   ✅ Redis working');
    } catch (error) {
        console.log('   ⚠️  Redis not available:', error.message);
        console.log('   📝 Note: Worker can run without Redis for basic tasks');
    }
    
    // Test 3: Directory structure
    try {
        console.log('3. Testing directory structure...');
        const fs = await import('fs/promises');
        
        // Check screenshots directory
        await fs.access('screenshots');
        console.log('   ✅ Screenshots directory exists');
        
        // Check BRAF directory
        try {
            await fs.access('BRAF');
            console.log('   ✅ BRAF directory exists');
        } catch {
            console.log('   ⚠️  BRAF directory not found - creating...');
            await fs.mkdir('BRAF', { recursive: true });
            await fs.mkdir('BRAF/data', { recursive: true });
            console.log('   ✅ BRAF directories created');
        }
        
    } catch (error) {
        console.log('   ❌ Directory setup failed:', error.message);
        allGood = false;
    }
    
    // Test 4: Environment variables
    console.log('4. Checking environment...');
    const envVars = {
        'HEADLESS': process.env.HEADLESS || 'true',
        'MAX_CONCURRENT': process.env.MAX_CONCURRENT || '3',
        'REDIS_HOST': process.env.REDIS_HOST || '127.0.0.1'
    };
    
    for (const [key, value] of Object.entries(envVars)) {
        console.log(`   ${key}=${value}`);
    }
    console.log('   ✅ Environment configured');
    
    console.log('\n' + '='.repeat(50));
    
    if (allGood) {
        console.log('🎉 Setup complete! Ready to run worker.');
        console.log('\nQuick start options:');
        console.log('• Basic worker:     node runWorker.js');
        console.log('• With manager:     npm run manager:start');
        console.log('• Simple test:      npm run run-worker');
    } else {
        console.log('⚠️  Some issues found. Check above for details.');
        console.log('The worker may still run with basic functionality.');
    }
}

testSetup().catch(console.error);