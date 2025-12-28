// Quick Redis connection test
import Redis from 'ioredis';

async function testRedis() {
    console.log('Testing Redis/Memurai connection...');
    
    try {
        const redis = new Redis({
            host: '127.0.0.1',
            port: 6379,
            lazyConnect: true
        });
        
        await redis.connect();
        console.log('✅ Connected to Redis/Memurai');
        
        const result = await redis.ping();
        console.log('✅ Ping result:', result);
        
        await redis.set('test', 'BRAF-Worker-Test');
        const value = await redis.get('test');
        console.log('✅ Set/Get test:', value);
        
        await redis.quit();
        console.log('✅ Redis connection test successful!');
        
        console.log('\n🚀 Redis is ready! You can now run:');
        console.log('npm run manager:start');
        
    } catch (error) {
        console.error('❌ Redis connection failed:', error.message);
    }
}

testRedis();