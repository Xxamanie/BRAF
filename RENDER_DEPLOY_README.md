# Deploying BRAF to Render for Testing

This guide helps you deploy BRAF to Render's free tier for testing the limitless withdrawal machine.

## Prerequisites
- GitHub account
- Render account (free)
- Project pushed to GitHub

## Step 1: Push Code to GitHub
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

## Step 2: Create Render Services
1. Go to [render.com](https://render.com)
2. Click "New" > "Web Service"
3. Choose "Deploy from GitHub repo"
4. Select your BRAF repository
5. Configure:
   - Runtime: Docker
   - Branch: main
   - Build Command: (leave default)
   - Start Command: (leave default)

## Step 3: Set Environment Variables
In the web service settings, add these variables:
- ENVIRONMENT=production
- DEBUG=false
- LOG_LEVEL=INFO
- SECRET_KEY=your-256-bit-production-secret-key-here
- ENCRYPTION_KEY=your-256-bit-encryption-key-here
- JWT_SECRET_KEY=your-jwt-secret-key-here
- MAXEL_API_KEY=pk_Eq8N27HLVFDrPFd34j7a7cpIJd6PncsW
- MAXEL_SECRET_KEY=sk_rI7pJyhIyaiU5js1BCpjYA53y5iS7Ny0
- MAXEL_BASE_URL=https://api.maxel.io/v1
- MAXEL_SANDBOX=false
- DATABASE_URL=postgresql://user:pass@host:5432/db (use free external Postgres if needed)
- REDIS_URL=redis://host:port (use Upstash free Redis)
- PORT=10000 (Render uses 10000 by default)

## Step 4: Add Free External Databases
- **Redis**: Use Upstash (upstash.com) - free tier available
- **Postgres**: Use Neon.tech or Supabase free tier (may require credit card, but free limits)

## Step 5: Deploy
1. Click "Create Web Service"
2. Render will build and deploy
3. Check logs for errors

## Step 6: Test Withdrawal
- Access at the Render domain
- Test health: `curl https://your-app.onrender.com/health`
- Run withdrawal tests via API

## Notes
- Free tier: 512MB RAM, 750 hours/month
- Databases: Use free external services to avoid costs
- If blocked, Render may flag automation features