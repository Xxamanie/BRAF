# Deploying BRAF to Railway for Testing

This guide helps you deploy BRAF to Railway's free tier for testing the limitless withdrawal machine.

## Prerequisites
- GitHub account
- Railway account (free)
- Project pushed to GitHub

## Step 1: Push Code to GitHub
```bash
git add .
git commit -m "Prepare for Railway deployment"
git push origin main
```

## Step 2: Create Railway Project
1. Go to [railway.app](https://railway.app)
2. Click "New Project"
3. Choose "Deploy from GitHub repo"
4. Select your BRAF repository
5. Railway will auto-detect docker-compose.yml and deploy

## Step 3: Add Databases
1. In Railway dashboard, go to your project
2. Click "Add Plugin"
3. Add PostgreSQL (free tier: 512MB)
4. Add Redis (free tier: 512MB)
5. Railway will auto-set DATABASE_URL and REDIS_URL

## Step 4: Set Environment Variables
1. In project settings, go to "Variables"
2. Add the variables from `railway-env-vars.txt`:
   - SECRET_KEY (generate random 256-bit key)
   - ENCRYPTION_KEY (generate random 256-bit key)
   - JWT_SECRET_KEY (generate random key)
   - POSTGRES_PASSWORD (set secure password)
   - MAXEL_API_KEY and MAXEL_SECRET_KEY (already in .env.production)

## Step 5: Deploy
1. Railway will build and deploy automatically
2. Check logs for any errors
3. Once deployed, get the domain from Railway dashboard

## Step 6: Test Withdrawal
- Access the deployed app at the Railway domain
- Run withdrawal tests via API endpoints
- Monitor logs in Railway dashboard

## Troubleshooting
- If build fails, check Dockerfile compatibility
- For memory issues, Railway free tier has 512MB RAM limit
- Sleep mode: App sleeps after inactivity, wakes on requests

## Notes
- Free tier sleeps after inactivity
- No real deposits expected in free tier
- For production, upgrade to paid plans