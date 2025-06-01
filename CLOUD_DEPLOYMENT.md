# ☁️ Cloud Deployment Guide for ML Service

## 🚀 Quick Deploy Options

### Option 1: Railway (Recommended - Easy)
1. **Visit**: https://railway.app
2. **Sign up** with GitHub
3. **New Project** → **Deploy from GitHub repo**
4. **Select** your F1 predictions repository
5. **Environment Variables**: 
   ```
   PORT=8001
   PYTHONPATH=/app
   ```
6. **Auto-deploy** from main branch
7. **Get URL**: `https://your-app.railway.app`

### Option 2: Render (Free Tier Available)
1. **Visit**: https://render.com
2. **New Web Service** → **Connect GitHub**
3. **Select repository**
4. **Configuration**:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python ml_production_service.py`
   - **Port**: 8001
5. **Deploy**
6. **Get URL**: `https://your-app.onrender.com`

### Option 3: Google Cloud Run (Serverless)
```bash
# Build and deploy to Google Cloud Run
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/f1-ml-service
gcloud run deploy f1-ml-service \
  --image gcr.io/YOUR_PROJECT_ID/f1-ml-service \
  --platform managed \
  --port 8001 \
  --allow-unauthenticated
```

## 🔧 After Deployment

### Update Worker Configuration
Once deployed, update the ML service URL in your Worker:

```bash
# Replace YOUR_ML_SERVICE_URL with actual deployed URL
cd packages/worker
echo "https://your-app.railway.app" | npx wrangler secret put ML_SERVICE_URL
```

### Test the Integration
```bash
# Test ML service health
curl https://your-app.railway.app/

# Test predictions endpoint
curl -X POST https://your-app.railway.app/predict/race/9

# Test Worker integration
curl https://f1-predictions-api.vprifntqe.workers.dev/api/ultra/status
```

## 📊 Service Comparison

| Service | Free Tier | Pros | Cons |
|---------|-----------|------|------|
| **Railway** | $5/month | Easy setup, great for ML | Paid only |
| **Render** | 512MB RAM free | Free tier, simple | Limited free resources |
| **Google Cloud Run** | Pay per request | Serverless, scalable | More complex setup |
| **AWS Lambda** | 1M requests free | Serverless | Cold starts, size limits |

## 🛠️ Development Setup

### Local Testing with Docker
```bash
# Build image
docker build -t f1-ml-service .

# Run locally
docker run -p 8001:8001 f1-ml-service

# Test
curl http://localhost:8001/
```

### Environment Variables for Production
```bash
# Required
PORT=8001
PYTHONPATH=/app

# Optional (for enhanced features)
WEATHER_API_KEY=your-weather-api-key
DATABASE_URL=sqlite:///f1_predictions_test.db
```

## 🔄 CI/CD Integration

### GitHub Actions Auto-Deploy (Railway)
Add this to `.github/workflows/deploy-ml.yml`:

```yaml
name: Deploy ML Service
on:
  push:
    branches: [main]
    paths: ['ml_production_service.py', 'requirements.txt']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Railway
        uses: railway-app/cli@v2
        with:
          railway-token: ${{ secrets.RAILWAY_TOKEN }}
        run: railway up
```

## 🚨 Important Notes

1. **Database**: Currently uses SQLite (included in container)
2. **Scaling**: For high traffic, consider PostgreSQL + Redis
3. **Monitoring**: Add health checks and logging
4. **Security**: API key authentication recommended for production
5. **Performance**: Cold starts may occur with serverless options

## 💡 Recommended Quick Start

**Railway** is the easiest option:
1. Connect GitHub repo to Railway
2. Set PORT=8001 environment variable  
3. Deploy automatically
4. Update Worker with new ML_SERVICE_URL
5. Test Ultra Predictor endpoints

The system will automatically fall back to 86% accuracy if the ML service is unavailable.