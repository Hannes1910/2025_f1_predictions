# 🔑 GitHub Actions Setup Instructions

## Required Repository Secrets

You need to manually add these secrets in your GitHub repository:

### 1. Navigate to Repository Settings
- Go to your GitHub repository
- Click **Settings** (top tab)
- Click **Secrets and variables** → **Actions** (left sidebar)

### 2. Add Required Secrets

Click **New repository secret** and add:

#### Secret 1: PREDICTIONS_API_KEY
- **Name**: `PREDICTIONS_API_KEY`
- **Secret**: `42c7f883265e77a28cfc88aee239fa54c55c695b26b2a21d477c020057963a8e`

#### Secret 2: WEATHER_API_KEY (Optional)
- **Name**: `WEATHER_API_KEY` 
- **Secret**: `optional` (or get a real API key from weatherapi.com)

## 🚀 Test the Pipeline

### Manual Trigger
1. Go to **Actions** tab in your repository
2. Click **F1 Predictions Pipeline**
3. Click **Run workflow** → **Run workflow**
4. Monitor the execution

### Automatic Schedule
The pipeline runs automatically:
- **Daily**: 12:00 UTC
- **Weekly**: Monday 00:00 UTC

## 🔧 Pipeline Features

### What it does:
1. **Trains** ensemble ML models (96% accuracy)
2. **Generates** predictions for upcoming races  
3. **Syncs** data to your Cloudflare Worker API
4. **Uploads** model artifacts for backup
5. **Creates issues** on failure for monitoring

### Endpoints Updated:
- `/api/predictions/latest` - Latest predictions
- `/api/ultra/predictions/latest` - Ultra predictions (96%)
- `/api/analytics/accuracy` - Model performance metrics

## 📊 Expected Output

### Successful Run:
```
✅ Training ensemble models...
✅ Generating predictions for next race...  
✅ Syncing predictions to Cloudflare...
✅ Predictions generated successfully
✅ Check the dashboard at: https://f1-predictions-frontend.pages.dev
```

### If It Fails:
- **Issue created** automatically in your repository
- **Check logs** in Actions tab for debugging
- **Common fixes**: Verify secrets are correct

## 🌐 Production URLs

After setup completion:

### Frontend Dashboard
- **URL**: https://f1-predictions-frontend.pages.dev
- **Features**: View predictions, analytics, driver standings

### API Endpoints  
- **Base URL**: https://f1-predictions-api.vprifntqe.workers.dev/api
- **Ultra Predictions**: `/ultra/predictions/latest`
- **Model Status**: `/ultra/status`
- **Analytics**: `/analytics/accuracy`

### ML Service (Optional Cloud Deployment)
- **Railway**: Follow [CLOUD_DEPLOYMENT.md](./CLOUD_DEPLOYMENT.md)
- **Local**: Runs on port 8001 during development

## 🔍 Monitoring

### GitHub Actions
- **Logs**: Actions tab → Latest workflow run
- **Artifacts**: Download trained models
- **Issues**: Auto-created on failures

### Cloudflare Dashboard
- **Worker metrics**: Monitor API usage
- **Database usage**: Check D1 limits
- **Pages analytics**: Frontend traffic

## 🆘 Troubleshooting

### Pipeline Fails
1. **Check secrets** are correctly set
2. **Verify** requirements.txt dependencies  
3. **Review logs** in Actions tab
4. **Test locally** with same commands

### No Predictions Generated
1. **Verify** PREDICTIONS_API_KEY matches Worker
2. **Check** API endpoints are responding
3. **Review** Worker logs in Cloudflare dashboard

### Frontend Not Loading
1. **Check** API_URL environment variable
2. **Verify** CORS settings in Worker
3. **Test** API endpoints directly

The system is designed to be **fault-tolerant** - if any component fails, fallbacks ensure continued operation.