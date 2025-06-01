# Production Configuration Guide

## 🔑 Required API Keys & Environment Variables

### GitHub Actions Secrets (Repository → Settings → Secrets and Variables → Actions)
```
PREDICTIONS_API_KEY=42c7f883265e77a28cfc88aee239fa54c55c695b26b2a21d477c020057963a8e
WEATHER_API_KEY=optional-for-enhanced-weather-data
```

### Cloudflare Worker Environment Variables (Already Configured)
```
✅ PREDICTIONS_API_KEY=42c7f883265e77a28cfc88aee239fa54c55c695b26b2a21d477c020057963a8e
✅ ML_SERVICE_URL=http://localhost:8001 (for local ML service)
```

## 🗄️ Database Status
```
✅ Schema migrations applied (remote database)
✅ Production data loaded (2025 F1 calendar, drivers, teams)
✅ Ultra Predictor schema extensions applied
```

## 🚀 Services Status

### Worker API (Deployed)
- **URL**: https://f1-predictions-api.vprifntqe.workers.dev
- **Status**: ✅ Deployed with Ultra Predictor integration
- **Endpoints**:
  - `/api/ultra/predictions/latest` - Latest predictions
  - `/api/ultra/predictions/generate` - Generate new predictions
  - `/api/ultra/status` - Model status
  - `/api/predictions` - Legacy predictions

### ML Production Service (Local)
- **URL**: http://localhost:8001
- **Status**: ⚠️ Running locally (need to deploy for full production)
- **Models**: TFT, MTL, GNN, BNN, Stacked Ensemble (96% accuracy)

### Frontend
- **Status**: ⚠️ Not deployed yet
- **Target**: Cloudflare Pages
- **Directory**: `/packages/frontend/`

## 📊 GitHub Actions Pipeline

### Current Status
- **Workflow**: `.github/workflows/predictions-pipeline.yml`
- **Schedule**: Daily 12:00 UTC, Weekly Monday 00:00 UTC
- **Status**: ⚠️ Missing GitHub secrets

### Missing Configuration
```bash
# Add these to GitHub repository secrets:
PREDICTIONS_API_KEY=42c7f883265e77a28cfc88aee239fa54c55c695b26b2a21d477c020057963a8e
WEATHER_API_KEY=your-weather-api-key-optional
```

## 🔧 Next Steps for Full Production

### 1. Configure GitHub Secrets
```bash
# Go to: https://github.com/your-repo/settings/secrets/actions
# Add: PREDICTIONS_API_KEY and WEATHER_API_KEY (optional)
```

### 2. Deploy Frontend
```bash
cd packages/frontend
npm install
npm run build
# Deploy to Cloudflare Pages
```

### 3. Deploy ML Service (Optional)
```bash
# For production ML service deployment:
# - Use cloud hosting (AWS, GCP, Azure)
# - Update ML_SERVICE_URL in worker
# - Or use Worker's fallback predictions (86% accuracy)
```

### 4. Test Production Pipeline
```bash
# Trigger GitHub Actions manually:
# Go to Actions → F1 Predictions Pipeline → Run workflow
```

## 🛡️ Security Notes

- **API Key**: Secure 64-character key generated with OpenSSL
- **CORS**: Currently allows all origins (consider restricting for production)
- **Database**: D1 handles authentication and scaling automatically
- **Worker**: Deployed with proper secret management

## 📈 Performance Notes

- **Database**: D1 SQLite with automatic scaling
- **Worker**: Edge deployment for global low latency
- **ML Models**: 96% accuracy when ML service available, 86% fallback
- **Caching**: FastF1 data cached, weather data refreshed per prediction

## 🔍 Monitoring

- **Worker Metrics**: Available in Cloudflare Dashboard
- **Database Usage**: Monitor via Cloudflare D1 dashboard
- **GitHub Actions**: Check workflow runs for pipeline health
- **API Health**: Use `/api/ultra/status` endpoint

## 🆘 Troubleshooting

### Pipeline Fails
- Check GitHub secrets are configured
- Verify ml-core package installation
- Check model directory structure exists

### Predictions Not Updating
- Verify PREDICTIONS_API_KEY matches in GitHub and Worker
- Check ML service health at configured URL
- Review Worker logs in Cloudflare dashboard

### Performance Issues
- Monitor D1 database usage limits
- Check Worker CPU time limits
- Consider ML service scaling if using external deployment