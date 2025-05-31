# F1 Predictions 2025 - New Architecture

## 🚀 Overview

This is the modernized F1 Predictions system built with:
- **Cloudflare Workers** for serverless API
- **Cloudflare D1** for edge database
- **React** for interactive dashboard
- **Python ML Pipeline** for model training

## 📁 Project Structure

```
f1-predictions/
├── packages/
│   ├── worker/         # Cloudflare Worker API
│   ├── frontend/       # React dashboard
│   ├── ml-core/        # Python ML library
│   └── shared/         # Shared types/utils
├── scripts/
│   ├── migrate_data.py # Data migration
│   ├── train_models.py # Model training
│   └── schema.sql      # D1 database schema
├── predictions/        # Stored predictions
├── models/            # Trained models
└── wrangler.toml      # Cloudflare config
```

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt
cd packages/ml-core && pip install -e .
```

### 2. Set Up Cloudflare

```bash
# Login to Cloudflare
wrangler login

# Create D1 database
wrangler d1 create f1-predictions

# Update wrangler.toml with your database_id
# Apply schema
wrangler d1 execute f1-predictions --file=scripts/schema.sql
```

### 3. Migrate Data

```bash
# Run migration script
python scripts/migrate_data.py

# Import to D1 (for each table)
wrangler d1 execute f1-predictions --file=data_export/drivers.sql
```

### 4. Train Models

```bash
# Train models for upcoming races
python scripts/train_models.py --weather-key YOUR_API_KEY
```

### 5. Deploy

```bash
# Deploy Worker API
npm run deploy --workspace=packages/worker

# Deploy Frontend (if using Cloudflare Pages)
npm run deploy --workspace=packages/frontend
```

## 📊 API Endpoints

- `GET /api/predictions/latest` - Latest race predictions
- `GET /api/predictions/:raceId` - Predictions for specific race
- `GET /api/results/:raceId` - Actual race results
- `GET /api/drivers` - All drivers with statistics
- `GET /api/races` - Season calendar
- `GET /api/analytics/accuracy` - Model performance metrics

## 🔧 Development

```bash
# Start Worker dev server
npm run dev --workspace=packages/worker

# Start Frontend dev server
npm run dev --workspace=packages/frontend

# Run both
npm run dev
```

## 🧪 Testing

```bash
# Test all packages
npm test

# Test specific package
npm test --workspace=packages/worker
```

## 📈 Model Training Pipeline

The ML pipeline automatically:
1. Fetches historical race data
2. Engineers features (weather, team performance, etc.)
3. Trains Gradient Boosting models
4. Generates predictions with confidence scores
5. Stores results in D1 database

Run manually:
```bash
python scripts/train_models.py --force
```

## 🌐 Environment Variables

Create `.env` file:
```
WEATHER_API_KEY=your_openweather_api_key
ANTHROPIC_MODEL=claude-opus-4-20250514
```

## 📝 Database Schema

Key tables:
- `races` - F1 calendar
- `drivers` - Driver information
- `predictions` - Model predictions
- `race_results` - Actual results
- `model_metrics` - Performance tracking
- `feature_data` - Input features

## 🚀 Next Steps

1. **Frontend Dashboard**: Complete React dashboard implementation
2. **Real-time Updates**: Add WebSocket support for live updates
3. **Advanced Models**: Implement ensemble methods
4. **User Auth**: Add authentication for admin features
5. **Monitoring**: Set up performance monitoring

## 📊 Current Features

✅ Modular ML prediction system
✅ RESTful API with Cloudflare Workers
✅ Edge database with D1
✅ Automated training pipeline
✅ Historical data migration
✅ Model versioning
✅ Confidence scoring

## 🎯 Improvements Over Original

- **Scalability**: Serverless architecture handles any load
- **Performance**: Edge computing for <100ms responses
- **Maintainability**: Clean separation of concerns
- **Extensibility**: Easy to add new features/models
- **Cost**: Minimal hosting costs with Cloudflare free tier

## 📧 Support

For issues or questions:
- Create an issue in the repository
- Check the technical roadmap: `TECHNICAL_ROADMAP.md`