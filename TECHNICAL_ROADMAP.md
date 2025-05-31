# F1 Predictions 2025 - Technical Roadmap

## Overview
This document outlines the technical implementation plan for modernizing the F1 Predictions system into a scalable web application using Cloudflare Workers and D1 database.

## Architecture Overview

### Tech Stack
- **Frontend**: React with TypeScript, hosted on Cloudflare Pages
- **Backend**: Cloudflare Workers (TypeScript)
- **Database**: Cloudflare D1 (SQLite)
- **ML Pipeline**: Python services for model training, results stored in D1
- **APIs**: FastF1, OpenWeatherMap
- **CDN**: Cloudflare CDN for static assets
- **Analytics**: Cloudflare Analytics

### System Architecture
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  React Frontend │────▶│ Cloudflare Worker│────▶│  Cloudflare D1  │
│ (CF Pages)      │     │    (API)         │     │   Database      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                           │
                               ▼                           ▼
                        ┌──────────────┐          ┌──────────────┐
                        │ Python ML    │          │ External APIs │
                        │ Service      │          │ (FastF1, etc) │
                        └──────────────┘          └──────────────┘
```

## Phase 1: Foundation (Week 1-2)

### 1.1 Project Structure
```
f1-predictions/
├── packages/
│   ├── worker/           # Cloudflare Worker API
│   ├── frontend/         # React dashboard
│   ├── ml-core/          # Python ML library
│   └── shared/           # Shared types/utils
├── scripts/
│   ├── migrate-data.py   # Data migration scripts
│   └── train-models.py   # Model training pipeline
├── wrangler.toml         # Cloudflare config
├── package.json          # Monorepo config
└── requirements.txt      # Python dependencies
```

### 1.2 Database Schema (D1)
```sql
-- Races table
CREATE TABLE races (
    id INTEGER PRIMARY KEY,
    season INTEGER NOT NULL,
    round INTEGER NOT NULL,
    name TEXT NOT NULL,
    date TEXT NOT NULL,
    circuit TEXT NOT NULL,
    UNIQUE(season, round)
);

-- Drivers table
CREATE TABLE drivers (
    id INTEGER PRIMARY KEY,
    code TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    team TEXT
);

-- Predictions table
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    race_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    predicted_position INTEGER,
    predicted_time REAL,
    confidence REAL,
    model_version TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES races(id),
    FOREIGN KEY (driver_id) REFERENCES drivers(id)
);

-- Race results table
CREATE TABLE race_results (
    id INTEGER PRIMARY KEY,
    race_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    position INTEGER,
    time REAL,
    points INTEGER,
    FOREIGN KEY (race_id) REFERENCES races(id),
    FOREIGN KEY (driver_id) REFERENCES drivers(id)
);

-- Model metrics table
CREATE TABLE model_metrics (
    id INTEGER PRIMARY KEY,
    model_version TEXT NOT NULL,
    race_id INTEGER,
    mae REAL,
    accuracy REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES races(id)
);

-- Feature data table
CREATE TABLE feature_data (
    id INTEGER PRIMARY KEY,
    race_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    qualifying_time REAL,
    sector1_time REAL,
    sector2_time REAL,
    sector3_time REAL,
    weather_temp REAL,
    rain_probability REAL,
    team_points INTEGER,
    FOREIGN KEY (race_id) REFERENCES races(id),
    FOREIGN KEY (driver_id) REFERENCES drivers(id)
);
```

### 1.3 Core ML Module Structure
```python
# ml-core/f1_predictor.py
class F1Predictor:
    def __init__(self, config: PredictorConfig):
        self.model = None
        self.features = config.features
        self.model_type = config.model_type
    
    def train(self, training_data: pd.DataFrame) -> ModelMetrics:
        """Train model with cross-validation"""
        pass
    
    def predict(self, race_data: RaceData) -> List[Prediction]:
        """Generate predictions for a race"""
        pass
    
    def evaluate(self, predictions: List[Prediction], 
                 actual_results: List[Result]) -> ModelMetrics:
        """Evaluate model performance"""
        pass
```

## Phase 2: API Development (Week 3-4)

### 2.1 Cloudflare Worker Endpoints
```typescript
// GET /api/predictions/:raceId
// Get predictions for a specific race

// GET /api/predictions/latest
// Get latest race predictions

// POST /api/predictions
// Generate new predictions (admin only)

// GET /api/results/:raceId
// Get actual race results

// GET /api/analytics/accuracy
// Get model accuracy over time

// GET /api/drivers
// Get all drivers with stats

// GET /api/races
// Get all races for the season
```

### 2.2 Worker Implementation
```typescript
// worker/src/index.ts
import { Router } from 'itty-router';
import { Env } from './types';

const router = Router();

router.get('/api/predictions/:raceId', async (request, env: Env) => {
  const { raceId } = request.params;
  const predictions = await env.DB.prepare(
    `SELECT p.*, d.name as driver_name, d.code as driver_code
     FROM predictions p
     JOIN drivers d ON p.driver_id = d.id
     WHERE p.race_id = ?
     ORDER BY p.predicted_position`
  ).bind(raceId).all();
  
  return new Response(JSON.stringify(predictions), {
    headers: { 'Content-Type': 'application/json' }
  });
});
```

## Phase 3: Frontend Dashboard (Week 5-6)

### 3.1 Component Structure
```
frontend/src/
├── components/
│   ├── PredictionTable/
│   ├── RaceSelector/
│   ├── AccuracyChart/
│   ├── FeatureImportance/
│   └── DriverComparison/
├── pages/
│   ├── Dashboard/
│   ├── Analytics/
│   ├── Historical/
│   └── Admin/
└── services/
    └── api.ts
```

### 3.2 Key Features
- Real-time predictions display
- Interactive charts (Chart.js)
- Historical accuracy tracking
- Driver performance comparison
- Model confidence visualization
- Mobile-responsive design

## Phase 4: ML Pipeline Integration (Week 7-8)

### 4.1 Automated Training Pipeline
```python
# scripts/train_models.py
async def train_and_deploy():
    # 1. Fetch latest data from FastF1
    # 2. Preprocess and engineer features
    # 3. Train multiple models
    # 4. Select best performer
    # 5. Store predictions in D1
    # 6. Update model metrics
```

### 4.2 Scheduled Jobs
- Daily: Fetch latest qualifying data
- Weekly: Retrain models with new race data
- Pre-race: Generate and store predictions

## Phase 5: Advanced Features (Week 9-10)

### 5.1 Enhanced Predictions
- Ensemble models combining multiple algorithms
- Weather-adjusted predictions
- Tire strategy analysis
- Safety car probability integration

### 5.2 User Features
- Email notifications for new predictions
- API access for developers
- Prediction explanations (SHAP values)
- Social sharing capabilities

## Implementation Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-2  | Foundation | Project setup, D1 schema, ML core module |
| 3-4  | API | Worker endpoints, data migration |
| 5-6  | Frontend | Dashboard, basic visualizations |
| 7-8  | ML Pipeline | Automated training, scheduled jobs |
| 9-10 | Advanced | Enhanced features, optimization |

## Performance Targets

- API Response Time: < 100ms (p95)
- Model Training: < 5 minutes per race
- Prediction Accuracy: MAE < 2.5 seconds
- Database Queries: < 50ms
- Frontend Load: < 2 seconds

## Security Considerations

1. **API Security**
   - Rate limiting on Workers
   - API key authentication for write operations
   - CORS configuration

2. **Data Protection**
   - Encrypted API keys in environment variables
   - Secure webhook endpoints
   - Input validation on all endpoints

## Monitoring & Observability

1. **Cloudflare Analytics**
   - Request metrics
   - Error tracking
   - Performance monitoring

2. **Custom Metrics**
   - Model accuracy tracking
   - Prediction confidence trends
   - Feature importance changes

## Cost Estimation

- Cloudflare Workers: Free tier (100k requests/day)
- D1 Database: Free tier (5GB storage)
- External APIs: ~$50/month (weather data)
- Total: < $100/month

## Next Steps

1. Set up Cloudflare account and configure Workers
2. Initialize project structure
3. Implement core ML module
4. Begin API development
5. Create basic frontend

This roadmap provides a comprehensive plan for transforming the F1 predictions system into a modern, scalable web application leveraging Cloudflare's edge computing platform.