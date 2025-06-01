# 🚀 Deploy Clean System to Production - Online Only

## 📋 Step-by-Step Production Deployment

### Step 1: Update Cloudflare Worker with Clean Endpoints

#### 1.1 Replace the current data-api handler
```bash
# Copy clean endpoints to replace current ones
cp packages/worker/src/handlers/data-api-clean.ts packages/worker/src/handlers/data-api.ts
```

#### 1.2 Update the Worker index.ts to use clean endpoints
Edit `packages/worker/src/index.ts`:
```typescript
// Replace existing data-api imports with clean versions
import {
  handleGetDriverStatsClean as handleGetDriverStats,
  handleGetTeamStatsClean as handleGetTeamStats,
  handleGetRaceFeaturesClean as handleGetRaceFeatures,
  handleGetCircuitPatternsClean as handleGetCircuitPatterns,
  handleGetMLPredictionDataClean as handleGetMLPredictionData
} from './handlers/data-api';
```

#### 1.3 Build and deploy Worker
```bash
cd packages/worker
npm run build
cd ../..
npx wrangler deploy
```

### Step 2: Update D1 Production Database Schema

#### 2.1 Create backup of current data (safety)
```bash
npx wrangler d1 execute f1-predictions --remote --command="SELECT COUNT(*) FROM drivers"
npx wrangler d1 execute f1-predictions --remote --command="SELECT COUNT(*) FROM races"
```

#### 2.2 Add clean schema elements
```bash
# Add team_entries table
npx wrangler d1 execute f1-predictions --remote --command="
CREATE TABLE IF NOT EXISTS team_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL CHECK (season >= 1950 AND season <= 2030),
    team_name TEXT NOT NULL,
    driver_id INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (driver_id) REFERENCES drivers(id) ON DELETE RESTRICT,
    UNIQUE(season, driver_id),
    CHECK (length(team_name) > 0)
)"

# Update drivers table with missing columns
npx wrangler d1 execute f1-predictions --remote --command="
ALTER TABLE drivers ADD COLUMN driver_ref TEXT;
ALTER TABLE drivers ADD COLUMN forename TEXT;
ALTER TABLE drivers ADD COLUMN surname TEXT;
ALTER TABLE drivers ADD COLUMN nationality TEXT;
ALTER TABLE drivers ADD COLUMN dob DATE;
"

# Update qualifying_results with proper constraints
npx wrangler d1 execute f1-predictions --remote --command="
ALTER TABLE qualifying_results ADD COLUMN qualifying_position INTEGER;
ALTER TABLE qualifying_results ADD COLUMN grid_penalty INTEGER DEFAULT 0;
ALTER TABLE qualifying_results ADD COLUMN session_date DATETIME;
"

# Update race_results with proper structure
npx wrangler d1 execute f1-predictions --remote --command="
ALTER TABLE race_results ADD COLUMN grid_position INTEGER;
ALTER TABLE race_results ADD COLUMN race_time_ms INTEGER;
ALTER TABLE race_results ADD COLUMN fastest_lap_time_ms INTEGER;
ALTER TABLE race_results ADD COLUMN status TEXT DEFAULT 'Finished';
ALTER TABLE race_results ADD COLUMN laps_completed INTEGER DEFAULT 0;
ALTER TABLE race_results ADD COLUMN session_date DATETIME;
"
```

#### 2.3 Create production indexes
```bash
npx wrangler d1 execute f1-predictions --remote --command="
CREATE INDEX IF NOT EXISTS idx_team_entries_season ON team_entries(season);
CREATE INDEX IF NOT EXISTS idx_team_entries_team ON team_entries(team_name);
CREATE INDEX IF NOT EXISTS idx_qualifying_session_date ON qualifying_results(session_date);
CREATE INDEX IF NOT EXISTS idx_qualifying_grid_position ON qualifying_results(race_id, grid_position);
CREATE INDEX IF NOT EXISTS idx_race_results_session_date ON race_results(session_date);
"
```

### Step 3: Populate Production with Real F1 Data

#### 3.1 Update existing drivers with proper data
```bash
# Update Max Verstappen
npx wrangler d1 execute f1-predictions --remote --command="
UPDATE drivers SET 
    driver_ref = 'max_verstappen',
    forename = 'Max',
    surname = 'Verstappen',
    nationality = 'Dutch'
WHERE code = 'VER'
"

# Update Lando Norris
npx wrangler d1 execute f1-predictions --remote --command="
UPDATE drivers SET 
    driver_ref = 'lando_norris',
    forename = 'Lando',
    surname = 'Norris',
    nationality = 'British'
WHERE code = 'NOR'
"

# Continue for other drivers...
```

#### 3.2 Add team entries for 2025 season
```bash
npx wrangler d1 execute f1-predictions --remote --command="
INSERT INTO team_entries (season, team_name, driver_id) VALUES
(2025, 'Red Bull Racing', 1),
(2025, 'Red Bull Racing', 2),
(2025, 'McLaren', 3),
(2025, 'McLaren', 4),
(2025, 'Ferrari', 5),
(2025, 'Ferrari', 6),
(2025, 'Mercedes', 7),
(2025, 'Mercedes', 8)
"
```

#### 3.3 Update races with proper format
```bash
npx wrangler d1 execute f1-predictions --remote --command="
UPDATE races SET 
    circuit = 'Albert Park',
    country = 'Australia'
WHERE id = 1 AND name = 'Australian Grand Prix'
"

npx wrangler d1 execute f1-predictions --remote --command="
UPDATE races SET 
    circuit = 'Shanghai',
    country = 'China'
WHERE id = 2 AND name = 'Chinese Grand Prix'
"
```

### Step 4: Deploy ML Service to Cloud (Production)

#### 4.1 Choose cloud platform and deploy

**Option A: Google Cloud Run**
```bash
# Create Dockerfile for ML service
cat > Dockerfile << EOF
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ml_service_production_clean.py .
COPY d1_data_client.py .

ENV WORKER_URL=https://f1-predictions-api.vprifntqe.workers.dev
ENV PORT=8080

CMD ["python", "ml_service_production_clean.py"]
EOF

# Deploy to Cloud Run
gcloud run deploy f1-ml-service \\
  --source . \\
  --region us-central1 \\
  --set-env-vars WORKER_URL=https://f1-predictions-api.vprifntqe.workers.dev \\
  --allow-unauthenticated
```

**Option B: Railway**
```bash
# Create railway.toml
cat > railway.toml << EOF
[build]
builder = "DOCKERFILE"

[deploy]
startCommand = "python ml_service_production_clean.py"

[env]
WORKER_URL = "https://f1-predictions-api.vprifntqe.workers.dev"
PORT = "8080"
EOF

# Deploy
railway login
railway deploy
```

**Option C: Fly.io**
```bash
# Create fly.toml
fly launch --name f1-ml-service
fly secrets set WORKER_URL=https://f1-predictions-api.vprifntqe.workers.dev
fly deploy
```

### Step 5: Update GitHub Actions Pipeline

#### 5.1 Update the pipeline to use clean endpoints
Edit `.github/workflows/predictions-pipeline.yml`:
```yaml
env:
  WORKER_URL: https://f1-predictions-api.vprifntqe.workers.dev
  ML_SERVICE_URL: https://your-ml-service-url.com  # From step 4
```

#### 5.2 Configure secrets
```bash
# In GitHub repository settings, add:
# PREDICTIONS_API_KEY = your-api-key
# ML_SERVICE_URL = your-deployed-ml-service-url
```

### Step 6: Test Production Deployment

#### 6.1 Test Worker endpoints
```bash
# Test driver stats
curl "https://f1-predictions-api.vprifntqe.workers.dev/api/data/driver/1"

# Test race features  
curl "https://f1-predictions-api.vprifntqe.workers.dev/api/data/race/1/features"

# Test ML prediction data
curl -X POST "https://f1-predictions-api.vprifntqe.workers.dev/api/data/ml-prediction-data" \\
  -H "Content-Type: application/json" \\
  -d '{"raceId": 1}'
```

#### 6.2 Test ML service
```bash
# Test health
curl "https://your-ml-service-url.com/health"

# Test prediction
curl -X POST "https://your-ml-service-url.com/predict/race/1"
```

#### 6.3 Test integration
```bash
# Test full pipeline
curl "https://your-ml-service-url.com/predict/next"
```

### Step 7: Cleanup (Remove Old Components)

#### 7.1 Remove deprecated database elements
```bash
# Drop feature_data table (if exists)
npx wrangler d1 execute f1-predictions --remote --command="DROP TABLE IF EXISTS feature_data"
```

#### 7.2 Update frontend to use new endpoints
Update any frontend code to use the clean API responses format.

## ✅ Deployment Checklist

- [ ] Step 1: Worker updated with clean endpoints
- [ ] Step 2: D1 schema updated with constraints  
- [ ] Step 3: Real F1 data populated
- [ ] Step 4: ML service deployed to cloud
- [ ] Step 5: GitHub Actions updated
- [ ] Step 6: Production testing completed
- [ ] Step 7: Cleanup completed

## 🎯 Expected Results

After deployment:
- ✅ All APIs return real F1 data only
- ✅ Proper error handling (404/422) when data unavailable
- ✅ ML predictions use real qualifying data
- ✅ No mock data anywhere in production
- ✅ Clean, maintainable architecture

## 🆘 Rollback Plan (if needed)

If something goes wrong:
```bash
# Rollback Worker
git checkout HEAD~1 packages/worker/src/
npx wrangler deploy

# Rollback D1 (harder - would need backup restore)
# This is why we take backups in Step 2.1
```

## 📞 Support

Monitor these endpoints after deployment:
- Worker: `https://f1-predictions-api.vprifntqe.workers.dev/api/health`
- ML Service: `https://your-ml-service-url.com/health`
- Full System: `https://your-ml-service-url.com/predict/next`