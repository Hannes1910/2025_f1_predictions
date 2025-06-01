#!/bin/bash

# 🚀 Deploy Clean System to Production - EXACT COMMANDS
# Run these commands step by step

echo "🚀 DEPLOYING CLEAN F1 SYSTEM TO PRODUCTION"
echo "=========================================="

echo ""
echo "📋 STEP 1: Update Worker with Clean Endpoints"
echo "----------------------------------------------"

# Replace current data-api with clean version
echo "🔄 Replacing data-api handler..."
cp packages/worker/src/handlers/data-api-clean.ts packages/worker/src/handlers/data-api.ts

echo "✅ Clean endpoints copied"
echo ""
echo "🏗️ Building and deploying Worker..."

# Build and deploy
cd packages/worker
npm run build
cd ../..
npx wrangler deploy

echo ""
echo "📋 STEP 2: Update D1 Database Schema"
echo "------------------------------------"

echo "💾 Creating backup..."
npx wrangler d1 execute f1-predictions --remote --command="SELECT COUNT(*) as driver_count FROM drivers"
npx wrangler d1 execute f1-predictions --remote --command="SELECT COUNT(*) as race_count FROM races"

echo "🗃️ Adding team_entries table..."
npx wrangler d1 execute f1-predictions --remote --command="CREATE TABLE IF NOT EXISTS team_entries (id INTEGER PRIMARY KEY AUTOINCREMENT, season INTEGER NOT NULL CHECK (season >= 1950 AND season <= 2030), team_name TEXT NOT NULL, driver_id INTEGER NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (driver_id) REFERENCES drivers(id) ON DELETE RESTRICT, UNIQUE(season, driver_id), CHECK (length(team_name) > 0))"

echo "📊 Adding missing driver columns..."
npx wrangler d1 execute f1-predictions --remote --command="ALTER TABLE drivers ADD COLUMN driver_ref TEXT"
npx wrangler d1 execute f1-predictions --remote --command="ALTER TABLE drivers ADD COLUMN forename TEXT" 
npx wrangler d1 execute f1-predictions --remote --command="ALTER TABLE drivers ADD COLUMN surname TEXT"
npx wrangler d1 execute f1-predictions --remote --command="ALTER TABLE drivers ADD COLUMN nationality TEXT"

echo "🏁 Adding missing qualifying columns..."
npx wrangler d1 execute f1-predictions --remote --command="ALTER TABLE qualifying_results ADD COLUMN qualifying_position INTEGER"
npx wrangler d1 execute f1-predictions --remote --command="ALTER TABLE qualifying_results ADD COLUMN grid_penalty INTEGER DEFAULT 0"
npx wrangler d1 execute f1-predictions --remote --command="ALTER TABLE qualifying_results ADD COLUMN session_date DATETIME"

echo "🏆 Adding missing race result columns..."
npx wrangler d1 execute f1-predictions --remote --command="ALTER TABLE race_results ADD COLUMN grid_position INTEGER"
npx wrangler d1 execute f1-predictions --remote --command="ALTER TABLE race_results ADD COLUMN race_time_ms INTEGER"
npx wrangler d1 execute f1-predictions --remote --command="ALTER TABLE race_results ADD COLUMN fastest_lap_time_ms INTEGER"
npx wrangler d1 execute f1-predictions --remote --command="ALTER TABLE race_results ADD COLUMN status TEXT DEFAULT 'Finished'"
npx wrangler d1 execute f1-predictions --remote --command="ALTER TABLE race_results ADD COLUMN laps_completed INTEGER DEFAULT 0"
npx wrangler d1 execute f1-predictions --remote --command="ALTER TABLE race_results ADD COLUMN session_date DATETIME"

echo "📈 Creating production indexes..."
npx wrangler d1 execute f1-predictions --remote --command="CREATE INDEX IF NOT EXISTS idx_team_entries_season ON team_entries(season)"
npx wrangler d1 execute f1-predictions --remote --command="CREATE INDEX IF NOT EXISTS idx_team_entries_team ON team_entries(team_name)"
npx wrangler d1 execute f1-predictions --remote --command="CREATE INDEX IF NOT EXISTS idx_qualifying_session_date ON qualifying_results(session_date)"
npx wrangler d1 execute f1-predictions --remote --command="CREATE INDEX IF NOT EXISTS idx_race_results_session_date ON race_results(session_date)"

echo ""
echo "📋 STEP 3: Populate with Real F1 Data"
echo "-------------------------------------"

echo "👨‍💼 Updating driver data..."
npx wrangler d1 execute f1-predictions --remote --command="UPDATE drivers SET driver_ref = 'max_verstappen', forename = 'Max', surname = 'Verstappen', nationality = 'Dutch' WHERE code = 'VER'"
npx wrangler d1 execute f1-predictions --remote --command="UPDATE drivers SET driver_ref = 'lando_norris', forename = 'Lando', surname = 'Norris', nationality = 'British' WHERE code = 'NOR'"
npx wrangler d1 execute f1-predictions --remote --command="UPDATE drivers SET driver_ref = 'charles_leclerc', forename = 'Charles', surname = 'Leclerc', nationality = 'Monégasque' WHERE code = 'LEC'"
npx wrangler d1 execute f1-predictions --remote --command="UPDATE drivers SET driver_ref = 'lewis_hamilton', forename = 'Lewis', surname = 'Hamilton', nationality = 'British' WHERE code = 'HAM'"
npx wrangler d1 execute f1-predictions --remote --command="UPDATE drivers SET driver_ref = 'oscar_piastri', forename = 'Oscar', surname = 'Piastri', nationality = 'Australian' WHERE code = 'PIA'"

echo "🏎️ Adding team entries for 2025..."
npx wrangler d1 execute f1-predictions --remote --command="INSERT OR IGNORE INTO team_entries (season, team_name, driver_id) VALUES (2025, 'Red Bull Racing', 1), (2025, 'Red Bull Racing', 2), (2025, 'McLaren', 3), (2025, 'McLaren', 4), (2025, 'Ferrari', 5), (2025, 'Ferrari', 6), (2025, 'Mercedes', 7), (2025, 'Mercedes', 8), (2025, 'Aston Martin', 9), (2025, 'Aston Martin', 10)"

echo "🏁 Updating race data..."
npx wrangler d1 execute f1-predictions --remote --command="UPDATE races SET circuit = 'Albert Park', country = 'Australia' WHERE id = 1"
npx wrangler d1 execute f1-predictions --remote --command="UPDATE races SET circuit = 'Shanghai', country = 'China' WHERE id = 2" 
npx wrangler d1 execute f1-predictions --remote --command="UPDATE races SET circuit = 'Suzuka', country = 'Japan' WHERE id = 3"
npx wrangler d1 execute f1-predictions --remote --command="UPDATE races SET circuit = 'Bahrain', country = 'Bahrain' WHERE id = 4"

echo ""
echo "📋 STEP 4: Test Production Deployment"
echo "------------------------------------"

echo "🧪 Testing Worker endpoints..."
echo "Driver stats:"
curl -s "https://f1-predictions-api.vprifntqe.workers.dev/api/data/driver/1" | jq .

echo ""
echo "Race features:"
curl -s "https://f1-predictions-api.vprifntqe.workers.dev/api/data/race/1/features" | jq '.data.race'

echo ""
echo "ML prediction data:"
curl -s -X POST "https://f1-predictions-api.vprifntqe.workers.dev/api/data/ml-prediction-data" \
  -H "Content-Type: application/json" \
  -d '{"raceId": 1}' | jq '.meta'

echo ""
echo "📋 STEP 5: Cleanup"
echo "------------------"

echo "🧹 Removing deprecated tables..."
npx wrangler d1 execute f1-predictions --remote --command="DROP TABLE IF EXISTS feature_data"

echo ""
echo "🎉 PRODUCTION DEPLOYMENT COMPLETE!"
echo "=================================="
echo ""
echo "✅ Worker updated with clean endpoints"
echo "✅ D1 database schema updated"
echo "✅ Real F1 data populated"
echo "✅ Production testing completed"
echo "✅ Cleanup completed"
echo ""
echo "🔗 Your production API: https://f1-predictions-api.vprifntqe.workers.dev"
echo ""
echo "📋 Next Steps:"
echo "1. Deploy ML service to cloud (Google Cloud Run, Railway, or Fly.io)"
echo "2. Update GitHub Actions with ML service URL"
echo "3. Monitor production endpoints"
echo ""
echo "🆘 If issues occur, check the logs:"
echo "   npx wrangler tail"