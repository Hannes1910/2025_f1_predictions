#!/bin/bash

echo "🔧 FIXING DEPLOYMENT ISSUES"
echo "============================"

echo "1. Adding missing country column to races table..."
npx wrangler d1 execute f1-predictions --remote --command="ALTER TABLE races ADD COLUMN country TEXT"

echo "2. Updating races with country data..."
npx wrangler d1 execute f1-predictions --remote --command="UPDATE races SET country = 'Australia' WHERE circuit = 'Albert Park'"
npx wrangler d1 execute f1-predictions --remote --command="UPDATE races SET country = 'China' WHERE circuit = 'Shanghai'"
npx wrangler d1 execute f1-predictions --remote --command="UPDATE races SET country = 'Japan' WHERE circuit = 'Suzuka'"
npx wrangler d1 execute f1-predictions --remote --command="UPDATE races SET country = 'Bahrain' WHERE circuit = 'Bahrain'"

echo "3. Building and deploying fixed Worker..."
cd packages/worker
npm run build
cd ../..
npx wrangler deploy

echo "4. Testing fixed endpoints..."
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
echo "✅ FIXES COMPLETE!"