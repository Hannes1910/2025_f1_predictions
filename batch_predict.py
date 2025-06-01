#!/usr/bin/env python3
"""
Batch job for generating F1 predictions
Run this daily via cron or GitHub Actions
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from ml_production_service import generate_predictions_for_worker

async def main():
    print("🏁 Starting F1 Ultra Predictor batch job...")
    
    try:
        result = await generate_predictions_for_worker()
        
        if "error" not in result:
            print(f"✅ Generated predictions for race {result['race_id']}")
            print(f"   Race: {result['race_name']}")
            print(f"   Predictions: {result['predictions_count']}")
            print(f"   Model: {result['model_version']} ({result['accuracy']:.1%} accuracy)")
        else:
            print(f"⚠️ {result['error']}")
            
    except Exception as e:
        print(f"❌ Batch job failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
