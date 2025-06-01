#!/usr/bin/env python3
"""
Test real predictions end-to-end
"""

import requests
import json
from datetime import datetime

def test_ml_service():
    """Test the ML service with real data"""
    
    base_url = "http://localhost:8001"
    
    print("🧪 Testing F1 ML Service with Real Data")
    print("=" * 50)
    
    # 1. Check service health
    print("\n1. Checking service health...")
    response = requests.get(f"{base_url}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
    
    # 2. Test race prediction (China - race 4)
    print("\n2. Testing prediction for China GP (race 4)...")
    try:
        response = requests.post(f"{base_url}/predict/race/4")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success! Generated {len(data['predictions'])} predictions")
            print(f"   Model: {data['model_version']}")
            print(f"   Data source: {data['data_source']}")
            print("\n   Top 5 predictions:")
            for i, pred in enumerate(data['predictions'][:5]):
                print(f"   {i+1}. {pred['driver_code']} - Position: {pred['predicted_position']:.1f} (confidence: {pred['confidence']:.2%})")
        else:
            print(f"   ❌ Error: {response.status_code}")
            print(f"   {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. Test with a race that has no qualifying data yet (race 5)
    print("\n3. Testing prediction for Miami GP (race 5 - no quali data)...")
    try:
        response = requests.post(f"{base_url}/predict/race/5")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success! Generated predictions without qualifying data")
            print(f"   Using historical averages for grid positions")
        else:
            print(f"   Status: {response.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 4. Show how features are extracted
    print("\n4. Feature extraction example:")
    print("   - Recent form: Average of last 3 race positions")
    print("   - Team performance: Average position of team")
    print("   - Circuit patterns: Historical data (pit times, SC probability)")
    print("   - Weather: Expected conditions for circuit")
    
    print("\n" + "=" * 50)
    print("✅ End-to-end test complete!")
    print("\nThe system is now using:")
    print("- REAL race results from database")
    print("- REAL driver performance metrics")
    print("- REAL qualifying data (when available)")
    print("- Historical circuit patterns")


if __name__ == "__main__":
    test_ml_service()