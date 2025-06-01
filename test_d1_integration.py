#!/usr/bin/env python3
"""
Test D1 Integration
Verify that ML service can access D1 data through Worker API
"""

import requests
import json
from d1_data_client import D1DataClient

def test_d1_client_direct():
    """Test D1 client directly"""
    print("🔍 Testing D1 Client Direct Access...")
    
    client = D1DataClient(
        worker_url="https://f1-predictions-api.vprifntqe.workers.dev",
        api_key="test-key-123"  # Replace with actual key if needed
    )
    
    # Test 1: Get driver stats
    print("\n1. Testing driver stats...")
    driver_stats = client.get_driver_stats(1)  # Verstappen
    if driver_stats:
        print(f"✅ Driver stats: {json.dumps(driver_stats, indent=2)}")
    else:
        print("❌ Failed to get driver stats")
    
    # Test 2: Get team stats
    print("\n2. Testing team stats...")
    team_stats = client.get_team_stats("Red Bull Racing")
    if team_stats:
        print(f"✅ Team stats: {json.dumps(team_stats, indent=2)}")
    else:
        print("❌ Failed to get team stats")
    
    # Test 3: Get race features
    print("\n3. Testing race features...")
    race_features = client.get_race_features(1)  # First race
    if race_features:
        print(f"✅ Race features available for {len(race_features.get('drivers', []))} drivers")
        if race_features.get('race'):
            print(f"   Race: {race_features['race'].get('name', 'Unknown')}")
    else:
        print("❌ Failed to get race features")
    
    # Test 4: Get ML prediction data
    print("\n4. Testing ML prediction data...")
    ml_data = client.get_ml_prediction_data(1)
    if ml_data:
        print(f"✅ ML data retrieved:")
        print(f"   - Race info: {'✓' if ml_data.get('race') else '✗'}")
        print(f"   - Drivers: {len(ml_data.get('drivers', []))}")
        print(f"   - Recent results: {len(ml_data.get('recentResults', []))}")
    else:
        print("❌ Failed to get ML prediction data")
    
    return bool(driver_stats or team_stats or race_features or ml_data)


def test_ml_service_local():
    """Test ML service running locally"""
    print("\n\n🔍 Testing ML Service (Local)...")
    
    try:
        # Check if service is running
        response = requests.get("http://localhost:8001/")
        if response.status_code == 200:
            info = response.json()
            print(f"✅ ML Service is running:")
            print(f"   - Version: {info.get('version')}")
            print(f"   - Database: {info.get('database')}")
            print(f"   - Worker URL: {info.get('worker_url')}")
        
        # Test health endpoint
        response = requests.get("http://localhost:8001/health")
        if response.status_code == 200:
            health = response.json()
            print(f"\n✅ Health check:")
            print(f"   - Status: {health.get('status')}")
            print(f"   - D1 Connected: {health.get('d1_connected')}")
            print(f"   - Model Loaded: {health.get('model_loaded')}")
        
        # Test data access
        response = requests.get("http://localhost:8001/data/test")
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Data access test:")
            print(f"   - Driver stats: {'✓' if data.get('driver_stats') else '✗'}")
            print(f"   - Team stats: {'✓' if data.get('team_stats') else '✗'}")
            print(f"   - Race features: {'✓' if data.get('race_features_available') else '✗'}")
            print(f"   - Driver count: {data.get('driver_count', 0)}")
        
        # Test prediction
        print("\n🏁 Testing race prediction...")
        response = requests.post("http://localhost:8001/predict/race/1")
        if response.status_code == 200:
            prediction = response.json()
            print(f"✅ Prediction generated:")
            print(f"   - Model: {prediction.get('model_version')}")
            print(f"   - Accuracy: {prediction.get('expected_accuracy')}")
            print(f"   - Data source: {prediction.get('data_source')}")
            print(f"   - Predictions: {len(prediction.get('predictions', []))}")
            
            # Show top 3
            if prediction.get('predictions'):
                print("\n   Top 3 predictions:")
                for i, pred in enumerate(prediction['predictions'][:3], 1):
                    print(f"   {i}. {pred['driver_code']} - Confidence: {pred['confidence']:.1%}")
        
    except requests.exceptions.ConnectionError:
        print("❌ ML Service is not running. Start it with: ./start_ml_service_d1.sh")
        return False
    except Exception as e:
        print(f"❌ Error testing ML service: {e}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("D1 + ML SERVICE INTEGRATION TEST")
    print("=" * 60)
    
    # Test D1 client
    d1_success = test_d1_client_direct()
    
    # Test ML service (if running)
    ml_success = test_ml_service_local()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"D1 Client Direct Access: {'✅ PASS' if d1_success else '❌ FAIL'}")
    print(f"ML Service Integration:  {'✅ PASS' if ml_success else '❓ Not tested (service not running)'}")
    
    if d1_success:
        print("\n✅ D1 integration is working! The ML service can now access production data.")
        print("\n📌 Next steps:")
        print("1. Start the ML service: ./start_ml_service_d1.sh")
        print("2. The service will use real D1 data for predictions")
        print("3. Deploy to cloud when ready")
    else:
        print("\n❌ D1 integration failed. Check:")
        print("1. Worker is deployed with data API endpoints")
        print("2. Worker URL is correct")
        print("3. Network connectivity")


if __name__ == "__main__":
    main()