#!/usr/bin/env python3
"""
Test script to verify pipeline components work locally
"""

import os
import sys

print("🧪 Testing F1 Prediction Pipeline Components")
print("=" * 50)

# Test 1: Check directories
print("\n1. Checking directories...")
directories = ['f1_cache', 'models/ensemble', 'packages/ml-core']
for dir_path in directories:
    exists = os.path.exists(dir_path)
    print(f"   {dir_path}: {'✅ exists' if exists else '❌ missing'}")
    if not exists and dir_path != 'packages/ml-core':
        os.makedirs(dir_path, exist_ok=True)
        print(f"   Created {dir_path}")

# Test 2: Check imports
print("\n2. Testing imports...")
try:
    import pandas as pd
    print("   ✅ pandas")
except ImportError:
    print("   ❌ pandas - run: pip install pandas")

try:
    import numpy as np
    print("   ✅ numpy")
except ImportError:
    print("   ❌ numpy - run: pip install numpy")

try:
    import sklearn
    print("   ✅ scikit-learn")
except ImportError:
    print("   ❌ scikit-learn - run: pip install scikit-learn")

try:
    import fastf1
    print("   ✅ fastf1")
    # Test cache
    fastf1.Cache.enable_cache('f1_cache')
    print("   ✅ fastf1 cache enabled")
except ImportError:
    print("   ❌ fastf1 - run: pip install fastf1")
except Exception as e:
    print(f"   ❌ fastf1 cache error: {e}")

# Test 3: Check ML package
print("\n3. Testing ML package imports...")
try:
    sys.path.append('./packages/ml-core')
    from f1_predictor.data_loader import DataLoader
    print("   ✅ DataLoader import")
except ImportError as e:
    print(f"   ❌ DataLoader import failed: {e}")

try:
    from f1_predictor.feature_engineering import FeatureEngineer
    print("   ✅ FeatureEngineer import")
except ImportError as e:
    print(f"   ❌ FeatureEngineer import failed: {e}")

# Test 4: Check environment variables
print("\n4. Checking environment variables...")
env_vars = ['WEATHER_API_KEY', 'PREDICTIONS_API_KEY', 'API_URL']
for var in env_vars:
    value = os.getenv(var)
    if value:
        print(f"   ✅ {var}: {'*' * 8} (set)")
    else:
        print(f"   ⚠️  {var}: not set")

# Test 5: Quick model test
print("\n5. Testing basic model creation...")
try:
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    # Create dummy data
    X = [[1, 2], [3, 4], [5, 6]]
    y = [1, 2, 3]
    model.fit(X, y)
    pred = model.predict([[7, 8]])
    print(f"   ✅ Model creation and prediction successful (pred: {pred[0]:.2f})")
except Exception as e:
    print(f"   ❌ Model test failed: {e}")

print("\n" + "=" * 50)
print("🏁 Test complete!")
print("\nTo run the full pipeline:")
print("1. Install all dependencies: pip install -r requirements.txt")
print("2. Set environment variables (optional)")
print("3. Run: python train_models_ensemble.py")