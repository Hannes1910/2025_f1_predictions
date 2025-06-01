# 🔄 ML Data Flow - Training vs Prediction

## Current Problem
The ML service is generating predictions with **random data** instead of real features!

```python
# WRONG - Current implementation
'skill_rating': np.random.uniform(0.6, 1.0, len(drivers)),
'recent_form': np.random.uniform(5, 15, len(drivers)),
```

## How It Should Work

### 1️⃣ Training Phase (GitHub Actions Pipeline)

```
GitHub Actions (Daily)
    │
    ▼
Load Historical Data
    ├── 2025 races (completed)
    ├── 2024 patterns
    │
    ▼
Feature Engineering
    ├── Driver average positions
    ├── Team performance
    ├── Weather conditions
    ├── Circuit characteristics
    │
    ▼
Train Models
    ├── Learn patterns
    ├── Save model weights (.pkl files)
    └── Save feature scalers
```

### 2️⃣ Prediction Phase (ML Service)

```
ML Service receives request
    │
    ▼
Load Pre-trained Models
    ├── ensemble_model.pkl
    ├── feature_scaler.pkl
    │
    ▼
Get Current Race Features
    ├── Latest driver form (from DB)
    ├── Current championship standings
    ├── Qualifying results (if available)
    ├── Weather forecast
    ├── Historical track patterns
    │
    ▼
Apply Models
    └── Predictions based on REAL features
```

## The Missing Link

The ML service needs to:

1. **Load real-time features** from the database
2. **Use the same feature engineering** as training
3. **Apply pre-trained models** to these features

## Correct Data Flow

### Training (Offline)
```python
# Pipeline loads ALL historical data
data = load_2025_races() + load_2024_patterns()

# Engineer features
features = calculate_driver_form(data)
features += calculate_team_momentum(data)

# Train and save
model.fit(features, results)
joblib.dump(model, 'ensemble_model.pkl')
```

### Prediction (Real-time)
```python
# Load pre-trained model
model = joblib.load('ensemble_model.pkl')

# Get CURRENT features for next race
features = {
    'driver_form': get_last_3_races(driver_id),
    'quali_position': get_qualifying_result(race_id, driver_id),
    'team_avg': get_team_performance(team_id),
    'weather': get_weather_forecast(circuit),
    'track_history': get_driver_circuit_history(driver_id, circuit)
}

# Predict with REAL data
prediction = model.predict(features)
```

## What Needs Fixing

1. **ML Service** should query the database for:
   - Latest race results
   - Current championship standings
   - Driver recent form
   - Team performance trends

2. **Feature Engineering** must be consistent:
   - Same calculations in training and prediction
   - Same feature names and scaling

3. **Model Loading** should include:
   - The trained model weights
   - Feature scalers/encoders
   - Feature column names

## Example: Getting Real Driver Form

```python
async def get_driver_recent_form(self, driver_id: int) -> float:
    """Get driver's average position from last 3 races"""
    
    # Query actual results from database
    query = """
    SELECT AVG(position) as avg_position
    FROM race_results
    WHERE driver_id = ?
    ORDER BY race_date DESC
    LIMIT 3
    """
    
    result = await self.db.execute(query, [driver_id])
    return result['avg_position'] or 10.0  # Default if no data
```

## Summary

- **Training**: Uses historical data to learn patterns
- **Prediction**: Uses CURRENT data with learned patterns
- **Problem**: ML service using random data instead of real features
- **Solution**: Query database for actual current performance metrics