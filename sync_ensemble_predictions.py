#!/usr/bin/env python3
"""
Sync Ensemble Predictions to Cloudflare
Integrates with existing prediction pipeline
"""

import json
import requests
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import fastf1
from ml_package.data_loader import DataLoader
from ml_package.feature_engineer import FeatureEngineer

# Enable FastF1 cache
fastf1.Cache.enable_cache('f1_cache')

class EnsemblePredictionSync:
    def __init__(self):
        self.api_url = os.getenv('API_URL', 'https://f1-predictions-api.lando19.workers.dev')
        self.api_key = os.getenv('PREDICTIONS_API_KEY')
        
        # Load ensemble model
        self.ensemble = joblib.load('models/ensemble/ensemble_model.pkl')
        
        # Load feature columns
        with open('models/ensemble/feature_columns.json', 'r') as f:
            self.feature_columns = json.load(f)
        
        # Load individual models for uncertainty quantification
        self.models = {}
        model_names = ['gradient_boost', 'xgboost', 'lightgbm', 'catboost', 'random_forest']
        for name in model_names:
            self.models[name] = joblib.load(f'models/ensemble/{name}_model.pkl')
        
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
    
    def get_upcoming_races(self):
        """Get races in the next 14 days that need predictions"""
        response = requests.get(f"{self.api_url}/api/races")
        races = response.json()['races']
        
        upcoming = []
        today = datetime.now()
        two_weeks = today + timedelta(days=14)
        
        for race in races:
            race_date = datetime.fromisoformat(race['date'].replace('Z', '+00:00'))
            if today <= race_date <= two_weeks:
                upcoming.append(race)
        
        return upcoming
    
    def prepare_race_features(self, race, drivers):
        """Prepare features for a specific race"""
        # Get historical data for feature engineering
        historical_data = self.data_loader.load_2024_season_data()
        
        # Create base features for each driver
        race_data = []
        for driver in drivers:
            driver_data = {
                'driver_id': driver['id'],
                'driver_code': driver['code'],
                'team': driver['team'],
                'circuit': race['circuit'],
                'race_distance': 305.0  # Standard F1 race distance
            }
            race_data.append(driver_data)
        
        race_df = pd.DataFrame(race_data)
        
        # Engineer features
        race_df = self.feature_engineer.create_features(race_df, historical_data)
        
        # Add weather data
        weather = self.get_weather_forecast(race['circuit'], race['date'])
        race_df['expected_temperature'] = weather['temperature']
        race_df['rain_probability'] = weather['rain_probability']
        
        # Add recent form
        race_df['driver_recent_form'] = self.calculate_recent_form(race_df['driver_id'])
        
        # Add team momentum
        race_df['team_momentum_change'] = self.calculate_team_momentum(race_df['team'])
        
        # Add circuit type features
        circuit_types = {
            'Monaco': 'street', 'Singapore': 'street', 'Las Vegas': 'street',
            'Italy': 'high_speed', 'Belgium': 'high_speed', 'Great Britain': 'high_speed',
            'Hungary': 'technical', 'Spain': 'technical', 'Japan': 'technical'
        }
        
        circuit_type = circuit_types.get(race['circuit'], 'balanced')
        for ct in ['street', 'high_speed', 'technical', 'balanced']:
            race_df[f'circuit_type_{ct}'] = 1 if ct == circuit_type else 0
        
        return race_df
    
    def get_weather_forecast(self, circuit, date):
        """Get weather forecast for race"""
        # Implement actual weather API call or use patterns
        weather_patterns = {
            'Spain': {'temperature': 26, 'rain_probability': 0.1},
            'Monaco': {'temperature': 24, 'rain_probability': 0.15},
            'Canada': {'temperature': 22, 'rain_probability': 0.25}
        }
        
        return weather_patterns.get(circuit, {'temperature': 20, 'rain_probability': 0.2})
    
    def calculate_recent_form(self, driver_ids):
        """Calculate driver's recent form from last 3 races"""
        # This would query actual recent results
        # For now, return placeholder
        return pd.Series([5.0] * len(driver_ids))
    
    def calculate_team_momentum(self, teams):
        """Calculate team momentum from recent races"""
        # This would analyze team performance trends
        # For now, return placeholder
        return pd.Series([0.0] * len(teams))
    
    def generate_predictions(self, race, race_features):
        """Generate ensemble predictions with uncertainty"""
        
        # Ensure we have all required features
        X = race_features[self.feature_columns]
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Get ensemble prediction
        ensemble_pred = self.ensemble.predict(X)
        
        # Calculate uncertainty (standard deviation across models)
        pred_array = np.array(list(predictions.values()))
        pred_std = np.std(pred_array, axis=0)
        
        # Create results
        results = pd.DataFrame({
            'driver_id': race_features['driver_id'],
            'predicted_time': ensemble_pred,
            'prediction_std': pred_std,
            'confidence': 1 / (1 + pred_std)
        })
        
        # Sort by predicted time to get positions
        results = results.sort_values('predicted_time').reset_index(drop=True)
        results['predicted_position'] = range(1, len(results) + 1)
        
        # Format for API
        predictions_list = []
        for _, row in results.iterrows():
            predictions_list.append({
                'driver_id': int(row['driver_id']),
                'predicted_position': int(row['predicted_position']),
                'predicted_time': float(row['predicted_time']),
                'confidence': float(row['confidence']),
                'model_version': 'ensemble_v1.0'
            })
        
        return predictions_list
    
    def sync_predictions(self, race_id, predictions):
        """Upload predictions to Cloudflare Worker"""
        
        payload = {
            'race_id': race_id,
            'predictions': predictions,
            'model_metrics': {
                'ensemble_models': len(self.models),
                'feature_count': len(self.feature_columns),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        headers = {
            'X-API-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f"{self.api_url}/api/admin/predictions",
            json=payload,
            headers=headers
        )
        
        if response.status_code == 200:
            print(f"✅ Successfully synced predictions for race {race_id}")
        else:
            print(f"❌ Failed to sync predictions: {response.text}")
            raise Exception(f"API error: {response.status_code}")

def main():
    """Main sync pipeline"""
    
    print("🏁 F1 Ensemble Prediction Sync")
    print("=" * 50)
    
    syncer = EnsemblePredictionSync()
    
    # Get upcoming races
    races = syncer.get_upcoming_races()
    print(f"\n📅 Found {len(races)} upcoming races")
    
    # Get drivers
    response = requests.get(f"{syncer.api_url}/api/drivers")
    drivers = response.json()['drivers']
    
    for race in races:
        print(f"\n🏎️ Generating predictions for {race['name']}...")
        
        try:
            # Prepare features
            race_features = syncer.prepare_race_features(race, drivers)
            
            # Generate predictions
            predictions = syncer.generate_predictions(race, race_features)
            
            # Sync to API
            syncer.sync_predictions(race['id'], predictions)
            
            # Show top 5 predictions
            print("\n📊 Top 5 Predictions:")
            for i, pred in enumerate(predictions[:5]):
                driver = next(d for d in drivers if d['id'] == pred['driver_id'])
                print(f"  {i+1}. {driver['code']} - Confidence: {pred['confidence']:.2%}")
                
        except Exception as e:
            print(f"❌ Error processing {race['name']}: {str(e)}")
            continue
    
    print("\n✅ Ensemble prediction sync complete!")

if __name__ == "__main__":
    main()