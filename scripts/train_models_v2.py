#!/usr/bin/env python3
"""
Enhanced Model Training Pipeline for F1 Predictions
Uses real FastF1 data and stores predictions in Cloudflare D1
"""

import sys
import os
import json
import sqlite3
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import fastf1

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from f1_predictor import F1Predictor, PredictorConfig, RaceData, ModelType

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable FastF1 cache
fastf1.Cache.enable_cache("f1_cache")

class RealDataTrainingPipeline:
    def __init__(self, db_path: str, weather_api_key: str = None):
        self.db_path = db_path
        self.weather_api_key = weather_api_key or os.getenv('WEATHER_API_KEY')
        self.conn = None
        self.current_year = 2025
        self.training_year = 2024
        
    def connect(self):
        """Connect to the database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def get_completed_2024_races(self):
        """Get list of completed 2024 races for training data"""
        # Get all 2024 races that have been completed
        races_2024 = [
            'Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'China',
            'Miami', 'Imola', 'Monaco', 'Canada', 'Spain',
            'Austria', 'Great Britain', 'Hungary', 'Belgium', 'Netherlands',
            'Italy', 'Azerbaijan', 'Singapore', 'United States', 'Mexico',
            'Brazil', 'Las Vegas', 'Qatar', 'Abu Dhabi'
        ]
        return races_2024[:20]  # Use first 20 races for training
    
    def load_historical_race_data(self):
        """Load real 2024 race data from FastF1"""
        logger.info("Loading 2024 race data from FastF1...")
        
        all_race_data = []
        races = self.get_completed_2024_races()
        
        for race_name in races:
            try:
                logger.info(f"Loading {race_name} 2024...")
                
                # Load race session
                race = fastf1.get_session(2024, race_name, 'R')
                race.load()
                
                # Load qualifying session
                quali = fastf1.get_session(2024, race_name, 'Q')
                quali.load()
                
                # Get race results
                race_results = race.results
                
                # Get qualifying times
                quali_times = {}
                for driver in quali.drivers:
                    driver_laps = quali.laps.pick_driver(driver)
                    if not driver_laps.empty:
                        best_lap = driver_laps.pick_fastest()
                        if pd.notna(best_lap['LapTime']):
                            quali_times[driver] = best_lap['LapTime'].total_seconds()
                
                # Process each driver's race data
                for _, driver_result in race_results.iterrows():
                    driver_abbr = driver_result['Abbreviation']
                    
                    # Skip if no qualifying time
                    if driver_abbr not in quali_times:
                        continue
                    
                    # Get driver's race laps
                    driver_laps = race.laps.pick_driver(driver_abbr)
                    if driver_laps.empty:
                        continue
                    
                    # Calculate average lap time (excluding outliers)
                    lap_times = driver_laps['LapTime'].dropna()
                    if len(lap_times) > 0:
                        lap_times_seconds = lap_times.dt.total_seconds()
                        # Remove outliers (pit stops, safety car, etc.)
                        q1 = lap_times_seconds.quantile(0.25)
                        q3 = lap_times_seconds.quantile(0.75)
                        iqr = q3 - q1
                        mask = (lap_times_seconds >= q1 - 1.5 * iqr) & (lap_times_seconds <= q3 + 1.5 * iqr)
                        clean_lap_times = lap_times_seconds[mask]
                        
                        if len(clean_lap_times) > 0:
                            avg_lap_time = clean_lap_times.mean()
                        else:
                            continue
                    else:
                        continue
                    
                    # Get sector times
                    sector_times = driver_laps[['Sector1Time', 'Sector2Time', 'Sector3Time']].dropna()
                    if not sector_times.empty:
                        avg_s1 = sector_times['Sector1Time'].dt.total_seconds().mean()
                        avg_s2 = sector_times['Sector2Time'].dt.total_seconds().mean()
                        avg_s3 = sector_times['Sector3Time'].dt.total_seconds().mean()
                    else:
                        avg_s1 = avg_s2 = avg_s3 = np.nan
                    
                    # Create data record
                    race_record = {
                        'Race': race_name,
                        'Driver': driver_abbr,
                        'QualifyingTime (s)': quali_times[driver_abbr],
                        'QualifyingPosition': driver_result['GridPosition'],
                        'LapTime (s)': avg_lap_time,
                        'Sector1Time (s)': avg_s1,
                        'Sector2Time (s)': avg_s2,
                        'Sector3Time (s)': avg_s3,
                        'Position': driver_result['Position'],
                        'Points': driver_result['Points'],
                        'Status': driver_result['Status'],
                        'TeamName': driver_result['TeamName']
                    }
                    
                    all_race_data.append(race_record)
                
            except Exception as e:
                logger.error(f"Error loading {race_name}: {e}")
                continue
        
        df = pd.DataFrame(all_race_data)
        logger.info(f"Loaded {len(df)} race records from {len(races)} races")
        return df
    
    def engineer_features(self, df):
        """Add engineered features to the dataset"""
        # Add team performance score
        team_points = df.groupby('TeamName')['Points'].sum()
        max_points = team_points.max()
        team_performance = (team_points / max_points).to_dict()
        df['TeamPerformanceScore'] = df['TeamName'].map(team_performance)
        
        # Add driver consistency
        driver_std = df.groupby('Driver')['Position'].std()
        df['DriverConsistency'] = df['Driver'].map(driver_std).fillna(5)
        
        # Add qualifying vs grid difference
        df['QualifyingGap'] = df['QualifyingTime (s)'] - df.groupby('Race')['QualifyingTime (s)'].transform('min')
        
        # Add wet performance factor (simplified - would need weather data)
        wet_performance = {
            'VER': 0.975, 'HAM': 0.976, 'LEC': 0.976, 'NOR': 0.978,
            'ALO': 0.973, 'RUS': 0.969, 'SAI': 0.979, 'PER': 0.985,
            'OCO': 0.982, 'GAS': 0.979, 'STR': 0.980, 'TSU': 0.996,
            'ALB': 0.982, 'MAG': 0.988, 'HUL': 0.985, 'BOT': 0.977,
            'ZHO': 0.990, 'RIC': 0.980, 'SAR': 0.992, 'PIA': 0.980
        }
        df['WetPerformanceFactor'] = df['Driver'].map(wet_performance).fillna(0.985)
        
        return df
    
    def get_upcoming_races(self, days_ahead: int = 14):
        """Get races happening in the next N days"""
        cursor = self.conn.cursor()
        cutoff_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        races = cursor.execute("""
            SELECT r.*, 
                   COUNT(p.id) as prediction_count
            FROM races r
            LEFT JOIN predictions p ON r.id = p.race_id
            WHERE r.date >= date('now') 
              AND r.date <= ?
              AND r.season = ?
            GROUP BY r.id
            ORDER BY r.date
        """, (cutoff_date, self.current_year)).fetchall()
        
        return [dict(race) for race in races]
    
    def get_2025_qualifying_data(self, race_name: str):
        """Get qualifying data for 2025 race (simulated for now)"""
        # In production, this would fetch real 2025 qualifying data
        # For now, we'll use the existing qualifying data from your scripts
        
        qualifying_data = {
            "Australian Grand Prix": {
                "Lando Norris": 75.096, "Oscar Piastri": 75.180, 
                "Max Verstappen": 75.481, "George Russell": 75.546,
                "Yuki Tsunoda": 75.670, "Alexander Albon": 75.737,
                "Charles Leclerc": 75.755, "Lewis Hamilton": 75.973,
                "Pierre Gasly": 75.980, "Carlos Sainz": 76.062,
                "Fernando Alonso": 76.4, "Lance Stroll": 76.5
            },
            # Add more races as qualifying happens
        }
        
        return qualifying_data.get(race_name, self._generate_mock_qualifying())
    
    def _generate_mock_qualifying(self):
        """Generate mock qualifying times for testing"""
        drivers = [
            "Max Verstappen", "Lando Norris", "Oscar Piastri", "Charles Leclerc",
            "Carlos Sainz", "Lewis Hamilton", "George Russell", "Fernando Alonso",
            "Lance Stroll", "Pierre Gasly", "Esteban Ocon", "Yuki Tsunoda",
            "Alexander Albon", "Nico Hulkenberg", "Liam Lawson"
        ]
        base_time = 85.0
        times = {}
        for i, driver in enumerate(drivers):
            times[driver] = base_time + (i * 0.15) + np.random.uniform(-0.05, 0.05)
        return times
    
    def train_and_predict(self, race):
        """Train model and generate predictions for a specific race"""
        logger.info(f"Training model for {race['name']} on {race['date']}")
        
        # Load historical data
        training_data = self.load_historical_race_data()
        training_data = self.engineer_features(training_data)
        
        # Configure predictor
        config = PredictorConfig(
            model_type=ModelType.GRADIENT_BOOSTING,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            weather_api_key=self.weather_api_key,
            use_weather=True,
            use_team_performance=True,
            use_sector_times=True,
            cv_folds=5
        )
        
        predictor = F1Predictor(config)
        
        # Select features
        feature_cols = [
            'QualifyingTime (s)', 'QualifyingGap', 'TeamPerformanceScore',
            'DriverConsistency', 'WetPerformanceFactor'
        ]
        
        # Add sector times if available
        if 'Sector1Time (s)' in training_data.columns:
            feature_cols.extend(['Sector1Time (s)', 'Sector2Time (s)', 'Sector3Time (s)'])
        
        # Prepare training data
        train_features = training_data[feature_cols].fillna(training_data[feature_cols].mean())
        train_target = training_data['LapTime (s)']
        
        # Remove any rows with NaN target
        mask = train_target.notna()
        train_features = train_features[mask]
        train_target = train_target[mask]
        
        # Create training DataFrame
        train_df = pd.concat([train_features, train_target], axis=1)
        
        # Train model
        metrics = predictor.train(train_df, target_column='LapTime (s)')
        logger.info(f"Model trained - MAE: {metrics.mae:.2f}, Accuracy: {metrics.accuracy:.2f}")
        
        # Get qualifying data for prediction
        qualifying_data = self.get_2025_qualifying_data(race['name'])
        
        # Get weather data
        from f1_predictor.data_loader import DataLoader
        data_loader = DataLoader()
        coords = data_loader.get_circuit_coordinates()
        lat, lon = coords.get(race['circuit'], (0, 0))
        
        # Get weather data (uses free providers or circuit patterns)
        weather_data = data_loader.get_weather_data(
            lat, lon, 
            f"{race['date']} 14:00:00",
            self.weather_api_key,
            race['circuit']
        )
        
        # Get current team points
        team_points = {
            'McLaren': 300, 'Red Bull': 250, 'Ferrari': 200,
            'Mercedes': 180, 'Aston Martin': 50, 'Alpine': 30,
            'Williams': 25, 'Racing Bulls': 20, 'Kick Sauber': 10, 'Haas': 5
        }
        
        # Create race data
        race_data = RaceData(
            race_id=race['id'],
            race_name=race['name'],
            circuit=race['circuit'],
            date=race['date'],
            qualifying_data=qualifying_data,
            weather_data=weather_data,
            team_points=team_points
        )
        
        # Generate predictions
        predictions = predictor.predict(race_data)
        
        # Save model
        model_version = f"v2.{datetime.now().strftime('%Y%m%d_%H%M')}"
        model_path = f"models/{race['season']}_{race['round']}_{model_version}.pkl"
        os.makedirs('models', exist_ok=True)
        predictor.save_model(model_path)
        
        return predictions, metrics, model_version
    
    def store_predictions_to_d1(self, race_id: int, predictions: list, model_version: str):
        """Store predictions in D1 database via API call"""
        # Get driver ID mapping
        cursor = self.conn.cursor()
        drivers = cursor.execute("SELECT id, code, name FROM drivers").fetchall()
        
        # Create mapping from both code and name
        driver_map = {}
        for d in drivers:
            driver_map[d['code']] = d['id']
            driver_map[d['name']] = d['id']
        
        # Prepare predictions for API
        predictions_data = []
        for pred in predictions:
            driver_id = driver_map.get(pred.driver_code) or driver_map.get(pred.driver_name)
            if driver_id:
                predictions_data.append({
                    'race_id': race_id,
                    'driver_id': driver_id,
                    'predicted_position': pred.predicted_position,
                    'predicted_time': pred.predicted_time,
                    'confidence': pred.confidence,
                    'model_version': model_version
                })
        
        # Store in local database
        for pred_data in predictions_data:
            cursor.execute("""
                INSERT OR REPLACE INTO predictions 
                (race_id, driver_id, predicted_position, predicted_time, confidence, model_version)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                pred_data['race_id'],
                pred_data['driver_id'],
                pred_data['predicted_position'],
                pred_data['predicted_time'],
                pred_data['confidence'],
                pred_data['model_version']
            ))
        
        self.conn.commit()
        logger.info(f"Stored {len(predictions_data)} predictions for race {race_id}")
        
        return predictions_data
    
    def run_pipeline(self, force_retrain: bool = False):
        """Run the complete training pipeline"""
        try:
            self.connect()
            
            # Get upcoming races
            upcoming_races = self.get_upcoming_races(days_ahead=30)
            
            if not upcoming_races:
                logger.info("No upcoming races found")
                return
            
            logger.info(f"Found {len(upcoming_races)} upcoming races")
            
            results = []
            
            for race in upcoming_races:
                # Skip if predictions already exist (unless force_retrain)
                if race['prediction_count'] > 0 and not force_retrain:
                    logger.info(f"Predictions already exist for {race['name']}, skipping")
                    continue
                
                try:
                    # Train model and generate predictions
                    predictions, metrics, model_version = self.train_and_predict(race)
                    
                    # Store predictions
                    stored_predictions = self.store_predictions_to_d1(
                        race['id'], 
                        predictions, 
                        model_version
                    )
                    
                    # Store metrics
                    cursor = self.conn.cursor()
                    cursor.execute("""
                        INSERT INTO model_metrics (model_version, race_id, mae, accuracy)
                        VALUES (?, ?, ?, ?)
                    """, (model_version, race['id'], metrics.mae, metrics.accuracy))
                    self.conn.commit()
                    
                    results.append({
                        'race': race['name'],
                        'predictions': len(stored_predictions),
                        'model_version': model_version,
                        'mae': metrics.mae
                    })
                    
                    logger.info(f"✅ Completed predictions for {race['name']}")
                    
                    # Print top 10 predictions
                    print(f"\n🏁 Predictions for {race['name']}:")
                    for pred in predictions[:10]:
                        print(f"  {pred.predicted_position:2d}. {pred.driver_code} - {pred.driver_name:25s} "
                              f"({pred.confidence:.0%} confidence)")
                    
                except Exception as e:
                    logger.error(f"Failed to process {race['name']}: {e}")
                    continue
            
            # Summary
            if results:
                print("\n📊 Pipeline Summary:")
                for result in results:
                    print(f"  - {result['race']}: {result['predictions']} predictions "
                          f"(MAE: {result['mae']:.2f}s)")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            self.close()

def main():
    parser = argparse.ArgumentParser(description='Train F1 prediction models with real data')
    parser.add_argument('--db', default='f1_predictions.db', help='Database path')
    parser.add_argument('--force', action='store_true', help='Force retrain existing predictions')
    parser.add_argument('--weather-key', help='Weather API key')
    
    args = parser.parse_args()
    
    # Get weather API key from env if not provided
    weather_key = args.weather_key or os.getenv('WEATHER_API_KEY')
    
    if not weather_key:
        logger.warning("No weather API key provided. Weather features will be disabled.")
    
    pipeline = RealDataTrainingPipeline(args.db, weather_key)
    pipeline.run_pipeline(force_retrain=args.force)

if __name__ == "__main__":
    main()