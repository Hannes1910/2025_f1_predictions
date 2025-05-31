#!/usr/bin/env python3
"""
Model training pipeline for F1 predictions
Trains models for upcoming races and stores predictions in the database
"""

import sys
import os
import json
import sqlite3
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from packages.ml_core.f1_predictor import F1Predictor, PredictorConfig, RaceData, ModelType

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, db_path: str, weather_api_key: str = None):
        self.db_path = db_path
        self.weather_api_key = weather_api_key or os.getenv('WEATHER_API_KEY')
        self.conn = None
        
    def connect(self):
        """Connect to the database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            
    def get_upcoming_races(self, days_ahead: int = 7) -> list:
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
            GROUP BY r.id
            ORDER BY r.date
        """, (cutoff_date,)).fetchall()
        
        return [dict(race) for race in races]
    
    def get_historical_data(self, season: int = 2024) -> pd.DataFrame:
        """Get historical race data for training"""
        cursor = self.conn.cursor()
        
        # For now, we'll create synthetic training data
        # In production, this would fetch from actual race results
        
        # Get all drivers
        drivers = cursor.execute("SELECT * FROM drivers").fetchall()
        
        # Create synthetic historical data
        historical_data = []
        
        for race_num in range(1, 10):  # Simulate 9 races
            for driver in drivers:
                # Generate realistic lap times based on driver/team performance
                base_time = 90.0  # Base lap time
                
                # Team performance factor
                team_factors = {
                    'Red Bull': -1.0, 'McLaren': -0.8, 'Ferrari': -0.5,
                    'Mercedes': -0.3, 'Aston Martin': 0.2, 'Alpine': 0.5,
                    'Williams': 0.8, 'Racing Bulls': 0.6, 'Kick Sauber': 1.0,
                    'Haas': 0.9
                }
                
                team_factor = team_factors.get(driver['team'], 0.5)
                
                # Add some randomness
                variation = np.random.normal(0, 0.5)
                
                lap_time = base_time + team_factor + variation
                
                historical_data.append({
                    'Driver': driver['code'],
                    'LapTime (s)': lap_time,
                    'QualifyingTime (s)': lap_time - np.random.uniform(0.5, 1.5),
                    'Sector1Time (s)': lap_time * 0.3 + np.random.normal(0, 0.1),
                    'Sector2Time (s)': lap_time * 0.35 + np.random.normal(0, 0.1),
                    'Sector3Time (s)': lap_time * 0.35 + np.random.normal(0, 0.1),
                    'Temperature': np.random.uniform(15, 35),
                    'RainProbability': np.random.choice([0, 0.1, 0.3, 0.8], p=[0.7, 0.15, 0.1, 0.05]),
                    'TeamPerformanceScore': 1.0 - (team_factor + 1.0) / 2.0,
                    'SeasonPoints': np.random.randint(0, 100)
                })
        
        return pd.DataFrame(historical_data)
    
    def get_qualifying_data(self, race_id: int) -> dict:
        """Get qualifying data for a specific race"""
        # In production, this would fetch actual qualifying results
        # For now, we'll generate realistic qualifying times
        
        cursor = self.conn.cursor()
        drivers = cursor.execute("SELECT * FROM drivers").fetchall()
        
        qualifying_data = {}
        base_time = 75.0  # Base qualifying time
        
        for i, driver in enumerate(drivers):
            # Generate realistic qualifying times with some spread
            time = base_time + (i * 0.15) + np.random.uniform(-0.1, 0.1)
            qualifying_data[driver['name']] = time
            
        return qualifying_data
    
    def train_model_for_race(self, race: dict) -> dict:
        """Train a model for a specific race and generate predictions"""
        logger.info(f"Training model for {race['name']} on {race['date']}")
        
        # Configure predictor
        config = PredictorConfig(
            model_type=ModelType.GRADIENT_BOOSTING,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            weather_api_key=self.weather_api_key,
            use_weather=True,
            use_team_performance=True,
            use_sector_times=True
        )
        
        predictor = F1Predictor(config)
        
        # Get training data
        training_data = self.get_historical_data()
        
        # Train model
        metrics = predictor.train(training_data)
        logger.info(f"Model trained - MAE: {metrics.mae:.2f}, Accuracy: {metrics.accuracy:.2f}")
        
        # Get qualifying data for prediction
        qualifying_data = self.get_qualifying_data(race['id'])
        
        # Get weather data (mock for now)
        weather_data = {
            'temperature': np.random.uniform(15, 35),
            'rain_probability': np.random.choice([0, 0.1, 0.3, 0.8], p=[0.7, 0.15, 0.1, 0.05])
        }
        
        # Get current team points
        team_points = {
            'McLaren': 279, 'Mercedes': 147, 'Red Bull': 131, 
            'Williams': 51, 'Ferrari': 114, 'Haas': 20, 
            'Aston Martin': 14, 'Kick Sauber': 6, 
            'Racing Bulls': 10, 'Alpine': 7
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
        model_path = f"models/{race['season']}_{race['round']}_{race['circuit']}.pkl"
        os.makedirs('models', exist_ok=True)
        predictor.save_model(model_path)
        
        return {
            'race': race,
            'predictions': predictions,
            'metrics': metrics,
            'model_path': model_path
        }
    
    def store_predictions(self, race_id: int, predictions: list, model_version: str):
        """Store predictions in the database"""
        cursor = self.conn.cursor()
        
        # Get driver ID mapping
        drivers = cursor.execute("SELECT id, code FROM drivers").fetchall()
        driver_map = {d['code']: d['id'] for d in drivers}
        
        # Insert predictions
        for pred in predictions:
            driver_id = driver_map.get(pred.driver_code)
            if driver_id:
                cursor.execute("""
                    INSERT OR REPLACE INTO predictions 
                    (race_id, driver_id, predicted_position, predicted_time, confidence, model_version)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    race_id,
                    driver_id,
                    pred.predicted_position,
                    pred.predicted_time,
                    pred.confidence,
                    model_version
                ))
        
        self.conn.commit()
        logger.info(f"Stored {len(predictions)} predictions for race {race_id}")
    
    def store_metrics(self, model_version: str, race_id: int, metrics):
        """Store model metrics in the database"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_metrics (model_version, race_id, mae, accuracy)
            VALUES (?, ?, ?, ?)
        """, (model_version, race_id, metrics.mae, metrics.accuracy))
        
        self.conn.commit()
        logger.info(f"Stored metrics for model {model_version}")
    
    def run_pipeline(self, force_retrain: bool = False):
        """Run the complete training pipeline"""
        try:
            self.connect()
            
            # Get upcoming races
            upcoming_races = self.get_upcoming_races(days_ahead=14)
            
            if not upcoming_races:
                logger.info("No upcoming races found")
                return
            
            logger.info(f"Found {len(upcoming_races)} upcoming races")
            
            for race in upcoming_races:
                # Skip if predictions already exist (unless force_retrain)
                if race['prediction_count'] > 0 and not force_retrain:
                    logger.info(f"Predictions already exist for {race['name']}, skipping")
                    continue
                
                # Train model and generate predictions
                result = self.train_model_for_race(race)
                
                # Generate model version
                model_version = f"v1.0.{datetime.now().strftime('%Y%m%d%H%M')}"
                
                # Store predictions
                self.store_predictions(
                    race['id'], 
                    result['predictions'], 
                    model_version
                )
                
                # Store metrics
                self.store_metrics(
                    model_version,
                    race['id'],
                    result['metrics']
                )
                
                # Export predictions to JSON
                predictions_data = [{
                    'position': p.predicted_position,
                    'driver': p.driver_name,
                    'driver_code': p.driver_code,
                    'predicted_time': round(p.predicted_time, 2),
                    'confidence': round(p.confidence, 2)
                } for p in result['predictions']]
                
                output_file = f"predictions/{race['season']}_{race['round']}_{race['circuit']}.json"
                os.makedirs('predictions', exist_ok=True)
                
                with open(output_file, 'w') as f:
                    json.dump({
                        'race': race,
                        'predictions': predictions_data,
                        'model_version': model_version,
                        'generated_at': datetime.now().isoformat()
                    }, f, indent=2)
                
                logger.info(f"✅ Completed predictions for {race['name']}")
                print(f"\n🏁 Predictions for {race['name']}:")
                for p in predictions_data[:10]:  # Show top 10
                    print(f"  {p['position']:2d}. {p['driver_code']} - {p['driver']:25s} ({p['confidence']:.0%} confidence)")
                
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            self.close()

def main():
    parser = argparse.ArgumentParser(description='Train F1 prediction models')
    parser.add_argument('--db', default='f1_predictions.db', help='Database path')
    parser.add_argument('--force', action='store_true', help='Force retrain existing predictions')
    parser.add_argument('--weather-key', help='Weather API key')
    
    args = parser.parse_args()
    
    pipeline = TrainingPipeline(args.db, args.weather_key)
    pipeline.run_pipeline(force_retrain=args.force)

if __name__ == "__main__":
    main()