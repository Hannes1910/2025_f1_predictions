#!/usr/bin/env python3
"""
Sync predictions from local database to Cloudflare Worker API
"""

import sqlite3
import requests
import json
import argparse
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictionSync:
    def __init__(self, db_path: str, api_url: str, api_key: str):
        self.db_path = db_path
        self.api_url = api_url
        self.api_key = api_key
        self.conn = None
    
    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
    
    def close(self):
        if self.conn:
            self.conn.close()
    
    def get_local_predictions(self, race_id: int):
        """Get predictions from local database"""
        cursor = self.conn.cursor()
        
        predictions = cursor.execute("""
            SELECT 
                p.driver_id,
                p.predicted_position,
                p.predicted_time,
                p.confidence,
                p.model_version
            FROM predictions p
            WHERE p.race_id = ?
            ORDER BY p.predicted_position
        """, (race_id,)).fetchall()
        
        return [dict(p) for p in predictions]
    
    def get_model_metrics(self, model_version: str, race_id: int):
        """Get model metrics for a specific version and race"""
        cursor = self.conn.cursor()
        
        metrics = cursor.execute("""
            SELECT mae, accuracy
            FROM model_metrics
            WHERE model_version = ? AND race_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (model_version, race_id)).fetchone()
        
        return dict(metrics) if metrics else None
    
    def sync_race_predictions(self, race_id: int):
        """Sync predictions for a specific race"""
        # Get local predictions
        predictions = self.get_local_predictions(race_id)
        
        if not predictions:
            logger.warning(f"No predictions found for race {race_id}")
            return False
        
        # Get model version from first prediction
        model_version = predictions[0]['model_version']
        
        # Get model metrics
        metrics = self.get_model_metrics(model_version, race_id)
        
        # Prepare payload
        payload = {
            'race_id': race_id,
            'predictions': predictions,
            'model_version': model_version
        }
        
        if metrics:
            payload['model_metrics'] = metrics
        
        # Send to API
        headers = {
            'Content-Type': 'application/json',
            'X-API-Key': self.api_key
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/api/admin/predictions",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Synced {result['predictions_stored']} predictions for race {race_id}")
                return True
            else:
                logger.error(f"Failed to sync predictions: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error syncing predictions: {e}")
            return False
    
    def sync_all_predictions(self):
        """Sync all predictions for upcoming races"""
        cursor = self.conn.cursor()
        
        # Get races with predictions
        races = cursor.execute("""
            SELECT DISTINCT r.id, r.name, COUNT(p.id) as prediction_count
            FROM races r
            JOIN predictions p ON r.id = p.race_id
            WHERE r.date >= date('now')
            GROUP BY r.id
            ORDER BY r.date
        """).fetchall()
        
        logger.info(f"Found {len(races)} races with predictions to sync")
        
        success_count = 0
        for race in races:
            logger.info(f"Syncing {race['name']} (ID: {race['id']}) - {race['prediction_count']} predictions")
            if self.sync_race_predictions(race['id']):
                success_count += 1
        
        logger.info(f"✅ Successfully synced {success_count}/{len(races)} races")
        return success_count == len(races)
    
    def run(self):
        """Run the sync process"""
        try:
            self.connect()
            return self.sync_all_predictions()
        finally:
            self.close()

def main():
    parser = argparse.ArgumentParser(description='Sync F1 predictions to Cloudflare')
    parser.add_argument('--db', default='f1_predictions.db', help='Local database path')
    parser.add_argument('--api-url', default='https://f1-predictions-api.vprifntqe.workers.dev', 
                        help='Worker API URL')
    parser.add_argument('--api-key', help='API key for authentication')
    parser.add_argument('--race-id', type=int, help='Sync specific race only')
    
    args = parser.parse_args()
    
    # Get API key from env if not provided
    api_key = args.api_key or os.getenv('PREDICTIONS_API_KEY')
    
    if not api_key:
        logger.error("API key required. Set PREDICTIONS_API_KEY env var or use --api-key")
        return 1
    
    syncer = PredictionSync(args.db, args.api_url, api_key)
    
    try:
        syncer.connect()
        
        if args.race_id:
            # Sync specific race
            success = syncer.sync_race_predictions(args.race_id)
        else:
            # Sync all races
            success = syncer.sync_all_predictions()
        
        return 0 if success else 1
        
    finally:
        syncer.close()

if __name__ == "__main__":
    exit(main())