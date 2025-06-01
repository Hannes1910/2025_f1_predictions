#!/usr/bin/env python3
"""
D1 Data Client
Accesses Cloudflare D1 database through Worker API
(D1 cannot be accessed directly from external services)
"""

import requests
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class D1DataClient:
    """Client to access D1 database through Worker API proxy"""
    
    def __init__(self, worker_url: str, api_key: Optional[str] = None):
        self.worker_url = worker_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json'
        }
        if api_key:
            self.headers['X-API-Key'] = api_key
    
    def get_driver_stats(self, driver_id: int) -> Dict:
        """Get driver statistics from D1"""
        try:
            response = requests.get(
                f"{self.worker_url}/api/data/driver/{driver_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get driver stats: {e}")
            return {}
    
    def get_team_stats(self, team: str) -> Dict:
        """Get team statistics from D1"""
        try:
            response = requests.get(
                f"{self.worker_url}/api/data/team/{team}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get team stats: {e}")
            return {}
    
    def get_race_features(self, race_id: int) -> Dict:
        """Get all features for a race from D1"""
        try:
            response = requests.get(
                f"{self.worker_url}/api/data/race/{race_id}/features",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get race features: {e}")
            return {}
    
    def get_circuit_patterns(self, circuit: str) -> Dict:
        """Get historical patterns for a circuit"""
        try:
            response = requests.get(
                f"{self.worker_url}/api/data/circuit/{circuit}/patterns",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get circuit patterns: {e}")
            return {}
    
    def get_ml_prediction_data(self, race_id: int) -> Dict:
        """Get all data needed for ML prediction in one call"""
        try:
            response = requests.post(
                f"{self.worker_url}/api/data/ml-prediction-data",
                json={"raceId": race_id},
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get ML prediction data: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    # Test the client
    client = D1DataClient(
        worker_url="https://f1-predictions-api.vprifntqe.workers.dev",
        api_key="your-api-key-here"
    )
    
    # Get driver stats
    print("Testing driver stats...")
    driver_stats = client.get_driver_stats(1)  # Verstappen
    print(f"Driver stats: {driver_stats}")
    
    # Get race features
    print("\nTesting race features...")
    race_features = client.get_race_features(4)  # China GP
    print(f"Race features: {race_features}")
    
    print("\n✅ D1 Data Client ready for ML service integration!")