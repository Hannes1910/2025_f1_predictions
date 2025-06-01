#!/usr/bin/env python3
"""
F1 ML Production Service - CLEAN VERSION
- No mock data anywhere
- Real F1 data only
- Proper error handling
- Clear validation
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import joblib
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Clean prediction result structure"""
    race_id: int
    driver_id: int
    driver_code: str
    driver_name: str
    team_name: str
    predicted_position: float
    confidence: float
    model_version: str
    data_source: str
    created_at: str

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_version: str
    accuracy: float
    mae: float
    features_used: List[str]
    training_data_size: int
    last_trained: str

class D1DataClientClean:
    """Clean D1 client - no fallbacks to mock data"""
    
    def __init__(self, worker_url: str, api_key: Optional[str] = None):
        self.worker_url = worker_url.rstrip('/')
        self.api_key = api_key
        self.headers = {'Content-Type': 'application/json'}
        if api_key:
            self.headers['X-API-Key'] = api_key
    
    def get_ml_prediction_data(self, race_id: int) -> Optional[Dict]:
        """Get ML prediction data - returns None if insufficient real data"""
        try:
            response = requests.post(
                f"{self.worker_url}/api/data/ml-prediction-data",
                json={"raceId": race_id},
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 404:
                logger.warning(f"Race {race_id} not found")
                return None
            elif response.status_code == 422:
                logger.warning(f"Insufficient real data for race {race_id}")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Validate data sufficiency
            meta = data.get('meta', {})
            sufficiency = meta.get('data_sufficiency', {})
            
            if sufficiency.get('drivers') != 'sufficient':
                logger.warning(f"Insufficient driver data for race {race_id}")
                return None
            
            return data['data']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get ML data for race {race_id}: {e}")
            return None
    
    def get_race_features(self, race_id: int) -> Optional[Dict]:
        """Get race features - returns None if insufficient real data"""
        try:
            response = requests.get(
                f"{self.worker_url}/api/data/race/{race_id}/features",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code in [404, 422]:
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Validate we have sufficient data
            race_data = data['data']
            if len(race_data['qualifying_results']) == 0:
                logger.warning(f"No qualifying data for race {race_id}")
                return None
            
            return race_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get race features for {race_id}: {e}")
            return None


class F1PredictorClean:
    """Clean F1 predictor - no mock data, real models only"""
    
    def __init__(self):
        self.model_version = "clean_v1.0"
        self.models = self.load_trained_models()
        self.feature_columns = self.load_feature_config()
    
    def load_trained_models(self) -> Optional[Dict]:
        """Load trained models - no fallbacks"""
        models_dir = Path("models/production")
        
        if not models_dir.exists():
            logger.warning("No trained models found - predictions unavailable")
            return None
        
        try:
            models = {}
            
            # Load ensemble model if available
            ensemble_path = models_dir / "ensemble_model.pkl"
            if ensemble_path.exists():
                models['ensemble'] = joblib.load(ensemble_path)
                logger.info("✅ Loaded ensemble model")
            
            # Load individual models
            for model_file in models_dir.glob("*.pkl"):
                if model_file.name != "ensemble_model.pkl":
                    model_name = model_file.stem
                    models[model_name] = joblib.load(model_file)
                    logger.info(f"✅ Loaded {model_name} model")
            
            return models if models else None
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return None
    
    def load_feature_config(self) -> List[str]:
        """Load feature configuration"""
        config_path = Path("models/production/feature_config.json")
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('features', [])
            except Exception as e:
                logger.error(f"Failed to load feature config: {e}")
        
        # Essential features for F1 prediction
        return [
            'grid_position',
            'qualifying_time_diff_ms',
            'driver_recent_form',
            'season_points',
            'team_performance',
            'circuit_experience'
        ]
    
    def extract_features(self, ml_data: Dict) -> Optional[pd.DataFrame]:
        """Extract features from real ML data"""
        try:
            race = ml_data['race']
            drivers = ml_data['drivers']
            qualifying_results = ml_data.get('qualifying_results', [])
            
            if not drivers:
                logger.error("No driver data available")
                return None
            
            # Create qualifying lookup
            qualifying_lookup = {qr['driver_id']: qr for qr in qualifying_results}
            
            features_list = []
            
            for driver in drivers:
                driver_id = driver['id']
                qualifying = qualifying_lookup.get(driver_id)
                
                # Skip if no qualifying data (essential for predictions)
                if not qualifying:
                    logger.warning(f"No qualifying data for driver {driver['code']}")
                    continue
                
                features = {
                    'driver_id': driver_id,
                    'driver_code': driver['code'],
                    'driver_name': f"{driver['forename']} {driver['surname']}",
                    'team_name': driver.get('team_name', 'Unknown'),
                    
                    # Grid position (most important feature)
                    'grid_position': qualifying['grid_position'],
                    
                    # Qualifying time difference from pole (milliseconds)
                    'qualifying_time_diff_ms': self._calculate_time_diff(
                        qualifying['best_time_ms'], qualifying_results
                    ),
                    
                    # Driver recent form
                    'driver_recent_form': driver.get('avg_recent_position') or 15.0,
                    
                    # Season points
                    'season_points': driver.get('season_points', 0),
                    
                    # Team performance (average of teammates)
                    'team_performance': self._calculate_team_performance(
                        driver['team_name'], drivers
                    ),
                    
                    # Circuit experience (simplified)
                    'circuit_experience': 1.0  # Would need historical circuit data
                }
                
                features_list.append(features)
            
            if not features_list:
                logger.error("No valid feature data extracted")
                return None
            
            return pd.DataFrame(features_list)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def _calculate_time_diff(self, lap_time_ms: int, qualifying_results: List[Dict]) -> float:
        """Calculate time difference from pole position"""
        if not qualifying_results:
            return 0.0
        
        pole_time = min(qr['best_time_ms'] for qr in qualifying_results)
        return lap_time_ms - pole_time
    
    def _calculate_team_performance(self, team_name: str, drivers: List[Dict]) -> float:
        """Calculate team performance average"""
        teammates = [d for d in drivers if d.get('team_name') == team_name]
        if not teammates:
            return 15.0
        
        recent_forms = [d.get('avg_recent_position') for d in teammates if d.get('avg_recent_position')]
        return np.mean(recent_forms) if recent_forms else 15.0
    
    def predict(self, features_df: pd.DataFrame) -> Optional[List[PredictionResult]]:
        """Make predictions using real features and trained models"""
        
        if features_df.empty:
            logger.error("No features provided for prediction")
            return None
        
        if not self.models:
            logger.error("No trained models available")
            return None
        
        try:
            # Prepare feature matrix
            X = features_df[self.feature_columns].fillna(0)
            
            # Use ensemble model if available, otherwise best available model
            model_name = 'ensemble' if 'ensemble' in self.models else list(self.models.keys())[0]
            model = self.models[model_name]
            
            # Make predictions
            predicted_positions = model.predict(X)
            
            # Calculate confidence based on feature quality
            confidence_scores = self._calculate_confidence(features_df, predicted_positions)
            
            # Create prediction results
            results = []
            current_time = datetime.now().isoformat()
            
            for i, (_, row) in enumerate(features_df.iterrows()):
                result = PredictionResult(
                    race_id=0,  # Will be set by caller
                    driver_id=int(row['driver_id']),
                    driver_code=row['driver_code'],
                    driver_name=row['driver_name'],
                    team_name=row['team_name'],
                    predicted_position=float(predicted_positions[i]),
                    confidence=float(confidence_scores[i]),
                    model_version=f"{self.model_version}_{model_name}",
                    data_source="real_f1_data",
                    created_at=current_time
                )
                results.append(result)
            
            # Sort by predicted position
            results.sort(key=lambda x: x.predicted_position)
            
            logger.info(f"✅ Generated {len(results)} predictions using {model_name} model")
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def _calculate_confidence(self, features_df: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
        """Calculate prediction confidence based on data quality"""
        base_confidence = 0.85  # Base confidence for real data
        
        confidences = np.full(len(features_df), base_confidence)
        
        # Adjust based on data quality
        for i, (_, row) in enumerate(features_df.iterrows()):
            # Higher confidence for better grid positions
            if row['grid_position'] <= 3:
                confidences[i] += 0.1
            elif row['grid_position'] >= 15:
                confidences[i] -= 0.1
            
            # Adjust for recent form quality
            if pd.notna(row['driver_recent_form']) and row['driver_recent_form'] < 20:
                confidences[i] += 0.05
            
            # Ensure confidence bounds
            confidences[i] = np.clip(confidences[i], 0.5, 0.95)
        
        return confidences


class MLServiceProductionClean:
    """Clean ML service - no mock data, real predictions only"""
    
    def __init__(self):
        # Configuration
        self.worker_url = os.getenv('WORKER_URL', 'https://f1-predictions-api.vprifntqe.workers.dev')
        self.api_key = os.getenv('PREDICTIONS_API_KEY')
        
        # Initialize components
        self.app = FastAPI(
            title="F1 ML Predictions - Production Clean",
            version="1.0.0",
            description="Real F1 predictions with no mock data"
        )
        self.d1_client = D1DataClientClean(self.worker_url, self.api_key)
        self.predictor = F1PredictorClean()
        
        self.setup_middleware()
        self.setup_routes()
    
    def setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "F1 ML Predictions - Production Clean",
                "version": "1.0.0",
                "models_available": self.predictor.models is not None,
                "model_count": len(self.predictor.models) if self.predictor.models else 0,
                "data_source": "real_f1_data_only",
                "worker_url": self.worker_url,
                "features": self.predictor.feature_columns
            }
        
        @self.app.post("/predict/race/{race_id}")
        async def predict_race(race_id: int):
            """Generate predictions for a race using real data only"""
            
            # Validate race ID
            if race_id <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid race ID"
                )
            
            # Get real ML data
            ml_data = self.d1_client.get_ml_prediction_data(race_id)
            if not ml_data:
                raise HTTPException(
                    status_code=422,
                    detail=f"Insufficient real data for race {race_id}. Predictions require real qualifying data."
                )
            
            # Extract features
            features_df = self.predictor.extract_features(ml_data)
            if features_df is None or features_df.empty:
                raise HTTPException(
                    status_code=422,
                    detail="Failed to extract sufficient features from real data"
                )
            
            # Generate predictions
            predictions = self.predictor.predict(features_df)
            if not predictions:
                raise HTTPException(
                    status_code=503,
                    detail="Prediction service unavailable - no trained models"
                )
            
            # Set race ID for all predictions
            for pred in predictions:
                pred.race_id = race_id
            
            return {
                "race_id": race_id,
                "race_info": {
                    "name": ml_data['race']['name'],
                    "date": ml_data['race']['date'],
                    "circuit": ml_data['race']['circuit']
                },
                "predictions": [asdict(pred) for pred in predictions],
                "metadata": {
                    "model_version": predictions[0].model_version,
                    "predictions_count": len(predictions),
                    "data_source": "real_f1_data",
                    "features_used": self.predictor.feature_columns,
                    "generated_at": datetime.now().isoformat()
                }
            }
        
        @self.app.get("/predict/next")
        async def predict_next_race():
            """Predict the next upcoming race"""
            
            # Find next race (simplified - would need race calendar logic)
            next_race_id = 9  # Spanish GP 2025 - current race
            
            try:
                # Use the race prediction endpoint
                return await predict_race(next_race_id)
            except HTTPException as e:
                if e.status_code == 422:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Cannot predict next race: {e.detail}"
                    )
                raise
        
        @self.app.get("/health")
        async def health():
            """Health check with data availability"""
            
            # Test data connection
            try:
                test_data = self.d1_client.get_ml_prediction_data(1)
                data_available = test_data is not None
            except:
                data_available = False
            
            status = "healthy" if (self.predictor.models and data_available) else "degraded"
            
            return {
                "status": status,
                "models_loaded": self.predictor.models is not None,
                "model_count": len(self.predictor.models) if self.predictor.models else 0,
                "data_connection": data_available,
                "worker_url": self.worker_url,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/models/status")
        async def models_status():
            """Get model status and metrics"""
            
            if not self.predictor.models:
                return {
                    "available": False,
                    "message": "No trained models available",
                    "suggestion": "Train models using real F1 data first"
                }
            
            return {
                "available": True,
                "models": list(self.predictor.models.keys()),
                "model_version": self.predictor.model_version,
                "features": self.predictor.feature_columns,
                "feature_count": len(self.predictor.feature_columns)
            }


def main():
    """Run the clean ML service"""
    logger.info("🚀 Starting F1 ML Production Service - CLEAN VERSION")
    logger.info("✅ NO MOCK DATA - Real F1 data only")
    
    service = MLServiceProductionClean()
    
    # Log configuration
    logger.info(f"Worker URL: {service.worker_url}")
    logger.info(f"Models available: {service.predictor.models is not None}")
    logger.info(f"Features: {len(service.predictor.feature_columns)}")
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )


if __name__ == "__main__":
    main()