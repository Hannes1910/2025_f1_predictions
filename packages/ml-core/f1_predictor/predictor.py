import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import logging
from datetime import datetime
import json

from .config import PredictorConfig, RaceData, Prediction, ModelMetrics, ModelType
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class F1Predictor:
    """Main F1 prediction model class"""
    
    def __init__(self, config: PredictorConfig):
        self.config = config
        self.model = None
        self.data_loader = DataLoader(config.cache_dir)
        self.feature_engineer = FeatureEngineer()
        self.feature_importance = {}
        self.scaler = None
        
    def _create_model(self):
        """Create model based on configuration"""
        if self.config.model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state
            )
        elif self.config.model_type == ModelType.RANDOM_FOREST:
            return RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def train(self, training_data: pd.DataFrame, target_column: str = "LapTime (s)") -> ModelMetrics:
        """Train model with cross-validation"""
        logger.info(f"Training {self.config.model_type.value} model with {len(training_data)} samples")
        
        # Prepare features and target
        feature_columns = [col for col in self.config.features if col in training_data.columns]
        X = training_data[feature_columns]
        y = training_data[target_column]
        
        # Remove any rows with missing values
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            raise ValueError("No valid training data after removing missing values")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X, y, cv=self.config.cv_folds, 
            scoring='neg_mean_absolute_error'
        )
        accuracy = -cv_scores.mean()
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(feature_columns, self.model.feature_importances_))
        
        logger.info(f"Model trained - MAE: {mae:.2f}, RMSE: {rmse:.2f}, CV Accuracy: {accuracy:.2f}")
        
        return ModelMetrics(
            mae=mae,
            rmse=rmse,
            accuracy=accuracy,
            feature_importance=self.feature_importance
        )
    
    def predict(self, race_data: RaceData) -> List[Prediction]:
        """Generate predictions for a race"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare prediction data
        drivers_data = []
        driver_mapping = self.data_loader.get_driver_mapping()
        
        for driver_name, qual_time in race_data.qualifying_data.items():
            driver_code = driver_mapping.get(driver_name, driver_name)
            
            row = {
                'Driver': driver_code,
                'DriverName': driver_name,
                'QualifyingTime (s)': qual_time
            }
            
            # Add weather data if available
            if race_data.weather_data and self.config.use_weather:
                row.update({
                    'Temperature': race_data.weather_data.get('temperature', 20),
                    'RainProbability': race_data.weather_data.get('rain_probability', 0)
                })
            
            # Add team points if available
            if race_data.team_points and self.config.use_team_performance:
                team_mapping = self.data_loader.get_team_mapping()
                team = team_mapping.get(driver_code)
                if team:
                    row['SeasonPoints'] = race_data.team_points.get(team, 0)
                    max_points = max(race_data.team_points.values()) if race_data.team_points else 1
                    row['TeamPerformanceScore'] = row['SeasonPoints'] / max_points
            
            drivers_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(drivers_data)
        
        # Engineer features
        df = self.feature_engineer.engineer_features(df, {
            'use_weather': self.config.use_weather,
            'use_team_performance': self.config.use_team_performance
        })
        
        # Select features for prediction
        feature_columns = [col for col in self.config.features if col in df.columns]
        X_pred = df[feature_columns].fillna(0)
        
        # Make predictions
        predicted_times = self.model.predict(X_pred)
        
        # Create prediction objects
        predictions = []
        df['PredictedTime'] = predicted_times
        df = df.sort_values('PredictedTime')
        
        for idx, (_, row) in enumerate(df.iterrows()):
            # Calculate confidence based on prediction variance
            confidence = self._calculate_confidence(row, df)
            
            predictions.append(Prediction(
                driver_code=row['Driver'],
                driver_name=row['DriverName'],
                predicted_position=idx + 1,
                predicted_time=row['PredictedTime'],
                confidence=confidence
            ))
        
        return predictions
    
    def _calculate_confidence(self, driver_row: pd.Series, all_predictions: pd.DataFrame) -> float:
        """Calculate prediction confidence"""
        # Base confidence on qualifying position vs predicted position
        qual_pos = all_predictions['QualifyingTime (s)'].rank().loc[driver_row.name]
        pred_pos = all_predictions['PredictedTime'].rank().loc[driver_row.name]
        
        position_diff = abs(qual_pos - pred_pos)
        
        # Higher confidence for smaller position changes
        confidence = max(0.5, 1.0 - (position_diff * 0.05))
        
        # Adjust based on weather if applicable
        if 'RainProbability' in driver_row and driver_row['RainProbability'] > 0.5:
            confidence *= 0.8  # Lower confidence in wet conditions
        
        return min(0.95, confidence)
    
    def evaluate(self, predictions: List[Prediction], actual_results: List[Dict]) -> ModelMetrics:
        """Evaluate model performance against actual results"""
        # Convert to DataFrames for easier comparison
        pred_df = pd.DataFrame([{
            'driver': p.driver_code,
            'predicted_position': p.predicted_position,
            'predicted_time': p.predicted_time
        } for p in predictions])
        
        actual_df = pd.DataFrame(actual_results)
        
        # Merge on driver
        comparison = pred_df.merge(actual_df, on='driver')
        
        # Calculate metrics
        position_mae = mean_absolute_error(
            comparison['actual_position'], 
            comparison['predicted_position']
        )
        
        time_mae = mean_absolute_error(
            comparison['actual_time'], 
            comparison['predicted_time']
        )
        
        # Calculate accuracy (predictions within 3 positions)
        correct_predictions = (
            abs(comparison['actual_position'] - comparison['predicted_position']) <= 3
        ).sum()
        accuracy = correct_predictions / len(comparison)
        
        return ModelMetrics(
            mae=position_mae,
            rmse=np.sqrt(mean_squared_error(
                comparison['actual_position'], 
                comparison['predicted_position']
            )),
            accuracy=accuracy,
            feature_importance=self.feature_importance
        )
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_importance': self.feature_importance,
            'version': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.config = model_data['config']
        self.feature_importance = model_data['feature_importance']
        logger.info(f"Model loaded from {filepath}")
    
    def export_predictions_json(self, predictions: List[Prediction], filepath: str):
        """Export predictions to JSON format"""
        predictions_data = [{
            'driver_code': p.driver_code,
            'driver_name': p.driver_name,
            'predicted_position': p.predicted_position,
            'predicted_time': p.predicted_time,
            'confidence': p.confidence
        } for p in predictions]
        
        with open(filepath, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        
        logger.info(f"Predictions exported to {filepath}")