#!/usr/bin/env python3
"""
Shared Feature Engineering
Ensures consistency between training and prediction phases
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class F1FeatureEngineer:
    """Consistent feature engineering for training and prediction"""
    
    def __init__(self):
        # Define all features used in the model
        self.feature_columns = [
            # Driver features
            'grid_position',
            'qualifying_time_diff',
            'driver_skill_rating',
            'avg_finish_position',
            'driver_recent_form',
            'dnf_rate',
            'consistency_score',
            
            # Team features
            'team_performance',
            'team_momentum_change',
            
            # Circuit features
            'circuit_type_high_speed',
            'circuit_type_street', 
            'circuit_type_technical',
            'circuit_type_balanced',
            
            # Weather features
            'expected_temperature',
            'rain_probability',
            
            # Historical patterns
            'hist_avg_pit_stop_time',
            'hist_safety_car_probability',
            'hist_dnf_rate',
            'hist_track_evolution',
            'hist_tire_degradation'
        ]
        
        # Circuit classifications
        self.circuit_types = {
            'Monaco': {'street': 1, 'technical': 1},
            'Singapore': {'street': 1},
            'Baku': {'street': 1, 'high_speed': 1},
            'Monza': {'high_speed': 1},
            'Spa-Francorchamps': {'high_speed': 1},
            'Silverstone': {'high_speed': 1},
            'Hungaroring': {'technical': 1},
            'Barcelona': {'technical': 1},
            'Suzuka': {'technical': 1},
            'Interlagos': {'technical': 1},
            'COTA': {'balanced': 1},
            'Melbourne': {'balanced': 1}
        }
        
        # Historical weather patterns
        self.circuit_weather = {
            'Bahrain': {'temp': 30, 'rain_prob': 0.02},
            'Saudi Arabia': {'temp': 28, 'rain_prob': 0.02},
            'Australia': {'temp': 22, 'rain_prob': 0.15},
            'China': {'temp': 20, 'rain_prob': 0.20},
            'Miami': {'temp': 28, 'rain_prob': 0.25},
            'Monaco': {'temp': 22, 'rain_prob': 0.10},
            'Spain': {'temp': 25, 'rain_prob': 0.10},
            'Canada': {'temp': 22, 'rain_prob': 0.20},
            'Austria': {'temp': 23, 'rain_prob': 0.25},
            'Britain': {'temp': 20, 'rain_prob': 0.25},
            'Hungary': {'temp': 28, 'rain_prob': 0.15},
            'Belgium': {'temp': 18, 'rain_prob': 0.30},
            'Netherlands': {'temp': 20, 'rain_prob': 0.20},
            'Italy': {'temp': 26, 'rain_prob': 0.15},
            'Singapore': {'temp': 30, 'rain_prob': 0.20},
            'Japan': {'temp': 22, 'rain_prob': 0.20},
            'Qatar': {'temp': 30, 'rain_prob': 0.02},
            'USA': {'temp': 25, 'rain_prob': 0.15},
            'Mexico': {'temp': 22, 'rain_prob': 0.10},
            'Brazil': {'temp': 25, 'rain_prob': 0.30},
            'Las Vegas': {'temp': 15, 'rain_prob': 0.05},
            'Abu Dhabi': {'temp': 28, 'rain_prob': 0.02}
        }
        
        # Historical track patterns
        self.track_patterns = {
            'Monaco': {'pit_time': 22, 'sc_prob': 0.40, 'dnf_rate': 0.20},
            'Singapore': {'pit_time': 23, 'sc_prob': 0.35, 'dnf_rate': 0.15},
            'Baku': {'pit_time': 24, 'sc_prob': 0.30, 'dnf_rate': 0.18},
            'Monza': {'pit_time': 21, 'sc_prob': 0.15, 'dnf_rate': 0.10},
            'Spa-Francorchamps': {'pit_time': 23, 'sc_prob': 0.20, 'dnf_rate': 0.12},
            'default': {'pit_time': 24, 'sc_prob': 0.20, 'dnf_rate': 0.10}
        }
    
    def create_features_for_training(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Create features from historical data for training"""
        
        features = raw_data.copy()
        
        # Driver features
        features = self._add_driver_features(features)
        
        # Team features
        features = self._add_team_features(features)
        
        # Circuit features
        features = self._add_circuit_features(features)
        
        # Weather features
        features = self._add_weather_features(features)
        
        # Historical patterns
        features = self._add_historical_patterns(features)
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0
                logger.warning(f"Missing feature column {col}, filled with zeros")
        
        return features[self.feature_columns + ['race_time_seconds']]  # Include target
    
    def create_features_for_prediction(self, driver_data: List[Dict], 
                                     race_info: Dict) -> pd.DataFrame:
        """Create features for prediction from current data"""
        
        features = pd.DataFrame(driver_data)
        
        # Add circuit information
        features['circuit'] = race_info['circuit']
        
        # Circuit type features
        circuit_type = self.circuit_types.get(race_info['circuit'], {'balanced': 1})
        for ctype in ['high_speed', 'street', 'technical', 'balanced']:
            features[f'circuit_type_{ctype}'] = circuit_type.get(ctype, 0)
        
        # Weather features
        weather = self.circuit_weather.get(race_info['circuit'], 
                                         {'temp': 25, 'rain_prob': 0.1})
        features['expected_temperature'] = weather['temp']
        features['rain_probability'] = weather['rain_prob']
        
        # Historical patterns
        patterns = self.track_patterns.get(race_info['circuit'], 
                                         self.track_patterns['default'])
        features['hist_avg_pit_stop_time'] = patterns['pit_time']
        features['hist_safety_car_probability'] = patterns['sc_prob']
        features['hist_dnf_rate'] = patterns['dnf_rate']
        features['hist_track_evolution'] = 0.02  # Default 2% improvement
        features['hist_tire_degradation'] = 0.05  # Default 0.05s/lap
        
        # Calculate consistency score
        if 'recent_positions' in features.columns:
            features['consistency_score'] = features['recent_positions'].apply(
                lambda x: np.std(x) if isinstance(x, list) else 3.0
            )
        else:
            features['consistency_score'] = 3.0
        
        # Ensure all required features exist
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0
                logger.warning(f"Missing feature {col} for prediction, using default")
        
        return features[self.feature_columns]
    
    def _add_driver_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add driver-specific features"""
        
        # Recent form (rolling average)
        data['driver_recent_form'] = data.groupby('driver_code')['final_position'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        ).fillna(10)
        
        # Average finish position
        data['avg_finish_position'] = data.groupby('driver_code')['final_position'].transform(
            lambda x: x.expanding().mean().shift(1)
        ).fillna(10)
        
        # DNF rate
        data['dnf'] = data['status'] != 'Finished'
        data['dnf_rate'] = data.groupby('driver_code')['dnf'].transform(
            lambda x: x.expanding().mean().shift(1)
        ).fillna(0.1)
        
        # Consistency score (std of positions)
        data['consistency_score'] = data.groupby('driver_code')['final_position'].transform(
            lambda x: x.rolling(window=5, min_periods=2).std().shift(1)
        ).fillna(3)
        
        # Skill rating based on points
        data['driver_skill_rating'] = data.groupby('driver_code')['points'].transform(
            lambda x: x.expanding().sum()
        ) / 400.0  # Normalize
        data['driver_skill_rating'] = data['driver_skill_rating'].clip(0, 1)
        
        return data
    
    def _add_team_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add team-specific features"""
        
        # Team average performance
        data['team_performance'] = data.groupby(['team', 'race_id'])['final_position'].transform('mean')
        
        # Team momentum (improvement trend)
        data['team_momentum'] = data.groupby('team')['team_performance'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        data['team_momentum_change'] = data.groupby('team')['team_momentum'].transform(
            lambda x: x.diff().rolling(window=2, min_periods=1).mean()
        ).fillna(0)
        
        return data
    
    def _add_circuit_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add circuit-specific features"""
        
        # Add circuit types
        for _, row in data.iterrows():
            circuit = row.get('circuit', 'default')
            circuit_type = self.circuit_types.get(circuit, {'balanced': 1})
            
            for ctype in ['high_speed', 'street', 'technical', 'balanced']:
                data.loc[row.name, f'circuit_type_{ctype}'] = circuit_type.get(ctype, 0)
        
        return data
    
    def _add_weather_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add weather features"""
        
        for _, row in data.iterrows():
            circuit = row.get('circuit', 'default')
            weather = self.circuit_weather.get(circuit, {'temp': 25, 'rain_prob': 0.1})
            
            # Use actual weather if available, otherwise use historical
            if 'temperature' not in data.columns or pd.isna(row.get('temperature')):
                data.loc[row.name, 'expected_temperature'] = weather['temp']
            else:
                data.loc[row.name, 'expected_temperature'] = row['temperature']
                
            if 'rain' not in data.columns:
                data.loc[row.name, 'rain_probability'] = weather['rain_prob']
            else:
                data.loc[row.name, 'rain_probability'] = float(row.get('rain', False))
        
        return data
    
    def _add_historical_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add historical track patterns"""
        
        for _, row in data.iterrows():
            circuit = row.get('circuit', 'default')
            patterns = self.track_patterns.get(circuit, self.track_patterns['default'])
            
            data.loc[row.name, 'hist_avg_pit_stop_time'] = patterns['pit_time']
            data.loc[row.name, 'hist_safety_car_probability'] = patterns['sc_prob']
            data.loc[row.name, 'hist_dnf_rate'] = patterns['dnf_rate']
            data.loc[row.name, 'hist_track_evolution'] = 0.02
            data.loc[row.name, 'hist_tire_degradation'] = 0.05
        
        return data
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for model"""
        return self.feature_columns
    
    def save_feature_config(self, path: str = "models/ensemble/"):
        """Save feature configuration for consistency"""
        import json
        import os
        
        os.makedirs(path, exist_ok=True)
        
        config = {
            'feature_columns': self.feature_columns,
            'circuit_types': self.circuit_types,
            'circuit_weather': self.circuit_weather,
            'track_patterns': self.track_patterns
        }
        
        with open(f"{path}/feature_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved feature configuration to {path}/feature_config.json")