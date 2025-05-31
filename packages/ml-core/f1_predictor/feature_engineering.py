import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Handles feature engineering for F1 predictions"""
    
    def __init__(self):
        self.wet_performance_factors = {
            "VER": 0.975196, "HAM": 0.976464, "LEC": 0.975862, "NOR": 0.978179, 
            "ALO": 0.972655, "RUS": 0.968678, "SAI": 0.978754, "TSU": 0.996338, 
            "OCO": 0.981810, "GAS": 0.978832, "STR": 0.979857, "HUL": 0.985000,
            "ALB": 0.982000, "PIA": 0.980000, "HAD": 0.990000, "ANT": 0.988000,
            "BEA": 0.992000, "DOO": 0.991000, "BOR": 0.993000, "LAW": 0.989000
        }
    
    def engineer_features(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Engineer features based on configuration"""
        df = data.copy()
        
        # Basic features
        if 'QualifyingTime (s)' in df.columns:
            df['QualifyingPosition'] = df['QualifyingTime (s)'].rank()
            df['QualifyingGap'] = df['QualifyingTime (s)'] - df['QualifyingTime (s)'].min()
        
        # Sector time features
        if all(col in df.columns for col in ['Sector1Time (s)', 'Sector2Time (s)', 'Sector3Time (s)']):
            df['TotalSectorTime (s)'] = df[['Sector1Time (s)', 'Sector2Time (s)', 'Sector3Time (s)']].sum(axis=1)
            df['SectorConsistency'] = df[['Sector1Time (s)', 'Sector2Time (s)', 'Sector3Time (s)']].std(axis=1)
        
        # Weather adjustments
        if 'RainProbability' in df.columns and 'Driver' in df.columns:
            df['WetPerformanceFactor'] = df['Driver'].map(self.wet_performance_factors).fillna(0.985)
            df['WeatherAdjustedTime'] = df.apply(
                lambda row: row['QualifyingTime (s)'] * row['WetPerformanceFactor'] 
                if row['RainProbability'] > 0.75 else row['QualifyingTime (s)'],
                axis=1
            )
        
        # Team performance features
        if 'TeamPerformanceScore' in df.columns:
            df['TeamPerformanceRank'] = df['TeamPerformanceScore'].rank(ascending=False)
        
        return df
    
    def create_historical_features(self, current_data: pd.DataFrame, 
                                 historical_results: pd.DataFrame) -> pd.DataFrame:
        """Create features based on historical performance"""
        df = current_data.copy()
        
        if historical_results.empty:
            return df
        
        # Calculate driver statistics
        driver_stats = historical_results.groupby('Driver').agg({
            'Position': ['mean', 'std', 'min'],
            'Points': 'sum'
        }).reset_index()
        
        driver_stats.columns = ['Driver', 'AvgPosition', 'PositionStd', 'BestPosition', 'TotalPoints']
        
        # Recent form (last 3 races)
        recent_races = historical_results.sort_values('Date').groupby('Driver').tail(3)
        recent_form = recent_races.groupby('Driver')['Position'].mean().reset_index()
        recent_form.columns = ['Driver', 'RecentForm']
        
        # Merge with current data
        df = df.merge(driver_stats, on='Driver', how='left')
        df = df.merge(recent_form, on='Driver', how='left')
        
        # Fill missing values
        df['AvgPosition'] = df['AvgPosition'].fillna(15)
        df['PositionStd'] = df['PositionStd'].fillna(5)
        df['BestPosition'] = df['BestPosition'].fillna(20)
        df['TotalPoints'] = df['TotalPoints'].fillna(0)
        df['RecentForm'] = df['RecentForm'].fillna(15)
        
        return df
    
    def create_circuit_specific_features(self, data: pd.DataFrame, 
                                       circuit_type: str) -> pd.DataFrame:
        """Create features specific to circuit characteristics"""
        df = data.copy()
        
        circuit_types = {
            'street': ['Monaco', 'Singapore', 'Baku', 'Jeddah'],
            'high_speed': ['Monza', 'Spa', 'Silverstone'],
            'technical': ['Hungary', 'Barcelona', 'Suzuka'],
            'mixed': ['Interlagos', 'COTA', 'Shanghai']
        }
        
        # Circuit type bonus/penalty for drivers
        street_circuit_bonus = {
            'LEC': -0.2, 'ALO': -0.15, 'HAM': -0.1, 'VER': 0.05
        }
        
        high_speed_bonus = {
            'VER': -0.2, 'HAM': -0.15, 'RUS': -0.1, 'NOR': -0.1
        }
        
        if circuit_type == 'street' and 'Driver' in df.columns:
            df['CircuitBonus'] = df['Driver'].map(street_circuit_bonus).fillna(0)
        elif circuit_type == 'high_speed' and 'Driver' in df.columns:
            df['CircuitBonus'] = df['Driver'].map(high_speed_bonus).fillna(0)
        else:
            df['CircuitBonus'] = 0
        
        return df
    
    def scale_features(self, data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Scale features to improve model performance"""
        from sklearn.preprocessing import StandardScaler
        
        df = data.copy()
        scaler = StandardScaler()
        
        # Only scale numeric features
        numeric_features = [col for col in feature_columns if col in df.columns and df[col].dtype in ['float64', 'int64']]
        
        if numeric_features:
            df[numeric_features] = scaler.fit_transform(df[numeric_features])
        
        return df