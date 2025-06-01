#!/usr/bin/env python3
"""
Smart F1 data loading that separates:
- Historical patterns (2024): track characteristics, weather, safety cars
- Current performance (2025): driver/car performance for accurate predictions
"""

import pandas as pd
import numpy as np
import fastf1
from datetime import datetime
import logging
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable FastF1 cache
fastf1.Cache.enable_cache('f1_cache')

sys.path.append('./packages/ml-core')
from f1_predictor.data_loader import DataLoader
from f1_predictor.feature_engineering import FeatureEngineer

class SmartF1DataLoader:
    """Smart data loading that uses 2024 for patterns, 2025 for performance"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        
    def load_2024_season_data(self):
        """Load SMART data: 2025 for performance, 2024 for historical patterns"""
        logger.info("Smart F1 data loading strategy:")
        logger.info("- 2025 data: Current driver/car performance")
        logger.info("- 2024 data: Historical track patterns only")
        
        # First, load 2025 data for current performance
        current_data = self._load_2025_performance_data()
        
        # Then, enrich with 2024 historical patterns
        historical_patterns = self._load_2024_pattern_data()
        
        # Merge intelligently
        final_data = self._merge_smart_data(current_data, historical_patterns)
        
        return final_data
    
    def _load_2025_performance_data(self):
        """Load 2025 data for current driver and car performance"""
        logger.info("\n📊 Loading 2025 performance data...")
        
        all_race_data = []
        
        try:
            schedule = fastf1.get_event_schedule(2025)
            races = schedule[schedule['EventFormat'] != 'testing']
            logger.info(f"Found {len(races)} races in 2025 season")
        except Exception as e:
            logger.error(f"Could not load 2025 schedule: {e}")
            # Return empty dataset if 2025 not available yet
            return pd.DataFrame()
        
        races_loaded = 0
        for idx, race in races.iterrows():
            try:
                race_name = race['EventName']
                race_round = race['RoundNumber']
                
                # Skip future races
                if pd.to_datetime(race['EventDate']) > datetime.now():
                    logger.info(f"Skipping future race: {race_name}")
                    continue
                
                logger.info(f"Loading 2025 Round {race_round}: {race_name}")
                
                # Load race session
                session = fastf1.get_session(2025, race_round, 'R')
                session.load()
                
                # Get race results
                results = session.results.copy()
                
                # Get driver lap data for CURRENT performance
                laps = session.laps.copy()
                
                if not laps.empty:
                    # Current lap time performance
                    driver_performance = laps.groupby('Driver').agg({
                        'LapTime': lambda x: x.dt.total_seconds().mean(),
                        'SpeedI1': 'mean',  # Speed in sector 1
                        'SpeedI2': 'mean',  # Speed in sector 2
                        'SpeedFL': 'mean',  # Speed at finish line
                        'Compound': lambda x: x.mode()[0] if len(x) > 0 else 'MEDIUM'
                    }).reset_index()
                    
                    # Merge with results
                    race_data = results.merge(driver_performance, left_on='Abbreviation', right_on='Driver', how='left')
                else:
                    race_data = results.copy()
                
                # Add race metadata
                race_data['race_id'] = f"2025_{race_round}"
                race_data['season'] = 2025
                race_data['round'] = race_round
                race_data['race_name'] = race_name
                race_data['circuit'] = race['Location']
                race_data['race_date'] = race['EventDate']
                
                # Load qualifying for CURRENT car performance
                try:
                    quali_session = fastf1.get_session(2025, race_round, 'Q')
                    quali_session.load()
                    quali_results = quali_session.results.copy()
                    
                    # Get best qualifying times (indicates current car speed)
                    quali_performance = quali_results[['Abbreviation', 'Q1', 'Q2', 'Q3', 'Position']].copy()
                    quali_performance['best_quali_time'] = quali_performance[['Q1', 'Q2', 'Q3']].min(axis=1)
                    quali_performance['quali_position'] = quali_performance['Position']
                    
                    # Convert to seconds
                    quali_performance['best_quali_seconds'] = pd.to_timedelta(quali_performance['best_quali_time']).dt.total_seconds()
                    
                    # Merge
                    race_data = race_data.merge(
                        quali_performance[['Abbreviation', 'quali_position', 'best_quali_seconds']], 
                        on='Abbreviation', 
                        how='left'
                    )
                    
                except Exception as e:
                    logger.warning(f"Could not load 2025 qualifying for {race_name}: {e}")
                
                # Current weather (affects current performance)
                weather_info = session.weather_data
                if not weather_info.empty:
                    race_data['current_temp'] = weather_info['AirTemp'].mean()
                    race_data['current_track_temp'] = weather_info['TrackTemp'].mean()
                    race_data['current_humidity'] = weather_info['Humidity'].mean()
                    race_data['current_pressure'] = weather_info['Pressure'].mean()
                
                all_race_data.append(race_data)
                races_loaded += 1
                
            except Exception as e:
                logger.error(f"Failed to load 2025 {race_name}: {e}")
                continue
        
        if not all_race_data:
            logger.warning("No 2025 race data available yet")
            return pd.DataFrame()
        
        # Combine all 2025 races
        performance_data = pd.concat(all_race_data, ignore_index=True)
        
        logger.info(f"✅ Loaded {races_loaded} races from 2025")
        logger.info(f"   Current drivers: {performance_data['Abbreviation'].nunique()}")
        
        return performance_data
    
    def _load_2024_pattern_data(self):
        """Load 2024 data ONLY for historical patterns, not driver performance"""
        logger.info("\n📚 Loading 2024 historical patterns...")
        
        pattern_data = {}
        
        try:
            schedule = fastf1.get_event_schedule(2024)
            races = schedule[schedule['EventFormat'] != 'testing']
        except Exception as e:
            logger.error(f"Could not load 2024 schedule: {e}")
            return pattern_data
        
        for idx, race in races.iterrows():
            try:
                circuit = race['Location']
                race_name = race['EventName']
                race_round = race['RoundNumber']
                
                # Skip if future
                if pd.to_datetime(race['EventDate']) > datetime.now():
                    continue
                
                logger.info(f"Extracting patterns from 2024 {race_name}")
                
                # Load session for patterns
                session = fastf1.get_session(2024, race_round, 'R')
                session.load()
                
                # Extract PATTERNS only (not driver-specific performance)
                patterns = {
                    'circuit': circuit,
                    'avg_pit_stop_time': self._calculate_avg_pit_stop(session),
                    'safety_car_probability': self._calculate_safety_car_prob(session),
                    'dnf_rate': self._calculate_circuit_dnf_rate(session),
                    'typical_lap_time': self._calculate_typical_lap_time(session),
                    'weather_patterns': self._extract_weather_patterns(session),
                    'track_evolution': self._calculate_track_evolution(session),
                    'tire_degradation': self._calculate_tire_degradation(session),
                    'drs_zones': len(session.laps['DRSEnabled'].unique()) if not session.laps.empty else 2,
                    'race_distance': race.get('EventDistance', 305)  # km
                }
                
                pattern_data[circuit] = patterns
                
            except Exception as e:
                logger.warning(f"Could not extract patterns from 2024 {race_name}: {e}")
                continue
        
        logger.info(f"✅ Extracted patterns from {len(pattern_data)} circuits")
        
        return pattern_data
    
    def _calculate_avg_pit_stop(self, session):
        """Calculate average pit stop time at circuit"""
        try:
            laps = session.laps
            pit_laps = laps[laps['PitInTime'].notna()]
            if not pit_laps.empty:
                # Calculate pit stop duration
                pit_times = (pit_laps['PitOutTime'] - pit_laps['PitInTime']).dt.total_seconds()
                return pit_times.mean()
        except:
            pass
        return 25.0  # Default pit stop time
    
    def _calculate_safety_car_prob(self, session):
        """Calculate safety car probability"""
        try:
            # Check for safety car periods
            track_status = session.track_status
            if not track_status.empty:
                sc_periods = track_status[track_status['Status'].isin(['4', '5'])]  # SC or VSC
                return min(len(sc_periods) / 10, 1.0)  # Normalize
        except:
            pass
        return 0.15  # Default 15% chance
    
    def _calculate_circuit_dnf_rate(self, session):
        """Calculate DNF rate at circuit"""
        try:
            results = session.results
            total_starters = len(results)
            dnf_count = len(results[~results['Status'].str.contains('Finished', na=False)])
            return dnf_count / total_starters if total_starters > 0 else 0.1
        except:
            pass
        return 0.1  # Default 10% DNF rate
    
    def _calculate_typical_lap_time(self, session):
        """Calculate typical racing lap time"""
        try:
            laps = session.laps
            # Get laps between 10-90% of race distance (avoid start/end anomalies)
            if not laps.empty:
                clean_laps = laps[(laps['LapNumber'] > 5) & (laps['LapNumber'] < laps['LapNumber'].max() - 5)]
                if not clean_laps.empty:
                    return clean_laps['LapTime'].dt.total_seconds().median()
        except:
            pass
        return 90.0  # Default lap time
    
    def _extract_weather_patterns(self, session):
        """Extract typical weather patterns"""
        try:
            weather = session.weather_data
            if not weather.empty:
                return {
                    'typical_temp': weather['AirTemp'].mean(),
                    'temp_variation': weather['AirTemp'].std(),
                    'typical_humidity': weather['Humidity'].mean(),
                    'rain_occurred': weather['Rainfall'].any()
                }
        except:
            pass
        return {
            'typical_temp': 25,
            'temp_variation': 5,
            'typical_humidity': 50,
            'rain_occurred': False
        }
    
    def _calculate_track_evolution(self, session):
        """Calculate how much track improves over session"""
        try:
            laps = session.laps
            if not laps.empty:
                # Compare early vs late lap times
                early_laps = laps[laps['LapNumber'] < 10]['LapTime'].dt.total_seconds().mean()
                late_laps = laps[laps['LapNumber'] > laps['LapNumber'].max() - 10]['LapTime'].dt.total_seconds().mean()
                return (early_laps - late_laps) / early_laps if early_laps > 0 else 0
        except:
            pass
        return 0.02  # Default 2% improvement
    
    def _calculate_tire_degradation(self, session):
        """Calculate average tire degradation per lap"""
        try:
            laps = session.laps
            if not laps.empty:
                # Group by stint and calculate degradation
                stint_deg = []
                for driver in laps['Driver'].unique():
                    driver_laps = laps[laps['Driver'] == driver]
                    stints = driver_laps['Stint'].unique()
                    
                    for stint in stints:
                        stint_laps = driver_laps[driver_laps['Stint'] == stint]
                        if len(stint_laps) > 5:
                            times = stint_laps['LapTime'].dt.total_seconds()
                            if times.notna().sum() > 5:
                                # Calculate degradation as time increase per lap
                                deg = np.polyfit(range(len(times)), times.fillna(times.mean()), 1)[0]
                                stint_deg.append(deg)
                
                if stint_deg:
                    return np.mean(stint_deg)
        except:
            pass
        return 0.05  # Default 0.05s per lap degradation
    
    def _merge_smart_data(self, current_data, historical_patterns):
        """Intelligently merge current performance with historical patterns"""
        
        if current_data.empty:
            logger.warning("No current data available, returning empty dataset")
            return pd.DataFrame()
        
        # Clean current data first
        current_data = self._clean_current_data(current_data)
        
        # Add historical patterns based on circuit
        for circuit, patterns in historical_patterns.items():
            mask = current_data['circuit'] == circuit
            if mask.any():
                for key, value in patterns.items():
                    if key != 'circuit' and key != 'weather_patterns':
                        current_data.loc[mask, f'hist_{key}'] = value
                
                # Add weather patterns
                if 'weather_patterns' in patterns:
                    for weather_key, weather_value in patterns['weather_patterns'].items():
                        current_data.loc[mask, f'hist_{weather_key}'] = weather_value
        
        # Fill missing historical data with defaults
        current_data['hist_avg_pit_stop_time'] = current_data.get('hist_avg_pit_stop_time', 25.0)
        current_data['hist_safety_car_probability'] = current_data.get('hist_safety_car_probability', 0.15)
        current_data['hist_dnf_rate'] = current_data.get('hist_dnf_rate', 0.1)
        
        logger.info(f"\n✅ Smart data merge complete:")
        logger.info(f"   Total records: {len(current_data)}")
        logger.info(f"   2025 races: {current_data['race_id'].nunique()}")
        logger.info(f"   Circuits with patterns: {len(historical_patterns)}")
        
        return current_data
    
    def _clean_current_data(self, data):
        """Clean and prepare current performance data"""
        # Rename columns
        column_mapping = {
            'Abbreviation': 'driver_code',
            'TeamName': 'team',
            'GridPosition': 'grid_position',
            'Position': 'final_position',
            'Points': 'points',
            'Status': 'status',
            'Time': 'race_time',
            'LapTime': 'avg_lap_time'
        }
        
        data = data.rename(columns=column_mapping)
        
        # Convert times
        if 'race_time' in data.columns:
            try:
                data['race_time_seconds'] = pd.to_timedelta(data['race_time']).dt.total_seconds()
            except:
                data['race_time_seconds'] = data['final_position'] * 90
        
        # Handle missing values
        data['grid_position'] = data['grid_position'].fillna(20)
        data['final_position'] = data['final_position'].fillna(20)
        
        # Create driver ID
        driver_mapping = {code: idx for idx, code in enumerate(data['driver_code'].unique())}
        data['driver_id'] = data['driver_code'].map(driver_mapping)
        
        # Calculate current form (2025 only!)
        data = data.sort_values(['driver_code', 'round'])
        data['current_form'] = data.groupby('driver_code')['final_position'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # Team performance (2025 only!)
        data['team_avg_position'] = data.groupby(['team', 'round'])['final_position'].transform('mean')
        
        return data


def test_smart_loading():
    """Test the smart data loading"""
    loader = SmartF1DataLoader()
    
    try:
        # Load smart data
        data = loader.load_2024_season_data()
        
        if not data.empty:
            print("\n🎯 Smart Data Loading Summary:")
            print(f"Total records: {len(data)}")
            print(f"2025 races loaded: {data['race_id'].nunique()}")
            print(f"\nCurrent performance columns (2025):")
            perf_cols = [c for c in data.columns if 'current_' in c or 'avg_lap_time' in c]
            print(f"  {perf_cols[:5]}...")
            print(f"\nHistorical pattern columns (2024):")
            hist_cols = [c for c in data.columns if 'hist_' in c]
            print(f"  {hist_cols[:5]}...")
            
            # Save sample
            data.to_csv('smart_f1_data_sample.csv', index=False)
            print("\n✅ Sample saved to smart_f1_data_sample.csv")
        else:
            print("\n⚠️ No 2025 data available yet - this is expected early in the season")
            print("The system will use historical patterns when 2025 races become available")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_smart_loading()