#!/usr/bin/env python3
"""
Load real F1 data from FastF1 for training
Production-ready data loading for ensemble models
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

class RealF1DataLoader:
    """Load real F1 data for training ensemble models"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        
    def load_2024_season_data(self):
        """Load full 2024 AND 2025 season data for training"""
        logger.info("Loading 2024 and 2025 F1 season data...")
        
        all_race_data = []
        
        # Load both 2024 and 2025 seasons
        for year in [2024, 2025]:
            logger.info(f"\nLoading {year} season...")
            
            # Get race calendar
            schedule = fastf1.get_event_schedule(year)
            races = schedule[schedule['EventFormat'] != 'testing']
            
            logger.info(f"Found {len(races)} races in {year} season")
            
            for idx, race in races.iterrows():
                try:
                    race_name = race['EventName']
                    race_round = race['RoundNumber']
                    
                    # Skip future races
                    if pd.to_datetime(race['EventDate']) > datetime.now():
                        logger.info(f"Skipping future race: {race_name}")
                        continue
                    
                    logger.info(f"Loading data for {year} Round {race_round}: {race_name}")
                
                    # Load race session
                    session = fastf1.get_session(year, race_round, 'R')
                    session.load()
                
                # Get race results
                results = session.results.copy()
                
                # Get driver lap data
                laps = session.laps.copy()
                
                # Calculate average lap times per driver
                driver_avg_laps = laps.groupby('Driver').agg({
                    'LapTime': lambda x: x.dt.total_seconds().mean(),
                    'Sector1Time': lambda x: x.dt.total_seconds().mean(),
                    'Sector2Time': lambda x: x.dt.total_seconds().mean(),
                    'Sector3Time': lambda x: x.dt.total_seconds().mean(),
                }).reset_index()
                
                # Merge with results
                race_data = results.merge(driver_avg_laps, left_on='Abbreviation', right_on='Driver', how='left')
                
                # Add race metadata
                race_data['race_id'] = race_round
                race_data['race_name'] = race_name
                race_data['circuit'] = race['Location']
                race_data['race_date'] = race['EventDate']
                
                # Load qualifying data
                try:
                    quali_session = fastf1.get_session(2024, race_round, 'Q')
                    quali_session.load()
                    quali_results = quali_session.results.copy()
                    
                    # Get qualifying times
                    quali_times = quali_results[['Abbreviation', 'Q1', 'Q2', 'Q3', 'Position']].copy()
                    quali_times.columns = ['Driver', 'Q1_time', 'Q2_time', 'Q3_time', 'quali_position']
                    
                    # Convert times to seconds
                    for col in ['Q1_time', 'Q2_time', 'Q3_time']:
                        quali_times[col] = pd.to_timedelta(quali_times[col]).dt.total_seconds()
                    
                    # Merge qualifying data
                    race_data = race_data.merge(quali_times, left_on='Abbreviation', right_on='Driver', how='left')
                    
                except Exception as e:
                    logger.warning(f"Could not load qualifying data for {race_name}: {e}")
                
                # Add weather data
                weather_info = session.weather_data
                if not weather_info.empty:
                    avg_temp = weather_info['AirTemp'].mean()
                    avg_humidity = weather_info['Humidity'].mean()
                    rain_detected = weather_info['Rainfall'].any()
                else:
                    avg_temp = 25
                    avg_humidity = 50
                    rain_detected = False
                
                race_data['temperature'] = avg_temp
                race_data['humidity'] = avg_humidity
                race_data['rain'] = rain_detected
                
                # Add to collection
                all_race_data.append(race_data)
                
            except Exception as e:
                logger.error(f"Failed to load data for {race_name}: {e}")
                continue
        
        if not all_race_data:
            raise ValueError("No race data could be loaded for 2024 season")
        
        # Combine all race data
        season_data = pd.concat(all_race_data, ignore_index=True)
        
        # Clean and prepare data
        season_data = self._clean_data(season_data)
        
        logger.info(f"Loaded {len(season_data)} driver-race records")
        logger.info(f"Races loaded: {season_data['race_id'].nunique()}")
        logger.info(f"Drivers: {season_data['Abbreviation'].nunique()}")
        
        return season_data
    
    def _clean_data(self, data):
        """Clean and prepare data for training"""
        # Rename columns for consistency
        column_mapping = {
            'Abbreviation': 'driver_code',
            'TeamName': 'team',
            'GridPosition': 'grid_position',
            'Position': 'final_position',
            'Points': 'points',
            'Status': 'status',
            'Time': 'race_time',
            'LapTime': 'avg_lap_time',
            'quali_position': 'qualifying_position'
        }
        
        data = data.rename(columns=column_mapping)
        
        # Convert race time to seconds
        if 'race_time' in data.columns:
            data['race_time_seconds'] = pd.to_timedelta(data['race_time']).dt.total_seconds()
        else:
            # If no race time, estimate from position
            data['race_time_seconds'] = data['final_position'] * 90  # Rough estimate
        
        # Handle missing values
        data['grid_position'] = data['grid_position'].fillna(20)
        data['qualifying_position'] = data['qualifying_position'].fillna(20)
        data['avg_lap_time'] = data['avg_lap_time'].fillna(data['avg_lap_time'].mean())
        
        # Create driver ID mapping
        driver_mapping = {code: idx for idx, code in enumerate(data['driver_code'].unique())}
        data['driver_id'] = data['driver_code'].map(driver_mapping)
        
        # Add round number
        data['round'] = data['race_id']
        
        # Filter out DNF/DNS for position-based training
        # But keep them for other features
        data['dnf'] = ~data['status'].str.contains('Finished', na=False)
        
        return data
    
    def load_upcoming_race_data(self):
        """Load data for the next upcoming race"""
        logger.info("Loading upcoming race data...")
        
        # Get 2025 schedule
        schedule = fastf1.get_event_schedule(2025)
        races = schedule[schedule['EventFormat'] != 'testing']
        
        # Find next race
        today = datetime.now()
        future_races = races[pd.to_datetime(races['EventDate']) > today]
        
        if future_races.empty:
            # If no 2025 races, use last 2024 race as template
            logger.warning("No upcoming 2025 races found, using 2024 template")
            return self._create_template_race_data()
        
        next_race = future_races.iloc[0]
        logger.info(f"Next race: {next_race['EventName']} on {next_race['EventDate']}")
        
        # Create prediction template
        return self._create_race_template(next_race)
    
    def _create_race_template(self, race_info):
        """Create template for upcoming race predictions"""
        # Get current driver list
        drivers = self.data_loader.get_driver_mapping()
        
        template_data = []
        for driver_name, driver_code in drivers.items():
            template_data.append({
                'driver_code': driver_code,
                'driver_name': driver_name,
                'race_id': race_info['RoundNumber'],
                'race_name': race_info['EventName'],
                'circuit': race_info['Location'],
                'race_date': race_info['EventDate']
            })
        
        return pd.DataFrame(template_data)


def test_data_loading():
    """Test the data loading functionality"""
    loader = RealF1DataLoader()
    
    try:
        # Load 2024 data
        season_data = loader.load_2024_season_data()
        
        print("\n📊 Data Loading Summary:")
        print(f"Total records: {len(season_data)}")
        print(f"Races: {season_data['race_id'].nunique()}")
        print(f"Drivers: {season_data['driver_code'].nunique()}")
        print(f"\nColumns: {list(season_data.columns)}")
        print(f"\nSample data:")
        print(season_data.head())
        
        # Save sample for inspection
        season_data.to_csv('sample_2024_data.csv', index=False)
        print("\n✅ Sample data saved to sample_2024_data.csv")
        
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_data_loading()