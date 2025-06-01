#!/usr/bin/env python3
"""
Load real F1 data from FastF1 for training
Production-ready data loading for ensemble models
Includes BOTH 2024 and 2025 seasons
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
            try:
                schedule = fastf1.get_event_schedule(year)
                races = schedule[schedule['EventFormat'] != 'testing']
                logger.info(f"Found {len(races)} races in {year} season")
            except Exception as e:
                logger.warning(f"Could not load {year} schedule: {e}")
                continue
            
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
                    if not laps.empty:
                        driver_avg_laps = laps.groupby('Driver').agg({
                            'LapTime': lambda x: x.dt.total_seconds().mean() if len(x) > 0 else np.nan,
                            'Sector1Time': lambda x: x.dt.total_seconds().mean() if len(x) > 0 else np.nan,
                            'Sector2Time': lambda x: x.dt.total_seconds().mean() if len(x) > 0 else np.nan,
                            'Sector3Time': lambda x: x.dt.total_seconds().mean() if len(x) > 0 else np.nan,
                        }).reset_index()
                        
                        # Merge with results
                        race_data = results.merge(driver_avg_laps, left_on='Abbreviation', right_on='Driver', how='left')
                    else:
                        race_data = results.copy()
                    
                    # Add race metadata
                    race_data['race_id'] = f"{year}_{race_round}"
                    race_data['season'] = year
                    race_data['round'] = race_round
                    race_data['race_name'] = race_name
                    race_data['circuit'] = race['Location']
                    race_data['race_date'] = race['EventDate']
                    
                    # Load qualifying data
                    try:
                        quali_session = fastf1.get_session(year, race_round, 'Q')
                        quali_session.load()
                        quali_results = quali_session.results.copy()
                        
                        # Get qualifying times
                        quali_times = quali_results[['Abbreviation', 'Q1', 'Q2', 'Q3', 'Position']].copy()
                        quali_times.columns = ['Driver', 'Q1_time', 'Q2_time', 'Q3_time', 'quali_position']
                        
                        # Convert times to seconds
                        for col in ['Q1_time', 'Q2_time', 'Q3_time']:
                            if col in quali_times.columns:
                                quali_times[col] = pd.to_timedelta(quali_times[col]).dt.total_seconds()
                        
                        # Merge qualifying data
                        race_data = race_data.merge(quali_times, left_on='Abbreviation', right_on='Driver', how='left')
                        
                    except Exception as e:
                        logger.warning(f"Could not load qualifying data for {year} {race_name}: {e}")
                    
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
                    
                    # Add year for tracking
                    race_data['year'] = year
                    
                    # Add to collection
                    all_race_data.append(race_data)
                    
                except Exception as e:
                    logger.error(f"Failed to load data for {year} {race_name}: {e}")
                    continue
        
        if not all_race_data:
            raise ValueError("No race data could be loaded for 2024/2025 seasons")
        
        # Combine all race data
        season_data = pd.concat(all_race_data, ignore_index=True)
        
        # Clean and prepare data
        season_data = self._clean_data(season_data)
        
        logger.info(f"\nTotal data loaded:")
        logger.info(f"  Records: {len(season_data)}")
        logger.info(f"  Races: {season_data['race_id'].nunique()}")
        logger.info(f"  2024 races: {season_data[season_data['year']==2024]['race_id'].nunique()}")
        logger.info(f"  2025 races: {season_data[season_data['year']==2025]['race_id'].nunique()}")
        logger.info(f"  Drivers: {season_data['driver_code'].nunique()}")
        
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
            # Handle both timedelta and string formats
            try:
                data['race_time_seconds'] = pd.to_timedelta(data['race_time']).dt.total_seconds()
            except:
                data['race_time_seconds'] = data['final_position'] * 90  # Estimate
        else:
            data['race_time_seconds'] = data['final_position'] * 90
        
        # Handle missing values
        data['grid_position'] = data['grid_position'].fillna(20)
        data['qualifying_position'] = data['qualifying_position'].fillna(20)
        
        if 'avg_lap_time' in data.columns:
            data['avg_lap_time'] = data['avg_lap_time'].fillna(data.groupby('race_id')['avg_lap_time'].transform('mean'))
        
        # Create driver ID mapping
        driver_mapping = {code: idx for idx, code in enumerate(data['driver_code'].unique())}
        data['driver_id'] = data['driver_code'].map(driver_mapping)
        
        # Filter out DNF/DNS for position-based training
        data['dnf'] = ~data['status'].str.contains('Finished', na=False)
        
        # Add recent form features (considering 2024-2025 continuity)
        data = data.sort_values(['driver_code', 'year', 'round'])
        data['driver_recent_avg'] = data.groupby('driver_code')['final_position'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        return data
    
    def load_upcoming_race_data(self):
        """Load data for the next upcoming race"""
        logger.info("Loading upcoming race data...")
        
        # Get 2025 schedule
        try:
            schedule = fastf1.get_event_schedule(2025)
            races = schedule[schedule['EventFormat'] != 'testing']
            
            # Find next race
            today = datetime.now()
            future_races = races[pd.to_datetime(races['EventDate']) > today]
            
            if not future_races.empty:
                next_race = future_races.iloc[0]
                logger.info(f"Next race: {next_race['EventName']} on {next_race['EventDate']}")
                return self._create_race_template(next_race)
        except Exception as e:
            logger.warning(f"Could not load 2025 schedule: {e}")
        
        # Fallback to template
        return self._create_template_race_data()
    
    def _create_race_template(self, race_info):
        """Create template for upcoming race predictions"""
        # Get current driver list
        drivers = self.data_loader.get_driver_mapping()
        teams = self.data_loader.get_team_mapping()
        
        template_data = []
        for driver_name, driver_code in drivers.items():
            template_data.append({
                'driver_code': driver_code,
                'driver_name': driver_name,
                'team': teams.get(driver_code, 'Unknown'),
                'race_id': f"2025_{race_info['RoundNumber']}",
                'race_name': race_info['EventName'],
                'circuit': race_info['Location'],
                'race_date': race_info['EventDate'],
                'year': 2025
            })
        
        return pd.DataFrame(template_data)
    
    def _create_template_race_data(self):
        """Create a template when no schedule available"""
        drivers = self.data_loader.get_driver_mapping()
        teams = self.data_loader.get_team_mapping()
        
        template_data = []
        for driver_name, driver_code in drivers.items():
            template_data.append({
                'driver_code': driver_code,
                'driver_name': driver_name,
                'team': teams.get(driver_code, 'Unknown'),
                'race_id': '2025_1',
                'race_name': 'Next Race',
                'circuit': 'TBD',
                'race_date': datetime.now().strftime('%Y-%m-%d'),
                'year': 2025
            })
        
        return pd.DataFrame(template_data)


def test_data_loading():
    """Test the data loading functionality"""
    loader = RealF1DataLoader()
    
    try:
        # Load 2024+2025 data
        season_data = loader.load_2024_season_data()
        
        print("\n📊 Data Loading Summary:")
        print(f"Total records: {len(season_data)}")
        print(f"Seasons: {sorted(season_data['year'].unique())}")
        print(f"Total races: {season_data['race_id'].nunique()}")
        print(f"Drivers: {season_data['driver_code'].nunique()}")
        
        # Show races by year
        print("\nRaces by year:")
        for year in sorted(season_data['year'].unique()):
            year_data = season_data[season_data['year'] == year]
            print(f"  {year}: {year_data['race_id'].nunique()} races")
            races = year_data[['race_name', 'race_date']].drop_duplicates()
            for _, race in races.head(3).iterrows():
                print(f"    - {race['race_name']} ({race['race_date']})")
        
        print(f"\nColumns: {list(season_data.columns[:10])}...")
        
        # Save sample for inspection
        season_data.to_csv('sample_2024_2025_data.csv', index=False)
        print("\n✅ Sample data saved to sample_2024_2025_data.csv")
        
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_data_loading()