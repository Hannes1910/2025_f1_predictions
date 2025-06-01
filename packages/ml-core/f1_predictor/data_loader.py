import fastf1
import pandas as pd
import requests
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles data loading from various sources"""
    
    def __init__(self, cache_dir: str = "f1_cache"):
        self.cache_dir = cache_dir
        fastf1.Cache.enable_cache(cache_dir)
    
    def load_race_data(self, year: int, race_identifier: str | int) -> pd.DataFrame:
        """Load race session data from FastF1"""
        try:
            session = fastf1.get_session(year, race_identifier, "R")
            session.load()
            
            laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
            laps.dropna(inplace=True)
            
            # Convert times to seconds
            for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
                laps[f"{col} (s)"] = laps[col].dt.total_seconds()
            
            return laps
        except Exception as e:
            logger.error(f"Error loading race data: {e}")
            return pd.DataFrame()
    
    def load_qualifying_data(self, year: int, race_identifier: str | int) -> pd.DataFrame:
        """Load qualifying session data from FastF1"""
        try:
            session = fastf1.get_session(year, race_identifier, "Q")
            session.load()
            
            # Get best lap times for each driver
            qualifying = session.laps.groupby('Driver')['LapTime'].min().reset_index()
            qualifying['QualifyingTime (s)'] = qualifying['LapTime'].dt.total_seconds()
            
            return qualifying[['Driver', 'QualifyingTime (s)']]
        except Exception as e:
            logger.error(f"Error loading qualifying data: {e}")
            return pd.DataFrame()
    
    def get_weather_data(self, lat: float, lon: float, date: str, api_key: str = None, circuit: str = None) -> Dict[str, float]:
        """Fetch weather data for race location and date"""
        try:
            # Import weather providers
            from .weather_providers import WeatherFactory, get_circuit_weather_estimate
            
            # Try to get real weather data
            if lat != 0 and lon != 0:
                provider = WeatherFactory.create_provider(api_key)
                weather_data = provider.get_weather_data(lat, lon, date)
                
                # If we got valid data, return it
                if weather_data.get("temperature", 0) != 20 or weather_data.get("rain_probability", 0) != 0:
                    return weather_data
            
            # Fallback to circuit-specific historical patterns
            if circuit:
                logger.info(f"Using historical weather patterns for {circuit}")
                return get_circuit_weather_estimate(circuit)
            
            # Final fallback to default
            return {"temperature": 20, "rain_probability": 0, "humidity": 50, "wind_speed": 5}
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            # Fallback to circuit patterns if available
            if circuit:
                from .weather_providers import get_circuit_weather_estimate
                return get_circuit_weather_estimate(circuit)
            return {"temperature": 20, "rain_probability": 0, "humidity": 50, "wind_speed": 5}
    
    def get_driver_mapping(self) -> Dict[str, str]:
        """Get mapping of driver names to codes"""
        return {
            "Lando Norris": "NOR", "Oscar Piastri": "PIA", "Max Verstappen": "VER", 
            "George Russell": "RUS", "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", 
            "Charles Leclerc": "LEC", "Lewis Hamilton": "HAM", "Pierre Gasly": "GAS", 
            "Carlos Sainz": "SAI", "Lance Stroll": "STR", "Fernando Alonso": "ALO",
            "Nico Hulkenberg": "HUL", "Esteban Ocon": "OCO", "Isack Hadjar": "HAD",
            "Andrea Kimi Antonelli": "ANT", "Oliver Bearman": "BEA", "Jack Doohan": "DOO",
            "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
        }
    
    def get_team_mapping(self) -> Dict[str, str]:
        """Get mapping of drivers to teams"""
        return {
            "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", 
            "RUS": "Mercedes", "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", 
            "TSU": "Racing Bulls", "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", 
            "STR": "Aston Martin", "ALB": "Williams", "HAD": "Red Bull", "ANT": "Mercedes",
            "BEA": "Haas", "DOO": "Alpine", "BOR": "Kick Sauber", "LAW": "Racing Bulls"
        }
    
    def load_2024_season_data(self) -> pd.DataFrame:
        """Load SMART data: 2025 for performance, 2024 for patterns only"""
        try:
            # Use smart loader that separates current performance from historical patterns
            from load_smart_f1_data import SmartF1DataLoader
            loader = SmartF1DataLoader()
            return loader.load_2024_season_data()
        except ImportError:
            # Fallback to v2 if smart not available
            try:
                from load_real_f1_data_v2 import RealF1DataLoader
            except ImportError:
                from load_real_f1_data import RealF1DataLoader
            loader = RealF1DataLoader()
            return loader.load_2024_season_data()
    
    def get_circuit_coordinates(self) -> Dict[str, Tuple[float, float]]:
        """Get GPS coordinates for F1 circuits"""
        return {
            "Australia": (-37.8497, 144.9680),
            "China": (31.3389, 121.2199),
            "Japan": (34.8434, 136.5408),
            "Bahrain": (26.0325, 50.5106),
            "Saudi Arabia": (21.6319, 39.1044),
            "Monaco": (43.7347, 7.4206),
            "Spain": (41.5700, 2.2611),
            "Canada": (45.5017, -73.5228),
            "Austria": (47.2197, 14.7647),
            "Great Britain": (52.0786, -1.0169),
            "Hungary": (47.5789, 19.2486),
            "Belgium": (50.4372, 5.9714),
            "Netherlands": (52.3888, 4.5409),
            "Italy": (45.6156, 9.2811),
            "Singapore": (1.2914, 103.8644),
            "USA": (30.1328, -97.6411),
            "Mexico": (19.4042, -99.0907),
            "Brazil": (-23.7036, -46.6973),
            "Las Vegas": (36.1716, -115.1391),
            "Qatar": (25.4901, 51.4542),
            "Abu Dhabi": (24.4672, 54.6031)
        }