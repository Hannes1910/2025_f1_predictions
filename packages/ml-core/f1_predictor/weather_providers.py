"""
Weather data providers for F1 predictions
Multiple free providers without payment requirements
"""

import requests
import logging
from typing import Dict, Optional
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class WeatherProvider(ABC):
    """Base class for weather providers"""
    
    @abstractmethod
    def get_weather_data(self, lat: float, lon: float, date: str) -> Dict[str, float]:
        """Get weather data for location and date"""
        pass

class OpenMeteoProvider(WeatherProvider):
    """Open-Meteo - Completely free, no API key required"""
    
    def get_weather_data(self, lat: float, lon: float, date: str) -> Dict[str, float]:
        try:
            # Parse date
            dt = datetime.strptime(date.split()[0], "%Y-%m-%d")
            
            # Open-Meteo API endpoint
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_probability_max", "windspeed_10m_max"],
                "timezone": "UTC",
                "start_date": dt.strftime("%Y-%m-%d"),
                "end_date": dt.strftime("%Y-%m-%d")
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                daily = data.get("daily", {})
                
                # Extract data
                temp_max = daily.get("temperature_2m_max", [20])[0]
                temp_min = daily.get("temperature_2m_min", [15])[0]
                rain_prob = daily.get("precipitation_probability_max", [0])[0]
                wind_speed = daily.get("windspeed_10m_max", [5])[0]
                
                return {
                    "temperature": (temp_max + temp_min) / 2,
                    "rain_probability": rain_prob / 100.0,  # Convert to 0-1 range
                    "humidity": 50,  # Not available in free tier, use default
                    "wind_speed": wind_speed / 3.6  # Convert km/h to m/s
                }
            else:
                logger.warning(f"Open-Meteo API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching Open-Meteo data: {e}")
        
        return self._default_weather()
    
    def _default_weather(self) -> Dict[str, float]:
        """Default weather conditions"""
        return {
            "temperature": 20,
            "rain_probability": 0,
            "humidity": 50,
            "wind_speed": 5
        }

class WeatherAPIProvider(WeatherProvider):
    """WeatherAPI.com - Free tier with 1M calls/month"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def get_weather_data(self, lat: float, lon: float, date: str) -> Dict[str, float]:
        try:
            # WeatherAPI endpoint
            url = "https://api.weatherapi.com/v1/forecast.json"
            params = {
                "key": self.api_key,
                "q": f"{lat},{lon}",
                "days": 1,
                "aqi": "no",
                "alerts": "no"
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                forecast = data.get("forecast", {}).get("forecastday", [{}])[0]
                day = forecast.get("day", {})
                
                return {
                    "temperature": day.get("avgtemp_c", 20),
                    "rain_probability": day.get("daily_chance_of_rain", 0) / 100.0,
                    "humidity": day.get("avghumidity", 50),
                    "wind_speed": day.get("maxwind_kph", 18) / 3.6
                }
            else:
                logger.warning(f"WeatherAPI error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching WeatherAPI data: {e}")
        
        return OpenMeteoProvider()._default_weather()

class WeatherBitProvider(WeatherProvider):
    """WeatherBit - Free tier with 500 calls/day"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def get_weather_data(self, lat: float, lon: float, date: str) -> Dict[str, float]:
        try:
            # WeatherBit endpoint
            url = "https://api.weatherbit.io/v2.0/current"
            params = {
                "lat": lat,
                "lon": lon,
                "key": self.api_key
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                obs = data.get("data", [{}])[0]
                
                return {
                    "temperature": obs.get("temp", 20),
                    "rain_probability": obs.get("precip", 0) / 10.0,  # Rough estimate
                    "humidity": obs.get("rh", 50),
                    "wind_speed": obs.get("wind_spd", 5)
                }
            else:
                logger.warning(f"WeatherBit error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching WeatherBit data: {e}")
        
        return OpenMeteoProvider()._default_weather()

class WeatherFactory:
    """Factory to create weather providers"""
    
    @staticmethod
    def create_provider(api_key: Optional[str] = None, provider_name: str = "open-meteo") -> WeatherProvider:
        """
        Create a weather provider instance
        
        Args:
            api_key: API key if required
            provider_name: Provider to use (open-meteo, weatherapi, weatherbit)
        
        Returns:
            WeatherProvider instance
        """
        if provider_name == "open-meteo" or not api_key:
            logger.info("Using Open-Meteo weather provider (no API key required)")
            return OpenMeteoProvider()
        elif provider_name == "weatherapi":
            logger.info("Using WeatherAPI.com provider")
            return WeatherAPIProvider(api_key)
        elif provider_name == "weatherbit":
            logger.info("Using WeatherBit provider")
            return WeatherBitProvider(api_key)
        else:
            logger.info("Defaulting to Open-Meteo provider")
            return OpenMeteoProvider()

# Circuit-specific weather patterns (historical averages)
CIRCUIT_WEATHER_PATTERNS = {
    "Monaco": {"rain_probability": 0.15, "temp_avg": 22},
    "Singapore": {"rain_probability": 0.40, "temp_avg": 30},
    "Spa": {"rain_probability": 0.35, "temp_avg": 18},
    "Interlagos": {"rain_probability": 0.45, "temp_avg": 23},
    "Suzuka": {"rain_probability": 0.30, "temp_avg": 20},
    "Silverstone": {"rain_probability": 0.25, "temp_avg": 18},
    "Montreal": {"rain_probability": 0.20, "temp_avg": 22},
    "Malaysia": {"rain_probability": 0.50, "temp_avg": 32},
    "Shanghai": {"rain_probability": 0.25, "temp_avg": 20},
    "Melbourne": {"rain_probability": 0.20, "temp_avg": 20},
    "Bahrain": {"rain_probability": 0.05, "temp_avg": 25},
    "Abu Dhabi": {"rain_probability": 0.02, "temp_avg": 28},
    "Jeddah": {"rain_probability": 0.03, "temp_avg": 27},
}

def get_circuit_weather_estimate(circuit: str) -> Dict[str, float]:
    """Get historical weather patterns for a circuit"""
    patterns = CIRCUIT_WEATHER_PATTERNS.get(circuit, {})
    return {
        "temperature": patterns.get("temp_avg", 22),
        "rain_probability": patterns.get("rain_probability", 0.10),
        "humidity": 50,
        "wind_speed": 5
    }