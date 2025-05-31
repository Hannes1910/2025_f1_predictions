#!/usr/bin/env python3
"""
Test weather providers to ensure they work without API keys
"""

import sys
from pathlib import Path

# Add both parent directory and ml-core package to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "packages" / "ml-core"))

from f1_predictor.weather_providers import WeatherFactory, get_circuit_weather_estimate

def test_open_meteo():
    """Test Open-Meteo (no API key required)"""
    print("🌤️  Testing Open-Meteo (free, no API key)...")
    
    provider = WeatherFactory.create_provider()
    
    # Test with Monaco coordinates
    lat, lon = 43.7347, 7.4206
    date = "2025-05-25"
    
    weather = provider.get_weather_data(lat, lon, date)
    
    print(f"Monaco weather: {weather}")
    print(f"Temperature: {weather['temperature']}°C")
    print(f"Rain probability: {weather['rain_probability']:.0%}")
    print("✅ Open-Meteo test successful!\n")

def test_circuit_patterns():
    """Test circuit-specific weather patterns"""
    print("🏁 Testing circuit weather patterns...")
    
    circuits = ['Monaco', 'Singapore', 'Spa', 'Silverstone', 'Bahrain']
    
    for circuit in circuits:
        weather = get_circuit_weather_estimate(circuit)
        print(f"{circuit}: {weather['temperature']}°C, {weather['rain_probability']:.0%} rain chance")
    
    print("✅ Circuit patterns test successful!\n")

def main():
    print("🧪 Testing F1 Weather Providers\n")
    
    try:
        test_open_meteo()
        test_circuit_patterns()
        
        print("🎉 All weather providers working!")
        print("💡 No API key needed - Open-Meteo is completely free")
        print("📊 Circuit patterns provide historical weather data as fallback")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())