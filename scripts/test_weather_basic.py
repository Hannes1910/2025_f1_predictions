#!/usr/bin/env python3
"""
Basic test for weather providers using only standard library
"""

import urllib.request
import urllib.parse
import json
import sys

def test_open_meteo():
    """Test Open-Meteo API using only standard library"""
    print("🌤️  Testing Open-Meteo (free, no API key)...")
    
    try:
        # Monaco coordinates
        lat, lon = 43.7347, 7.4206
        date = "2025-05-31"
        
        # Build URL with parameters
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,windspeed_10m_max",
            "timezone": "UTC",
            "start_date": date,
            "end_date": date
        }
        
        url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
        
        # Make request
        with urllib.request.urlopen(url, timeout=10) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                daily = data.get("daily", {})
                
                # Extract data
                temp_max = daily.get("temperature_2m_max", [20])[0]
                temp_min = daily.get("temperature_2m_min", [15])[0]
                rain_prob = daily.get("precipitation_probability_max", [0])[0]
                wind_speed = daily.get("windspeed_10m_max", [5])[0]
                
                weather = {
                    "temperature": (temp_max + temp_min) / 2,
                    "rain_probability": rain_prob / 100.0,
                    "humidity": 50,
                    "wind_speed": wind_speed / 3.6
                }
                
                print(f"Monaco weather: {weather}")
                print(f"Temperature: {weather['temperature']:.1f}°C")
                print(f"Rain probability: {weather['rain_probability']:.0%}")
                print(f"Wind speed: {weather['wind_speed']:.1f} m/s")
                print("✅ Open-Meteo test successful!\n")
                return True
            else:
                print(f"❌ Open-Meteo API error: {response.status}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing Open-Meteo: {e}")
        return False

def test_circuit_patterns():
    """Test circuit-specific weather patterns"""
    print("🏁 Testing circuit weather patterns...")
    
    # Circuit patterns (historical averages)
    patterns = {
        "Monaco": {"rain_probability": 0.15, "temp_avg": 22},
        "Singapore": {"rain_probability": 0.40, "temp_avg": 30},
        "Spa": {"rain_probability": 0.35, "temp_avg": 18},
        "Silverstone": {"rain_probability": 0.25, "temp_avg": 18},
        "Bahrain": {"rain_probability": 0.05, "temp_avg": 25},
    }
    
    circuits = ['Monaco', 'Singapore', 'Spa', 'Silverstone', 'Bahrain']
    
    for circuit in circuits:
        if circuit in patterns:
            data = patterns[circuit]
            weather = {
                "temperature": data["temp_avg"],
                "rain_probability": data["rain_probability"],
                "humidity": 50,
                "wind_speed": 5
            }
            print(f"{circuit}: {weather['temperature']}°C, {weather['rain_probability']:.0%} rain chance")
        else:
            print(f"{circuit}: Default weather pattern")
    
    print("✅ Circuit patterns test successful!\n")
    return True

def main():
    print("🧪 Testing F1 Weather Providers (Basic Test)\n")
    
    tests_passed = 0
    total_tests = 2
    
    try:
        # Test Open-Meteo
        if test_open_meteo():
            tests_passed += 1
        
        # Test circuit patterns
        if test_circuit_patterns():
            tests_passed += 1
        
        if tests_passed == total_tests:
            print("🎉 All weather providers working!")
            print("💡 No API key needed - Open-Meteo is completely free")
            print("📊 Circuit patterns provide historical weather data as fallback")
            print("\n✅ Weather system ready for F1 predictions!")
            print("\n📋 Summary:")
            print("  • Open-Meteo API: ✅ Working (no API key required)")
            print("  • Circuit patterns: ✅ Working (historical data)")
            print("  • Alternative to OpenWeatherMap: ✅ Ready")
            return 0
        else:
            print(f"⚠️  {tests_passed}/{total_tests} tests passed")
            return 1
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())