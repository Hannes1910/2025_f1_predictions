#!/usr/bin/env python3
"""
Add sample qualifying data via API
"""

import requests
import json

# API configuration
API_URL = "https://f1-predictions-api.vprifntqe.workers.dev"
API_KEY = "test-key-123"  # Replace with actual key

# Sample qualifying results for Australian GP
qualifying_data = {
    "race_id": 1,
    "results": [
        {"driver_id": 1, "q1_time": 82.456, "q2_time": 81.789, "q3_time": 81.123, "qualifying_time": 81.123, "grid_position": 1},
        {"driver_id": 3, "q1_time": 82.567, "q2_time": 81.890, "q3_time": 81.234, "qualifying_time": 81.234, "grid_position": 2},
        {"driver_id": 5, "q1_time": 82.678, "q2_time": 81.901, "q3_time": 81.345, "qualifying_time": 81.345, "grid_position": 3},
        {"driver_id": 7, "q1_time": 82.789, "q2_time": 82.012, "q3_time": 81.456, "qualifying_time": 81.456, "grid_position": 4},
        {"driver_id": 9, "q1_time": 82.890, "q2_time": 82.123, "q3_time": 81.567, "qualifying_time": 81.567, "grid_position": 5},
        {"driver_id": 11, "q1_time": 83.001, "q2_time": 82.234, "q3_time": 81.678, "qualifying_time": 81.678, "grid_position": 6},
        {"driver_id": 4, "q1_time": 83.112, "q2_time": 82.345, "q3_time": 81.789, "qualifying_time": 81.789, "grid_position": 7},
        {"driver_id": 2, "q1_time": 83.223, "q2_time": 82.456, "q3_time": 81.890, "qualifying_time": 81.890, "grid_position": 8},
        {"driver_id": 8, "q1_time": 83.334, "q2_time": 82.567, "q3_time": 81.901, "qualifying_time": 81.901, "grid_position": 9},
        {"driver_id": 6, "q1_time": 83.445, "q2_time": 82.678, "q3_time": 82.012, "qualifying_time": 82.012, "grid_position": 10},
        {"driver_id": 12, "q1_time": 83.556, "q2_time": 82.789, "qualifying_time": 82.789, "grid_position": 11},
        {"driver_id": 10, "q1_time": 83.667, "q2_time": 82.890, "qualifying_time": 82.890, "grid_position": 12},
        {"driver_id": 13, "q1_time": 83.778, "q2_time": 83.001, "qualifying_time": 83.001, "grid_position": 13},
        {"driver_id": 14, "q1_time": 83.889, "q2_time": 83.112, "qualifying_time": 83.112, "grid_position": 14},
        {"driver_id": 15, "q1_time": 84.000, "q2_time": 83.223, "qualifying_time": 83.223, "grid_position": 20, "qualifying_position": 15, "grid_penalty": 5},
        {"driver_id": 16, "q1_time": 84.111, "qualifying_time": 84.111, "grid_position": 16},
        {"driver_id": 17, "q1_time": 84.222, "qualifying_time": 84.222, "grid_position": 17},
        {"driver_id": 18, "q1_time": 84.333, "qualifying_time": 84.333, "grid_position": 18},
        {"driver_id": 19, "q1_time": 84.444, "qualifying_time": 84.444, "grid_position": 19},
        {"driver_id": 20, "q1_time": 84.555, "qualifying_time": 84.555, "grid_position": 15}
    ]
}

# Send request
headers = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

response = requests.post(
    f"{API_URL}/api/admin/qualifying",
    json=qualifying_data,
    headers=headers
)

if response.status_code == 200:
    print("✅ Qualifying data added successfully!")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"❌ Failed to add qualifying data: {response.status_code}")
    print(response.text)