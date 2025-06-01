#!/usr/bin/env python3
"""
Set up local database with test data for ML service
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def create_database():
    """Create and populate test database"""
    
    conn = sqlite3.connect('f1_predictions_test.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.executescript("""
    -- Drivers table
    CREATE TABLE IF NOT EXISTS drivers (
        id INTEGER PRIMARY KEY,
        code TEXT NOT NULL,
        name TEXT NOT NULL,
        team TEXT NOT NULL,
        points REAL DEFAULT 0
    );
    
    -- Races table
    CREATE TABLE IF NOT EXISTS races (
        id INTEGER PRIMARY KEY,
        season INTEGER NOT NULL,
        round INTEGER NOT NULL,
        name TEXT NOT NULL,
        circuit TEXT NOT NULL,
        date TEXT NOT NULL
    );
    
    -- Race results table
    CREATE TABLE IF NOT EXISTS race_results (
        id INTEGER PRIMARY KEY,
        race_id INTEGER NOT NULL,
        driver_id INTEGER NOT NULL,
        position INTEGER NOT NULL,
        points REAL DEFAULT 0,
        grid_position INTEGER,
        status TEXT DEFAULT 'Finished',
        FOREIGN KEY (race_id) REFERENCES races (id),
        FOREIGN KEY (driver_id) REFERENCES drivers (id)
    );
    
    -- Qualifying results
    CREATE TABLE IF NOT EXISTS qualifying_results (
        id INTEGER PRIMARY KEY,
        race_id INTEGER NOT NULL,
        driver_id INTEGER NOT NULL,
        grid_position INTEGER NOT NULL,
        qualifying_time REAL,
        FOREIGN KEY (race_id) REFERENCES races (id),
        FOREIGN KEY (driver_id) REFERENCES drivers (id)
    );
    """)
    
    # Insert 2025 drivers
    drivers_data = [
        (1, 'VER', 'Max Verstappen', 'Red Bull', 100),
        (2, 'PER', 'Sergio Perez', 'Red Bull', 50),
        (3, 'HAM', 'Lewis Hamilton', 'Ferrari', 80),
        (4, 'LEC', 'Charles Leclerc', 'Ferrari', 75),
        (5, 'NOR', 'Lando Norris', 'McLaren', 90),
        (6, 'PIA', 'Oscar Piastri', 'McLaren', 60),
        (7, 'RUS', 'George Russell', 'Mercedes', 70),
        (8, 'ANT', 'Andrea Kimi Antonelli', 'Mercedes', 20),
        (9, 'ALO', 'Fernando Alonso', 'Aston Martin', 40),
        (10, 'STR', 'Lance Stroll', 'Aston Martin', 25),
        (11, 'GAS', 'Pierre Gasly', 'Alpine', 35),
        (12, 'DOO', 'Jack Doohan', 'Alpine', 10),
        (13, 'HUL', 'Nico Hulkenberg', 'Kick Sauber', 30),
        (14, 'BOR', 'Gabriel Bortoleto', 'Kick Sauber', 5),
        (15, 'TSU', 'Yuki Tsunoda', 'Racing Bulls', 32),
        (16, 'HAD', 'Isack Hadjar', 'Racing Bulls', 8),
        (17, 'BEA', 'Oliver Bearman', 'Haas', 15),
        (18, 'OCO', 'Esteban Ocon', 'Haas', 28),
        (19, 'ALB', 'Alexander Albon', 'Williams', 22),
        (20, 'SAI', 'Carlos Sainz', 'Williams', 45)
    ]
    
    cursor.executemany("INSERT OR REPLACE INTO drivers VALUES (?, ?, ?, ?, ?)", drivers_data)
    
    # Insert 2025 races
    races_data = []
    race_names = [
        ('Bahrain', 'Bahrain International Circuit'),
        ('Saudi Arabia', 'Jeddah Street Circuit'),
        ('Australia', 'Melbourne'),
        ('China', 'Shanghai'),
        ('Miami', 'Miami International Autodrome'),
        ('Monaco', 'Monaco'),
        ('Spain', 'Barcelona'),
        ('Canada', 'Circuit Gilles Villeneuve')
    ]
    
    base_date = datetime(2025, 3, 1)
    for i, (name, circuit) in enumerate(race_names):
        race_date = base_date + timedelta(days=i*14)
        races_data.append((i+1, 2025, i+1, f"{name} Grand Prix", circuit, race_date.strftime('%Y-%m-%d')))
    
    cursor.executemany("INSERT OR REPLACE INTO races VALUES (?, ?, ?, ?, ?, ?)", races_data)
    
    # Insert some race results for the first 3 races
    # This gives the ML model some historical data to work with
    results_data = []
    
    # Race 1 - Bahrain
    positions_r1 = [1, 3, 2, 5, 4, 8, 6, 15, 7, 12, 9, 18, 10, 20, 11, 19, 13, 14, 16, 17]
    for i, pos in enumerate(positions_r1):
        points = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][pos-1]
        results_data.append((None, 1, i+1, pos, points, pos, 'Finished'))
    
    # Race 2 - Saudi Arabia  
    positions_r2 = [1, 4, 3, 2, 5, 6, 8, 12, 9, 11, 7, 16, 13, 19, 10, 20, 14, 15, 17, 18]
    for i, pos in enumerate(positions_r2):
        points = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][pos-1]
        status = 'Finished' if i != 15 else 'DNF'  # One DNF
        results_data.append((None, 2, i+1, pos, points, pos+1, status))
    
    # Race 3 - Australia
    positions_r3 = [2, 5, 1, 3, 4, 7, 6, 14, 8, 13, 9, 17, 11, 18, 10, 19, 12, 16, 15, 20]
    for i, pos in enumerate(positions_r3):
        points = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][pos-1]
        results_data.append((None, 3, i+1, pos, points, pos, 'Finished'))
    
    cursor.executemany("INSERT INTO race_results VALUES (?, ?, ?, ?, ?, ?, ?)", results_data)
    
    # Add some qualifying results for upcoming race (China - race 4)
    quali_data = []
    quali_times = [90.123, 90.234, 90.156, 90.278, 90.189, 90.345, 90.301, 90.567, 
                   90.423, 90.678, 90.489, 90.789, 90.534, 90.890, 90.612, 90.923,
                   90.701, 90.812, 90.756, 90.845]
    
    for i, time in enumerate(quali_times):
        quali_data.append((None, 4, i+1, i+1, time))
    
    cursor.executemany("INSERT INTO qualifying_results VALUES (?, ?, ?, ?, ?)", quali_data)
    
    conn.commit()
    conn.close()
    
    print("✅ Database created: f1_predictions_test.db")
    print("   - 20 drivers (2025 lineup)")
    print("   - 8 races scheduled")
    print("   - 3 races with results") 
    print("   - Qualifying data for race 4")


def test_database():
    """Test the database contents"""
    conn = sqlite3.connect('f1_predictions_test.db')
    
    # Check drivers
    drivers = pd.read_sql_query("SELECT * FROM drivers", conn)
    print(f"\nDrivers: {len(drivers)}")
    print(drivers.head())
    
    # Check races
    races = pd.read_sql_query("SELECT * FROM races", conn)
    print(f"\nRaces: {len(races)}")
    print(races)
    
    # Check results
    results = pd.read_sql_query("SELECT COUNT(*) as count FROM race_results", conn)
    print(f"\nRace results: {results['count'].iloc[0]} records")
    
    # Check driver standings
    standings = pd.read_sql_query("""
        SELECT d.code, d.name, d.team, SUM(rr.points) as total_points
        FROM drivers d
        LEFT JOIN race_results rr ON d.id = rr.driver_id
        GROUP BY d.id
        ORDER BY total_points DESC
        LIMIT 10
    """, conn)
    print("\nCurrent standings:")
    print(standings)
    
    conn.close()


if __name__ == "__main__":
    create_database()
    test_database()