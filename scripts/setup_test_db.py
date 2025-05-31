#!/usr/bin/env python3
"""
Set up a local test database with sample F1 data for testing
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path

def create_database(db_path: str):
    """Create test database with sample data"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables using the schema
    schema_files = [
        "scripts/schema.sql",
        "scripts/schema_update_v2.sql"
    ]
    
    for schema_file in schema_files:
        if Path(schema_file).exists():
            print(f"Executing {schema_file}...")
            with open(schema_file, 'r') as f:
                cursor.executescript(f.read())
    
    # Insert sample drivers
    drivers = [
        (1, "VER", "Max Verstappen", "Red Bull Racing"),
        (2, "NOR", "Lando Norris", "McLaren"),
        (3, "PIA", "Oscar Piastri", "McLaren"),
        (4, "LEC", "Charles Leclerc", "Ferrari"),
        (5, "SAI", "Carlos Sainz", "Ferrari"),
        (6, "HAM", "Lewis Hamilton", "Mercedes"),
        (7, "RUS", "George Russell", "Mercedes"),
        (8, "ALO", "Fernando Alonso", "Aston Martin"),
        (9, "STR", "Lance Stroll", "Aston Martin"),
        (10, "GAS", "Pierre Gasly", "Alpine"),
        (11, "OCO", "Esteban Ocon", "Alpine"),
        (12, "TSU", "Yuki Tsunoda", "Racing Bulls"),
        (13, "LAW", "Liam Lawson", "Racing Bulls"),
        (14, "ALB", "Alexander Albon", "Williams"),
        (15, "COL", "Franco Colapinto", "Williams"),
        (16, "HUL", "Nico Hulkenberg", "Haas"),
        (17, "BEA", "Oliver Bearman", "Haas"),
        (18, "BOT", "Valtteri Bottas", "Kick Sauber"),
        (19, "ZHO", "Zhou Guanyu", "Kick Sauber")
    ]
    
    cursor.executemany("""
        INSERT OR REPLACE INTO drivers (id, code, name, team) 
        VALUES (?, ?, ?, ?)
    """, drivers)
    
    # Insert sample races
    races = [
        (1, 2025, 1, "Australian Grand Prix", "2025-03-16", "Melbourne"),
        (2, 2025, 2, "Chinese Grand Prix", "2025-03-23", "Shanghai"),
        (3, 2025, 3, "Japanese Grand Prix", "2025-04-13", "Suzuka"),
        (4, 2025, 4, "Bahrain Grand Prix", "2025-04-20", "Bahrain"),
        (5, 2025, 5, "Saudi Arabian Grand Prix", "2025-05-04", "Jeddah"),
        (6, 2025, 6, "Miami Grand Prix", "2025-05-11", "Miami"),
        (7, 2025, 7, "Monaco Grand Prix", "2025-05-25", "Monaco"),
        (8, 2025, 8, "Spanish Grand Prix", "2025-06-01", "Barcelona"),
        (9, 2025, 9, "Canadian Grand Prix", "2025-06-15", "Montreal"),
        (10, 2025, 10, "Austrian Grand Prix", "2025-06-29", "Red Bull Ring")
    ]
    
    cursor.executemany("""
        INSERT OR REPLACE INTO races (id, season, round, name, date, circuit) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, races)
    
    # Create teams table and insert data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            constructor TEXT NOT NULL
        )
    """)
    
    teams = [
        (1, "Red Bull Racing", "Red Bull"),
        (2, "McLaren", "McLaren"),
        (3, "Ferrari", "Ferrari"),
        (4, "Mercedes", "Mercedes"),
        (5, "Aston Martin", "Aston Martin"),
        (6, "Alpine", "Alpine"),
        (7, "Racing Bulls", "Racing Bulls"),
        (8, "Williams", "Williams"),
        (9, "Haas", "Haas"),
        (10, "Kick Sauber", "Kick Sauber")
    ]
    
    cursor.executemany("""
        INSERT OR REPLACE INTO teams (id, name, constructor) 
        VALUES (?, ?, ?)
    """, teams)
    
    conn.commit()
    conn.close()
    
    print(f"✅ Test database created: {db_path}")
    print(f"📊 Added {len(drivers)} drivers, {len(races)} races, {len(teams)} teams")

def main():
    db_path = "f1_predictions_test.db"
    
    # Remove existing database
    if Path(db_path).exists():
        Path(db_path).unlink()
        print(f"🗑️  Removed existing database: {db_path}")
    
    create_database(db_path)
    
    print(f"\n🎯 Test database ready!")
    print(f"📁 Database file: {db_path}")
    print(f"🔧 You can now run: python3 scripts/train_models_v2.py --db {db_path}")

if __name__ == "__main__":
    main()