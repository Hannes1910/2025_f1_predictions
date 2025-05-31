#!/usr/bin/env python3
"""
Data migration script to populate Cloudflare D1 database with initial data
"""

import sqlite3
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from f1_predictor import DataLoader

class DataMigrator:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.data_loader = DataLoader()
        self.conn = None
        
    def connect(self):
        """Connect to the SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            
    def create_tables(self):
        """Create all necessary tables"""
        with open(Path(__file__).parent / 'schema.sql', 'r') as f:
            schema = f.read()
        
        self.conn.executescript(schema)
        self.conn.commit()
        print("✓ Database schema created")
        
    def populate_drivers(self):
        """Populate drivers table with 2025 season drivers"""
        drivers = [
            ("VER", "Max Verstappen", "Red Bull"),
            ("NOR", "Lando Norris", "McLaren"),
            ("PIA", "Oscar Piastri", "McLaren"),
            ("LEC", "Charles Leclerc", "Ferrari"),
            ("SAI", "Carlos Sainz", "Ferrari"),
            ("HAM", "Lewis Hamilton", "Mercedes"),
            ("RUS", "George Russell", "Mercedes"),
            ("ALO", "Fernando Alonso", "Aston Martin"),
            ("STR", "Lance Stroll", "Aston Martin"),
            ("GAS", "Pierre Gasly", "Alpine"),
            ("OCO", "Esteban Ocon", "Alpine"),
            ("TSU", "Yuki Tsunoda", "Racing Bulls"),
            ("LAW", "Liam Lawson", "Racing Bulls"),
            ("ALB", "Alexander Albon", "Williams"),
            ("HAD", "Isack Hadjar", "Red Bull"),
            ("ANT", "Andrea Kimi Antonelli", "Mercedes"),
            ("HUL", "Nico Hulkenberg", "Kick Sauber"),
            ("BOR", "Gabriel Bortoleto", "Kick Sauber"),
            ("BEA", "Oliver Bearman", "Haas"),
            ("DOO", "Jack Doohan", "Alpine")
        ]
        
        cursor = self.conn.cursor()
        cursor.executemany(
            "INSERT OR IGNORE INTO drivers (code, name, team) VALUES (?, ?, ?)",
            drivers
        )
        self.conn.commit()
        print(f"✓ Inserted {cursor.rowcount} drivers")
        
    def populate_races(self):
        """Populate races table with 2025 season calendar"""
        races = [
            (2025, 1, "Australian Grand Prix", "2025-03-16", "Melbourne"),
            (2025, 2, "Chinese Grand Prix", "2025-03-23", "Shanghai"),
            (2025, 3, "Japanese Grand Prix", "2025-04-06", "Suzuka"),
            (2025, 4, "Bahrain Grand Prix", "2025-04-13", "Bahrain"),
            (2025, 5, "Saudi Arabian Grand Prix", "2025-04-20", "Jeddah"),
            (2025, 6, "Miami Grand Prix", "2025-05-04", "Miami"),
            (2025, 7, "Emilia Romagna Grand Prix", "2025-05-18", "Imola"),
            (2025, 8, "Monaco Grand Prix", "2025-05-25", "Monaco"),
            (2025, 9, "Spanish Grand Prix", "2025-06-01", "Barcelona"),
            (2025, 10, "Canadian Grand Prix", "2025-06-15", "Montreal"),
            (2025, 11, "Austrian Grand Prix", "2025-06-29", "Red Bull Ring"),
            (2025, 12, "British Grand Prix", "2025-07-06", "Silverstone"),
            (2025, 13, "Belgian Grand Prix", "2025-07-27", "Spa"),
            (2025, 14, "Hungarian Grand Prix", "2025-08-03", "Budapest"),
            (2025, 15, "Dutch Grand Prix", "2025-08-31", "Zandvoort"),
            (2025, 16, "Italian Grand Prix", "2025-09-07", "Monza"),
            (2025, 17, "Azerbaijan Grand Prix", "2025-09-21", "Baku"),
            (2025, 18, "Singapore Grand Prix", "2025-10-05", "Singapore"),
            (2025, 19, "United States Grand Prix", "2025-10-19", "Austin"),
            (2025, 20, "Mexican Grand Prix", "2025-10-26", "Mexico City"),
            (2025, 21, "Brazilian Grand Prix", "2025-11-09", "Interlagos"),
            (2025, 22, "Las Vegas Grand Prix", "2025-11-22", "Las Vegas"),
            (2025, 23, "Qatar Grand Prix", "2025-11-30", "Lusail"),
            (2025, 24, "Abu Dhabi Grand Prix", "2025-12-07", "Yas Marina")
        ]
        
        cursor = self.conn.cursor()
        cursor.executemany(
            "INSERT OR IGNORE INTO races (season, round, name, date, circuit) VALUES (?, ?, ?, ?, ?)",
            races
        )
        self.conn.commit()
        print(f"✓ Inserted {cursor.rowcount} races")
        
    def migrate_existing_predictions(self):
        """Migrate predictions from existing Python files to database"""
        # This would parse the existing prediction files and extract results
        # For now, we'll add sample data
        
        cursor = self.conn.cursor()
        
        # Get race and driver IDs
        races = cursor.execute("SELECT id, round FROM races WHERE season = 2025").fetchall()
        drivers = cursor.execute("SELECT id, code FROM drivers").fetchall()
        
        driver_map = {d['code']: d['id'] for d in drivers}
        race_map = {r['round']: r['id'] for r in races}
        
        # Sample predictions for completed races
        completed_predictions = [
            # Australian GP (Round 1)
            {"race": 1, "driver": "LEC", "position": 1, "time": 82.67},
            {"race": 1, "driver": "NOR", "position": 2, "time": 83.12},
            {"race": 1, "driver": "VER", "position": 3, "time": 83.45},
            # Add more as needed
        ]
        
        for pred in completed_predictions:
            if pred["driver"] in driver_map and pred["race"] in race_map:
                cursor.execute("""
                    INSERT OR IGNORE INTO predictions 
                    (race_id, driver_id, predicted_position, predicted_time, confidence, model_version)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    race_map[pred["race"]], 
                    driver_map[pred["driver"]], 
                    pred["position"],
                    pred["time"],
                    0.85,  # Default confidence
                    "v1.0.0"
                ))
        
        self.conn.commit()
        print(f"✓ Migrated predictions")
        
    def add_sample_metrics(self):
        """Add sample model metrics"""
        cursor = self.conn.cursor()
        
        metrics = [
            ("v1.0.0", 1, 3.22, 0.75),
            ("v1.0.0", 2, 2.98, 0.78),
            ("v1.0.0", 3, 3.45, 0.72),
        ]
        
        for version, race_id, mae, accuracy in metrics:
            cursor.execute("""
                INSERT INTO model_metrics (model_version, race_id, mae, accuracy)
                VALUES (?, ?, ?, ?)
            """, (version, race_id, mae, accuracy))
        
        self.conn.commit()
        print(f"✓ Added model metrics")
        
    def export_to_json(self, output_dir: str):
        """Export database to JSON files for easy import to D1"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        cursor = self.conn.cursor()
        
        # Export each table
        tables = ['drivers', 'races', 'predictions', 'race_results', 'model_metrics', 'feature_data']
        
        for table in tables:
            rows = cursor.execute(f"SELECT * FROM {table}").fetchall()
            data = [dict(row) for row in rows]
            
            with open(output_path / f"{table}.json", 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Exported {len(data)} rows from {table}")
            
    def run_migration(self):
        """Run the complete migration process"""
        try:
            self.connect()
            self.create_tables()
            self.populate_drivers()
            self.populate_races()
            self.migrate_existing_predictions()
            self.add_sample_metrics()
            self.export_to_json('data_export')
            print("\n✅ Migration completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Migration failed: {e}")
            raise
        finally:
            self.close()

def main():
    # Create local SQLite database for initial migration
    db_path = "f1_predictions.db"
    
    migrator = DataMigrator(db_path)
    migrator.run_migration()
    
    print("\n📝 Next steps:")
    print("1. Create D1 database: wrangler d1 create f1-predictions")
    print("2. Apply schema: wrangler d1 execute f1-predictions --file=scripts/schema.sql")
    print("3. Import data: wrangler d1 execute f1-predictions --file=data_export/[table].sql")

if __name__ == "__main__":
    main()