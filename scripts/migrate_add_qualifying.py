#!/usr/bin/env python3
"""
Migration script to add qualifying_results table
and populate with sample data for testing
"""

import sqlite3
import sys
from pathlib import Path

def apply_migration(db_path: str):
    """Apply the qualifying results table migration"""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Read and execute the schema
        schema_path = Path(__file__).parent / 'schema_add_qualifying.sql'
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        cursor.executescript(schema_sql)
        print("✅ Created qualifying_results table")
        
        # Check if we should add sample data
        cursor.execute("SELECT COUNT(*) FROM qualifying_results")
        if cursor.fetchone()[0] == 0:
            print("📝 Adding sample qualifying data for testing...")
            add_sample_qualifying_data(cursor)
        
        conn.commit()
        print("✅ Migration completed successfully!")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Migration failed: {e}")
        sys.exit(1)
    finally:
        conn.close()


def add_sample_qualifying_data(cursor):
    """Add sample qualifying data for the first race"""
    
    # Sample qualifying results for Australian GP (race_id = 1)
    # Based on typical qualifying times and grid positions
    sample_data = [
        # (driver_id, q1_time, q2_time, q3_time, qualifying_time, grid_position, qualifying_position, grid_penalty)
        (1, 82.456, 81.789, 81.123, 81.123, 1, 1, 0),     # VER - Pole position
        (3, 82.567, 81.890, 81.234, 81.234, 2, 2, 0),     # NOR - P2
        (5, 82.678, 81.901, 81.345, 81.345, 3, 3, 0),     # LEC - P3
        (7, 82.789, 82.012, 81.456, 81.456, 4, 4, 0),     # HAM - P4
        (9, 82.890, 82.123, 81.567, 81.567, 5, 5, 0),     # ALO - P5
        (11, 83.001, 82.234, 81.678, 81.678, 6, 6, 0),    # GAS - P6
        (4, 83.112, 82.345, 81.789, 81.789, 7, 7, 0),     # PIA - P7
        (2, 83.223, 82.456, 81.890, 81.890, 8, 8, 0),     # HAD - P8
        (8, 83.334, 82.567, 81.901, 81.901, 9, 9, 0),     # RUS - P9
        (6, 83.445, 82.678, 82.012, 82.012, 10, 10, 0),   # SAI - P10
        (12, 83.556, 82.789, None, 82.789, 11, 11, 0),    # DOO - Eliminated in Q2
        (10, 83.667, 82.890, None, 82.890, 12, 12, 0),    # STR - Eliminated in Q2
        (13, 83.778, 83.001, None, 83.001, 13, 13, 0),    # HUL - Eliminated in Q2
        (14, 83.889, 83.112, None, 83.112, 14, 14, 0),    # MAG - Eliminated in Q2
        (15, 84.000, 83.223, None, 83.223, 20, 15, 5),    # TSU - 5 place grid penalty
        (16, 84.111, None, None, 84.111, 16, 16, 0),      # LIA - Eliminated in Q1
        (17, 84.222, None, None, 84.222, 17, 17, 0),      # DEV - Eliminated in Q1
        (18, 84.333, None, None, 84.333, 18, 18, 0),      # OFT - Eliminated in Q1
        (19, 84.444, None, None, 84.444, 19, 19, 0),      # BOT - Eliminated in Q1
        (20, 84.555, None, None, 84.555, 15, 20, 0),      # ZHO - Eliminated in Q1 (TSU penalty moves him up)
    ]
    
    for data in sample_data:
        cursor.execute("""
            INSERT INTO qualifying_results 
            (race_id, driver_id, q1_time, q2_time, q3_time, qualifying_time, 
             grid_position, qualifying_position, grid_penalty)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
    
    print(f"✅ Added {len(sample_data)} qualifying results for Australian GP")


def main():
    """Run the migration"""
    
    # Check if database path is provided
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        # Default to test database
        db_path = "f1_predictions_test.db"
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)
    
    print(f"🔄 Applying migration to: {db_path}")
    apply_migration(db_path)


if __name__ == "__main__":
    main()