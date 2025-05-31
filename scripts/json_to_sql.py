#!/usr/bin/env python3
"""Convert JSON exports to SQL for D1 import"""

import json
import os
from pathlib import Path

def json_to_sql(table_name, json_file, output_file):
    """Convert JSON data to SQL INSERT statements"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if not data:
        print(f"No data in {json_file}")
        return
    
    with open(output_file, 'w') as f:
        for row in data:
            # Get columns and values
            columns = ', '.join(row.keys())
            values = []
            
            for v in row.values():
                if v is None:
                    values.append('NULL')
                elif isinstance(v, (int, float)):
                    values.append(str(v))
                else:
                    # Escape single quotes in strings
                    escaped = str(v).replace("'", "''")
                    values.append(f"'{escaped}'")
            
            values_str = ', '.join(values)
            f.write(f"INSERT INTO {table_name} ({columns}) VALUES ({values_str});\n")
    
    print(f"✓ Created {output_file}")

def main():
    input_dir = Path('data_export')
    output_dir = Path('data_export/sql')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Looking for JSON files in: {input_dir.absolute()}")
    
    # Define tables in order of dependencies
    tables = [
        'drivers',
        'races', 
        'predictions',
        'race_results',
        'model_metrics',
        'feature_data'
    ]
    
    for table in tables:
        json_file = input_dir / f'{table}.json'
        sql_file = output_dir / f'{table}.sql'
        
        if json_file.exists():
            json_to_sql(table, json_file, sql_file)
        else:
            print(f"❌ File not found: {json_file}")

if __name__ == '__main__':
    main()