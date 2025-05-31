#!/usr/bin/env python3
"""
Generate demo predictions for upcoming races using API
"""

import urllib.request
import urllib.parse
import json
import random
from datetime import datetime

API_BASE = "https://f1-predictions-api.vprifntqe.workers.dev"

def get_upcoming_races():
    """Get upcoming races from API"""
    try:
        url = f"{API_BASE}/api/races"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            races = data.get('races', [])
            # Filter for upcoming races with no predictions
            upcoming = [r for r in races if r['status'] == 'upcoming' and r['prediction_count'] == 0]
            return upcoming[:3]  # Get next 3 races
    except Exception as e:
        print(f"Error getting races: {e}")
        return []

def get_drivers():
    """Get all drivers from API"""
    try:
        url = f"{API_BASE}/api/drivers"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            return data.get('drivers', [])
    except Exception as e:
        print(f"Error getting drivers: {e}")
        return []

def generate_race_predictions(race, drivers):
    """Generate realistic predictions for a race"""
    # Driver performance tiers based on 2024 season
    tier1 = ['VER', 'NOR', 'PIA', 'LEC', 'HAM']  # Top contenders
    tier2 = ['RUS', 'SAI', 'ALO', 'GAS', 'STR']  # Strong midfield
    tier3 = ['TSU', 'ALB', 'HUL', 'LAW', 'DOO']  # Competitive drivers
    tier4 = ['HAD', 'ANT', 'BEA', 'BOT', 'BOR']  # New/developing drivers
    
    # Create driver lookup
    driver_lookup = {d['code']: d for d in drivers}
    
    predictions = []
    position = 1
    
    # Assign positions based on tiers with some randomness
    all_tiers = [tier1, tier2, tier3, tier4]
    
    for tier in all_tiers:
        tier_drivers = [code for code in tier if code in driver_lookup]
        random.shuffle(tier_drivers)
        
        for driver_code in tier_drivers:
            driver = driver_lookup[driver_code]
            
            # Calculate confidence based on tier and position
            if position <= 5:
                confidence = random.uniform(0.75, 0.90)
            elif position <= 10:
                confidence = random.uniform(0.65, 0.80)
            else:
                confidence = random.uniform(0.50, 0.70)
            
            # Generate realistic lap time (base + position variance)
            base_time = 75.0  # Base lap time in seconds
            time_variance = (position - 1) * 0.1 + random.uniform(-0.2, 0.2)
            predicted_time = base_time + time_variance
            
            predictions.append({
                "driver_id": driver['id'],
                "predicted_position": position,
                "predicted_time": round(predicted_time, 3),
                "confidence": round(confidence, 3)
            })
            
            position += 1
            if position > 20:  # Limit to 20 drivers
                break
        
        if position > 20:
            break
    
    return predictions

def create_predictions_direct(race, predictions):
    """Create predictions directly in D1 database via SQL"""
    timestamp = datetime.now().isoformat()
    model_version = f"demo_v1.{datetime.now().strftime('%m%d')}"
    
    sql_statements = []
    
    # Delete existing predictions for this race
    sql_statements.append(f"DELETE FROM predictions WHERE race_id = {race['id']};")
    
    # Insert new predictions
    for pred in predictions:
        sql = f"""INSERT INTO predictions 
                  (race_id, driver_id, predicted_position, predicted_time, confidence, model_version, created_at)
                  VALUES ({race['id']}, {pred['driver_id']}, {pred['predicted_position']}, 
                         {pred['predicted_time']}, {pred['confidence']}, '{model_version}', '{timestamp}');"""
        sql_statements.append(sql)
    
    # Insert model metrics
    mae = round(random.uniform(1.5, 2.5), 2)
    accuracy = round(random.uniform(0.75, 0.85), 2)
    sql_statements.append(f"""INSERT INTO model_metrics 
                              (model_version, race_id, mae, accuracy, created_at)
                              VALUES ('{model_version}', {race['id']}, {mae}, {accuracy}, '{timestamp}');""")
    
    return sql_statements

def main():
    print("🤖 Generating Demo F1 Predictions\n")
    
    # Get data
    races = get_upcoming_races()
    drivers = get_drivers()
    
    if not races:
        print("❌ No upcoming races found or all races already have predictions")
        return
    
    if not drivers:
        print("❌ No drivers found")
        return
    
    print(f"📊 Found {len(races)} upcoming races and {len(drivers)} drivers")
    
    all_sql = []
    
    for race in races:
        print(f"\n🏁 Generating predictions for: {race['name']} ({race['date']})")
        
        predictions = generate_race_predictions(race, drivers)
        sql_statements = create_predictions_direct(race, predictions)
        all_sql.extend(sql_statements)
        
        # Show top 5 predictions
        print("   Top 5 predictions:")
        for i, pred in enumerate(predictions[:5]):
            driver = next(d for d in drivers if d['id'] == pred['driver_id'])
            print(f"      {pred['predicted_position']}. {driver['code']} - {driver['name']:25} ({pred['confidence']:.0%})")
    
    # Write SQL file
    sql_file = "scripts/demo_predictions.sql"
    with open(sql_file, 'w') as f:
        f.write("-- Demo predictions for upcoming F1 races\n\n")
        f.write('\n'.join(all_sql))
    
    print(f"\n✅ Generated SQL file: {sql_file}")
    print("📋 To apply predictions, run:")
    print(f"   cd packages/worker && npx wrangler d1 execute f1-predictions --file=../../{sql_file} --remote")

if __name__ == "__main__":
    main()