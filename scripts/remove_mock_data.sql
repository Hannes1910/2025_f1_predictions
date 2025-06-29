-- Remove all mock/demo data from the database
-- Only keep the schema and real data fetched from APIs

-- Create tables if they don't exist
CREATE TABLE IF NOT EXISTS qualifying_times (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    q1_time REAL,
    q2_time REAL,
    q3_time REAL,
    final_position INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES races(id),
    FOREIGN KEY (driver_id) REFERENCES drivers(id),
    UNIQUE(race_id, driver_id)
);

CREATE TABLE IF NOT EXISTS feature_explanations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT NOT NULL UNIQUE,
    explanation TEXT NOT NULL,
    importance REAL DEFAULT 0.0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Clear all mock race results
DELETE FROM race_results;

-- Clear all mock qualifying times
DELETE FROM qualifying_times;

-- Clear mock model metrics
DELETE FROM model_metrics WHERE model_version LIKE 'v%' OR model_version = 'demo_v1.05';

-- Clear mock feature explanations
DELETE FROM feature_explanations;

-- Keep the predictions as they are generated by your ML models
-- Keep the drivers and races tables as they contain real 2025 season data

-- Reset auto-increment counters
DELETE FROM sqlite_sequence WHERE name IN ('race_results', 'qualifying_times', 'model_metrics', 'feature_explanations');

-- Verify cleanup
SELECT 'race_results count:', COUNT(*) FROM race_results
UNION ALL
SELECT 'model_metrics count:', COUNT(*) FROM model_metrics;