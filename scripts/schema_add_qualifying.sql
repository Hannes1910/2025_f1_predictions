-- Add qualifying_results table for storing qualifying session data
-- This is crucial for predictions as grid position is a key predictor

CREATE TABLE IF NOT EXISTS qualifying_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    q1_time REAL,                    -- Q1 lap time in seconds
    q2_time REAL,                    -- Q2 lap time in seconds (NULL if eliminated in Q1)
    q3_time REAL,                    -- Q3 lap time in seconds (NULL if eliminated in Q2)
    qualifying_time REAL NOT NULL,    -- Best qualifying time
    grid_position INTEGER NOT NULL,   -- Final grid position (including penalties)
    qualifying_position INTEGER,      -- Position based on qualifying time (before penalties)
    grid_penalty INTEGER DEFAULT 0,   -- Grid position penalty applied
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES races(id),
    FOREIGN KEY (driver_id) REFERENCES drivers(id),
    UNIQUE(race_id, driver_id)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_qualifying_results_race_id ON qualifying_results(race_id);
CREATE INDEX IF NOT EXISTS idx_qualifying_results_driver_id ON qualifying_results(driver_id);
CREATE INDEX IF NOT EXISTS idx_qualifying_results_grid_position ON qualifying_results(race_id, grid_position);

-- Example data structure:
-- Driver qualifies P3 but has 5-place grid penalty = qualifying_position: 3, grid_position: 8
-- Q1: 1:23.456, Q2: 1:22.789, Q3: 1:22.123 = qualifying_time: 1:22.123