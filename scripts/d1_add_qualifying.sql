-- D1 Migration: Add qualifying_results table
-- Run this in Cloudflare Dashboard or via wrangler d1 execute

-- Create the qualifying_results table
CREATE TABLE IF NOT EXISTS qualifying_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    q1_time REAL,
    q2_time REAL,
    q3_time REAL,
    qualifying_time REAL NOT NULL,
    grid_position INTEGER NOT NULL,
    qualifying_position INTEGER,
    grid_penalty INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES races(id),
    FOREIGN KEY (driver_id) REFERENCES drivers(id),
    UNIQUE(race_id, driver_id)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_qualifying_results_race_id ON qualifying_results(race_id);
CREATE INDEX IF NOT EXISTS idx_qualifying_results_driver_id ON qualifying_results(driver_id);
CREATE INDEX IF NOT EXISTS idx_qualifying_results_grid_position ON qualifying_results(race_id, grid_position);

-- Add sample qualifying data for Australian GP (race_id = 1)
INSERT INTO qualifying_results (race_id, driver_id, q1_time, q2_time, q3_time, qualifying_time, grid_position, qualifying_position, grid_penalty) VALUES
(1, 1, 82.456, 81.789, 81.123, 81.123, 1, 1, 0),    -- VER Pole
(1, 3, 82.567, 81.890, 81.234, 81.234, 2, 2, 0),    -- NOR P2
(1, 5, 82.678, 81.901, 81.345, 81.345, 3, 3, 0),    -- LEC P3
(1, 7, 82.789, 82.012, 81.456, 81.456, 4, 4, 0),    -- HAM P4
(1, 9, 82.890, 82.123, 81.567, 81.567, 5, 5, 0),    -- ALO P5
(1, 11, 83.001, 82.234, 81.678, 81.678, 6, 6, 0),   -- GAS P6
(1, 4, 83.112, 82.345, 81.789, 81.789, 7, 7, 0),    -- PIA P7
(1, 2, 83.223, 82.456, 81.890, 81.890, 8, 8, 0),    -- HAD P8
(1, 8, 83.334, 82.567, 81.901, 81.901, 9, 9, 0),    -- RUS P9
(1, 6, 83.445, 82.678, 82.012, 82.012, 10, 10, 0),  -- SAI P10
(1, 12, 83.556, 82.789, NULL, 82.789, 11, 11, 0),   -- DOO Q2 elim
(1, 10, 83.667, 82.890, NULL, 82.890, 12, 12, 0),   -- STR Q2 elim
(1, 13, 83.778, 83.001, NULL, 83.001, 13, 13, 0),   -- HUL Q2 elim
(1, 14, 83.889, 83.112, NULL, 83.112, 14, 14, 0),   -- MAG Q2 elim
(1, 15, 84.000, 83.223, NULL, 83.223, 20, 15, 5),   -- TSU 5-place penalty
(1, 16, 84.111, NULL, NULL, 84.111, 16, 16, 0),     -- LIA Q1 elim
(1, 17, 84.222, NULL, NULL, 84.222, 17, 17, 0),     -- DEV Q1 elim
(1, 18, 84.333, NULL, NULL, 84.333, 18, 18, 0),     -- OFT Q1 elim
(1, 19, 84.444, NULL, NULL, 84.444, 19, 19, 0),     -- BOT Q1 elim
(1, 20, 84.555, NULL, NULL, 84.555, 15, 20, 0);     -- ZHO Q1 elim