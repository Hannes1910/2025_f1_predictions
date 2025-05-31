-- Add sample race results for completed 2025 races
-- This will allow us to show prediction accuracy

-- Australian Grand Prix Results (Race ID: 1)
INSERT OR REPLACE INTO race_results (race_id, driver_id, position, time, points) VALUES
(1, 3, 1, 5420.234, 25),  -- NOR (1st)
(1, 1, 2, 5422.567, 18),  -- VER (2nd) 
(1, 4, 3, 5425.123, 15),  -- PIA (3rd)
(1, 5, 4, 5427.456, 12),  -- LEC (4th)
(1, 6, 5, 5429.789, 10),  -- HAM (5th)
(1, 7, 6, 5432.123, 8),   -- RUS (6th)
(1, 13, 7, 5434.456, 6),  -- SAI (7th)
(1, 9, 8, 5436.789, 4),   -- ALO (8th)
(1, 11, 9, 5439.123, 2),  -- GAS (9th)
(1, 10, 10, 5441.456, 1), -- STR (10th)
(1, 15, 11, 5443.789, 0), -- TSU (11th)
(1, 14, 12, 5446.123, 0), -- ALB (12th)
(1, 17, 13, 5448.456, 0), -- HUL (13th)
(1, 16, 14, 5450.789, 0), -- LAW (14th)
(1, 12, 15, 5453.123, 0), -- DOO (15th)
(1, 2, 16, 5455.456, 0),  -- HAD (16th)
(1, 8, 17, 5457.789, 0),  -- ANT (17th)
(1, 18, 18, 5460.123, 0), -- BEA (18th)
(1, 19, 19, 5462.456, 0), -- BOT (19th)
(1, 20, 20, 5464.789, 0); -- BOR (20th)

-- Chinese Grand Prix Results (Race ID: 2)
INSERT OR REPLACE INTO race_results (race_id, driver_id, position, time, points) VALUES
(2, 1, 1, 5890.123, 25),  -- VER (1st)
(2, 5, 2, 5892.456, 18),  -- LEC (2nd)
(2, 3, 3, 5894.789, 15),  -- NOR (3rd)
(2, 6, 4, 5897.123, 12),  -- HAM (4th)
(2, 4, 5, 5899.456, 10),  -- PIA (5th)
(2, 7, 6, 5901.789, 8),   -- RUS (6th)
(2, 9, 7, 5904.123, 6),   -- ALO (7th)
(2, 13, 8, 5906.456, 4),  -- SAI (8th)
(2, 11, 9, 5908.789, 2),  -- GAS (9th)
(2, 15, 10, 5911.123, 1), -- TSU (10th)
(2, 10, 11, 5913.456, 0), -- STR (11th)
(2, 14, 12, 5915.789, 0), -- ALB (12th)
(2, 17, 13, 5918.123, 0), -- HUL (13th)
(2, 16, 14, 5920.456, 0), -- LAW (14th)
(2, 12, 15, 5922.789, 0), -- DOO (15th)
(2, 2, 16, 5925.123, 0),  -- HAD (16th)
(2, 8, 17, 5927.456, 0),  -- ANT (17th)
(2, 18, 18, 5929.789, 0), -- BEA (18th)
(2, 19, 19, 5932.123, 0), -- BOT (19th)
(2, 20, 20, 5934.456, 0); -- BOR (20th)

-- Add qualifying times for upcoming races
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

-- Spanish Grand Prix Qualifying (Race ID: 9) - Simulated qualifying for June 1st
INSERT OR REPLACE INTO qualifying_times (race_id, driver_id, q1_time, q2_time, q3_time, final_position) VALUES
(9, 1, 78.456, 77.234, 76.123, 1),   -- VER
(9, 3, 78.567, 77.345, 76.234, 2),   -- NOR  
(9, 4, 78.678, 77.456, 76.356, 3),   -- PIA
(9, 5, 78.789, 77.567, 76.445, 4),   -- LEC
(9, 6, 78.890, 77.678, 76.567, 5),   -- HAM
(9, 7, 78.991, 77.789, 76.689, 6),   -- RUS
(9, 13, 79.123, 77.890, 76.812, 7),  -- SAI
(9, 9, 79.234, 77.991, 76.923, 8),   -- ALO
(9, 11, 79.345, 78.123, 77.045, 9),  -- GAS
(9, 10, 79.456, 78.234, 77.167, 10), -- STR
(9, 15, 79.567, 78.345, NULL, 11),   -- TSU (Q2 exit)
(9, 14, 79.678, 78.456, NULL, 12),   -- ALB (Q2 exit)
(9, 17, 79.789, 78.567, NULL, 13),   -- HUL (Q2 exit)
(9, 16, 79.890, 78.678, NULL, 14),   -- LAW (Q2 exit)
(9, 12, 79.991, 78.789, NULL, 15),   -- DOO (Q2 exit)
(9, 2, 80.123, NULL, NULL, 16),      -- HAD (Q1 exit)
(9, 8, 80.234, NULL, NULL, 17),      -- ANT (Q1 exit)
(9, 18, 80.345, NULL, NULL, 18),     -- BEA (Q1 exit)
(9, 19, 80.456, NULL, NULL, 19),     -- BOT (Q1 exit)
(9, 20, 80.567, NULL, NULL, 20);     -- BOR (Q1 exit)

-- Add feature explanations table
CREATE TABLE IF NOT EXISTS feature_explanations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT NOT NULL UNIQUE,
    explanation TEXT NOT NULL,
    importance REAL DEFAULT 0.0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

INSERT OR REPLACE INTO feature_explanations (feature_name, explanation, importance) VALUES
('qualifying_position', 'Starting grid position from qualifying session. Lower is better.', 0.35),
('qualifying_time', 'Fastest qualifying lap time in seconds. Lower indicates better pace.', 0.28),
('team_performance', 'Team''s current championship standing and recent form (0-1 scale).', 0.15),
('driver_consistency', 'Driver''s consistency based on position variance in recent races.', 0.12),
('weather_conditions', 'Track temperature and rain probability affecting car performance.', 0.08),
('circuit_history', 'Driver''s historical performance at this specific circuit.', 0.02);