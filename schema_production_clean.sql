-- F1 Predictions Database Schema - PRODUCTION CLEAN VERSION
-- No mock data, no redundancy, proper constraints

-- ===================================================================
-- CORE TABLES
-- ===================================================================

-- Races: Official F1 calendar data only
CREATE TABLE IF NOT EXISTS races (
    id INTEGER PRIMARY KEY,
    season INTEGER NOT NULL CHECK (season >= 1950 AND season <= 2030),
    round INTEGER NOT NULL CHECK (round >= 1 AND round <= 25),
    name TEXT NOT NULL,
    date DATE NOT NULL,
    circuit TEXT NOT NULL,
    country TEXT NOT NULL,
    sprint_weekend BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(season, round),
    CHECK (length(name) > 0),
    CHECK (length(circuit) > 0),
    CHECK (length(country) > 0)
);

-- Drivers: Real F1 drivers only
CREATE TABLE IF NOT EXISTS drivers (
    id INTEGER PRIMARY KEY,
    driver_ref TEXT UNIQUE NOT NULL,  -- 'max_verstappen', 'lewis_hamilton'
    code TEXT UNIQUE NOT NULL,        -- 'VER', 'HAM'
    forename TEXT NOT NULL,
    surname TEXT NOT NULL,
    dob DATE,
    nationality TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(forename, surname),
    CHECK (length(driver_ref) >= 3),
    CHECK (length(code) = 3),
    CHECK (length(forename) > 0),
    CHECK (length(surname) > 0)
);

-- Team entries per season (drivers change teams)
CREATE TABLE IF NOT EXISTS team_entries (
    id INTEGER PRIMARY KEY,
    season INTEGER NOT NULL CHECK (season >= 1950 AND season <= 2030),
    team_name TEXT NOT NULL,
    driver_id INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (driver_id) REFERENCES drivers(id) ON DELETE RESTRICT,
    UNIQUE(season, driver_id),
    CHECK (length(team_name) > 0)
);

-- Qualifying results: MOST IMPORTANT for predictions
CREATE TABLE IF NOT EXISTS qualifying_results (
    id INTEGER PRIMARY KEY,
    race_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    q1_time_ms INTEGER CHECK (q1_time_ms > 0),              -- NULL if session not run
    q2_time_ms INTEGER CHECK (q2_time_ms > 0),              -- NULL if eliminated in Q1
    q3_time_ms INTEGER CHECK (q3_time_ms > 0),              -- NULL if eliminated in Q2
    best_time_ms INTEGER NOT NULL CHECK (best_time_ms > 0), -- Best qualifying time
    qualifying_position INTEGER NOT NULL CHECK (qualifying_position >= 1 AND qualifying_position <= 20),
    grid_position INTEGER NOT NULL CHECK (grid_position >= 1 AND grid_position <= 20),
    grid_penalty INTEGER DEFAULT 0 CHECK (grid_penalty >= 0),
    session_date DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES races(id) ON DELETE RESTRICT,
    FOREIGN KEY (driver_id) REFERENCES drivers(id) ON DELETE RESTRICT,
    UNIQUE(race_id, driver_id),
    UNIQUE(race_id, qualifying_position),
    UNIQUE(race_id, grid_position)
);

-- Race results: Actual race outcomes
CREATE TABLE IF NOT EXISTS race_results (
    id INTEGER PRIMARY KEY,
    race_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    grid_position INTEGER NOT NULL CHECK (grid_position >= 1 AND grid_position <= 20),
    final_position INTEGER CHECK (final_position >= 1 AND final_position <= 20), -- NULL for DNF
    race_time_ms INTEGER CHECK (race_time_ms > 0),         -- Total race time, NULL for DNF
    fastest_lap_time_ms INTEGER CHECK (fastest_lap_time_ms > 0), -- Fastest lap time
    points INTEGER NOT NULL DEFAULT 0 CHECK (points >= 0 AND points <= 26), -- Max 26 points (25 + fastest lap)
    status TEXT NOT NULL CHECK (status IN ('Finished', 'DNF', 'DSQ', 'DNS')),
    laps_completed INTEGER NOT NULL CHECK (laps_completed >= 0),
    session_date DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES races(id) ON DELETE RESTRICT,
    FOREIGN KEY (driver_id) REFERENCES drivers(id) ON DELETE RESTRICT,
    UNIQUE(race_id, driver_id),
    -- If finished, must have position
    CHECK ((status = 'Finished' AND final_position IS NOT NULL) OR status != 'Finished'),
    -- If finished, must have race time
    CHECK ((status = 'Finished' AND race_time_ms IS NOT NULL) OR status != 'Finished')
);

-- Predictions: ML model outputs ONLY
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY,
    race_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    predicted_position REAL NOT NULL CHECK (predicted_position >= 1 AND predicted_position <= 20),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES races(id) ON DELETE RESTRICT,
    FOREIGN KEY (driver_id) REFERENCES drivers(id) ON DELETE RESTRICT,
    UNIQUE(race_id, driver_id, model_version),
    CHECK (length(model_version) > 0)
);

-- ===================================================================
-- PERFORMANCE INDEXES
-- ===================================================================

-- Race queries
CREATE INDEX IF NOT EXISTS idx_races_season_round ON races(season, round);
CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);
CREATE INDEX IF NOT EXISTS idx_races_season ON races(season);

-- Driver queries
CREATE INDEX IF NOT EXISTS idx_drivers_code ON drivers(code);
CREATE INDEX IF NOT EXISTS idx_drivers_ref ON drivers(driver_ref);

-- Team entries
CREATE INDEX IF NOT EXISTS idx_team_entries_season ON team_entries(season);
CREATE INDEX IF NOT EXISTS idx_team_entries_team ON team_entries(team_name);

-- Qualifying results (CRITICAL for ML)
CREATE INDEX IF NOT EXISTS idx_qualifying_race_id ON qualifying_results(race_id);
CREATE INDEX IF NOT EXISTS idx_qualifying_driver_id ON qualifying_results(driver_id);
CREATE INDEX IF NOT EXISTS idx_qualifying_session_date ON qualifying_results(session_date);
CREATE INDEX IF NOT EXISTS idx_qualifying_grid_position ON qualifying_results(race_id, grid_position);

-- Race results
CREATE INDEX IF NOT EXISTS idx_race_results_race_id ON race_results(race_id);
CREATE INDEX IF NOT EXISTS idx_race_results_driver_id ON race_results(driver_id);
CREATE INDEX IF NOT EXISTS idx_race_results_session_date ON race_results(session_date);
CREATE INDEX IF NOT EXISTS idx_race_results_points ON race_results(points);

-- Predictions
CREATE INDEX IF NOT EXISTS idx_predictions_race_id ON predictions(race_id);
CREATE INDEX IF NOT EXISTS idx_predictions_driver_id ON predictions(driver_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);

-- ===================================================================
-- VIEWS FOR COMMON QUERIES
-- ===================================================================

-- Current season driver standings
CREATE VIEW IF NOT EXISTS current_driver_standings AS
SELECT 
    d.id,
    d.code,
    d.forename || ' ' || d.surname as full_name,
    te.team_name,
    COALESCE(SUM(rr.points), 0) as total_points,
    COUNT(rr.id) as races_completed
FROM drivers d
JOIN team_entries te ON d.id = te.driver_id
LEFT JOIN race_results rr ON d.id = rr.driver_id
JOIN races r ON rr.race_id = r.id AND r.season = te.season
WHERE te.season = (SELECT MAX(season) FROM races)
GROUP BY d.id, d.code, d.forename, d.surname, te.team_name
ORDER BY total_points DESC;

-- Driver recent form (last 3 races)
CREATE VIEW IF NOT EXISTS driver_recent_form AS
SELECT 
    d.id as driver_id,
    d.code,
    AVG(
        CASE 
            WHEN rr.final_position IS NOT NULL THEN rr.final_position
            ELSE 21  -- DNF penalty for average
        END
    ) as avg_recent_position,
    COUNT(rr.id) as recent_races
FROM drivers d
JOIN race_results rr ON d.id = rr.driver_id
JOIN races r ON rr.race_id = r.id
WHERE r.session_date >= (
    SELECT MAX(session_date) - INTERVAL '60 days'
    FROM race_results 
    WHERE driver_id = d.id
)
GROUP BY d.id, d.code
HAVING COUNT(rr.id) > 0;