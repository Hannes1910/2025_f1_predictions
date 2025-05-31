-- Production 2025 F1 Data Setup for Cloudflare D1
-- Real 2025 season data

-- Insert 2025 drivers with current teams
INSERT OR REPLACE INTO drivers (id, code, name, team) VALUES
-- Red Bull Racing
(1, 'VER', 'Max Verstappen', 'Red Bull Racing'),
(2, 'HAD', 'Isack Hadjar', 'Red Bull Racing'),

-- McLaren
(3, 'NOR', 'Lando Norris', 'McLaren'),
(4, 'PIA', 'Oscar Piastri', 'McLaren'),

-- Ferrari
(5, 'LEC', 'Charles Leclerc', 'Ferrari'),
(6, 'HAM', 'Lewis Hamilton', 'Ferrari'),

-- Mercedes
(7, 'RUS', 'George Russell', 'Mercedes'),
(8, 'ANT', 'Andrea Kimi Antonelli', 'Mercedes'),

-- Aston Martin
(9, 'ALO', 'Fernando Alonso', 'Aston Martin'),
(10, 'STR', 'Lance Stroll', 'Aston Martin'),

-- Alpine
(11, 'GAS', 'Pierre Gasly', 'Alpine'),
(12, 'DOO', 'Jack Doohan', 'Alpine'),

-- Williams
(13, 'SAI', 'Carlos Sainz', 'Williams'),
(14, 'ALB', 'Alexander Albon', 'Williams'),

-- Racing Bulls
(15, 'TSU', 'Yuki Tsunoda', 'Racing Bulls'),
(16, 'LAW', 'Liam Lawson', 'Racing Bulls'),

-- Haas
(17, 'HUL', 'Nico Hulkenberg', 'Haas'),
(18, 'BEA', 'Oliver Bearman', 'Haas'),

-- Kick Sauber
(19, 'BOT', 'Valtteri Bottas', 'Kick Sauber'),
(20, 'BOR', 'Gabriel Bortoleto', 'Kick Sauber');

-- Insert 2025 F1 Calendar (real dates)
INSERT OR REPLACE INTO races (id, season, round, name, date, circuit) VALUES
(1, 2025, 1, 'Australian Grand Prix', '2025-03-16', 'Australia'),
(2, 2025, 2, 'Chinese Grand Prix', '2025-03-23', 'China'),
(3, 2025, 3, 'Japanese Grand Prix', '2025-04-13', 'Japan'),
(4, 2025, 4, 'Bahrain Grand Prix', '2025-04-20', 'Bahrain'),
(5, 2025, 5, 'Saudi Arabian Grand Prix', '2025-05-04', 'Saudi Arabia'),
(6, 2025, 6, 'Miami Grand Prix', '2025-05-11', 'USA'),
(7, 2025, 7, 'Emilia Romagna Grand Prix', '2025-05-18', 'Italy'),
(8, 2025, 8, 'Monaco Grand Prix', '2025-05-25', 'Monaco'),
(9, 2025, 9, 'Spanish Grand Prix', '2025-06-01', 'Spain'),
(10, 2025, 10, 'Canadian Grand Prix', '2025-06-15', 'Canada'),
(11, 2025, 11, 'Austrian Grand Prix', '2025-06-29', 'Austria'),
(12, 2025, 12, 'British Grand Prix', '2025-07-06', 'Great Britain'),
(13, 2025, 13, 'Hungarian Grand Prix', '2025-07-20', 'Hungary'),
(14, 2025, 14, 'Belgian Grand Prix', '2025-07-27', 'Belgium'),
(15, 2025, 15, 'Dutch Grand Prix', '2025-08-31', 'Netherlands'),
(16, 2025, 16, 'Italian Grand Prix', '2025-09-07', 'Italy'),
(17, 2025, 17, 'Azerbaijan Grand Prix', '2025-09-21', 'Azerbaijan'),
(18, 2025, 18, 'Singapore Grand Prix', '2025-10-05', 'Singapore'),
(19, 2025, 19, 'United States Grand Prix', '2025-10-19', 'USA'),
(20, 2025, 20, 'Mexican Grand Prix', '2025-10-26', 'Mexico'),
(21, 2025, 21, 'Brazilian Grand Prix', '2025-11-09', 'Brazil'),
(22, 2025, 22, 'Las Vegas Grand Prix', '2025-11-22', 'Las Vegas'),
(23, 2025, 23, 'Qatar Grand Prix', '2025-11-30', 'Qatar'),
(24, 2025, 24, 'Abu Dhabi Grand Prix', '2025-12-07', 'Abu Dhabi');

-- Create teams table and insert current teams
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    constructor TEXT NOT NULL
);

INSERT OR REPLACE INTO teams (id, name, constructor) VALUES
(1, 'Red Bull Racing', 'Red Bull'),
(2, 'McLaren', 'McLaren'),
(3, 'Ferrari', 'Ferrari'),
(4, 'Mercedes', 'Mercedes'),
(5, 'Aston Martin', 'Aston Martin'),
(6, 'Alpine', 'Alpine'),
(7, 'Williams', 'Williams'),
(8, 'Racing Bulls', 'Racing Bulls'),
(9, 'Haas', 'Haas'),
(10, 'Kick Sauber', 'Kick Sauber');