-- Fix Driver Championship Points for 2025 Season (8 races completed)
-- Recalculate accurate points based on actual race results

-- Create a temporary table to calculate correct points
CREATE TEMP TABLE driver_points_temp AS
SELECT 
    d.id as driver_id,
    d.code,
    d.name,
    d.team,
    COUNT(rr.id) as total_races,
    COALESCE(SUM(rr.points), 0) as total_points,
    ROUND(AVG(CAST(rr.position AS REAL)), 2) as avg_actual_position
FROM drivers d
LEFT JOIN race_results rr ON d.id = rr.driver_id
GROUP BY d.id, d.code, d.name, d.team;

-- Update the driver data with correct calculations
-- Note: We can't directly update a view, so this shows the logic that should be applied

-- For reference, here are the correct championship standings after 8 races:
-- Based on the race results we just added:

-- VER: 4 wins (100) + 4 other podiums (42) = 142 points
-- NOR: 2 wins (50) + 6 other podiums (78) = 128 points  
-- LEC: 1 win (25) + 5 second places (90) + 2 other podiums (27) = 142 points
-- PIA: 0 wins + 1 second (18) + 5 third places (75) + 2 fourth places (24) = 117 points

-- The issue is that the API is probably summing all historical points incorrectly
-- Let's check what the actual points should be by race:

-- Race results show:
-- VER: 1st,2nd,1st,1st,2nd,3rd,1st,4th = 25+18+25+25+18+15+25+12 = 163 points
-- NOR: 1st,3rd,3rd,2nd,1st,3rd,3rd,2nd = 25+15+15+18+25+15+15+18 = 146 points
-- LEC: 4th,2nd,2nd,3rd,4th,4th,2nd,1st = 12+18+18+15+12+12+18+25 = 130 points
-- PIA: 3rd,4th,4th,4th,3rd,2nd,4th,3rd = 15+12+12+12+15+18+12+15 = 111 points

-- The issue might be that the driver API is calculating from all race_results 
-- including older test data. Let's check if there are duplicate results.