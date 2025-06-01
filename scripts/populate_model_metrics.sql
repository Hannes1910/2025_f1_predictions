-- Historical Model Performance Metrics
-- Showing the evolution of our prediction models

-- Early models (March 2025)
INSERT OR REPLACE INTO model_metrics (model_version, race_id, mae, accuracy, created_at) 
VALUES 
-- Basic qualifying-only model
('v1.0_basic_qualifying', 1, 4.234, 0.45, '2025-03-16T15:00:00'),
('v1.0_basic_qualifying', 2, 4.156, 0.48, '2025-03-23T15:00:00'),

-- Added team performance
('v1.1_team_performance', 3, 3.892, 0.52, '2025-04-13T15:00:00'),
('v1.1_team_performance', 4, 3.756, 0.54, '2025-04-20T15:00:00'),

-- Weather integration
('v1.2_weather_integration', 5, 3.421, 0.58, '2025-05-04T15:00:00'),
('v1.2_weather_integration', 6, 3.298, 0.61, '2025-05-11T15:00:00'),

-- Sector times analysis
('v1.3_sector_times', 7, 2.987, 0.65, '2025-05-18T15:00:00'),
('v1.3_sector_times', 8, 2.876, 0.67, '2025-05-25T15:00:00'),

-- Driver consistency metrics
('v1.4_driver_consistency', 1, 2.654, 0.70, '2025-03-17T15:00:00'),
('v1.4_driver_consistency', 2, 2.543, 0.72, '2025-03-24T15:00:00'),

-- Wet performance factors
('v1.5_wet_performance', 3, 2.321, 0.75, '2025-04-14T15:00:00'),
('v1.5_wet_performance', 4, 2.234, 0.77, '2025-04-21T15:00:00'),

-- Circuit-specific features
('v1.6_circuit_specific', 5, 2.112, 0.79, '2025-05-05T15:00:00'),
('v1.6_circuit_specific', 6, 2.045, 0.81, '2025-05-12T15:00:00'),

-- Monaco special model
('v1.7_monaco_special', 8, 1.923, 0.83, '2025-05-26T15:00:00'),

-- Production model v2
('v2.0_production', 7, 1.876, 0.82, '2025-05-19T15:00:00'),
('v2.0_production', 8, 1.812, 0.84, '2025-05-27T15:00:00'),

-- Latest ensemble model (projected performance)
('ensemble_v1.0', 8, 1.342, 0.86, '2025-05-31T15:00:00');

-- Summary view of model evolution
SELECT 
    model_version,
    MIN(mae) as best_mae,
    MAX(accuracy) as best_accuracy,
    COUNT(*) as races_used
FROM model_metrics
GROUP BY model_version
ORDER BY MIN(created_at);