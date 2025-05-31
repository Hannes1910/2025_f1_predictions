-- Add prediction triggers table for tracking automated runs
CREATE TABLE IF NOT EXISTS prediction_triggers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    races_count INTEGER NOT NULL,
    trigger_type TEXT NOT NULL, -- 'cron', 'manual', 'api'
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Add index for created_at
CREATE INDEX IF NOT EXISTS idx_prediction_triggers_created ON prediction_triggers(created_at);