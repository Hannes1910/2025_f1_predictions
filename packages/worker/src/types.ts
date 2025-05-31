export interface Env {
  DB: D1Database;
  ENVIRONMENT: string;
}

export interface Driver {
  id: number;
  code: string;
  name: string;
  team: string;
}

export interface Race {
  id: number;
  season: number;
  round: number;
  name: string;
  date: string;
  circuit: string;
}

export interface Prediction {
  id: number;
  race_id: number;
  driver_id: number;
  predicted_position: number;
  predicted_time: number;
  confidence: number;
  model_version: string;
  created_at: string;
}

export interface RaceResult {
  id: number;
  race_id: number;
  driver_id: number;
  position: number;
  time: number;
  points: number;
}

export interface ModelMetrics {
  id: number;
  model_version: string;
  race_id: number | null;
  mae: number;
  accuracy: number;
  created_at: string;
}