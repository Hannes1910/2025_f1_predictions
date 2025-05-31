export interface Driver {
  id: number
  code: string
  name: string
  team: string
  total_predictions?: number
  avg_predicted_position?: number
  total_races?: number
  avg_actual_position?: number
  total_points?: number
}

export interface Race {
  id: number
  season: number
  round: number
  name: string
  date: string
  circuit: string
  status?: 'completed' | 'today' | 'upcoming'
  prediction_count?: number
  result_count?: number
}

export interface Prediction {
  id: number
  race_id: number
  driver_id: number
  driver_name: string
  driver_code: string
  driver_team: string
  predicted_position: number
  predicted_time: number
  confidence: number
  model_version: string
  created_at: string
  race_name?: string
  race_date?: string
  race_circuit?: string
}

export interface RaceResult {
  id: number
  race_id: number
  driver_id: number
  driver_name: string
  driver_code: string
  driver_team: string
  position: number
  time: number
  points: number
  race_name?: string
  race_date?: string
}

export interface ModelMetrics {
  id: number
  model_version: string
  race_id: number | null
  mae: number
  accuracy: number
  created_at: string
  race_name?: string
  race_date?: string
}

export interface PredictionAccuracy {
  race_name: string
  race_date: string
  avg_position_error: number
  avg_time_error: number
  prediction_count: number
}

export interface TeamColors {
  [key: string]: string
}