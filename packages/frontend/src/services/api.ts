import axios from 'axios'
import type { Driver, Race, Prediction, RaceResult, ModelMetrics, PredictionAccuracy } from '@/types'

const api = axios.create({
  baseURL: (import.meta as any).env?.VITE_API_URL || '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

export const predictionsApi = {
  getLatest: async (): Promise<{ predictions: Prediction[] }> => {
    const { data } = await api.get('/predictions/latest')
    return data
  },

  getByRace: async (raceId: string): Promise<{ predictions: Prediction[] }> => {
    const { data } = await api.get(`/predictions/${raceId}`)
    return data
  },
}

export const resultsApi = {
  getByRace: async (raceId: string): Promise<{ results: RaceResult[] }> => {
    const { data } = await api.get(`/results/${raceId}`)
    return data
  },
}

export const driversApi = {
  getAll: async (): Promise<{ drivers: Driver[] }> => {
    const { data } = await api.get('/drivers')
    return data
  },
}

export const racesApi = {
  getAll: async (season?: number): Promise<{ races: Race[] }> => {
    const params = season ? { season } : {}
    const { data } = await api.get('/races', { params })
    return data
  },
}

export const analyticsApi = {
  getAccuracy: async (): Promise<{
    model_metrics: ModelMetrics[]
    prediction_accuracy: PredictionAccuracy[]
  }> => {
    const { data } = await api.get('/analytics/accuracy')
    return data
  },
}