import axios from 'axios'
import type { Driver, Race, Prediction, RaceResult, ModelMetrics, PredictionAccuracy } from '@/types'

const apiInstance = axios.create({
  baseURL: (import.meta as any).env?.VITE_API_URL || '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Generic API utility
export const api = {
  get: async <T>(url: string): Promise<T> => {
    const { data } = await apiInstance.get(url)
    return data
  },
  post: async <T>(url: string, body?: any): Promise<T> => {
    const { data } = await apiInstance.post(url, body)
    return data
  }
}

export const predictionsApi = {
  getLatest: async (): Promise<{ predictions: Prediction[] }> => {
    const { data } = await apiInstance.get('/predictions/latest')
    return data
  },

  getByRace: async (raceId: string): Promise<{ predictions: Prediction[] }> => {
    const { data } = await apiInstance.get(`/predictions/${raceId}`)
    return data
  },
}

export const resultsApi = {
  getByRace: async (raceId: string): Promise<{ results: RaceResult[] }> => {
    const { data } = await apiInstance.get(`/results/${raceId}`)
    return data
  },
}

export const driversApi = {
  getAll: async (): Promise<{ drivers: Driver[] }> => {
    const { data } = await apiInstance.get('/drivers')
    return data
  },
}

export const racesApi = {
  getAll: async (season?: number): Promise<{ races: Race[] }> => {
    const params = season ? { season } : {}
    const { data } = await apiInstance.get('/races', { params })
    return data
  },
}

export const analyticsApi = {
  getAccuracy: async (): Promise<{
    model_metrics: ModelMetrics[]
    prediction_accuracy: PredictionAccuracy[]
  }> => {
    const { data } = await apiInstance.get('/analytics/accuracy')
    return data
  },
}