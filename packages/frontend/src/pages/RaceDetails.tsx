import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { api } from '@/services/api'
import LoadingState from '@/components/LoadingState'
import ErrorState from '@/components/ErrorState'
import { formatTime, formatPercent } from '@/utils/format'
import { getTeamColor } from '@/utils/colors'

interface RaceDetail {
  race: {
    id: number
    name: string
    date: string
    circuit: string
    prediction_count: number
    result_count: number
  }
  predictions: Array<{
    id: number
    driver_id: number
    predicted_position: number
    predicted_time: number
    confidence: number
    driver_code: string
    driver_name: string
    driver_team: string
    model_version: string
  }>
  results: Array<{
    id: number
    driver_id: number
    position: number
    time: number
    points: number
    driver_code: string
    driver_name: string
    driver_team: string
  }>
  qualifying: Array<{
    id: number
    driver_id: number
    q1_time: number
    q2_time: number
    q3_time: number
    final_position: number
    driver_code: string
    driver_name: string
    driver_team: string
  }>
  accuracy?: {
    correct_predictions: number
    total_predictions: number
    accuracy_percentage: number
    average_position_error: number
  }
  weather: {
    temperature: number
    rain_probability: number
    conditions: string
  }
  features: Array<{
    feature_name: string
    explanation: string
    importance: number
  }>
}

export default function RaceDetails() {
  const { raceId } = useParams<{ raceId: string }>()
  const [raceDetail, setRaceDetail] = useState<RaceDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'predictions' | 'results' | 'qualifying' | 'analysis'>('predictions')

  useEffect(() => {
    const fetchRaceDetails = async () => {
      if (!raceId) return
      
      try {
        setLoading(true)
        const data = await api.get<RaceDetail>(`/race/${raceId}`)
        setRaceDetail(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch race details')
      } finally {
        setLoading(false)
      }
    }

    fetchRaceDetails()
  }, [raceId])

  if (loading) return <LoadingState />
  if (error) return <ErrorState message={error} />
  if (!raceDetail) return <ErrorState message="Race not found" />

  const { race, predictions, results, qualifying, accuracy, weather, features } = raceDetail
  const isCompleted = results.length > 0
  const hasQualifying = qualifying.length > 0

  return (
    <div className="space-y-8">
      {/* Race Header */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">{race.name}</h1>
            <p className="text-lg text-gray-600 mt-2">
              {new Date(race.date).toLocaleDateString('en-US', { 
                weekday: 'long', 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric' 
              })}
            </p>
            <p className="text-gray-500">📍 {race.circuit}</p>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-500">Status</div>
            <div className={`text-lg font-semibold ${
              isCompleted ? 'text-green-600' : 'text-blue-600'
            }`}>
              {isCompleted ? '🏁 Completed' : '⏳ Upcoming'}
            </div>
          </div>
        </div>

        {/* Weather Info */}
        <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 bg-gray-50 rounded-lg p-4">
          <div>
            <div className="text-sm text-gray-500">Temperature</div>
            <div className="text-lg font-semibold">🌡️ {weather.temperature}°C</div>
          </div>
          <div>
            <div className="text-sm text-gray-500">Rain Chance</div>
            <div className="text-lg font-semibold">🌧️ {formatPercent(weather.rain_probability)}</div>
          </div>
          <div>
            <div className="text-sm text-gray-500">Conditions</div>
            <div className="text-lg font-semibold">☀️ {weather.conditions}</div>
          </div>
        </div>
      </div>

      {/* Prediction Accuracy (if race is completed) */}
      {accuracy && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">🎯 Prediction Accuracy</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {accuracy.correct_predictions}/{accuracy.total_predictions}
              </div>
              <div className="text-sm text-gray-500">Correct Predictions</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {accuracy.accuracy_percentage.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-500">Accuracy Rate</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                ±{accuracy.average_position_error.toFixed(1)}
              </div>
              <div className="text-sm text-gray-500">Avg Position Error</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {predictions[0]?.model_version || 'N/A'}
              </div>
              <div className="text-sm text-gray-500">Model Version</div>
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="bg-white rounded-lg shadow-md">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6">
            {[
              { key: 'predictions', label: '🔮 Predictions', count: predictions.length },
              { key: 'results', label: '🏆 Results', count: results.length, disabled: !isCompleted },
              { key: 'qualifying', label: '⏱️ Qualifying', count: qualifying.length, disabled: !hasQualifying },
              { key: 'analysis', label: '📊 Analysis', count: features.length }
            ].map(({ key, label, count, disabled }) => (
              <button
                key={key}
                onClick={() => !disabled && setActiveTab(key as any)}
                disabled={disabled}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === key
                    ? 'border-blue-500 text-blue-600'
                    : disabled
                    ? 'border-transparent text-gray-400 cursor-not-allowed'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {label} {count > 0 && `(${count})`}
              </button>
            ))}
          </nav>
        </div>

        <div className="p-6">
          {/* Predictions Tab */}
          {activeTab === 'predictions' && (
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold">Race Predictions</h3>
                <div className="text-sm text-gray-500">
                  Model: {predictions[0]?.model_version || 'N/A'}
                </div>
              </div>
              
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Position
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Driver
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Team
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Predicted Time
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Confidence
                      </th>
                      {isCompleted && (
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Actual Result
                        </th>
                      )}
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {predictions.map((prediction) => {
                      const actualResult = results.find(r => r.driver_id === prediction.driver_id)
                      const positionDiff = actualResult ? actualResult.position - prediction.predicted_position : null
                      
                      return (
                        <tr key={prediction.id} className="hover:bg-gray-50">
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="flex items-center">
                              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm ${
                                prediction.predicted_position <= 3 ? 'bg-yellow-500' :
                                prediction.predicted_position <= 10 ? 'bg-gray-400' : 'bg-gray-300'
                              }`}>
                                {prediction.predicted_position}
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm font-medium text-gray-900">
                              {prediction.driver_code}
                            </div>
                            <div className="text-sm text-gray-500">
                              {prediction.driver_name}
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span 
                              className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium text-white"
                              style={{ backgroundColor: getTeamColor(prediction.driver_team) }}
                            >
                              {prediction.driver_team}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {formatTime(prediction.predicted_time)}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="text-sm text-gray-900">
                              {formatPercent(prediction.confidence)}
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-1.5">
                              <div 
                                className="bg-blue-600 h-1.5 rounded-full" 
                                style={{ width: `${prediction.confidence * 100}%` }}
                              ></div>
                            </div>
                          </td>
                          {isCompleted && (
                            <td className="px-6 py-4 whitespace-nowrap">
                              {actualResult ? (
                                <div className="flex items-center">
                                  <span className="text-sm font-medium">P{actualResult.position}</span>
                                  {positionDiff !== null && (
                                    <span className={`ml-2 text-xs px-2 py-1 rounded ${
                                      positionDiff === 0 ? 'bg-green-100 text-green-800' :
                                      Math.abs(positionDiff) <= 2 ? 'bg-yellow-100 text-yellow-800' :
                                      'bg-red-100 text-red-800'
                                    }`}>
                                      {positionDiff > 0 ? `+${positionDiff}` : positionDiff}
                                    </span>
                                  )}
                                </div>
                              ) : (
                                <span className="text-gray-400">-</span>
                              )}
                            </td>
                          )}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Results Tab */}
          {activeTab === 'results' && isCompleted && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Race Results</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Position
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Driver
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Team
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Time
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Points
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {results.map((result) => (
                      <tr key={result.id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm ${
                            result.position <= 3 ? 'bg-yellow-500' :
                            result.position <= 10 ? 'bg-gray-400' : 'bg-gray-300'
                          }`}>
                            {result.position}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900">
                            {result.driver_code}
                          </div>
                          <div className="text-sm text-gray-500">
                            {result.driver_name}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span 
                            className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium text-white"
                            style={{ backgroundColor: getTeamColor(result.driver_team) }}
                          >
                            {result.driver_team}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {formatTime(result.time)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {result.points}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Qualifying Tab */}
          {activeTab === 'qualifying' && hasQualifying && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Qualifying Results</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Position
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Driver
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Q1
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Q2
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Q3
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {qualifying.map((qual) => (
                      <tr key={qual.id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm ${
                            qual.final_position <= 3 ? 'bg-yellow-500' :
                            qual.final_position <= 10 ? 'bg-gray-400' : 'bg-gray-300'
                          }`}>
                            {qual.final_position}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900">
                            {qual.driver_code}
                          </div>
                          <div className="text-sm text-gray-500">
                            {qual.driver_name}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {qual.q1_time ? formatTime(qual.q1_time) : '-'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {qual.q2_time ? formatTime(qual.q2_time) : '-'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {qual.q3_time ? formatTime(qual.q3_time) : '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Analysis Tab */}
          {activeTab === 'analysis' && (
            <div className="space-y-6">
              <h3 className="text-lg font-semibold">🧠 Prediction Analysis</h3>
              
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 className="font-medium text-blue-900 mb-2">How Predictions Are Made</h4>
                <p className="text-sm text-blue-800">
                  Our ML model analyzes multiple factors to predict race outcomes. Each factor contributes 
                  differently to the final prediction based on historical data and statistical analysis.
                </p>
              </div>

              <div className="space-y-4">
                <h4 className="font-medium text-gray-900">Feature Importance</h4>
                {features.map((feature) => (
                  <div key={feature.feature_name} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex justify-between items-start mb-2">
                      <h5 className="font-medium text-gray-900 capitalize">
                        {feature.feature_name.replace('_', ' ')}
                      </h5>
                      <span className="text-sm font-semibold text-blue-600">
                        {(feature.importance * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full" 
                        style={{ width: `${feature.importance * 100}%` }}
                      ></div>
                    </div>
                    <p className="text-sm text-gray-600">{feature.explanation}</p>
                  </div>
                ))}
              </div>

              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <h4 className="font-medium text-yellow-900 mb-2">💡 Understanding the Model</h4>
                <ul className="text-sm text-yellow-800 space-y-1">
                  <li>• Predictions are based on gradient boosting machine learning algorithms</li>
                  <li>• Weather data is integrated from free Open-Meteo API</li>
                  <li>• Model is trained on real FastF1 telemetry data from 2024 season</li>
                  <li>• Confidence scores reflect the model's certainty in each prediction</li>
                  <li>• Features are weighted by their historical predictive power</li>
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}