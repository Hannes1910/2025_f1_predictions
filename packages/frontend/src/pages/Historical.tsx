import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useQuery } from 'react-query'
import { api, racesApi } from '@/services/api'
import LoadingState from '@/components/LoadingState'
import ErrorState from '@/components/ErrorState'
import { formatDate } from '@/utils/format'

interface HistoricalRace {
  id: number
  name: string
  date: string
  circuit: string
  prediction_count?: number
  result_count?: number
  status?: string
}

interface RaceWithAccuracy extends HistoricalRace {
  accuracy?: {
    correct_predictions: number
    total_predictions: number
    accuracy_percentage: number
    average_position_error: number
  }
}

export default function Historical() {
  const [racesWithAccuracy, setRacesWithAccuracy] = useState<RaceWithAccuracy[]>([])
  const [loading, setLoading] = useState(false)

  const { data: racesData, isLoading: racesLoading, error: racesError } = useQuery(
    'races',
    () => racesApi.getAll()
  )

  useEffect(() => {
    const fetchRaceAccuracy = async () => {
      if (!racesData?.races) return
      
      setLoading(true)
      try {
        const completedRaces = racesData.races.filter(race => race.status === 'completed')
        const racesWithAccuracyData: RaceWithAccuracy[] = []

        for (const race of completedRaces) {
          try {
            const raceDetails: any = await api.get(`/race/${race.id}`)
            racesWithAccuracyData.push({
              ...race,
              accuracy: raceDetails.accuracy
            } as RaceWithAccuracy)
          } catch (error) {
            // If we can't get accuracy data, just add the race without it
            racesWithAccuracyData.push(race as RaceWithAccuracy)
          }
        }

        setRacesWithAccuracy(racesWithAccuracyData)
      } catch (error) {
        console.error('Error fetching race accuracy:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchRaceAccuracy()
  }, [racesData])

  if (racesLoading) return <LoadingState />
  if (racesError) return <ErrorState message="Failed to load historical data" />

  const races = racesData?.races || []
  const completedRaces = racesWithAccuracy.filter(race => race.status === 'completed')
  const upcomingRaces = races.filter(race => race.status === 'upcoming')

  // Calculate overall statistics
  const totalPredictions = completedRaces.reduce((sum, race) => sum + (race.prediction_count || 0), 0)
  const avgAccuracy = completedRaces.length > 0 
    ? completedRaces.reduce((sum, race) => sum + (race.accuracy?.accuracy_percentage || 0), 0) / completedRaces.length
    : 0
  const avgPositionError = completedRaces.length > 0
    ? completedRaces.reduce((sum, race) => sum + (race.accuracy?.average_position_error || 0), 0) / completedRaces.length
    : 0

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-4"
      >
        <h1 className="text-4xl md:text-5xl font-bold">Historical Analysis</h1>
        <p className="text-gray-600">Track record of prediction accuracy across the season</p>
      </motion.div>

      {/* Season Overview */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-md p-6"
      >
        <h2 className="text-xl font-bold text-gray-900 mb-4">📈 2025 Season Performance</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{completedRaces.length}</div>
            <div className="text-sm text-gray-500">Completed Races</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{totalPredictions}</div>
            <div className="text-sm text-gray-500">Total Predictions</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">{avgAccuracy.toFixed(1)}%</div>
            <div className="text-sm text-gray-500">Average Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">±{avgPositionError.toFixed(1)}</div>
            <div className="text-sm text-gray-500">Avg Position Error</div>
          </div>
        </div>
      </motion.div>

      {/* Historical Accuracy Explanation */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-green-50 border border-green-200 rounded-lg p-6"
      >
        <h2 className="text-lg font-semibold text-green-900 mb-4">📊 How We Measure Accuracy</h2>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div>
            <h3 className="font-medium text-green-800 mb-2">🎯 Position Accuracy</h3>
            <p className="text-green-700">
              We count predictions as "correct" if they're within 2 positions of the actual result. 
              This accounts for the unpredictable nature of racing - crashes, strategy calls, and 
              mechanical failures can dramatically change outcomes.
            </p>
          </div>
          <div>
            <h3 className="font-medium text-green-800 mb-2">📈 Continuous Learning</h3>
            <p className="text-green-700">
              Our model learns from each race result. Early season accuracy may be lower as the 
              model adapts to new regulations, driver lineups, and team performance levels. 
              Accuracy typically improves as the season progresses.
            </p>
          </div>
        </div>
      </motion.div>

      {/* Completed Races Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-md overflow-hidden"
      >
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">🏁 Completed Races</h3>
        </div>
        
        {loading ? (
          <div className="p-8">
            <LoadingState />
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Race
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Predictions
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Accuracy
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Position Error
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Action
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {completedRaces.map((race) => (
                  <tr key={race.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{race.name}</div>
                      <div className="text-sm text-gray-500">📍 {race.circuit}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {formatDate(race.date)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">
                        {race.accuracy ? `${race.accuracy.correct_predictions}/${race.accuracy.total_predictions}` : 
                         race.prediction_count || '0'}
                      </div>
                      <div className="text-xs text-gray-500">
                        {race.accuracy ? 'correct/total' : 'predictions'}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {race.accuracy ? (
                        <div className="flex items-center">
                          <div className="text-sm font-medium text-gray-900">
                            {race.accuracy.accuracy_percentage.toFixed(1)}%
                          </div>
                          <div className="ml-2 w-16 bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-green-600 h-2 rounded-full" 
                              style={{ width: `${Math.min(race.accuracy.accuracy_percentage, 100)}%` }}
                            ></div>
                          </div>
                        </div>
                      ) : (
                        <span className="text-gray-400">-</span>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {race.accuracy ? `±${race.accuracy.average_position_error.toFixed(1)}` : '-'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-blue-600">
                      <a href={`/race/${race.id}`} className="hover:text-blue-900">
                        View Details →
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </motion.div>

      {/* Upcoming Races */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-md p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">⏳ Upcoming Races</h3>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {upcomingRaces.slice(0, 6).map((race) => (
            <div key={race.id} className="border border-gray-200 rounded-lg p-4">
              <div className="font-medium text-gray-900">{race.name}</div>
              <div className="text-sm text-gray-500 mt-1">📍 {race.circuit}</div>
              <div className="text-sm text-gray-500">{formatDate(race.date)}</div>
              <div className="mt-2">
                {(race.prediction_count || 0) > 0 ? (
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                    ✅ {race.prediction_count || 0} predictions
                  </span>
                ) : (
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                    ⏳ Pending predictions
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Model Evolution Timeline */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-md p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">🔬 Model Evolution</h3>
        <div className="space-y-4">
          <div className="bg-blue-50 border-l-4 border-blue-400 p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <div className="w-5 h-5 bg-blue-400 rounded-full"></div>
              </div>
              <div className="ml-3">
                <p className="text-sm text-blue-700">
                  <strong>May 2025:</strong> Integrated weather data from Open-Meteo API and enhanced qualifying analysis. 
                  Improved accuracy by 8% compared to baseline model.
                </p>
              </div>
            </div>
          </div>
          <div className="bg-green-50 border-l-4 border-green-400 p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <div className="w-5 h-5 bg-green-400 rounded-full"></div>
              </div>
              <div className="ml-3">
                <p className="text-sm text-green-700">
                  <strong>March 2025:</strong> Initial model deployment using 2024 FastF1 training data. 
                  Gradient boosting algorithm with sector time analysis and team performance metrics.
                </p>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}