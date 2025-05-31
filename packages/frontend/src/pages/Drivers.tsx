import { useQuery } from 'react-query'
import { motion } from 'framer-motion'
import { Users, Trophy, TrendingUp, Target } from 'lucide-react'
import { driversApi } from '@/services/api'
import LoadingState from '@/components/LoadingState'
import ErrorState from '@/components/ErrorState'
import { getTeamColor } from '@/utils/colors'
import { formatPosition } from '@/utils/format'

export default function Drivers() {
  const { data, isLoading, error, refetch } = useQuery('drivers', driversApi.getAll)

  if (isLoading) return <LoadingState />
  if (error) return <ErrorState message="Failed to load drivers" onRetry={refetch} />

  const drivers = data?.drivers || []

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-4"
      >
        <h1 className="text-4xl md:text-5xl font-bold">Drivers</h1>
        <p className="text-f1-gray-400">2025 Season Grid</p>
      </motion.div>

      {/* Driver Stats Explanation */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-yellow-50 border border-yellow-200 rounded-lg p-6"
      >
        <h2 className="text-lg font-semibold text-yellow-900 mb-4">🏎️ Understanding Driver Performance</h2>
        <div className="grid md:grid-cols-3 gap-4 text-sm">
          <div>
            <h3 className="font-medium text-yellow-800 mb-2">📊 Performance Score</h3>
            <p className="text-yellow-700">
              Shows how much better (+) or worse (-) a driver performs compared to predictions. 
              For example, +2.0 means the driver typically finishes 2 positions better than predicted, 
              indicating strong race pace and wheel-to-wheel skills.
            </p>
          </div>
          <div>
            <h3 className="font-medium text-yellow-800 mb-2">🎯 Average Position</h3>
            <p className="text-yellow-700">
              The driver's typical finishing position across all completed races. 
              This reflects overall performance throughout the season, combining 
              qualifying speed, race craft, and consistency.
            </p>
          </div>
          <div>
            <h3 className="font-medium text-yellow-800 mb-2">🔮 Predicted vs Actual</h3>
            <p className="text-yellow-700">
              Compares our AI predictions with actual race results. Green values indicate 
              drivers who consistently outperform expectations, while red shows those 
              who underperform relative to car capability and qualifying position.
            </p>
          </div>
        </div>
      </motion.div>

      {/* Stats Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="glass-effect rounded-xl p-6 text-center"
        >
          <Users className="w-8 h-8 text-f1-red mx-auto mb-2" />
          <p className="text-2xl font-bold">{drivers.length}</p>
          <p className="text-sm text-f1-gray-400">Total Drivers</p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="glass-effect rounded-xl p-6 text-center"
        >
          <Trophy className="w-8 h-8 text-yellow-500 mx-auto mb-2" />
          <p className="text-2xl font-bold">
            {drivers.reduce((sum, d) => sum + (d.total_points || 0), 0)}
          </p>
          <p className="text-sm text-f1-gray-400">Total Points</p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="glass-effect rounded-xl p-6 text-center"
        >
          <Target className="w-8 h-8 text-green-500 mx-auto mb-2" />
          <p className="text-2xl font-bold">
            {drivers.filter(d => (d.total_predictions || 0) > 0).length}
          </p>
          <p className="text-sm text-f1-gray-400">With Predictions</p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="glass-effect rounded-xl p-6 text-center"
        >
          <TrendingUp className="w-8 h-8 text-blue-500 mx-auto mb-2" />
          <p className="text-2xl font-bold">
            {Math.round(
              drivers.reduce((sum, d) => sum + (d.avg_predicted_position || 15), 0) /
              drivers.length
            )}
          </p>
          <p className="text-sm text-f1-gray-400">Avg Predicted Pos</p>
        </motion.div>
      </div>

      {/* Drivers Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {drivers.map((driver, index) => {
          const teamColor = getTeamColor(driver.team)
          const avgPredictedPos = driver.avg_predicted_position || 15
          const avgActualPos = driver.avg_actual_position || 15
          const performanceDiff = avgPredictedPos - avgActualPos

          return (
            <motion.div
              key={driver.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className="glass-effect rounded-xl p-6 hover:shadow-xl transition-all duration-300"
              style={{ borderColor: teamColor + '40' }}
            >
              {/* Driver Header */}
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="font-bold text-xl">{driver.name}</h3>
                  <div className="flex items-center space-x-3 mt-1">
                    <span
                      className="text-sm font-medium px-2 py-0.5 rounded"
                      style={{ backgroundColor: teamColor + '20', color: teamColor }}
                    >
                      {driver.team}
                    </span>
                    <span className="text-f1-gray-400 font-mono">{driver.code}</span>
                  </div>
                </div>
                {(driver.total_points ?? 0) > 0 && (
                  <div className="text-right">
                    <p className="text-2xl font-bold">{driver.total_points}</p>
                    <p className="text-xs text-f1-gray-400">points</p>
                  </div>
                )}
              </div>

              {/* Stats */}
              <div className="space-y-3">
                {(driver.total_races ?? 0) > 0 && (
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-f1-gray-400">Avg Position</span>
                    <span className="font-mono">
                      {formatPosition(Math.round(avgActualPos))}
                    </span>
                  </div>
                )}
                {(driver.total_predictions ?? 0) > 0 && (
                  <>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-f1-gray-400">Avg Predicted</span>
                      <span className="font-mono">
                        {formatPosition(Math.round(avgPredictedPos))}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-f1-gray-400">Performance</span>
                      <span
                        className={`font-mono ${
                          performanceDiff > 0
                            ? 'text-green-400'
                            : performanceDiff < 0
                            ? 'text-red-400'
                            : 'text-f1-gray-400'
                        }`}
                      >
                        {performanceDiff > 0 ? '+' : ''}{performanceDiff.toFixed(1)}
                      </span>
                    </div>
                  </>
                )}
              </div>

              {/* Progress Bars */}
              {(driver.total_races ?? 0) > 0 && (
                <div className="mt-4 space-y-2">
                  <div>
                    <div className="flex justify-between text-xs text-f1-gray-400 mb-1">
                      <span>Races</span>
                      <span>{driver.total_races ?? 0}/24</span>
                    </div>
                    <div className="h-1 bg-f1-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-blue-500 to-blue-600 rounded-full"
                        style={{ width: `${((driver.total_races ?? 0) / 24) * 100}%` }}
                      />
                    </div>
                  </div>
                  {(driver.total_predictions ?? 0) > 0 && (
                    <div>
                      <div className="flex justify-between text-xs text-f1-gray-400 mb-1">
                        <span>Predictions</span>
                        <span>{driver.total_predictions ?? 0}</span>
                      </div>
                      <div className="h-1 bg-f1-gray-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-green-500 to-green-600 rounded-full"
                          style={{
                            width: `${Math.min(
                              ((driver.total_predictions ?? 0) / (driver.total_races ?? 1)) * 100,
                              100
                            )}%`,
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}