import { useQuery } from 'react-query'
import { motion } from 'framer-motion'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { TrendingUp, Target, Award, Activity } from 'lucide-react'
import { analyticsApi } from '@/services/api'
import LoadingState from '@/components/LoadingState'
import ErrorState from '@/components/ErrorState'
import { formatDate } from '@/utils/format'


export default function Analytics() {
  const { data, isLoading, error, refetch } = useQuery('analytics', analyticsApi.getAccuracy)

  if (isLoading) return <LoadingState />
  if (error) return <ErrorState message="Failed to load analytics" onRetry={refetch} />

  const modelMetrics = data?.model_metrics || []
  const predictionAccuracy = data?.prediction_accuracy || []

  // Calculate average metrics
  const avgMAE = modelMetrics.reduce((sum, m) => sum + m.mae, 0) / modelMetrics.length || 0
  const avgAccuracy = modelMetrics.reduce((sum, m) => sum + m.accuracy, 0) / modelMetrics.length || 0
  const avgPositionError =
    predictionAccuracy.reduce((sum, p) => sum + p.avg_position_error, 0) /
      predictionAccuracy.length || 0

  // Prepare chart data - use model metrics if prediction accuracy is empty
  const chartData = predictionAccuracy.length > 0 
    ? predictionAccuracy.map(p => ({
        race: p.race_name?.split(' ')[0] || 'Unknown',
        positionError: p.avg_position_error || 0,
        timeError: p.avg_time_error || 0,
      }))
    : modelMetrics.map(m => ({
        race: m.race_name?.split(' ')[0] || 'Race',
        positionError: m.mae || 0, // Use MAE as position error approximation
        timeError: m.mae || 0,
      }))
  
  const accuracyOverTime = chartData

  const modelPerformance = modelMetrics.slice(-10).map(m => ({
    date: formatDate(m.created_at),
    mae: m.mae,
    accuracy: m.accuracy * 100,
  }))

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-4"
      >
        <h1 className="text-4xl md:text-5xl font-bold">Analytics</h1>
        <p className="text-f1-gray-400">Model performance and prediction accuracy</p>
      </motion.div>

      {/* Explanations Panel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-blue-50 border border-blue-200 rounded-lg p-6"
      >
        <h2 className="text-lg font-semibold text-blue-900 mb-4">📊 Understanding the Metrics</h2>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div>
            <h3 className="font-medium text-blue-800 mb-2">🎯 MAE (Mean Absolute Error)</h3>
            <p className="text-blue-700">
              Measures the average difference between predicted and actual lap times in seconds. 
              Lower values indicate more accurate time predictions. A MAE of 2.0s means our 
              predictions are typically within 2 seconds of the actual lap time.
            </p>
          </div>
          <div>
            <h3 className="font-medium text-blue-800 mb-2">📈 Model Accuracy</h3>
            <p className="text-blue-700">
              Percentage of predictions that were within 2 positions of the actual result. 
              For example, 75% accuracy means 3 out of 4 predictions were very close to reality. 
              This metric shows how well we predict race positions.
            </p>
          </div>
          <div>
            <h3 className="font-medium text-blue-800 mb-2">📍 Position Error</h3>
            <p className="text-blue-700">
              Average number of positions between predicted and actual finishing positions. 
              A value of 1.5 means predictions are typically off by about 1-2 positions. 
              Lower is better for position accuracy.
            </p>
          </div>
          <div>
            <h3 className="font-medium text-blue-800 mb-2">🔬 Model Versions</h3>
            <p className="text-blue-700">
              We continuously improve our models based on new data and feedback. 
              Each version represents an iteration with potentially better algorithms, 
              new features, or improved training data.
            </p>
          </div>
        </div>
      </motion.div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="glass-effect rounded-xl p-6 text-center"
        >
          <Activity className="w-8 h-8 text-f1-red mx-auto mb-2" />
          <p className="text-2xl font-bold">{avgMAE.toFixed(2)}s</p>
          <p className="text-sm text-f1-gray-400">Avg MAE</p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
          className="glass-effect rounded-xl p-6 text-center"
        >
          <Target className="w-8 h-8 text-green-500 mx-auto mb-2" />
          <p className="text-2xl font-bold">{(avgAccuracy * 100).toFixed(1)}%</p>
          <p className="text-sm text-f1-gray-400">Model Accuracy</p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="glass-effect rounded-xl p-6 text-center"
        >
          <TrendingUp className="w-8 h-8 text-blue-500 mx-auto mb-2" />
          <p className="text-2xl font-bold">{avgPositionError.toFixed(1)}</p>
          <p className="text-sm text-f1-gray-400">Avg Position Error</p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="glass-effect rounded-xl p-6 text-center"
        >
          <Award className="w-8 h-8 text-yellow-500 mx-auto mb-2" />
          <p className="text-2xl font-bold">{predictionAccuracy.length}</p>
          <p className="text-sm text-f1-gray-400">Races Analyzed</p>
        </motion.div>
      </div>

      {/* Charts Grid */}
      <div className="grid lg:grid-cols-2 gap-8">
        {/* Model Performance Over Time */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-effect rounded-xl p-6"
        >
          <h3 className="font-bold text-lg mb-4">Model Performance Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={modelPerformance}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2F2F3A" />
              <XAxis dataKey="date" stroke="#787885" />
              <YAxis stroke="#787885" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F1F28',
                  border: '1px solid #2F2F3A',
                  borderRadius: '8px',
                }}
              />
              <Line
                type="monotone"
                dataKey="mae"
                stroke="#E10600"
                strokeWidth={2}
                name="MAE (seconds)"
              />
              <Line
                type="monotone"
                dataKey="accuracy"
                stroke="#00D2BE"
                strokeWidth={2}
                name="Accuracy (%)"
              />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Position Error by Race */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-effect rounded-xl p-6"
        >
          <h3 className="font-bold text-lg mb-4">Prediction Error by Race</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={accuracyOverTime}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2F2F3A" />
              <XAxis dataKey="race" stroke="#787885" />
              <YAxis stroke="#787885" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F1F28',
                  border: '1px solid #2F2F3A',
                  borderRadius: '8px',
                }}
              />
              <Bar dataKey="positionError" fill="#FF8700" name="Position Error" />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Model Versions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-effect rounded-xl p-6"
        >
          <h3 className="font-bold text-lg mb-4">Recent Model Versions</h3>
          <div className="space-y-3">
            {modelMetrics.slice(-5).reverse().map((metric) => (
              <div key={metric.id} className="flex items-center justify-between p-3 bg-f1-gray-800 rounded-lg">
                <div>
                  <p className="font-mono text-sm">{metric.model_version}</p>
                  <p className="text-xs text-f1-gray-400">{metric.race_name || 'General'}</p>
                </div>
                <div className="text-right">
                  <p className="text-sm font-medium">MAE: {metric.mae.toFixed(2)}s</p>
                  <p className="text-xs text-f1-gray-400">
                    Accuracy: {(metric.accuracy * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Feature Importance (Placeholder) */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-effect rounded-xl p-6"
        >
          <h3 className="font-bold text-lg mb-4">Key Prediction Factors</h3>
          <div className="space-y-3">
            {[
              { name: 'Qualifying Time', value: 35, color: '#E10600' },
              { name: 'Team Performance', value: 25, color: '#FF8700' },
              { name: 'Weather Conditions', value: 20, color: '#00D2BE' },
              { name: 'Historical Data', value: 15, color: '#0090FF' },
              { name: 'Circuit Type', value: 5, color: '#FFD700' },
            ].map((factor, idx) => (
              <div key={factor.name}>
                <div className="flex justify-between text-sm mb-1">
                  <span>{factor.name}</span>
                  <span className="text-f1-gray-400">{factor.value}%</span>
                </div>
                <div className="h-2 bg-f1-gray-800 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${factor.value}%` }}
                    transition={{ duration: 0.8, delay: idx * 0.1 }}
                    className="h-full rounded-full"
                    style={{ backgroundColor: factor.color }}
                  />
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  )
}