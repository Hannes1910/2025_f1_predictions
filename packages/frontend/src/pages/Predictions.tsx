import { useQuery } from 'react-query'
import { motion } from 'framer-motion'
import { Calendar, MapPin, Brain, RefreshCw } from 'lucide-react'
import { predictionsApi } from '@/services/api'
import PredictionCard from '@/components/PredictionCard'
import { LoadingSkeleton } from '@/components/LoadingState'
import ErrorState from '@/components/ErrorState'

export default function Predictions() {
  const { data, isLoading, error, refetch } = useQuery(
    'latest-predictions',
    predictionsApi.getLatest,
    {
      refetchInterval: 60000, // Refetch every minute
    }
  )

  if (isLoading) {
    return (
      <div className="space-y-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold mb-4">Latest Predictions</h1>
          <p className="text-f1-gray-400">AI-powered race forecasts</p>
        </div>
        <LoadingSkeleton />
      </div>
    )
  }

  if (error) {
    return <ErrorState message="Failed to load predictions" onRetry={refetch} />
  }

  const predictions = data?.predictions || []
  const raceInfo = predictions[0]

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-4"
      >
        <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-f1-red to-f1-white bg-clip-text text-transparent">
          Race Predictions
        </h1>
        {raceInfo && (
          <div className="flex flex-wrap items-center justify-center gap-4 text-f1-gray-300">
            <div className="flex items-center space-x-2">
              <Calendar className="w-4 h-4" />
              <span>{raceInfo.race_name}</span>
            </div>
            <div className="flex items-center space-x-2">
              <MapPin className="w-4 h-4" />
              <span>{raceInfo.race_circuit}</span>
            </div>
            <div className="flex items-center space-x-2">
              <Brain className="w-4 h-4" />
              <span>{raceInfo.model_version}</span>
            </div>
          </div>
        )}
      </motion.div>

      {/* Refresh Button */}
      <div className="flex justify-end">
        <button
          onClick={() => refetch()}
          className="flex items-center space-x-2 px-4 py-2 bg-f1-gray-800 hover:bg-f1-gray-700 rounded-lg transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Refresh</span>
        </button>
      </div>

      {/* Predictions Grid */}
      {predictions.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-1">
          {predictions.map((prediction, index) => (
            <PredictionCard
              key={prediction.id}
              prediction={prediction}
              index={index}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <p className="text-f1-gray-400">No predictions available yet</p>
        </div>
      )}

      {/* Info Section */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="mt-12 grid md:grid-cols-3 gap-6"
      >
        <div className="glass-effect rounded-xl p-6 text-center">
          <h3 className="font-bold text-lg mb-2">Machine Learning</h3>
          <p className="text-f1-gray-400 text-sm">
            Powered by Gradient Boosting models trained on historical F1 data
          </p>
        </div>
        <div className="glass-effect rounded-xl p-6 text-center">
          <h3 className="font-bold text-lg mb-2">Real-time Data</h3>
          <p className="text-f1-gray-400 text-sm">
            Integrates qualifying times, weather conditions, and team performance
          </p>
        </div>
        <div className="glass-effect rounded-xl p-6 text-center">
          <h3 className="font-bold text-lg mb-2">Confidence Scores</h3>
          <p className="text-f1-gray-400 text-sm">
            Each prediction includes a confidence rating based on model certainty
          </p>
        </div>
      </motion.div>
    </div>
  )
}