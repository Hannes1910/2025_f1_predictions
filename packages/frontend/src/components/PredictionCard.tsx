import { motion } from 'framer-motion'
import { TrendingUp, Clock, Award } from 'lucide-react'
import type { Prediction } from '@/types'
import { getTeamColor, getPositionColor, getConfidenceColor } from '@/utils/colors'
import { formatRaceTime, formatConfidence } from '@/utils/format'
import clsx from 'clsx'

interface PredictionCardProps {
  prediction: Prediction
  index: number
  showRaceInfo?: boolean
}

export default function PredictionCard({ prediction, index, showRaceInfo = false }: PredictionCardProps) {
  const teamColor = getTeamColor(prediction.driver_team)
  const positionColor = getPositionColor(prediction.predicted_position)
  const confidenceColor = getConfidenceColor(prediction.confidence)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className="glass-effect rounded-xl p-6 hover:shadow-xl transition-all duration-300 group"
      style={{
        borderColor: teamColor + '40',
      }}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-4">
          {/* Position */}
          <div
            className="w-12 h-12 rounded-lg flex items-center justify-center font-bold text-lg"
            style={{ backgroundColor: positionColor + '20', color: positionColor }}
          >
            {prediction.predicted_position <= 3 && (
              <Award className="w-8 h-8" />
            )}
            <span className={clsx(prediction.predicted_position <= 3 && 'absolute text-f1-black text-sm')}>
              {prediction.predicted_position}
            </span>
          </div>

          {/* Driver Info */}
          <div>
            <h3 className="font-bold text-lg group-hover:text-f1-red transition-colors">
              {prediction.driver_name}
            </h3>
            <div className="flex items-center space-x-3 mt-1">
              <span
                className="text-sm font-medium px-2 py-0.5 rounded"
                style={{ backgroundColor: teamColor + '20', color: teamColor }}
              >
                {prediction.driver_team}
              </span>
              <span className="text-f1-gray-400 text-sm">{prediction.driver_code}</span>
            </div>

            {showRaceInfo && prediction.race_name && (
              <p className="text-sm text-f1-gray-400 mt-2">
                {prediction.race_name} • {prediction.race_circuit}
              </p>
            )}
          </div>
        </div>

        {/* Stats */}
        <div className="text-right space-y-2">
          <div className="flex items-center justify-end space-x-2">
            <Clock className="w-4 h-4 text-f1-gray-400" />
            <span className="font-mono text-sm">{formatRaceTime(prediction.predicted_time)}</span>
          </div>
          <div className="flex items-center justify-end space-x-2">
            <TrendingUp className="w-4 h-4" style={{ color: confidenceColor }} />
            <span
              className="text-sm font-medium"
              style={{ color: confidenceColor }}
            >
              {formatConfidence(prediction.confidence)}
            </span>
          </div>
        </div>
      </div>

      {/* Progress bar for confidence */}
      <div className="mt-4 h-1 bg-f1-gray-800 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${prediction.confidence * 100}%` }}
          transition={{ duration: 0.8, delay: index * 0.05 + 0.3 }}
          className="h-full rounded-full"
          style={{ backgroundColor: confidenceColor }}
        />
      </div>
    </motion.div>
  )
}