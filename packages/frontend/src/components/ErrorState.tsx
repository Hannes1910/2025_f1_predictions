import { AlertTriangle } from 'lucide-react'
import { motion } from 'framer-motion'

interface ErrorStateProps {
  message?: string
  onRetry?: () => void
}

export default function ErrorState({ message = 'Something went wrong', onRetry }: ErrorStateProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex flex-col items-center justify-center min-h-[400px] text-center"
    >
      <div className="w-20 h-20 bg-f1-red/20 rounded-full flex items-center justify-center mb-6">
        <AlertTriangle className="w-10 h-10 text-f1-red" />
      </div>
      <h3 className="text-2xl font-bold mb-2">Oops!</h3>
      <p className="text-f1-gray-400 mb-6 max-w-md">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="px-6 py-3 bg-f1-red text-white font-medium rounded-lg hover:bg-red-700 transition-colors"
        >
          Try Again
        </button>
      )}
    </motion.div>
  )
}