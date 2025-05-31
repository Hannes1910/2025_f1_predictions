import { motion } from 'framer-motion'
import { Trophy } from 'lucide-react'

export default function LoadingState() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[400px]">
      <motion.div
        animate={{
          scale: [1, 1.2, 1],
          rotate: [0, 360],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
        className="mb-8"
      >
        <Trophy className="w-16 h-16 text-f1-red" />
      </motion.div>
      <div className="space-y-2 text-center">
        <h3 className="text-xl font-bold">Loading predictions...</h3>
        <p className="text-f1-gray-400">Analyzing race data with AI</p>
      </div>
      <div className="mt-8 flex space-x-2">
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            animate={{
              opacity: [0.3, 1, 0.3],
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              delay: i * 0.2,
            }}
            className="w-3 h-3 bg-f1-red rounded-full"
          />
        ))}
      </div>
    </div>
  )
}

export function LoadingSkeleton() {
  return (
    <div className="space-y-4">
      {[...Array(5)].map((_, i) => (
        <div key={i} className="glass-effect rounded-xl p-6">
          <div className="flex items-start justify-between">
            <div className="flex items-start space-x-4">
              <div className="w-12 h-12 rounded-lg loading-shimmer" />
              <div className="space-y-2">
                <div className="h-6 w-32 rounded loading-shimmer" />
                <div className="h-4 w-24 rounded loading-shimmer" />
              </div>
            </div>
            <div className="space-y-2">
              <div className="h-4 w-20 rounded loading-shimmer" />
              <div className="h-4 w-16 rounded loading-shimmer" />
            </div>
          </div>
          <div className="mt-4 h-1 bg-f1-gray-800 rounded-full" />
        </div>
      ))}
    </div>
  )
}