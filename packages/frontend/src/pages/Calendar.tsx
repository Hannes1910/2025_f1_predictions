import { useQuery } from 'react-query'
import { motion } from 'framer-motion'
import { Calendar as CalendarIcon, MapPin, Trophy, Clock, ChevronRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import { racesApi } from '@/services/api'
import LoadingState from '@/components/LoadingState'
import ErrorState from '@/components/ErrorState'
import { formatDate } from '@/utils/format'
import clsx from 'clsx'

export default function Calendar() {
  const { data, isLoading, error, refetch } = useQuery(
    ['races', 2025],
    () => racesApi.getAll(2025)
  )

  if (isLoading) return <LoadingState />
  if (error) return <ErrorState message="Failed to load calendar" onRetry={refetch} />

  const races = data?.races || []
  const completedRaces = races.filter(r => r.status === 'completed')
  const upcomingRaces = races.filter(r => r.status === 'upcoming' || r.status === 'today')

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-4"
      >
        <h1 className="text-4xl md:text-5xl font-bold">2025 Season Calendar</h1>
        <p className="text-f1-gray-400">
          {races.length} races • {completedRaces.length} completed • {upcomingRaces.length} upcoming
        </p>
      </motion.div>

      {/* Season Progress */}
      <div className="glass-effect rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-bold">Season Progress</h3>
          <span className="text-f1-gray-400">
            {Math.round((completedRaces.length / races.length) * 100)}% Complete
          </span>
        </div>
        <div className="h-4 bg-f1-gray-800 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${(completedRaces.length / races.length) * 100}%` }}
            transition={{ duration: 1, ease: 'easeOut' }}
            className="h-full bg-gradient-to-r from-f1-red to-orange-500 rounded-full"
          />
        </div>
      </div>

      {/* Race Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {races.map((race, index) => (
          <motion.div
            key={race.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
          >
            <Link
              to={`/race/${race.id}`}
              className={clsx(
                'block glass-effect rounded-xl p-6 hover:shadow-xl transition-all duration-300 group',
                race.status === 'today' && 'ring-2 ring-f1-red',
                race.status === 'completed' && 'opacity-75'
              )}
            >
              {/* Race Header */}
              <div className="flex items-start justify-between mb-4">
                <div>
                  <div className="flex items-center space-x-2 text-f1-gray-400 text-sm mb-1">
                    <span>Round {race.round}</span>
                    {race.status === 'today' && (
                      <span className="px-2 py-0.5 bg-f1-red text-white text-xs rounded-full">
                        TODAY
                      </span>
                    )}
                  </div>
                  <h3 className="font-bold text-lg group-hover:text-f1-red transition-colors">
                    {race.name}
                  </h3>
                </div>
                <ChevronRight className="w-5 h-5 text-f1-gray-400 group-hover:text-f1-red transition-colors" />
              </div>

              {/* Race Info */}
              <div className="space-y-2 text-sm">
                <div className="flex items-center space-x-2 text-f1-gray-300">
                  <MapPin className="w-4 h-4" />
                  <span>{race.circuit}</span>
                </div>
                <div className="flex items-center space-x-2 text-f1-gray-300">
                  <CalendarIcon className="w-4 h-4" />
                  <span>{formatDate(race.date)}</span>
                </div>
                {(race.prediction_count ?? 0) > 0 && (
                  <div className="flex items-center space-x-2 text-green-400">
                    <Trophy className="w-4 h-4" />
                    <span>Predictions available</span>
                  </div>
                )}
              </div>

              {/* Status Badge */}
              <div className="mt-4">
                {race.status === 'completed' ? (
                  <div className="flex items-center space-x-2 text-f1-gray-400">
                    <div className="w-2 h-2 bg-f1-gray-400 rounded-full" />
                    <span className="text-sm">Completed</span>
                  </div>
                ) : race.status === 'today' ? (
                  <div className="flex items-center space-x-2 text-f1-red">
                    <div className="w-2 h-2 bg-f1-red rounded-full animate-pulse" />
                    <span className="text-sm font-medium">Race Day</span>
                  </div>
                ) : (
                  <div className="flex items-center space-x-2 text-blue-400">
                    <Clock className="w-4 h-4" />
                    <span className="text-sm">Upcoming</span>
                  </div>
                )}
              </div>
            </Link>
          </motion.div>
        ))}
      </div>
    </div>
  )
}