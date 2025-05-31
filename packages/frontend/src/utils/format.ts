import { format, formatDistance, parseISO } from 'date-fns'

export const formatDate = (date: string): string => {
  return format(parseISO(date), 'MMM d, yyyy')
}

export const formatDateTime = (date: string): string => {
  return format(parseISO(date), 'MMM d, yyyy HH:mm')
}

export const formatRaceTime = (seconds: number): string => {
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = (seconds % 60).toFixed(3)
  return `${minutes}:${remainingSeconds.padStart(6, '0')}`
}

export const formatPosition = (position: number): string => {
  const suffix = ['th', 'st', 'nd', 'rd']
  const v = position % 100
  return position + (suffix[(v - 20) % 10] || suffix[v] || suffix[0])
}

export const formatConfidence = (confidence: number): string => {
  return `${(confidence * 100).toFixed(0)}%`
}

export const formatTimeFromNow = (date: string): string => {
  return formatDistance(parseISO(date), new Date(), { addSuffix: true })
}

export const formatTime = (seconds: number): string => {
  const minutes = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  const ms = Math.floor((seconds % 1) * 1000)
  return `${minutes}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`
}

export const formatPercent = (value: number): string => {
  return `${(value * 100).toFixed(0)}%`
}