export const teamColors: Record<string, string> = {
  'Red Bull': '#0600EF',
  'McLaren': '#FF8700',
  'Ferrari': '#DC0000',
  'Mercedes': '#00D2BE',
  'Aston Martin': '#006F62',
  'Alpine': '#0090FF',
  'Williams': '#005AFF',
  'Racing Bulls': '#2B4562',
  'Kick Sauber': '#00E701',
  'Haas': '#B6BABD',
}

export const getTeamColor = (team: string): string => {
  return teamColors[team] || '#787885'
}

export const getPositionColor = (position: number): string => {
  if (position === 1) return '#FFD700'
  if (position === 2) return '#C0C0C0'
  if (position === 3) return '#CD7F32'
  if (position <= 10) return '#00D2BE'
  return '#787885'
}

export const getConfidenceColor = (confidence: number): string => {
  if (confidence >= 0.8) return '#00E701'
  if (confidence >= 0.6) return '#FFD700'
  if (confidence >= 0.4) return '#FF8700'
  return '#E10600'
}