import { Routes, Route } from 'react-router-dom'
import Layout from '@/components/Layout'
import Predictions from '@/pages/Predictions'
import Calendar from '@/pages/Calendar'
import Drivers from '@/pages/Drivers'
import Analytics from '@/pages/Analytics'
import RaceDetails from '@/pages/RaceDetails'
import Historical from '@/pages/Historical'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Predictions />} />
        <Route path="calendar" element={<Calendar />} />
        <Route path="drivers" element={<Drivers />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="historical" element={<Historical />} />
        <Route path="race/:raceId" element={<RaceDetails />} />
      </Route>
    </Routes>
  )
}

export default App